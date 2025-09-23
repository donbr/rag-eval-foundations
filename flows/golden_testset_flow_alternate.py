# flows/golden_testset_flow.py
# Prefect 2.x flow that maps tasks.yaml phases/tasks 1:1 and executes shell commands with retries.
# Usage:
#   uv run prefect deploy --name golden-testset
#   uv run prefect run -p flows/golden_testset_flow.py:orchestrate --params '{"only_phase":"phase4"}'
#   uv run python flows/golden_testset_flow.py --help

from __future__ import annotations

import os
import sys
import time
import yaml
import shlex
import signal
import typing as t
import subprocess
from pathlib import Path
from dataclasses import dataclass, field

from prefect import flow, task, get_run_logger
from prefect.tasks import task_input_hash
from prefect.states import State
from prefect.futures import PrefectFuture
from prefect.exceptions import PrefectException

# --------------------
# Data models
# --------------------

@dataclass
class TaskSpec:
    id: str
    name: str
    run: t.List[str] = field(default_factory=list)
    verify: t.List[str] = field(default_factory=list)
    perf_budget_ms: t.Optional[int] = None
    accepts: t.List[str] = field(default_factory=list)
    artifacts: t.List[str] = field(default_factory=list)

@dataclass
class PhaseSpec:
    id: str
    name: str
    depends_on: t.List[str] = field(default_factory=list)
    tasks: t.List[TaskSpec] = field(default_factory=list)

@dataclass
class Plan:
    schema: str
    version: str
    project: str
    description: str
    providers: dict
    global_: dict
    phases: t.List[PhaseSpec]

# --------------------
# Utilities
# --------------------

def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def _parse_plan(doc: dict) -> Plan:
    phases: t.List[PhaseSpec] = []
    for p in doc.get("phases", []):
        tasks = []
        for tsk in p.get("tasks", []):
            tasks.append(
                TaskSpec(
                    id=tsk["id"],
                    name=tsk.get("name", tsk["id"]),
                    run=list(tsk.get("run", []) or []),
                    verify=list(tsk.get("verify", []) or []),
                    perf_budget_ms=tsk.get("perf_budget_ms"),
                    accepts=list(tsk.get("accepts", []) or []),
                    artifacts=list(tsk.get("artifacts", []) or []),
                )
            )
        phases.append(
            PhaseSpec(
                id=p["id"],
                name=p.get("name", p["id"]),
                depends_on=list(p.get("depends_on", []) or []),
                tasks=tasks,
            )
        )
    return Plan(
        schema=doc.get("schema", ""),
        version=str(doc.get("version", "")),
        project=doc.get("project", ""),
        description=doc.get("description", ""),
        providers=doc.get("providers", {}),
        global_=doc.get("global", {}),
        phases=phases,
    )

def _apply_globals(env: dict, plan: Plan) -> None:
    # export env (non-secret)
    for k, v in (plan.global_.get("env") or {}).items():
        if isinstance(v, str):
            # naive ${VAR} expansion using current process env
            os.environ[k] = os.path.expandvars(v)
        else:
            os.environ[k] = str(v)
    # ensure secrets exist in environment
    missing = []
    for s in (plan.global_.get("secrets") or []):
        if os.environ.get(s) is None:
            missing.append(s)
    if missing:
        raise RuntimeError(f"Missing required secrets in environment: {', '.join(missing)}")

def _format_seconds(ms: int | float) -> str:
    return f"{ms/1000:.3f}s" if ms else "n/a"

# --------------------
# Shell execution task
# --------------------

@task(retries=1, retry_delay_seconds=5, cache_key_fn=task_input_hash)
def run_commands(cmds: t.List[str], cwd: str | None = None, env: dict | None = None) -> t.Tuple[int, str, str, float]:
    """
    Run a list of shell commands as a single step; fail-fast on first non-zero.
    Returns (exitcode, last_stdout, last_stderr, elapsed_ms).
    """
    logger = get_run_logger()
    start = time.perf_counter()
    last_out, last_err = "", ""
    code = 0
    for c in cmds:
        pretty = c if len(c) < 180 else c[:177] + "..."
        logger.info(f"$ {pretty}")
        proc = subprocess.Popen(
            c, cwd=cwd, env={**os.environ, **(env or {})},
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, preexec_fn=os.setsid
        )
        out_b, err_b = proc.communicate()
        code = proc.returncode
        last_out, last_err = out_b.decode(errors="ignore"), err_b.decode(errors="ignore")
        logger.info(last_out.strip())
        if code != 0:
            logger.error(last_err.strip())
            # attempt to kill process group if something leaked
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except Exception:
                pass
            raise PrefectException(f"Command failed (exit {code}): {c}\n{last_err}")
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    logger.info(f"Step completed in {_format_seconds(elapsed_ms)}")
    return code, last_out, last_err, elapsed_ms

# --------------------
# Single task runner
# --------------------

@flow(name="golden_testset_task")
def run_task(task_spec: TaskSpec, workdir: str | None = None) -> dict:
    logger = get_run_logger()
    logger.info(f"▶ Running task: {task_spec.id} — {task_spec.name}")

    # RUN commands
    if task_spec.run:
        _, _, _, elapsed_ms = run_commands.submit(task_spec.run, cwd=workdir).result()
        if task_spec.perf_budget_ms and elapsed_ms > task_spec.perf_budget_ms:
            raise PrefectException(
                f"Perf budget exceeded for {task_spec.id}: "
                f"{_format_seconds(elapsed_ms)} > {_format_seconds(task_spec.perf_budget_ms)}"
            )
    else:
        logger.warning("No 'run:' commands defined; skipping RUN step.")

    # VERIFY commands
    if task_spec.verify:
        run_commands.submit(task_spec.verify, cwd=workdir).result()
    else:
        logger.info("No 'verify:' commands defined; skipping VERIFY step.")

    # ACCEPTS (human-readable assertions; log for visibility)
    for cond in task_spec.accepts:
        logger.info(f"[accepts] {cond}")

    # ARTIFACTS (log presence)
    missing_artifacts = [p for p in task_spec.artifacts if not Path(p).exists()]
    if missing_artifacts:
        logger.warning(f"Declared artifacts missing: {missing_artifacts}")

    logger.info(f"✅ Task complete: {task_spec.id}")
    return {
        "task_id": task_spec.id,
        "elapsed_ms": elapsed_ms if task_spec.run else 0.0,
        "artifacts": task_spec.artifacts,
    }

# --------------------
# Phase runner
# --------------------

@flow(name="golden_testset_phase")
def run_phase(phase_spec: PhaseSpec, workdir: str | None = None) -> dict:
    logger = get_run_logger()
    logger.info(f"=== PHASE {phase_spec.id}: {phase_spec.name} ===")
    results = []
    for tsk in phase_spec.tasks:
        res = run_task.submit(tsk, workdir=workdir).result()
        results.append(res)
    logger.info(f"=== PHASE {phase_spec.id} complete ===")
    return {"phase_id": phase_spec.id, "tasks": results}

# --------------------
# Orchestrator
# --------------------

@flow(name="golden_testset_orchestrator")
def orchestrate(
    tasks_yaml_path: str = "tasks.yaml",
    workdir: str | None = None,
    only_phase: str | None = None,
    only_task: str | None = None,
) -> dict:
    """
    Orchestrate phases/tasks 1:1 from tasks.yaml.
    Params:
      - tasks_yaml_path: path to normalized tasks.yaml
      - workdir: working directory for command execution (repo root)
      - only_phase: if set, run only this phase id
      - only_task: if set (with only_phase), run only this task id
    """
    logger = get_run_logger()
    doc = _load_yaml(Path(tasks_yaml_path))
    plan = _parse_plan(doc)

    # apply global env + secrets
    _apply_globals(os.environ, plan)
    logger.info(f"Project: {plan.project}  Version: {plan.version}")
    logger.info(f"Schema: {plan.schema}  Description: {plan.description}")

    # simple dependency check (phases listed in order; we’ll trust tasks.yaml topology)
    phase_index = {p.id: idx for idx, p in enumerate(plan.phases)}
    for p in plan.phases:
        for dep in p.depends_on:
            if phase_index.get(dep, -1) >= phase_index[p.id]:
                raise RuntimeError(f"Phase ordering invalid: {p.id} depends on {dep} which appears after it")

    outputs = []
    for p in plan.phases:
        if only_phase and p.id != only_phase:
            continue

        if only_task:
            # run just the single requested task inside the phase
            target = next((t for t in p.tasks if t.id == only_task), None)
            if not target:
                raise RuntimeError(f"Task {only_task} not found in phase {p.id}")
            res = run_task.submit(target, workdir=workdir).result()
            outputs.append({"phase": p.id, "tasks": [res]})
            break
        else:
            res = run_phase.submit(p, workdir=workdir).result()
            outputs.append(res)

    logger.info("🎉 Orchestration complete.")
    return {"outputs": outputs}

# --------------------
# CLI shim
# --------------------

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Run golden testset Prefect orchestrator.")
    ap.add_argument("--tasks", default="tasks.yaml", help="Path to tasks.yaml")
    ap.add_argument("--workdir", default=".", help="Working directory (repo root)")
    ap.add_argument("--only-phase", default=None, help="Run only this phase id")
    ap.add_argument("--only-task", default=None, help="Run only this task id (requires --only-phase)")
    args = ap.parse_args()

    # Run as plain Python (no Prefect server required)
    state: State = orchestrate.serve if False else None  # keep mypy quiet

    orchestrate(tasks_yaml_path=args.tasks, workdir=args.workdir,
                only_phase=args.only_phase, only_task=args.only_task)
