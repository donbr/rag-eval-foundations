# flows/golden_testset_flow.py
# Prefect 3.x orchestrator for tasks.yaml (1:1 phases/tasks).
# Run:
#   uv run python flows/golden_testset_flow.py --tasks tasks.yaml --only-phase phase4
#   # or as a library flow
#   # from flows.golden_testset_flow import orchestrate; orchestrate()

from __future__ import annotations

import os
import time
import yaml
import signal
import typing as t
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from prefect import flow, task
from prefect.exceptions import PrefectException
from prefect.logging import get_run_logger
from prefect.tasks import task_input_hash  # cache key helper (still available in v3)

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

def _apply_globals(plan: Plan) -> None:
    # export env (non-secret)
    for k, v in (plan.global_.get("env") or {}).items():
        os.environ[k] = os.path.expandvars(v) if isinstance(v, str) else str(v)

    # ensure secrets exist in environment
    missing = [s for s in (plan.global_.get("secrets") or []) if os.environ.get(s) is None]
    if missing:
        raise RuntimeError(f"Missing required secrets in environment: {', '.join(missing)}")

def _fmt_secs(ms: float | None) -> str:
    return f"{(ms or 0)/1000:.3f}s"

# --------------------
# Shell exec task (Prefect 3)
# --------------------

@task(retries=1, retry_delay_seconds=5, cache_key_fn=task_input_hash)
def run_commands(cmds: t.List[str], cwd: str | None = None, extra_env: dict | None = None) -> dict:
    """
    Run a list of shell commands; fail-fast on first non-zero exit.
    Returns a dict with code/out/err/elapsed_ms.
    """
    logger = get_run_logger()
    start = time.perf_counter()
    last_out, last_err, code = "", "", 0

    for c in cmds:
        pretty = c if len(c) < 180 else c[:177] + "..."
        logger.info(f"$ {pretty}")
        proc = subprocess.Popen(
            c,
            cwd=cwd,
            env={**os.environ, **(extra_env or {})},
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=True,
            preexec_fn=os.setsid,
        )
        out_b, err_b = proc.communicate()
        code = proc.returncode
        last_out, last_err = out_b.decode(errors="ignore"), err_b.decode(errors="ignore")
        if last_out.strip():
            logger.info(last_out.strip())
        if code != 0:
            if last_err.strip():
                logger.error(last_err.strip())
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except Exception:
                pass
            raise PrefectException(f"Command failed (exit {code}): {c}\n{last_err}")

    elapsed_ms = (time.perf_counter() - start) * 1000.0
    logger.info(f"Step completed in {_fmt_secs(elapsed_ms)}")
    return {"exit_code": code, "stdout": last_out, "stderr": last_err, "elapsed_ms": elapsed_ms}

# --------------------
# Single task runner
# --------------------

@flow(name="golden_testset_task")
def run_task(task_spec: TaskSpec, workdir: str | None = None) -> dict:
    logger = get_run_logger()
    logger.info(f"▶ Running task: {task_spec.id} — {task_spec.name}")

    elapsed = 0.0

    # RUN commands — Prefect 3 requires explicit resolution of futures before returning
    # (we call .result() implicitly by awaiting the sub-flow return). :contentReference[oaicite:1]{index=1}
    if task_spec.run:
        run_res = run_commands.submit(task_spec.run, cwd=workdir).result()
        elapsed = float(run_res.get("elapsed_ms", 0.0))
        if task_spec.perf_budget_ms and elapsed > task_spec.perf_budget_ms:
            raise PrefectException(
                f"Perf budget exceeded for {task_spec.id}: {_fmt_secs(elapsed)} > {_fmt_secs(task_spec.perf_budget_ms)}"
            )
    else:
        logger.warning("No 'run:' commands defined; skipping RUN step.")

    # VERIFY commands
    if task_spec.verify:
        run_commands.submit(task_spec.verify, cwd=workdir).result()
    else:
        logger.info("No 'verify:' commands defined; skipping VERIFY step.")

    for cond in task_spec.accepts:
        logger.info(f"[accepts] {cond}")

    missing_artifacts = [p for p in task_spec.artifacts if not Path(p).exists()]
    if missing_artifacts:
        logger.warning(f"Declared artifacts missing: {missing_artifacts}")

    logger.info(f"✅ Task complete: {task_spec.id}")
    return {"task_id": task_spec.id, "elapsed_ms": elapsed, "artifacts": task_spec.artifacts}

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
# Orchestrator (top-level flow)
# --------------------

@flow(name="golden_testset_orchestrator")
def orchestrate(
    tasks_yaml_path: str = "tasks.yaml",
    workdir: str | None = ".",
    only_phase: str | None = None,
    only_task: str | None = None,
) -> dict:
    """
    Orchestrate phases/tasks 1:1 from tasks.yaml.
    Note: In Prefect 3, ensure all futures are resolved before returning (we do). :contentReference[oaicite:2]{index=2}
    """
    logger = get_run_logger()
    plan_doc = _load_yaml(Path(tasks_yaml_path))
    plan = _parse_plan(plan_doc)

    _apply_globals(plan)
    logger.info(f"Project: {plan.project}  Version: {plan.version}")
    logger.info(f"Schema: {plan.schema}  Description: {plan.description}")

    # Validate declared ordering vs. depends_on
    index = {p.id: i for i, p in enumerate(plan.phases)}
    for p in plan.phases:
        for dep in p.depends_on:
            if index.get(dep, -1) >= index[p.id]:
                raise RuntimeError(f"Phase ordering invalid: {p.id} depends on {dep} which appears after it")

    outputs = []
    for p in plan.phases:
        if only_phase and p.id != only_phase:
            continue

        if only_task:
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

    ap = argparse.ArgumentParser(description="Run golden testset Prefect 3 orchestrator.")
    ap.add_argument("--tasks", default="tasks.yaml", help="Path to tasks.yaml")
    ap.add_argument("--workdir", default=".", help="Working directory (repo root)")
    ap.add_argument("--only-phase", default=None, help="Run only this phase id")
    ap.add_argument("--only-task", default=None, help="Run only this task id (requires --only-phase)")
    args = ap.parse_args()

    orchestrate(tasks_yaml_path=args.tasks, workdir=args.workdir,
                only_phase=args.only_phase, only_task=args.only_task)
