#!/bin/bash
# Helper script to check RAG evaluation service status and detect conflicts

echo "RAG Evaluation Services Status Check"
echo "===================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default ports (can be overridden by environment variables)
POSTGRES_PORT=${POSTGRES_PORT:-6024}
PHOENIX_UI_PORT=${PHOENIX_UI_PORT:-6006}
PHOENIX_OTLP_PORT=${PHOENIX_OTLP_PORT:-4317}

echo -e "\nChecking for RAG evaluation containers..."
echo "----------------------------------------"

# Check for our specific containers
PGVECTOR_RUNNING=$(docker ps --filter "name=rag-eval-pgvector" --format "{{.Names}}" 2>/dev/null)
PHOENIX_RUNNING=$(docker ps --filter "name=rag-eval-phoenix" --format "{{.Names}}" 2>/dev/null)

if [ -n "$PGVECTOR_RUNNING" ]; then
    echo -e "${GREEN}✓${NC} PostgreSQL (rag-eval-pgvector) is running"
else
    echo -e "${RED}✗${NC} PostgreSQL (rag-eval-pgvector) is not running"
fi

if [ -n "$PHOENIX_RUNNING" ]; then
    echo -e "${GREEN}✓${NC} Phoenix (rag-eval-phoenix) is running"
else
    echo -e "${RED}✗${NC} Phoenix (rag-eval-phoenix) is not running"
fi

echo -e "\nChecking for port conflicts..."
echo "------------------------------"

# Function to check if a port is in use
check_port() {
    local port=$1
    local service=$2
    local port_check=$(docker ps --format "table {{.Names}}\t{{.Ports}}" | grep -E ":$port->" | grep -v "rag-eval-")
    
    if [ -n "$port_check" ]; then
        echo -e "${RED}✗${NC} Port $port ($service) is in use by another container:"
        echo "  $port_check"
        return 1
    else
        # Check if our service is using the port
        local our_service=$(docker ps --format "table {{.Names}}\t{{.Ports}}" | grep -E ":$port->" | grep "rag-eval-")
        if [ -n "$our_service" ]; then
            echo -e "${GREEN}✓${NC} Port $port ($service) is in use by our service"
        else
            echo -e "${YELLOW}○${NC} Port $port ($service) is available"
        fi
        return 0
    fi
}

# Check each port
check_port $POSTGRES_PORT "PostgreSQL"
POSTGRES_CONFLICT=$?

check_port $PHOENIX_UI_PORT "Phoenix UI"
PHOENIX_UI_CONFLICT=$?

check_port $PHOENIX_OTLP_PORT "Phoenix OTLP"
PHOENIX_OTLP_CONFLICT=$?

echo -e "\nChecking network..."
echo "------------------"
NETWORK_EXISTS=$(docker network ls --filter "name=rag-eval-network" --format "{{.Name}}" 2>/dev/null)
if [ -n "$NETWORK_EXISTS" ]; then
    echo -e "${GREEN}✓${NC} Network 'rag-eval-network' exists"
else
    echo -e "${YELLOW}○${NC} Network 'rag-eval-network' does not exist (will be created)"
fi

echo -e "\nChecking volumes..."
echo "------------------"
VOLUME_EXISTS=$(docker volume ls --filter "name=rag_eval_postgres_data" --format "{{.Name}}" 2>/dev/null)
if [ -n "$VOLUME_EXISTS" ]; then
    echo -e "${GREEN}✓${NC} Volume 'rag_eval_postgres_data' exists"
else
    echo -e "${YELLOW}○${NC} Volume 'rag_eval_postgres_data' does not exist (will be created)"
fi

# Summary and recommendations
echo -e "\nSummary"
echo "======="

if [ $POSTGRES_CONFLICT -eq 1 ] || [ $PHOENIX_UI_CONFLICT -eq 1 ] || [ $PHOENIX_OTLP_CONFLICT -eq 1 ]; then
    echo -e "${RED}Port conflicts detected!${NC}"
    echo -e "\nTo resolve conflicts, you can:"
    echo "1. Stop the conflicting containers"
    echo "2. Use different ports by setting environment variables:"
    echo "   export POSTGRES_PORT=6025"
    echo "   export PHOENIX_UI_PORT=6007"
    echo "   export PHOENIX_OTLP_PORT=4318"
    echo "3. Create a .env file with custom ports"
elif [ -n "$PGVECTOR_RUNNING" ] && [ -n "$PHOENIX_RUNNING" ]; then
    echo -e "${GREEN}All services are running!${NC}"
    echo -e "\nServices available at:"
    echo "- PostgreSQL: localhost:$POSTGRES_PORT"
    echo "- Phoenix UI: http://localhost:$PHOENIX_UI_PORT"
    echo "- Phoenix OTLP: localhost:$PHOENIX_OTLP_PORT"
else
    echo -e "${YELLOW}Services are not running.${NC}"
    echo -e "\nTo start services:"
    echo "  docker-compose up -d"
fi

echo -e "\nTo view all Docker containers using project labels:"
echo "  docker ps --filter 'label=project=rag-eval-foundations'"