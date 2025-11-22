#!/bin/bash
# Pre-commit hook for pyscn code quality check
# This script runs pyscn check to ensure code quality before commit

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Running pyscn code quality check...${NC}"

# Try different methods to run pyscn
if command -v pyscn &> /dev/null; then
    echo "Using installed pyscn..."
    pyscn check . || {
        echo -e "${RED}pyscn check failed. Please fix code quality issues before committing.${NC}"
        echo "Run 'pyscn analyze .' for detailed report."
        exit 1
    }
elif command -v uvx &> /dev/null; then
    echo "Using uvx to run pyscn..."
    uvx pyscn check . || {
        echo -e "${RED}pyscn check failed. Please fix code quality issues before committing.${NC}"
        echo "Run 'uvx pyscn analyze .' for detailed report."
        exit 1
    }
elif command -v pipx &> /dev/null; then
    echo "Installing pyscn via pipx..."
    pipx install pyscn && pyscn check . || {
        echo -e "${RED}pyscn check failed. Please fix code quality issues before committing.${NC}"
        exit 1
    }
else
    echo -e "${YELLOW}Warning: pyscn not found. Install it with: pipx install pyscn${NC}"
    echo -e "${YELLOW}Skipping pyscn check for this commit.${NC}"
    exit 0  # Don't fail commit if pyscn is not installed
fi

echo -e "${GREEN}pyscn check passed!${NC}"
exit 0

