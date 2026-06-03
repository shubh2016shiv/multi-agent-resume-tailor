#!/usr/bin/env bash
# graphify-full.sh — Full graphify build with LLM (community naming + report + visualization)
# Usage: ./scripts/graphify-full.sh
#        OPENAI_API_KEY=sk-... ./scripts/graphify-full.sh
#        ./scripts/graphify-full.sh --model gpt-4o
set -euo pipefail

MODEL="${GRAPHIFY_MODEL:-gpt-4o}"

# Parse optional --model flag
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model) MODEL="$2"; shift 2 ;;
        --model=*) MODEL="${1#*=}"; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Resolve API key: explicit env > .env file > prompt
if [ -z "${OPENAI_API_KEY:-}" ]; then
    if [ -f .env ]; then
        # Extract key with grep + sed — tolerates CRLF line endings that break `source`
        OPENAI_API_KEY="$(grep '^OPENAI_API_KEY=' .env 2>/dev/null | sed 's/^OPENAI_API_KEY=//; s/["'"'"']//g; s/\r$//' | head -1)"
        [ -n "$OPENAI_API_KEY" ] && export OPENAI_API_KEY
    fi
fi

if [ -z "${OPENAI_API_KEY:-}" ]; then
    printf "Enter your OpenAI API key: "
    read -rs OPENAI_API_KEY
    echo
    if [ -z "$OPENAI_API_KEY" ]; then
        echo "ERROR: No API key provided. Set OPENAI_API_KEY env var or create a .env file."
        exit 1
    fi
    export OPENAI_API_KEY
fi

echo "Running full graphify build (AST + LLM report + visualization)..."
echo "Backend: openai | Model: $MODEL"
echo ""

uv run graphify extract . --backend openai --model "$MODEL"

echo ""
echo "Done. Output in graphify-out/:"
echo "  graph.json         — queryable knowledge graph"
echo "  GRAPH_REPORT.md    — human-readable architecture summary"
echo "  graph.html         — interactive visualization"
