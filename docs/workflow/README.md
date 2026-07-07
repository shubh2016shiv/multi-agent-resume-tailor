# Resume Tailor Workflow Architecture

This folder explains the Resume Tailor multi-agent workflow as a system design.
Each document focuses on one architectural concern so readers can understand the
project without loading every source file into their head at once.

The current system is a deterministic workflow MAS: LangGraph owns state and
routing, while CrewAI agents work as bounded specialists inside graph nodes.
The architecture is designed for traceable resume tailoring, not open-ended
agent conversation.

## Reading Order

1. [Agent Roles](agent-roles.md)
2. [Orchestration Graph](orchestration-graph.md)
3. [Tool Contracts](tool-contracts.md)
4. [State Management](state-management.md)
5. [Memory Boundaries](memory-boundaries.md)
6. [Retry Strategy](retry-strategy.md)
7. [Idempotency](idempotency.md)
8. [Human In The Loop](human-in-the-loop.md)
9. [Evaluation](evaluation.md)
10. [Observability](observability.md)
11. [Token Cost Reduction](token-cost-reduction.md)
12. [Failure Handling](failure-handling.md)

## System Shape

```text
        documents
            |
            v
    +----------------+
    | LangGraph flow |
    +----------------+
            |
            +--> CrewAI section agents
            |
            +--> deterministic review tools
            |
            +--> checkpoint + HITL resume
            |
            +--> renderable artifacts
```

The important design choice is separation of responsibility. Agents draft or
interpret content, but the graph decides when a stage is allowed to run. Tools
and evaluators provide deterministic pressure around truthfulness, ATS structure,
renderability, and final quality.

