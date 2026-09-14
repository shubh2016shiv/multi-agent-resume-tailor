"""Drift-detection test for the manual msgpack allowlist.

Why this test exists instead of relying on LANGGRAPH_STRICT_MSGPACK:
--------------------------------------------------------------------
LangGraph can auto-derive this exact allowlist from the compiled graph's
schema when the ``LANGGRAPH_STRICT_MSGPACK`` environment variable is set
*before* ``langgraph.checkpoint.serde._msgpack`` is first imported anywhere
in the process (that flag is read once, at import time, into a module-level
constant -- not re-read per call).

Several modules in this repo import LangGraph directly, reached from multiple
entry points (CLI, web app, tests) that can import them in different orders.
Guaranteeing the env var is set before the *first* such import, in *every*
entry point and every test collection order, is exactly the kind of
cross-cutting, order-dependent behavior this project's own guidance says to
avoid touching without an isolated impact analysis.

So instead this test gets the same safety net a different way: it calls
LangGraph's own schema-walker directly (read-only, at test time) and asserts
the manual allowlist in ``src/orchestration/checkpoint_allowlist.py`` covers
everything the walker finds reachable from the real compiled graph. If a new
Pydantic model/Enum becomes reachable from state and nobody updates the manual
list, this test fails immediately instead of silently degrading checkpoint
deserialization.

Keeping the private-API dependency (``langgraph._internal``) confined to this
test is deliberate: deriving the list at runtime in production code would put a
private API on the hot path of every run.

Note: the manual list is allowed to contain harmless *extra* entries (models
not currently reachable through the graph's serialized channels, e.g. an
agent's own output type before it lands in state). Only a *missing* entry
is a real problem, so this test only fails in that direction.
"""

from langgraph._internal import _serde

from src.orchestration.checkpoint_allowlist import CHECKPOINT_ALLOWED_MSGPACK_MODULES
from src.orchestration.graph import build_resume_enhancement_graph


def _derive_reachable_custom_types() -> set[tuple[str, str]]:
    """Return every (module, class) LangGraph can reach from the real graph's schema.

    Mirrors exactly what ``StateGraph.compile()`` computes internally when
    ``LANGGRAPH_STRICT_MSGPACK`` is enabled (see ``langgraph.graph.state``),
    minus the curated core LangChain message types, which are not this
    repo's concern.
    """
    graph = build_resume_enhancement_graph()
    builder = graph.builder
    schema_types = [builder.state_schema, builder.input_schema, builder.output_schema]
    for node in builder.nodes.values():
        schema_types.append(node.input_schema)
    for branches in builder.branches.values():
        for branch in branches.values():
            if branch.input_schema is not None:
                schema_types.append(branch.input_schema)
    reachable = _serde.build_serde_allowlist(schemas=schema_types, channels=builder.channels)
    return {(entry[0], entry[1]) for entry in reachable if entry[0].startswith("src.")}


def test_manual_allowlist_covers_every_type_reachable_from_pipeline_state() -> None:
    """The manual allowlist must be a superset of what LangGraph derives from state.

    A failure here means a Pydantic model or Enum became reachable from
    ResumeEnhancementPipelineState (directly or nested) without a matching
    entry in checkpoint_allowlist.py -- exactly the drift that file warns about.
    Fix by adding the missing (module, class) tuple.
    """
    reachable = _derive_reachable_custom_types()
    manual = set(CHECKPOINT_ALLOWED_MSGPACK_MODULES)

    missing = reachable - manual
    assert not missing, (
        "src/orchestration/checkpoint_allowlist.py is missing entries LangGraph can reach "
        f"from the pipeline state: {sorted(missing)}. Add these to "
        "CHECKPOINT_ALLOWED_MSGPACK_MODULES or a paused run using them will "
        "fail to resume once LangGraph enforces strict msgpack by default."
    )
