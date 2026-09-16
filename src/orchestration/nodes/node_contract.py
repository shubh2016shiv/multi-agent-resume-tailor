"""Reusable type contracts for state-driven graph nodes."""

from collections.abc import Mapping
from typing import Protocol, TypeVar

StateT = TypeVar("StateT", contravariant=True)
ResultT = TypeVar("ResultT", covariant=True)

type StateUpdate = Mapping[str, object]


class StateNode(Protocol[StateT, ResultT]):
    """Accept graph state and return a result understood by the graph runtime."""

    def __call__(self, state: StateT) -> ResultT: ...


__all__ = ["StateNode", "StateUpdate"]
