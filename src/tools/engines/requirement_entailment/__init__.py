"""Bounded semantic requirement-entailment judgment."""

from .entailment_judge import EntailmentVerdict, judge_requirement_entailment

__all__ = [
    "EntailmentVerdict",
    "judge_requirement_entailment",
]
