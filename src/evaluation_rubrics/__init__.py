"""Public surface for the evaluation rubrics package.

    from src.evaluation_rubrics import grade_accuracy, grade_relevance, blend_overall

Deterministic, code-owned grading of quality dimensions from evidence. Each rubric
turns evidence into one QA dimension's metrics; blend_overall combines the dimension
scores with the documented weights. No LLM, so the QA gate reads grounded numbers.
"""

from src.evaluation_rubrics.accuracy_rubric import grade_accuracy
from src.evaluation_rubrics.ats_rubric import grade_ats
from src.evaluation_rubrics.relevance_rubric import grade_relevance
from src.evaluation_rubrics.score_weighting import blend_overall

__all__ = ["blend_overall", "grade_accuracy", "grade_ats", "grade_relevance"]
