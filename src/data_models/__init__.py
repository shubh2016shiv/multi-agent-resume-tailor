"""
This __init__.py file serves two purposes:

1.  It marks the `data_models` directory as a Python package, allowing you to
    import its modules from other parts of the application.

2.  It can be used to create a more convenient, flattened namespace for the data
    models, making imports cleaner.

By importing the key models here, you can change an import from:
`from src.data_models.resume import Resume`

to the shorter and more convenient:
`from src.data_models import Resume`

This is a common practice in Python packages to improve the developer experience.
"""

# Import key models from each module to make them directly accessible
# from the `data_models` package.

# From resume.py
# From evaluation.py
from .evaluation import (
    AccuracyMetrics,
    ATSMetrics,
    QualityReport,
    RelevanceMetrics,
)

# From job.py
from .job import (
    JobDescription,
    JobLevel,
    JobRequirement,
    SkillImportance,
)
from .resume import (
    Education,
    Experience,
    Resume,
    Skill,
)

# From strategy.py
from .strategy import (
    AlignmentStrategy,
    SkillGap,
    SkillMatch,
)

# The __all__ variable defines the public API of this package.
# When a user writes `from src.data_models import *`, only the names
# listed in __all__ will be imported. This prevents cluttering the
# namespace with unintended modules or variables.
__all__ = [
    # Resume models
    "Resume",
    "Experience",
    "Education",
    "Skill",
    # Job models
    "JobDescription",
    "JobRequirement",
    "JobLevel",
    "SkillImportance",
    # Strategy models
    "AlignmentStrategy",
    "SkillMatch",
    "SkillGap",
    # Evaluation models
    "QualityReport",
    "AccuracyMetrics",
    "RelevanceMetrics",
    "ATSMetrics",
]

