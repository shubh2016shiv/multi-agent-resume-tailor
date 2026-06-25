"""Project-root wrapper for running the sample resume tailoring pipeline."""

import sys
from collections.abc import Sequence

from src.main import main as run_pipeline

DEFAULT_RESUME_PATH = "sample_documents/Shubham_Resume_2026_April_version2.pdf"
DEFAULT_JOB_DESCRIPTION_PATH = "sample_documents/job_descriptions/DataScientist_JD_1BuyAI .docx.pdf"


def main(argv: Sequence[str] | None = None) -> int:
    """Run `src.main` with sample defaults unless arguments are supplied."""
    supplied_args = list(sys.argv[1:] if argv is None else argv)
    return run_pipeline(args_for_main(supplied_args))


def args_for_main(supplied_args: Sequence[str]) -> list[str]:
    """Return explicit args, or the default sample resume/JD pair."""
    if supplied_args:
        return list(supplied_args)
    return [DEFAULT_RESUME_PATH, DEFAULT_JOB_DESCRIPTION_PATH]


if __name__ == "__main__":
    raise SystemExit(main())
