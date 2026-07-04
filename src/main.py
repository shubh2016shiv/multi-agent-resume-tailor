"""Command-line entry point for the resume tailoring pipeline."""

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.data_models.orchestration import OrchestrationResult


@dataclass(frozen=True)
class PipelineInputs:
    """Validated input files for one pipeline run."""

    resume_path: Path | None = None
    job_description_path: Path | None = None
    resume_from_path: Path | None = None


def main(argv: Sequence[str] | None = None) -> int:
    """Run the pipeline and return a process exit code."""
    inputs = parse_inputs(argv)

    from src.orchestration import resume_paused_run, tailor_resume

    print_run_header(inputs)
    if inputs.resume_from_path is not None:
        result = resume_paused_run(str(inputs.resume_from_path))
    else:
        assert inputs.resume_path is not None
        assert inputs.job_description_path is not None
        result = tailor_resume(
            str(inputs.resume_path),
            str(inputs.job_description_path),
        )
    print_run_summary(result)
    return 0


def parse_inputs(argv: Sequence[str] | None) -> PipelineInputs:
    """Parse and validate CLI input paths."""
    parser = build_parser()
    args = parser.parse_args(argv)
    return resolve_inputs(args, parser)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description="Run the resume tailoring pipeline for one resume and job description.",
    )
    parser.add_argument(
        "--resume",
        metavar="PATH",
        help="Path to the source resume document.",
    )
    parser.add_argument(
        "--job-description",
        "--jd",
        dest="job_description",
        metavar="PATH",
        help="Path to the target job-description document.",
    )
    parser.add_argument(
        "--resume-from",
        metavar="PATH",
        help=(
            "Path to a paused_run_<id> directory. Answer clarifications_sheet.json in "
            "that folder, then resume from it."
        ),
    )
    parser.add_argument(
        "paths",
        nargs="*",
        metavar="PATH",
        help="Optional positional form for fresh runs: RESUME_PATH JOB_DESCRIPTION_PATH",
    )
    return parser


def resolve_inputs(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> PipelineInputs:
    """Resolve positional or named input paths."""
    if args.resume_from:
        if args.paths or args.resume or args.job_description:
            parser.error("use either --resume-from or fresh-run resume/JD inputs, not both")
        return PipelineInputs(
            resume_from_path=existing_directory(
                Path(args.resume_from).expanduser(),
                "paused run",
                parser,
            )
        )

    has_named_input = bool(args.resume or args.job_description)

    if args.paths and has_named_input:
        parser.error("use either positional paths or --resume/--job-description, not both")
    if len(args.paths) not in {0, 2}:
        parser.error("expected exactly two positional paths: RESUME_PATH JOB_DESCRIPTION_PATH")
    if has_named_input and not (args.resume and args.job_description):
        parser.error("provide both --resume and --job-description")
    if not args.paths and not has_named_input:
        parser.error("provide a resume path and a job-description path")

    resume_raw, job_raw = args.paths if args.paths else (args.resume, args.job_description)
    return PipelineInputs(
        resume_path=existing_file(Path(resume_raw).expanduser(), "resume", parser),
        job_description_path=existing_file(Path(job_raw).expanduser(), "job description", parser),
    )


def existing_file(
    path: Path,
    label: str,
    parser: argparse.ArgumentParser,
) -> Path:
    """Return a normalized file path or stop with a parser error."""
    if not path.is_file():
        parser.error(f"the {label} path does not exist or is not a file: {path}")
    return path.resolve()


def existing_directory(
    path: Path,
    label: str,
    parser: argparse.ArgumentParser,
) -> Path:
    """Return a normalized directory path or stop with a parser error."""
    if not path.is_dir():
        parser.error(f"the {label} path does not exist or is not a directory: {path}")
    return path.resolve()


def print_run_header(inputs: PipelineInputs) -> None:
    """Print the selected inputs before orchestration logs begin."""
    print("Resume Tailor pipeline")
    if inputs.resume_from_path is not None:
        print(f"  resume from: {inputs.resume_from_path}")
    else:
        print(f"  resume: {inputs.resume_path}")
        print(f"  job description: {inputs.job_description_path}")
    print()


def print_run_summary(result: "OrchestrationResult") -> None:
    """Print high-signal outputs for visual data-flow debugging."""
    quality = result.quality_report
    artifacts = result.rendered_artifacts

    print()
    print("Pipeline result")
    print_field("disposition", result.disposition.value)
    print_field("candidate", result.original_resume.full_name)
    print_field("target job", result.job_description.job_title)
    print_field("company", result.job_description.company_name)
    print_field("strategy score", result.strategy.overall_fit_score)
    if quality is not None:
        print_field("quality score", quality.overall_quality_score)
        print_field("gate passed", quality.passes_quality_gate)
        print_field("accuracy", quality.accuracy.accuracy_score)
        print_field("relevance", quality.relevance.relevance_score)
        print_field("ats", quality.ats_optimization.ats_score)
    if result.optimized_resume is not None:
        print_field("summary", result.optimized_resume.final_resume.professional_summary)

    if result.clarifications_requested:
        print_field(
            "clarifications",
            f"{len(result.clarifications_requested)} question(s) for you -- answer "
            "clarifications_sheet.json in the paused run folder and resume with --resume-from",
        )
    print_field("paused run", result.paused_run_path)

    if artifacts is None:
        print_field("artifacts", "none")
        return

    print_field("markdown", artifacts.markdown_path)
    print_field("docx", artifacts.docx_path)
    print_field("pdf", artifacts.pdf_path or "not produced")
    print_field("pdf reason", artifacts.pdf_skipped_reason)


def print_field(label: str, value: object | None) -> None:
    """Print one summary field when it has a meaningful value."""
    if value not in {None, ""}:
        print(f"  {label}: {value}")


if __name__ == "__main__":
    raise SystemExit(main())
