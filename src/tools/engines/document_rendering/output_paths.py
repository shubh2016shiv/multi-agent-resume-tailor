"""Output location for rendered resume artifacts: the directory and the file name.

One place owns WHERE a tailored resume is written and WHAT it is called, so every
renderer (markdown, docx, pdf) agrees and the convention lives in a single file
instead of being hardcoded per renderer. Pure path math -- no filesystem writes, no
config reads -- so the caller passes the configured base dir and a timestamp, keeping
this trivially unit-testable.

Layout (organised for browsing locally, named for portability when sent):
    <base_dir>/<candidate>/<designation>/<candidate>_<designation>_<date>_<time>.<ext>
The candidate and designation appear in BOTH the folders and the file name on purpose:
the folders group a candidate's resumes by the role they targeted, while the self-
describing file name still identifies the resume once detached (e.g. emailed).
"""

import re
from datetime import datetime
from pathlib import Path

from src.data_models.job import JobDescription
from src.data_models.resume import Resume


def resume_output_dir(resume: Resume, job: JobDescription, base_dir: Path) -> Path:
    """Return the directory a tailored resume's files belong in (not created here).

    Expects: a resume with full_name and a job with job_title (both required fields).
    Returns: base_dir/<candidate>/<designation>, each component sanitized for any OS.
    """
    ####################################################
    # STEP 1: SANITIZE THE CANDIDATE AND ROLE NAMES FOR FILESYSTEM USE#
    ####################################################
    # Human names and job titles may contain spaces, punctuation, or slashes,
    # so we normalize them before using them as path components.
    return base_dir / _sanitize_component(resume.full_name) / _sanitize_component(job.job_title)


def resume_filename(resume: Resume, job: JobDescription, extension: str, when: datetime) -> str:
    """Return the self-describing file name for one rendered artifact.

    Expects: extension with or without a leading dot (e.g. 'pdf' or '.pdf').
    Returns: '<candidate>_<designation>_<YYYYMMDD>_<HHMMSS>.<ext>', all sanitized.
             The timestamp makes repeated runs land on distinct, sortable names.
    """
    ####################################################
    # STEP 1: BUILD SAFE, SELF-DESCRIBING NAME PARTS#
    ####################################################
    candidate = _sanitize_component(resume.full_name)
    designation = _sanitize_component(job.job_title)

    ####################################################
    # STEP 2: ADD A TIMESTAMP SO REPEATED RUNS DO NOT COLLIDE#
    ####################################################
    stamp = when.strftime("%Y%m%d_%H%M%S")

    ####################################################
    # STEP 3: NORMALIZE THE EXTENSION AND BUILD THE FINAL FILE NAME#
    ####################################################
    ext = extension.lstrip(".")
    return f"{candidate}_{designation}_{stamp}.{ext}"


def _sanitize_component(text: str) -> str:
    """Make `text` safe as a single cross-platform filesystem path component.

    WHAT: keeps word characters (Unicode letters/digits/underscore, so international
        names survive) and collapses every other run -- spaces, punctuation, and path
        separators like '/' or '\\' that would otherwise create unintended nesting --
        into one '_'. Leading/trailing '_' are trimmed.
    WHY THIS APPROACH: a whitelist of safe characters is safer than a blacklist of
        forbidden ones, which differs per OS (Windows forbids <>:\"/\\|?* and more).
    RETURNS: the sanitized component, or 'unknown' if nothing usable remains, so a path
        component is always produced rather than an empty string.

    TODO: Windows reserved device names (CON, PRN, NUL, ...) are not special-cased.
          Proposed: suffix such names with '_'. Deferred: implausible for a real
          person's name or job title.
    """
    ####################################################
    # STEP 1: COLLAPSE UNSAFE CHARACTERS INTO UNDERSCORES#
    ####################################################
    # This keeps one portable path component instead of letting punctuation
    # or separators create invalid or nested paths.
    collapsed = re.sub(r"\W+", "_", text).strip("_")

    ####################################################
    # STEP 2: FALL BACK TO A SAFE NAME IF NOTHING USABLE REMAINS#
    ####################################################
    return collapsed or "unknown"
