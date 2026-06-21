"""
PII redaction: mask personal data before any text reaches an external LLM.

Detection uses Microsoft Presidio (NER + validated recognizers), not free regex,
because names and addresses cannot be matched reliably by pattern alone. The
function is stateless: it returns the redacted text plus a placeholder->value
map, and the orchestrator owns that map to rehydrate the final document.
"""

from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer, RecognizerResult

from src.core.logger import get_logger

logger = get_logger(__name__)

# Entities to redact. DATE_TIME is deliberately excluded: it is one bucket for
# all dates, so redacting it would also remove employment dates the review needs.
# ORGANIZATION is excluded too: employer names are required for the review.
REDACTED_ENTITIES = [
    "PERSON",
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "LOCATION",
    "URL",
    "US_SSN",
    "US_PASSPORT",
    "US_DRIVER_LICENSE",
    "US_ITIN",
    "US_BANK_NUMBER",
    "UK_NHS",
    "CREDIT_CARD",
    "IBAN_CODE",
    "CRYPTO",
    "IP_ADDRESS",
    "MAC_ADDRESS",
    "MEDICAL_LICENSE",
    "DATE_OF_BIRTH",  # custom, label-anchored (see _build_date_of_birth_recognizer)
    "AGE",  # custom, suffix-anchored (see _build_age_recognizer)
]

# Drop low-confidence hits. Label-anchored DOB matches clear this only when their
# context words boost the score; bare dates stay below and are not redacted.
MIN_CONFIDENCE_SCORE = 0.5
PHONE_PATTERN_SCORE = 0.85
ENTITY_PRIORITY = {
    "PHONE_NUMBER": 100,
    "EMAIL_ADDRESS": 90,
    "PERSON": 80,
}

# TODO: Education completion year.
#       Proposed: redact dates that fall inside the Education section.
#       Deferred: sections are unknown at this raw-markdown stage; needs a
#       section-aware pass after structured extraction.
# TODO: Indian government IDs (Aadhaar, PAN) are not in Presidio's default set.
#       Proposed: add custom recognizers if resumes from India carry them.
#       Deferred: the current sample resume has none; add on real need.

_analyzer: AnalyzerEngine | None = None


def redact_pii(markdown: str) -> tuple[str, dict[str, str]]:
    """Replace PII in the text with stable placeholders.

    Args:
        markdown: Resume text to redact.

    Returns:
        A tuple of (redacted_text, mapping), where mapping is placeholder->value
        so the orchestrator can restore the original text later. Same original
        value always maps to the same placeholder.
    """
    ####################################################
    # STEP 1: EXIT EARLY IF THERE IS NO TEXT TO REDACT#
    ####################################################
    if not markdown.strip():
        return markdown, {}

    ####################################################
    # STEP 2: DETECT THE TEXT SPANS THAT LOOK LIKE PII#
    ####################################################
    # At this stage we only identify the character ranges to replace.
    spans = _detect_pii_spans(markdown)

    ####################################################
    # STEP 3: ASSIGN A STABLE PLACEHOLDER TO EACH UNIQUE PII VALUE#
    ####################################################
    # If the same email or phone number appears twice, it should get
    # the same placeholder both times.
    placeholder_by_key = _build_placeholder_map(markdown, spans)

    ####################################################
    # STEP 4: REPLACE EACH DETECTED SPAN WITH ITS PLACEHOLDER#
    ####################################################
    redacted_text = _apply_redaction(markdown, spans, placeholder_by_key)

    ####################################################
    # STEP 5: BUILD THE RESTORATION MAP FOR LATER REHYDRATION#
    ####################################################
    # The orchestrator stores this map so it can restore the original values
    # after the LLM-based stages are finished.
    mapping = {placeholder: value for (_entity, value), placeholder in placeholder_by_key.items()}
    logger.info(f"Redacted {len(mapping)} unique PII value(s)")
    return redacted_text, mapping


def _get_analyzer() -> AnalyzerEngine:
    """Lazily build and cache the Presidio analyzer (loads the spaCy model once)."""
    global _analyzer
    ####################################################
    # STEP 1: BUILD THE ANALYZER ONLY ON FIRST USE#
    ####################################################
    # Presidio setup is heavier than a normal function call, so we reuse
    # one analyzer instead of rebuilding it for every document.
    if _analyzer is None:
        engine = AnalyzerEngine()

        ####################################################
        # STEP 2: ADD THE CUSTOM RECOGNIZERS THIS PROJECT NEEDS#
        ####################################################
        # Presidio already knows many entity types, but resumes benefit from
        # extra patterns for phone numbers, age, and date of birth.
        engine.registry.add_recognizer(_build_resume_phone_recognizer())
        engine.registry.add_recognizer(_build_date_of_birth_recognizer())
        engine.registry.add_recognizer(_build_age_recognizer())
        _analyzer = engine
    return _analyzer


def _detect_pii_spans(text: str) -> list[RecognizerResult]:
    """Run Presidio, keep confident hits, and drop overlaps so spans never collide."""
    ####################################################
    # STEP 1: ASK PRESIDIO TO FIND THE ENTITY TYPES WE CARE ABOUT#
    ####################################################
    results = _get_analyzer().analyze(text=text, entities=REDACTED_ENTITIES, language="en")

    ####################################################
    # STEP 2: DROP LOW-CONFIDENCE RESULTS#
    ####################################################
    # Weak guesses create noisy redaction, so we keep only spans
    # that clear the minimum confidence score.
    confident_spans = [result for result in results if result.score >= MIN_CONFIDENCE_SCORE]

    ####################################################
    # STEP 3: RESOLVE ANY OVERLAPPING DETECTIONS#
    ####################################################
    # Two entities can claim the same text region. We keep one per region
    # so later replacement stays correct and deterministic.
    return _resolve_overlaps(confident_spans)


def _resolve_overlaps(spans: list[RecognizerResult]) -> list[RecognizerResult]:
    """Keep the highest-confidence spans, dropping any that overlap an accepted one.

    Presidio can return overlapping hits (e.g. a URL 'gmail.com' inside the email
    'a@gmail.com'). Redacting both corrupts offsets, so we keep one per region:
    explicit entity priority first, then score, then longer span.
    """
    accepted: list[RecognizerResult] = []
    ####################################################
    # STEP 1: PREFER THE RESUME-SPECIFIC ENTITY TYPE BEFORE RAW SCORE#
    ####################################################
    sorted_spans = sorted(
        spans,
        key=lambda span: (
            ENTITY_PRIORITY.get(span.entity_type, 0),
            span.score,
            span.end - span.start,
        ),
        reverse=True,
    )

    ####################################################
    # STEP 2: KEEP A SPAN ONLY IF IT DOES NOT COLLIDE WITH A BETTER ONE#
    ####################################################
    for span in sorted_spans:
        if not any(_spans_overlap(span, kept) for kept in accepted):
            accepted.append(span)
    return accepted


def _spans_overlap(first: RecognizerResult, second: RecognizerResult) -> bool:
    """True if the two character spans share any range."""
    return first.start < second.end and second.start < first.end


def _build_placeholder_map(text: str, spans: list[RecognizerResult]) -> dict[tuple[str, str], str]:
    """Map each unique (entity_type, value) to a stable placeholder like '[EMAIL_ADDRESS_1]'."""
    ####################################################
    # STEP 1: WALK THROUGH SPANS FROM LEFT TO RIGHT#
    ####################################################
    # Left-to-right numbering gives placeholders a stable, human-readable order.
    counts_by_entity: dict[str, int] = {}
    placeholder_by_key: dict[tuple[str, str], str] = {}
    for span in sorted(spans, key=lambda s: s.start):
        value = text[span.start : span.end]
        key = (span.entity_type, value)

        ####################################################
        # STEP 2: REUSE THE SAME PLACEHOLDER FOR REPEATED VALUES#
        ####################################################
        # The same email address or phone number should not become
        # different placeholders in different places.
        if key not in placeholder_by_key:
            counts_by_entity[span.entity_type] = counts_by_entity.get(span.entity_type, 0) + 1
            placeholder_by_key[key] = f"[{span.entity_type}_{counts_by_entity[span.entity_type]}]"
    return placeholder_by_key


def _apply_redaction(
    text: str, spans: list[RecognizerResult], placeholder_by_key: dict[tuple[str, str], str]
) -> str:
    """Replace each detected span with its placeholder, right-to-left so offsets stay valid.

    Assumes spans are non-overlapping (see _resolve_overlaps), so right-to-left
    replacement keeps every remaining span's offsets valid.
    """
    ####################################################
    # STEP 1: REPLACE FROM RIGHT TO LEFT SO OFFSETS DO NOT SHIFT AHEAD OF US#
    ####################################################
    # If we replaced from left to right, every earlier replacement could move
    # the later text and break the remaining span positions.
    for span in sorted(spans, key=lambda s: s.start, reverse=True):
        value = text[span.start : span.end]
        placeholder = placeholder_by_key[(span.entity_type, value)]
        text = text[: span.start] + placeholder + text[span.end :]
    return text


def _build_date_of_birth_recognizer() -> PatternRecognizer:
    """Detect a date only when birth-context words are nearby, not any date."""
    ####################################################
    # STEP 1: DEFINE A GENERIC DATE SHAPE WITH A LOW BASE SCORE#
    ####################################################
    # A plain date alone is not enough because resumes contain many valid dates
    # that should stay visible, such as employment timelines.
    date_pattern = Pattern(
        name="dob_date",
        regex=r"\b\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}\b",
        score=0.4,  # below threshold alone; context words boost it past MIN_CONFIDENCE_SCORE
    )

    ####################################################
    # STEP 2: REQUIRE BIRTH CONTEXT WORDS TO MAKE IT A DOB MATCH#
    ####################################################
    return PatternRecognizer(
        supported_entity="DATE_OF_BIRTH",
        patterns=[date_pattern],
        context=["dob", "birth", "born"],
    )


def _build_resume_phone_recognizer() -> PatternRecognizer:
    """Detect phone formats common in resume headers.

    Presidio's default phone recognizer can miss human-spaced numbers such as
    '+1 555 123 4567'. These custom patterns keep phone detection inside the
    Presidio recognizer system instead of doing a separate redaction pass.
    """
    ####################################################
    # STEP 1: COMBINE THE PHONE FORMATS COMMONLY SEEN IN RESUMES#
    ####################################################
    # We keep them under one recognizer so Presidio can score them
    # consistently with the rest of the entity pipeline.
    return PatternRecognizer(
        supported_entity="PHONE_NUMBER",
        patterns=[
            _north_american_phone_pattern(),
            _international_phone_pattern(),
            _india_mobile_phone_pattern(),
        ],
        context=["phone", "mobile", "cell", "tel", "contact"],
    )


def _north_american_phone_pattern() -> Pattern:
    """Return a pattern for US/Canada resume phone formats."""
    return Pattern(
        name="north_american_resume_phone",
        regex=(
            r"(?<!\w)(?:\+?1[\s.-]+)?"
            r"(?:\(\d{3}\)|\d{3})[\s.-]+"
            r"\d{3}[\s.-]+\d{4}(?!\w)"
        ),
        score=PHONE_PATTERN_SCORE,
    )


def _international_phone_pattern() -> Pattern:
    """Return a pattern for separated international phone numbers."""
    return Pattern(
        name="international_resume_phone",
        regex=r"(?<!\w)\+\d{1,3}[\s.-]+\d{2,5}(?:[\s.-]+\d{2,5}){1,3}(?!\w)",
        score=PHONE_PATTERN_SCORE,
    )


def _india_mobile_phone_pattern() -> Pattern:
    """Return a pattern for Indian mobile numbers, with or without country code."""
    return Pattern(
        name="india_mobile_resume_phone",
        regex=r"(?<!\w)(?:\+?91[\s.-]+)?[6-9]\d{4}[\s.-]?\d{5}(?!\w)",
        score=PHONE_PATTERN_SCORE,
    )


def _build_age_recognizer() -> PatternRecognizer:
    """Detect an age written with an explicit suffix, e.g. '32 years old'."""
    ####################################################
    # STEP 1: MATCH AGE ONLY WHEN THE WORDING MAKES IT EXPLICIT#
    ####################################################
    # A standalone number should not be redacted as age, so the pattern
    # requires wording like "years old" or "yrs old".
    age_pattern = Pattern(
        name="age_value",
        regex=r"\b\d{1,2}\s*(?:years?\s*old|yrs?\s*old)\b",
        score=0.7,
    )
    return PatternRecognizer(
        supported_entity="AGE",
        patterns=[age_pattern],
        context=["age"],
    )
