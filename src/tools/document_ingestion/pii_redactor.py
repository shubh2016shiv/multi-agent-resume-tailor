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
    "AGE",            # custom, suffix-anchored (see _build_age_recognizer)
]

# Drop low-confidence hits. Label-anchored DOB matches clear this only when their
# context words boost the score; bare dates stay below and are not redacted.
MIN_CONFIDENCE_SCORE = 0.5

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
    if not markdown.strip():
        return markdown, {}
    spans = _detect_pii_spans(markdown)
    placeholder_by_key = _build_placeholder_map(markdown, spans)
    redacted_text = _apply_redaction(markdown, spans, placeholder_by_key)
    mapping = {placeholder: value for (_entity, value), placeholder in placeholder_by_key.items()}
    logger.info(f"Redacted {len(mapping)} unique PII value(s)")
    return redacted_text, mapping


def _get_analyzer() -> AnalyzerEngine:
    """Lazily build and cache the Presidio analyzer (loads the spaCy model once)."""
    global _analyzer
    if _analyzer is None:
        engine = AnalyzerEngine()
        engine.registry.add_recognizer(_build_date_of_birth_recognizer())
        engine.registry.add_recognizer(_build_age_recognizer())
        _analyzer = engine
    return _analyzer


def _detect_pii_spans(text: str) -> list[RecognizerResult]:
    """Run Presidio, keep confident hits, and drop overlaps so spans never collide."""
    results = _get_analyzer().analyze(text=text, entities=REDACTED_ENTITIES, language="en")
    confident_spans = [result for result in results if result.score >= MIN_CONFIDENCE_SCORE]
    return _resolve_overlaps(confident_spans)


def _resolve_overlaps(spans: list[RecognizerResult]) -> list[RecognizerResult]:
    """Keep the highest-confidence spans, dropping any that overlap an accepted one.

    Presidio can return overlapping hits (e.g. a URL 'gmail.com' inside the email
    'a@gmail.com'). Redacting both corrupts offsets, so we keep one per region:
    highest score first, longer span breaking ties.
    """
    accepted: list[RecognizerResult] = []
    for span in sorted(spans, key=lambda s: (s.score, s.end - s.start), reverse=True):
        if not any(_spans_overlap(span, kept) for kept in accepted):
            accepted.append(span)
    return accepted


def _spans_overlap(first: RecognizerResult, second: RecognizerResult) -> bool:
    """True if the two character spans share any range."""
    return first.start < second.end and second.start < first.end


def _build_placeholder_map(
    text: str, spans: list[RecognizerResult]
) -> dict[tuple[str, str], str]:
    """Map each unique (entity_type, value) to a stable placeholder like '[EMAIL_ADDRESS_1]'."""
    counts_by_entity: dict[str, int] = {}
    placeholder_by_key: dict[tuple[str, str], str] = {}
    for span in sorted(spans, key=lambda s: s.start):
        value = text[span.start : span.end]
        key = (span.entity_type, value)
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
    for span in sorted(spans, key=lambda s: s.start, reverse=True):
        value = text[span.start : span.end]
        placeholder = placeholder_by_key[(span.entity_type, value)]
        text = text[: span.start] + placeholder + text[span.end :]
    return text


def _build_date_of_birth_recognizer() -> PatternRecognizer:
    """Detect a date only when birth-context words are nearby, not any date."""
    date_pattern = Pattern(
        name="dob_date",
        regex=r"\b\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}\b",
        score=0.4,  # below threshold alone; context words boost it past MIN_CONFIDENCE_SCORE
    )
    return PatternRecognizer(
        supported_entity="DATE_OF_BIRTH",
        patterns=[date_pattern],
        context=["dob", "birth", "born"],
    )


def _build_age_recognizer() -> PatternRecognizer:
    """Detect an age written with an explicit suffix, e.g. '32 years old'."""
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
