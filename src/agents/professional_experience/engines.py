"""
Post-hoc quality engines for experience bullet validation.

These are NOT called during agent execution. They validate the agent's
output after the fact — useful for tests, monitoring, and QA pipelines.

All functions are pure: list/dict/str in, dict out. No framework, no LLM, no I/O.
"""

import re

# ── generic phrases to flag ───────────────────────────────────────────────────

_GENERIC_PHRASES = [
    "responsible for",
    "worked on",
    "participated in",
    "helped with",
    "assisted in",
    "involved in",
    "part of team",
    "various tasks",
    "multiple projects",
    "day-to-day operations",
    "duties included",
    "handled",
]

_PASSIVE_PATTERNS = [
    r"\b(?:was|were|been|being)\s+\w+ed\b",
    r"\b(?:was|were)\s+responsible\s+for\b",
    r"\b(?:was|were)\s+tasked\s+with\b",
]

_STRONG_VERBS = {
    "architected",
    "automated",
    "built",
    "delivered",
    "designed",
    "developed",
    "drove",
    "engineered",
    "established",
    "implemented",
    "improved",
    "increased",
    "launched",
    "led",
    "optimized",
    "orchestrated",
    "reduced",
    "scaled",
    "spearheaded",
    "transformed",
}

_IMPACT_PATTERNS = {
    "revenue_cost": r"(\$\d+[KMB]?|\d+%\s+(?:cost|revenue|profit|savings|budget))",
    "efficiency": r"(?:reduced|decreased|improved|accelerated|optimized|streamlined).*\d+%",
    "scale": r"(?:\d+\s*(?:users|customers|requests|transactions|records))",
    "time": r"(?:reduced|cut|shortened).*(?:time|latency|duration|cycle).*\d+",
    "automation": r"(?:automated|self-service|hands-free)",
}


# ── action verbs ──────────────────────────────────────────────────────────────


def analyze_action_verbs(achievements: list[str]) -> dict:
    """Score how many achievement bullets start with a strong action verb.

    Expects: list of achievement strings.
    Returns: dict with strong_count, weak_count, strong_ratio, per_bullet list.
    """
    results = []
    strong = 0
    for achievement in achievements:
        first_word = (
            achievement.strip().split()[0].lower().rstrip(".,;:") if achievement.strip() else ""
        )
        is_strong = first_word in _STRONG_VERBS
        if is_strong:
            strong += 1
        results.append(
            {"bullet": achievement[:60], "first_word": first_word, "is_strong": is_strong}
        )

    total = len(achievements)
    return {
        "total_bullets": total,
        "strong_verbs": strong,
        "weak_verbs": total - strong,
        "strong_ratio": round(strong / total, 2) if total > 0 else 0.0,
        "per_bullet": results,
    }


# ── quantified achievements ───────────────────────────────────────────────────


def count_quantified_achievements(achievements: list[str]) -> dict:
    """Count how many achievements include numbers, percentages, or metrics.

    Expects: list of achievement strings.
    Returns: dict with quantified_count, quantified_ratio, per_bullet list.
    """
    results = []
    quantified = 0
    for achievement in achievements:
        has_number = bool(re.search(r"\d+", achievement))
        has_percent = bool(re.search(r"\d+%", achievement))
        is_quantified = has_number or has_percent
        if is_quantified:
            quantified += 1
        results.append(
            {"bullet": achievement[:60], "has_number": has_number, "has_percent": has_percent}
        )

    total = len(achievements)
    return {
        "total_bullets": total,
        "quantified": quantified,
        "unquantified": total - quantified,
        "quantified_ratio": round(quantified / total, 2) if total > 0 else 0.0,
        "per_bullet": results,
    }


# ── bullet structure ──────────────────────────────────────────────────────────


def validate_bullet_structure(achievement: str) -> dict:
    """Check if a bullet follows impact-first structure (metric + method + outcome).

    Expects: a single achievement string.
    Returns: dict with has_metric, has_method, starts_strong, issues.
    """
    issues = []

    has_metric = bool(re.search(r"\d+%|\$\d+|\d+\s*(?:users|customers|requests)", achievement))
    has_method = bool(re.search(r"(?:using|with|via|through)\s+[A-Za-z]", achievement))
    first_word = (
        achievement.strip().split()[0].lower().rstrip(".,;:") if achievement.strip() else ""
    )
    starts_strong = first_word in _STRONG_VERBS

    if not starts_strong:
        issues.append("Does not start with a strong action verb")
    if not has_metric:
        issues.append("Missing measurable outcome (number, percentage, or scale)")
    if not has_method:
        issues.append("Missing method or technology used")

    return {
        "has_metric": has_metric,
        "has_method": has_method,
        "starts_strong": starts_strong,
        "issues": issues,
        "is_valid": len(issues) == 0,
    }


# ── impact level ──────────────────────────────────────────────────────────────


def assess_impact_level(achievement: str) -> dict:
    """Assess which business impact categories a bullet hits.

    Expects: a single achievement string.
    Returns: dict with impact_categories (list) and has_any_impact (bool).
    """
    categories = []
    for category, pattern in _IMPACT_PATTERNS.items():
        if re.search(pattern, achievement.lower()):
            categories.append(category)

    return {
        "impact_categories": categories,
        "has_any_impact": len(categories) > 0,
    }


# ── verb-noun pairs ───────────────────────────────────────────────────────────


def analyze_verb_noun_pairs(achievement: str) -> dict:
    """Check if the opening verb is paired with an impactful noun.

    Expects: a single achievement string.
    Returns: dict with verb, noun, has_power_pair.
    """
    words = achievement.strip().split()
    if len(words) < 2:
        return {"has_power_pair": False, "verb": "", "noun": ""}

    verb = words[0].lower().rstrip(".,;:")
    noun = words[1].lower().rstrip(".,;:")
    has_power = verb in _STRONG_VERBS and len(noun) > 2

    return {"has_power_pair": has_power, "verb": verb, "noun": noun}


# ── passive voice ─────────────────────────────────────────────────────────────


def detect_passive_voice(achievements: list[str]) -> dict:
    """Detect passive voice constructions across a list of achievements.

    Expects: list of achievement strings.
    Returns: dict with passive_count, passive_bullets list.
    """
    flagged = []
    for achievement in achievements:
        matches = []
        for pattern in _PASSIVE_PATTERNS:
            found = re.findall(pattern, achievement.lower())
            matches.extend(found)
        if matches:
            flagged.append({"bullet": achievement[:80], "patterns_found": matches})

    return {
        "total_bullets": len(achievements),
        "passive_count": len(flagged),
        "passive_bullets": flagged,
    }


# ── specificity ───────────────────────────────────────────────────────────────


def check_specificity(achievement: str) -> dict:
    """Check if a bullet is specific enough to be valuable.

    Expects: a single achievement string.
    Returns: dict with generic_phrases_found, issues, is_specific.
    """
    lower = achievement.lower()
    found = [phrase for phrase in _GENERIC_PHRASES if phrase in lower]
    issues = []
    if found:
        issues.append(f"Generic phrasing: {', '.join(found)}")
    if len(achievement.split()) < 6:
        issues.append("Too short — likely missing context or outcome")

    return {
        "generic_phrases_found": found,
        "issues": issues,
        "is_specific": len(issues) == 0,
    }
