"""
LaTeX escaping -- the reliability core of the renderer.

Any resume field can contain LaTeX-special characters. A single unescaped "%"
silently comments out the rest of a line, "&" breaks tabular alignment, and
"$ _ ^" flip into math mode. So every user-supplied string is escaped before it
reaches the template. Backslash is replaced FIRST -- otherwise the backslashes
introduced while escaping the other characters would themselves be escaped again.
"""

# Order matters: backslash must come first.
_LATEX_SPECIALS: list[tuple[str, str]] = [
    ("\\", r"\textbackslash{}"),
    ("&", r"\&"),
    ("%", r"\%"),
    ("$", r"\$"),
    ("#", r"\#"),
    ("_", r"\_"),
    ("{", r"\{"),
    ("}", r"\}"),
    ("~", r"\textasciitilde{}"),
    ("^", r"\textasciicircum{}"),
]

# In a \href URL argument only these are LaTeX-special; the rest of the address
# (slashes, "_", "-", "?", "=") must stay verbatim or the link breaks.
_URL_SPECIALS = ("%", "#", "&")


def escape_latex(text: str) -> str:
    """Escape LaTeX-special characters in body text so it compiles verbatim.

    Args:
        text: Any user-supplied string (a name, bullet, skill, etc.).

    Returns:
        The text with \\ & % $ # _ { } ~ ^ replaced by LaTeX-safe sequences.
    """
    ####################################################
    # STEP 1: REPLACE EACH LATEX-SPECIAL CHARACTER IN THE REQUIRED ORDER#
    ####################################################
    # Backslash goes first so the escaping we introduce later does not
    # get escaped a second time.
    for char, replacement in _LATEX_SPECIALS:
        text = text.replace(char, replacement)
    return text


def escape_latex_url(url: str) -> str:
    """Escape only the characters that are special inside a \\href{...} target.

    URLs legitimately contain "%", "#", "_", "~" etc.; escaping them the way body
    text is escaped would corrupt the address. Only "% # &" actually need a leading
    backslash inside an href argument, so only those are touched.
    """
    ####################################################
    # STEP 1: ESCAPE ONLY THE CHARACTERS THAT BREAK HREF TARGETS#
    ####################################################
    # We intentionally do less here than in normal body text because a URL
    # must stay as close to its original form as possible.
    for char in _URL_SPECIALS:
        url = url.replace(char, "\\" + char)
    return url
