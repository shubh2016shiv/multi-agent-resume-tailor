"""Typed exceptions for the orchestration pipeline.

Two exception types separate three ways a run can die:

- PipelineQualityGateError: an LLM stage produced WELL-FORMED output that failed
  a content/style judgment (banned phrase, too short, generic opener). This is
  an expected, input-dependent outcome -- most often a resume too thin in
  measurable evidence for the target job -- not a bug. user_action tells the
  user what to change in THEIR input.
- AgentOutputError: the LLM's raw response could not even be parsed into the
  expected structure (no JSON object, or it does not match the output schema).
  This is not a judgment about content quality -- it is a formatting failure,
  most often a transient LLM hiccup. user_action tells the user to retry rather
  than to change their resume.
- Everything else (ValueError, RuntimeError, ...): a programming error -- a
  pipeline invariant was violated (e.g. state a node should have populated is
  still None). Those still crash loudly with a full traceback so a developer
  sees them; there is no user-facing action for a bug.

The CLI catches only the first two types and shows the user what failed and
what THEY can do about it, without a stack trace. The fail-fast philosophy is
unchanged: any of these still stops the run. Only the presentation differs.
"""


class PipelineQualityGateError(Exception):
    """A pipeline stage's WELL-FORMED output failed its content/style quality gate.

    Carries what the CLI needs to explain the failure to the user:
    the stage that failed, the gate's findings, and the action the
    user can take to make the next run succeed.
    """

    def __init__(self, stage: str, findings: list[str], user_action: str):
        self.stage = stage
        self.findings = findings
        self.user_action = user_action
        super().__init__(f"{stage} failed the quality gate: {'; '.join(findings)}")


class AgentOutputError(Exception):
    """An agent's raw response could not be parsed into its expected output model.

    Distinct from PipelineQualityGateError: this is a formatting/schema failure,
    not a content judgment, so the right user action is different (retry, not
    "add more detail to your resume").
    """

    def __init__(self, stage: str, reason: str, user_action: str):
        self.stage = stage
        self.reason = reason
        self.user_action = user_action
        super().__init__(f"{stage} produced unparseable output: {reason}")
