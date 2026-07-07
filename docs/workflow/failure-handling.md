# Failure Handling

Failure handling in Resume Tailor should preserve evidence and recover from the
smallest safe boundary. A multi-agent workflow should not treat every failure as
"start over." That wastes tokens and erases the local cause.

## Failure Taxonomy

```text
failure
  |
  +--> schema or JSON formatting
  |
  +--> transient provider/tool issue
  |
  +--> missing candidate-owned fact
  |
  +--> unsupported or inflated claim
  |
  +--> rendered ATS structure failure
  |
  +--> quality gate failure
  |
  +--> artifact rendering failure
```

## Current Handling Patterns

The system already has several concrete handling paths:

- malformed agent output raises an agent output error with user-facing guidance
- experience clarification pauses the graph and resumes later
- ATS rendered structure failure can route through deterministic patching
- failed quality can stop rendering unless draft rendering is enabled
- PII cleanup is tied to terminal outcomes
- checkpoint files are archived or removed based on run disposition

This is stronger than a simple try/except wrapper around the whole pipeline.

## Recovery Ladder

```text
1. validate output
2. retry same node when failure is transient
3. rerun upstream producer when input is bad
4. patch deterministically when safe
5. pause for human facts when required
6. stop with persisted evidence when unsafe
```

The ladder is important because it prevents overusing the LLM. If a deterministic
patch can restore an empty section from typed upstream state, that is better than
asking the model to regenerate the entire resume.

## Evidence First

Every failure should leave enough evidence to answer three questions:

```text
what node failed?
what input did it receive?
what output or exception did it produce?
```

Without that evidence, retries become superstition. With it, the system can
evolve from manual debugging toward explicit node-level recovery policies.

