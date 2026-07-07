# Human In The Loop
## Two Different Reasons a Machine Hands Control to a Person — and Why They Must Never Be Confused

> **Scope:** `src/hitl/professional_experience/` (the candidate clarification
> loop), `src/orchestration/nodes/experience.py`'s pause boundary,
> `src/orchestration/human_review_policy.py` (the human review escalation),
> and `src/main.py` (the actual CLI round-trip a person drives).
> **Audience:** Contributors touching the experience pipeline or the quality
> gate; anyone who needs to reason precisely about when this system stops and
> waits for a person, why, and what happens next.

---

## Table of Contents

1. [What Problem HITL Solves — and the Philosophy Behind It](#1-what-problem-hitl-solves--and-the-philosophy-behind-it)
2. [In Plain Terms, First](#2-in-plain-terms-first)
3. [Two Distinct Mechanisms, Named Clearly](#3-two-distinct-mechanisms-named-clearly)
4. [The Candidate Clarification Loop — Full Architecture](#4-the-candidate-clarification-loop--full-architecture)
5. [The Human Review Escalation — Full Architecture](#5-the-human-review-escalation--full-architecture)
6. [Case Study: One Role, Two Rounds, End to End](#6-case-study-one-role-two-rounds-end-to-end)
7. [Design Rule — When Does Something Become a HITL Boundary?](#7-design-rule--when-does-something-become-a-hitl-boundary)
8. [Future Considerations](#8-future-considerations)

---

## 1. What Problem HITL Solves — and the Philosophy Behind It

Somewhere in nearly every resume-tailoring system, an LLM eventually faces a
bullet like *"Worked on backend services for claims operations"* and is asked
to make it sound like an accomplishment. The model can restructure the
sentence, tighten the verb, name the domain more precisely — but if the
bullet's real weakness is that it has no result, no scale, and no named
system, no amount of rewriting fixes that, because the missing thing was
never in the input to begin with. The only two honest paths forward are: ship
something modest but true, or ask the one party who actually knows the
missing fact. This codebase's entire human-in-the-loop design exists because
its authors chose never to take a third path — quietly inventing the number,
the system name, or the scale to make the sentence read better.

That framing matters because it means HITL, in this system, is not a
usability nicety layered on top of the pipeline. It is one of the load-bearing
pillars of the project's truthfulness guarantee, standing beside the
deterministic truth floor and the code-owned quality engines documented in
[Orchestration Graph §11](orchestration-graph.md#11-case-study-how-experience-optimization-actually-runs)
and [Tool Contracts](tool-contracts.md). Where those mechanisms answer "how do
we stop a model from inventing a fact," HITL answers the question those
mechanisms cannot: "what do we do when a fact is missing and no amount of
rewriting can supply it honestly?" The answer this system gives is: stop, and
ask the person who actually knows.

---

## 2. In Plain Terms, First

Before any diagram, here is what actually happens, described the way you'd
explain it to someone who has never seen the code. While rewriting one
person's work history, the system sometimes notices that a bullet is honest
but thin — it says real work happened, but not what changed, for whom, at
what scale, or using what artifact. Rather than guessing at those details, the
system writes down a specific question about exactly that missing piece and
sets the whole run aside, mid-flight, exactly where it stopped. It writes that
question — and any others like it — into a small file the candidate can open
and edit by hand. Once the candidate has answered as many of those questions
as they can, in their own words, they run the tool again, pointing it at that
same paused folder. The system picks up exactly where it left off, folds the
candidate's answers in as real evidence, and tries the rewrite again — this
time with something true to work with instead of a blank. If that round still
leaves some bullets thin, it can pause again and ask again; it only ever
stops when either every bullet has enough to work with, or the candidate has
answered everything they're going to.

That is one of the two things this document calls "human in the loop." The
other is entirely different and far more blunt: sometimes, after the whole
resume has been assembled and checked, the system simply cannot certify that
the result is safe to hand back — the file wouldn't even render into a form
it could inspect, for instance. There is no question to ask anyone in that
case. The run just stops, flagged, with whatever the system managed to
produce, and whoever operates this system has to look at it directly. No
sheet, no resume path, no follow-up round — that path is a dead end by
design, and Section 5 explains why that's the correct shape for it.

---

## 3. Two Distinct Mechanisms, Named Clearly

```text
                    CANDIDATE CLARIFICATION LOOP           HUMAN REVIEW ESCALATION
                    -----------------------------           ------------------------
WHO is asked        the candidate                           an operator/reviewer
                                                              (the code never says
                                                              which -- see 5.3)

WHAT triggers it    a bullet is truthful but missing a       the system cannot
                    candidate-owned fact (artifact,          CERTIFY the final
                    result, user/scope, or scale)            artifact is safe --
                                                              unrenderable, or ATS
                                                              recovery exhausted

WHEN in the         BEFORE ATS assembly -- there is no       AFTER assembly AND
pipeline            assembled resume yet at all              quality evaluation --
                                                              a (possibly flawed)
                                                              resume already exists

MECHANISM           a real LangGraph interrupt(); a          a state flag
                    checkpoint archived to a resumable        (human_review_required)
                    directory                                folded into the run's
                                                              final disposition

CAN THE RUN         YES -- resume_paused_run(), first-       NO -- there is no
CONTINUE?           class, documented in Section 4           resumption path at all;
                                                              this is a terminal
                                                              disposition (Section 5.2)

WHAT THE PIPELINE   more truthful evidence, so the           nothing -- the pipeline
GAINS               rewrite can improve                      does not change; a
                                                              human acts outside it
```

Both mechanisms are, in the broadest sense, "the system asking a person for
help." Treating them as one concept would be a mistake: one is a designed,
resumable, multi-round conversation with the candidate that makes the output
better; the other is a safety valve that admits the system's own limits and
stops. [Orchestration Graph's human-review-policy section](orchestration-graph.md#8-the-human-review-escalation-policy--one-documented-home)
already covers the escalation policy's *rules*; this document covers *both*
mechanisms in full, and specifically what makes them structurally different.

---

## 4. The Candidate Clarification Loop — Full Architecture

### 4.1 Why Experience Bullets Are Where This Lives

Of everything the pipeline rewrites, work-experience bullets are where a
model is most tempted to manufacture specificity — a fabricated metric or a
named system makes a bullet instantly more compelling, and the pressure to
produce one is highest exactly where evidence is thinnest. The project's own
node-level documentation states the mechanism's purpose plainly: substance is
best-effort, and "whatever is still thin afterward is surfaced to the
candidate, never invented away" (see
[Orchestration Graph §11](orchestration-graph.md#11-case-study-how-experience-optimization-actually-runs)).
No other section of the resume — summary, skills, education — carries this
same fabrication risk at the level of a single, concrete, checkable claim, so
no other section has a clarification loop built around it.

### 4.2 The Fact-Gap Taxonomy — Four Kinds of Candidate-Owned Facts

`ExperienceBulletMissingFactCategory` (`src/hitl/professional_experience/models.py`)
closes the space of what "missing" can mean to exactly four categories:

```text
ARTIFACT     what the candidate built, changed, shipped, or operated
RESULT       what improved because of the work, with or without a metric
USER_SCOPE   who or what the work served -- users, teams, systems, clients
SCALE        size/context -- data volume, request load, team size, rollout scope
```

This closed taxonomy matters because it is what keeps the clarification
question itself precise. A candidate is never asked a vague "can you say more
about this?" — the system has already decided, per bullet, which *one* of
these four kinds of fact is missing, which is what makes the eventual
candidate-facing question specific enough to answer in one sentence.

### 4.3 One LLM Call Makes Two Judgments At Once, Deliberately

`audit_experience_candidate_fact_gaps` (`src/hitl/professional_experience/clarifications.py`)
is a single `request_structured_output` call, at temperature `0.0`, that
decides two things about every shipped bullet in one pass: *whether* a
candidate-owned fact is still missing, and, if so, the *exact question* to
ask for it. The module's own docstring explains why these were deliberately
kept as one judgment rather than split into two calls: "the rubric owns both
decisions... because they are one judgment: what to ask follows directly from
what is missing." Splitting gap-detection from question-phrasing into two
separate calls would risk the two disagreeing with each other — a detector
flagging `SCALE` as missing while a separate phrasing call asks about
`ARTIFACT` instead. Making it one call removes that entire class of
inconsistency by construction.

```text
   INPUT to the one LLM call, per role:
     role context (title, company, description, skills_used)
     every shipped bullet, with its bullet_id, source text, shipped text,
       declared ownership level, and supporting evidence the writer cited

   OUTPUT: ExperienceBulletFactGapReview
     one ExperienceBulletFactGapFinding PER bullet:
       requires_candidate_input: bool
       gap_category: one of the four (Section 4.2), if input is required
       missing_fact_summary, why_candidate_input_is_needed
       question: the exact candidate-facing question, phrased in the SAME pass
```

### 4.4 The Deterministic Join — Code Owns the Sheet's Integrity

The LLM's findings are not shipped to a candidate directly. `clarifications_from_findings`
is pure code — no model call — that joins each finding back to its rewrite
record by `bullet_id` and builds the actual sheet entry. It is explicitly
defensive about a finding that is malformed or incomplete: a finding that
names a `bullet_id` with no matching rewrite record, or that claims
`requires_candidate_input=True` without a `gap_category` or a non-empty
`question`, is **dropped with a logged warning** rather than shipped as a
broken sheet entry. The model is trusted to *judge* what's missing and *phrase*
the question; the *engineering guarantee* that every sheet entry a candidate
sees is well-formed and answerable is entirely code-owned, and it fails
closed — a bad finding simply never reaches the candidate, rather than
reaching them malformed.

### 4.5 What the Candidate Actually Sees (and Why It Matches Reality)

`_shipped_bullet_text` resolves exactly one string per clarification: the
bullet text as it actually shipped, whichever of two possible outcomes that
turned out to be. Recall from
[Orchestration Graph §11](orchestration-graph.md#11-case-study-how-experience-optimization-actually-runs)
that a rewrite failing the truth floor twice causes the pipeline to fall back
to the **original** source bullet rather than ship an unsafe rewrite. This
function specifically checks whether the rewritten text is present in
`shipped_bullets` (the bullets that actually made it into the final section)
— if it is, the candidate is shown the rewrite; if the source-preserved
fallback fired instead, the candidate is shown the *original* bullet, because
that is what genuinely shipped. A candidate is never asked to elaborate on
text that doesn't exist anywhere in their actual resume.

### 4.6 The Pause — A Real Graph Interrupt, Not a Queued Retry

`await_candidate_clarifications` (`src/orchestration/nodes/experience.py`) is
the node where all of the above becomes an actual pause:

```text
   clarifications = state["experience_clarifications"]  (from THIS pass)
   clarification_answers = state["clarification_answers"]  (from a RESUME, if any)

   no clarifications at all?
     -> return {} immediately; nothing to ask, proceed normally

   clarifications exist AND answers are already present (a resumed run)?
     -> return {} immediately; the routing function sends control back to
        optimize_experience to use those answers (Section 4.8)

   clarifications exist AND no answers yet?
     -> interrupt({"type": "candidate_clarifications_required",
                    "questions": [... every clarification, as JSON ...]})
     -> LangGraph genuinely suspends execution HERE. Nothing continues to run
        in this process until the graph is explicitly resumed with a Command.
```

This is a true suspension, not a background retry or a queued job — the
`interrupt()` call is LangGraph's own mechanism for exactly this pattern, and
it is why the pipeline can pause for minutes or for days with zero process
resource consumption while it waits: nothing is running, there is nothing to
poll, and the entire suspended state lives durably on disk (see
[State Management §6](state-management.md#6-durable-state--langgraphs-checkpointer-and-the-msgpack-allowlist)).

### 4.7 The Paused-Run Directory and the CLI Round-Trip

`src/orchestration/runner.py` detects the interrupt (`"__interrupt__" in output`)
and packages the paused run into one self-contained, movable directory (see
[State Management §7](state-management.md#7-the-paused-run-directory--three-files-one-resumable-unit)
for the full mechanics):

```text
   paused_run_<run_id>/
     clarifications_sheet.json    -- every open question, plus instructions:
                                     "Answer the questions below in your own
                                     words using only real facts from your
                                     work..."
     paused_run_manifest.json     -- run_id, original resume/JD paths, status
     checkpoints.sqlite3          -- the actual LangGraph checkpoint history
```

The person on the other end of this pause interacts with it through exactly
one CLI flag, `--resume-from <paused_run_dir>` (`src/main.py`). The workflow
is entirely file-based: open `clarifications_sheet.json` in any text editor,
fill in the `answer` field for as many questions as you can in your own
words, save the file, and re-run the CLI pointed at that same directory. The
CLI's own run summary is explicit about this the first time a run pauses —
printing exactly how many questions are open and exactly which file and flag
to use next.

### 4.8 The Resume — Command(resume=..., update=...) and Re-Entering the Rewrite

`resume_paused_run` loads the manifest and reopens the archived checkpoint
database, loads only the **answered** entries from the sheet (any
clarification whose `answer` field is still blank is silently excluded — see
`load_answered_clarifications`), and resumes the graph with:

```python
pipeline.invoke(
    Command(
        resume={"status": "candidate_answers_submitted"},
        update={"clarification_answers": answered_clarifications},
    ),
    config=config,
)
```

This resumes execution from the exact `interrupt()` boundary inside
`await_candidate_clarifications`, but now with `clarification_answers`
populated in state. The node's own body takes the second branch this time
("clarifications exist AND answers are already present") and returns
immediately, letting `_route_after_candidate_clarifications` see a non-empty
`clarification_answers` and route back to `optimize_experience` — this time
to *re-run* the rewrite, not run it for the first time.

### 4.9 Answers Become Evidence, Not a Bypass — the Truth Floor Still Applies

This is one of the most important architectural guarantees in the whole
system, and it is easy to state wrong. A candidate's answer is **not**
spliced directly into a bullet, and it does **not** skip any part of the
truth-floor or quality-review machinery documented in
[Orchestration Graph §11](orchestration-graph.md#11-case-study-how-experience-optimization-actually-runs).
Instead, `experience_with_candidate_answers` (`src/hitl/professional_experience/answers.py`)
folds each answer into the **role's description** as explicit, attributed
evidence:

```text
   original description:
     "Backend engineer on the claims platform team."

   augmented description (what the rewrite pass actually reads):
     "Backend engineer on the claims platform team.

     Candidate-provided clarifications (their own words, first-class facts):
     - BULLET_ID exp_001::bullet::2 | About "Worked on backend services for
       claims operations.": We handled about 40,000 claims a day across three
       regional teams."
```

The rewrite pass then runs again — the **exact same** truth-floor and
quality-review pipeline from
[Orchestration Graph §11](orchestration-graph.md#11-case-study-how-experience-optimization-actually-runs)
runs on this fresh attempt, with no shortcuts: bullet count parity is still
checked, `detect_claim_inflation` still runs, the one-repair budget still
applies. The candidate's answer earns the rewrite **better evidence to draw
from**, not a bypass of any safety mechanism. If the model still cannot
produce a truthful, specific rewrite from that evidence, the same fallback
rules apply exactly as they would have on a first attempt.

`answers_for_role` is the routing function that makes this safe in the
per-role, parallelized rewrite fan-out documented in
[Orchestration Graph §11](orchestration-graph.md#11-case-study-how-experience-optimization-actually-runs):
it filters the full list of answered clarifications down to only the ones
whose `bullet_id` belongs to *this* role, so a role that received no
clarifications, or whose clarifications weren't yet answered, sees no
injected evidence at all.

### 4.10 The Deepest Design Decision: Answered Facts Permanently Upgrade the Canonical Resume

Look closely at what `optimize_experience` actually returns after a resume:

```python
return {
    "resume": source_resume_for_downstream_review,   # <-- NOT the original resume
    "optimized_experience": optimized_experience,
    "experience_clarifications": clarifications,
    "clarification_answers": [],
}
```

`source_resume_for_downstream_review` is built by
`_resume_with_candidate_answers_as_source`, which applies
`experience_with_candidate_answers` to **every** role in `resume.work_experience`
and writes the result back as `state["resume"]` — overwriting the pipeline's
own copy of the "original" resume, permanently, for the rest of the run.

This has a consequence worth stating explicitly, because it is not obvious
from any single function in isolation: `state["resume"]` is exactly the
object [Stage 5's quality evaluation](orchestration-graph.md#4-stage-by-stage-walkthrough)
later uses as the `original_resume` argument to `evaluate_resume_truthfulness(original_resume, revised_resume)`
— the truthfulness diff that compares the final tailored resume against "the
source of truth." Because candidate-provided facts were folded into that same
`resume` object's role descriptions the moment they were answered, they are
now treated, for every downstream purpose including the final truthfulness
check, exactly as if the candidate had written them into their resume from
day one. A specific number the candidate supplied through a clarification
answer will **not** be flagged later as an unsupported figure "introduced
during optimization" — because by the time that check runs, it is already
part of the accepted, canonical evidence. This is a deliberate architectural
choice: the clarification loop doesn't just inform one rewrite attempt, it
**upgrades what the rest of the system is willing to call true** for the
remainder of the run.

### 4.11 Multi-Round Support — Refining a Simplification From Orchestration Graph

[Orchestration Graph §3.3](orchestration-graph.md#33-zoom-in-the-one-loop-in-the-graph)
described this loop as running "at most once per unanswered question." That
framing is a useful simplification for understanding the graph's shape, but
it undersells what the mechanism actually supports, and this document should
say so plainly: **the loop can pause more than once**, in full generality, and
the code is built to handle that correctly, not merely tolerate it.

After a resumed `optimize_experience` run completes, control always returns
to `await_candidate_clarifications` again — the same node, via the same
graph edge every pass through this section uses. If that fresh pass produced
**new** unresolved clarifications (a different bullet in the same role turned
out to need a fact the first round's answers didn't cover, for instance), and
no answers for *those* new questions exist yet, the node calls `interrupt()`
again, exactly as it did the first time.

```text
   round 1: pause -> candidate answers Q1, Q2 -> resume
   round 1 rewrite: Q1's answer fixes bullet A; bullet B still asks
                    a DIFFERENT question this time (Q3) -- maybe the improved
                    evidence around bullet A surfaced a new gap on bullet B,
                    or bullet B's question from round 1 went unanswered
   round 2: pause AGAIN -> candidate answers Q3 -> resume
   round 2 rewrite: no bullets left needing candidate input
   -> falls through to assemble_ats_resume
```

Two facts keep this from ever becoming an unbounded loop, consistent with the
"no unbounded retry" discipline documented in
[Retry Strategy §10](retry-strategy.md#10-the-no-unbounded-retry-rule-recap):
first, each round strictly requires a real human action to proceed — nothing
in this loop can spin without a person editing a file and re-invoking the
CLI, so there is no risk of automatic, runaway repetition. Second, the set of
addressable gaps is monotonically shrinking: every answered fact permanently
upgrades the canonical resume (Section 4.10), so the same bullet can never
ask the same resolved question twice. The loop terminates because it is
converging on "every bullet has enough evidence," not because of a hardcoded
round limit.

Because the paused-run directory is keyed by the stable `run_id` (not by
round number), a second pause during the same run reuses the identical
`paused_run_<run_id>/` folder — `save_paused_run_state` overwrites
`clarifications_sheet.json` with exactly the current round's open questions
(never a merged history of every round), and `archive_checkpoint_database`'s
same-path guard, documented in
[Idempotency §4](idempotency.md#4-storage-key-idempotency--sets-guarded-no-ops-best-effort-deletes),
makes re-archiving an already-archived checkpoint a safe no-op.

---

## 5. The Human Review Escalation — Full Architecture

### 5.1 The Two Triggers, Precisely

[Orchestration Graph §8](orchestration-graph.md#8-the-human-review-escalation-policy--one-documented-home)
documents this policy's mechanics in full; the summary relevant here is that
`human_review_required` becomes `True` in exactly two situations, both
concerning the rendered-ATS check, and both discovered *after* a resume has
already been assembled:

```text
UNVERIFIABLE  -- the resume could not even be rendered to inspect (no .tex
                 to check at all). Discovered at the QA stage. Nothing to
                 patch, because there is nothing to look at.

UNRECOVERABLE -- the ATS check FAILed, the deterministic section-restore in
                 patch_ats_assembly ran, and the re-graded result is STILL
                 not PASS. Discovered at the patch stage, after automated
                 recovery already ran and failed.
```

Both triggers share a common shape worth naming: they are not questions a
candidate could answer even if asked. Neither "the file wouldn't render" nor
"an essential section is still empty after we tried to restore it from typed
state we already had" is a fact any candidate possesses. This is exactly why
this path cannot be, and structurally is not, a variant of the clarification
loop — there is no question to phrase, because the missing thing here is not
a fact about the candidate's work; it is the system's own inability to
certify its output.

### 5.2 Why This Path Has No Resume Mechanism (And What That Means)

Compare what state exists at the moment each mechanism triggers:

```text
   CANDIDATE CLARIFICATION triggers                HUMAN REVIEW triggers
   (inside optimize_experience,                    (inside evaluate_resume_quality
    BEFORE assembly)                                or patch_ats_assembly,
                                                     AFTER assembly + QA)
   ---------------------------------                ---------------------------------
   state["optimized_resume"] is None                state["optimized_resume"] EXISTS
   -- there is nothing to show anyone yet            -- a (possibly flawed) resume
                                                       already exists to look at

   pausing here and asking a targeted                there is no equivalent targeted
   question genuinely improves what the              question this system could ask
   rewrite can produce next                          that would improve the outcome
```

There is no `interrupt()` call anywhere in `evaluate_resume_quality` or
`patch_ats_assembly`. `human_review_required` is set as a plain boolean field
on state and folded into `derive_run_disposition`'s output — the graph simply
continues running to its natural end (through the render gate, or straight to
`END` if the quality gate also failed, per
[Orchestration Graph §5](orchestration-graph.md#5-the-three-routing-decisions)).
The run **completes**, not pauses. `OrchestrationResult.disposition` comes
back as `NEEDS_HUMAN_REVIEW`, the caller sees whatever artifacts did or didn't
render, and that is the end of this run's story. There is no
`resume_something_after_human_review()` entry point, no manifest, no sheet —
because there is nothing this system knows how to do differently even if a
human confirms the resume is fine. Whatever correction happens next — fixing
a document conversion bug, investigating why a section rendered empty,
manually salvaging the resume — happens entirely outside this pipeline's own
mechanics. This is by design: the escalation exists to stop the system from
quietly shipping something it cannot vouch for, not to model a repair
workflow it has no way to execute automatically.

### 5.3 Who Is "The Human" Here? A Different Audience Than the Candidate Loop

It's worth being honest about a gap the code itself leaves open: nothing in
`human_review_policy.py`, `resume_quality.py`, or `ats_patch.py` specifies
*who* "human review" actually means. The candidate clarification loop is
unambiguously addressed to the candidate — the sheet's own instructions say
"answer the questions below... using only real facts from your work." The
human-review escalation carries no equivalent audience signal; nothing in the
`OrchestrationResult` or the CLI's summary output tells a candidate what to
*do* about a `NEEDS_HUMAN_REVIEW` disposition beyond seeing the word. In
practice this almost certainly means an operator or engineer running the
pipeline, not the candidate — a candidate has no ability to fix "the PDF
render pipeline failed to produce a `.tex` file" — but the codebase does not
say so anywhere, and that ambiguity is worth naming rather than assuming away.

---

## 6. Case Study: One Role, Two Rounds, End to End

Tracing one concrete run through both a first pause and a second pause makes
the whole mechanism concrete:

```text
1. optimize_experience runs for "Senior Engineer, Acme Corp."
   Bullet exp_004::bullet::1 rewrites cleanly (truth floor + quality floor
   both pass). Bullet exp_004::bullet::2 ships as "Improved claims processing
   reliability" -- truthful, but the fact-gap review flags it: RESULT missing,
   asks "What changed as a result -- e.g. error rate, latency, or throughput?"

2. await_candidate_clarifications sees one open question, no answers yet.
   interrupt() fires. runner.py archives the run into
   paused_run_<run_id>/, writes clarifications_sheet.json with that one
   question. CLI prints: "1 question(s) for you -- answer
   clarifications_sheet.json ... and resume with --resume-from".

3. Candidate opens the file, answers: "Cut the claims-processing error rate
   from about 4% to under 1% after the change." Saves. Re-runs the CLI with
   --resume-from paused_run_<run_id>/.

4. resume_paused_run loads that one answered clarification, resumes the
   graph with Command(resume=..., update={"clarification_answers": [...]}).
   await_candidate_clarifications sees answers present, routes back to
   optimize_experience.

5. experience_with_candidate_answers folds the answer into the role's
   description as attributed evidence. The rewrite pass runs AGAIN for this
   role. exp_004::bullet::2 now rewrites to something like "Reduced claims-
   processing error rate from ~4% to under 1% by [supported detail]" --
   passes the truth floor (the figure is now sourced from evidence, not
   invented) and the quality floor.

   But suppose this same pass's fact-gap review now flags a DIFFERENT
   bullet, exp_004::bullet::3, that was never in question before --
   maybe the richer role description surfaced that its ownership claim
   ("led the migration") needs a SCALE fact to back it up.

6. optimize_experience returns experience_clarifications with this ONE
   new question. state["resume"] now permanently carries the FIRST answer
   baked into the role description -- it will never be asked again.

7. await_candidate_clarifications sees a new open question, clarification_
   answers now empty (cleared in step 6). interrupt() fires AGAIN. The
   SAME paused_run_<run_id>/ directory is reused; clarifications_sheet.json
   is overwritten with just this second question.

8. Candidate answers again, resumes again. This round's rewrite succeeds
   for every bullet. optimize_experience returns an EMPTY
   experience_clarifications. await_candidate_clarifications finds nothing
   to ask, falls through. The graph proceeds to assemble_ats_resume, using
   a resume object in which BOTH candidate answers are now permanent,
   trusted evidence -- indistinguishable, to every later truthfulness
   check, from facts the candidate had written down from the start.
```

---

## 7. Design Rule — When Does Something Become a HITL Boundary?

```text
Is the missing thing a FACT only the candidate could possibly know
(a number, a named system, a scope, a scale) -- not a matter of the
model trying harder or being prompted differently?
  YES -> this belongs in the candidate clarification loop. Add it as a
         new ExperienceBulletMissingFactCategory value ONLY if it is
         genuinely a new closed category of candidate-owned fact, not
         a rephrasing of one of the existing four (Section 4.2).

Is the run at a point where NO assembled output exists yet, and asking
a targeted question would give the pipeline better evidence to continue
producing something?
  YES -> this can be a real interrupt() pause with a resumable checkpoint,
         following the pattern in Section 4.6-4.8. It must be resumable
         through a first-class entry point, never a dead end.

Is the problem that the SYSTEM cannot certify its own output is safe or
correct, discovered AFTER the artifact already exists (or failed to
render)?
  YES -> this is a human review escalation (Section 5), a terminal
         disposition flag, not a pause. Do not build a fake "resume"
         path for this category unless the system genuinely has a next
         automated step to take once a human has looked -- today it
         does not, and pretending otherwise would be misleading.

Would answering the question BYPASS the truth floor or quality review
rather than feed it better evidence?
  STOP -> that is not what a clarification answer is for (Section 4.9).
          An answer must always re-enter the SAME validation path a
          model-only rewrite would have to pass, never skip it.
```

---

## 8. Future Considerations

**Whether the human-review escalation should name its intended audience.**
Section 5.3 documents that nothing in the code or the disposition value
specifies who "human review" addresses. As this system's operational usage
grows, deciding explicitly whether `NEEDS_HUMAN_REVIEW` is a candidate-facing
or operator-facing signal — and adjusting the CLI's messaging and the
`OrchestrationResult` schema to say so — would close a real ambiguity rather
than leaving it to be inferred from context each time.

**Whether the `has_candidate_questions` branch in `derive_run_disposition` is
reachable given the current graph topology.** Tracing the graph's edges
carefully (Section 4.11) suggests that `await_candidate_clarifications`
always pauses via `interrupt()` whenever `experience_clarifications` is
non-empty and no fresh answers exist — which would mean a run can only ever
reach `_build_completed_orchestration_result` with an *empty*
`clarifications_requested` list, making the `NEEDS_CANDIDATE_INPUT` branch of
`derive_run_disposition`'s precedence chain unreachable in practice for a
*completed* run (as opposed to a *paused* one, which takes an entirely
separate code path via `_build_paused_orchestration_result`). This is offered
as an observation from static tracing, not a confirmed defect — it is worth a
maintainer's direct verification, since either outcome is informative: if
confirmed unreachable, it's dead defensive code worth removing or
documenting as intentionally defensive; if some path does reach it, that path
is worth naming explicitly, because it isn't obvious from the graph topology
alone.

**Whether a second, narrower escalation category belongs between the two
mechanisms documented here.** Today, a genuinely candidate-answerable gap and
a genuinely system-side certification failure are the only two recognized
shapes. Whether there's a third category worth naming explicitly — a failure
that a *human reviewer* (not the candidate) could plausibly resolve and then
signal the pipeline to continue from, which would need its own first-class
resume mechanism analogous to Section 4 — is an open architectural question
this document surfaces without resolving, since no such case has yet forced
the distinction to be drawn.
