# Elegant Refactoring Principles

This document is a reusable guide for refactoring code into something easier to read,
safer to change, and easier to explain to the next developer.

It is not tied to one framework, one language feature, or one project. The goal is to
capture the principles behind readable code so they can be reused across many systems.

Use this when:
- restructuring a package
- splitting mixed-responsibility files
- reducing overengineering
- deciding whether an abstraction is helping or hurting
- reviewing whether code is readable without a long explanation

---

## 1. The Main Goal

Elegant code is not code that is merely short.

Elegant code is code that:
- reveals its purpose quickly
- keeps responsibilities separated
- avoids forcing readers to hold too much in their heads
- makes decisions visible instead of hiding them
- is easy to modify without fear

The standard is simple:

> A new developer should be able to open a file and understand what it owns, why it
> exists, and how it fits into the system without needing a private explanation.

---

## 2. The Core Idea

The most important rule is:

> Organize code by meaning first, not by implementation accident.

Most ugly codebases are not ugly because the algorithms are impossible. They become ugly
because structure follows historical accidents:
- where the first code happened to be written
- what was easiest to patch
- what was copied from somewhere else
- what one helper file silently absorbed over time

Elegant refactoring replaces accidental structure with intentional structure.

---

## 3. Elegance Principles

### 3.1 One file should own one clear thing

A file should have one positive identity.

Good:
- `payment_validation.py`
- `resume_extraction.py`
- `section_header_checks.py`

Bad:
- `helpers.py`
- `common.py`
- `manager.py`
- `processor.py`

If you cannot describe the file in one short sentence, it owns too much.

Wrong:
```python
# helpers.py
def read_user():
    ...

def validate_invoice():
    ...

def build_email():
    ...
```

Right:
```python
# user_loading.py
def read_user():
    ...

# invoice_validation.py
def validate_invoice():
    ...

# email_rendering.py
def build_email():
    ...
```

---

### 3.2 Separate what the system exposes from how it works internally

A clean system separates:
- public entry points
- internal implementation

Readers should quickly know:
- “this is something callers are meant to use”
- “this is supporting machinery”

Wrong:
```python
# same file mixes public endpoint, parsing, validation, persistence,
# and response formatting
def handle_request(payload):
    ...

def _parse(payload):
    ...

def _validate(data):
    ...

def _save(data):
    ...

def _format_response(result):
    ...
```

Right:
```python
# api_handler.py
def handle_request(payload):
    parsed_request = parse_request(payload)
    validated_request = validate_request(parsed_request)
    saved_record = save_record(validated_request)
    return format_response(saved_record)
```

And the lower-level operations live in modules that own those concerns.

---

### 3.3 Name things by domain meaning, not technical role

A name should explain what the thing is about, not just what kind of thing it is.

Wrong:
- `validator.py`
- `processor.py`
- `handler.py`
- `tool.py`

Right:
- `section_header_checks.py`
- `keyword_coverage.py`
- `invoice_status_fetcher.py`
- `customer_profile_renderer.py`

Why this matters:
- role names repeat everywhere
- domain names help a reader search, navigate, and remember

If a name is clear only after opening the file, the name is weak.

---

### 3.4 Put structure where readers expect to find it

The directory layout should teach the system.

A reader should not need tribal knowledge like:
- “this folder is called `utils`, but the real business logic is there”
- “these files are validators, except some are actually wrappers”
- “this module is named after one thing but secretly owns three others”

Wrong:
```text
src/
  utils/
    parser.py
    validator.py
    service.py
    formatter.py
```

Right:
```text
src/
  parsing/
    request_parsing.py
  validation/
    request_validation.py
  rendering/
    response_rendering.py
  services/
    customer_lookup.py
```

The directory tree is the first documentation most developers read.

---

### 3.5 Prefer plain functions unless a class clearly earns its place

Classes are not automatically cleaner.

Use a class only when it truly represents:
- durable state
- identity
- interchangeable behavior
- a meaningful object in the domain

Do not invent classes to make code “look structured”.

Wrong:
```python
class EmailFormatter:
    def format_email(self, user_name: str) -> str:
        return f"Hello {user_name}"
```

Right:
```python
def format_email(user_name: str) -> str:
    return f"Hello {user_name}"
```

Use a class when it carries real value:

```python
class Invoice:
    def __init__(self, line_items: list[LineItem]) -> None:
        self.line_items = line_items

    def total_amount(self) -> Decimal:
        ...
```

The rule:

> If a function is enough, a class is extra weight.

---

### 3.6 Do not create abstraction before there is real duplication pain

Abstraction is justified by repeated pressure, not by anticipation.

Wrong:
```python
class BaseProcessor:
    def run(self):
        raise NotImplementedError

class UserProcessor(BaseProcessor):
    ...
```

If there is only one processor, this is ceremony.

Right:
```python
def process_user(user_data: dict) -> ProcessedUser:
    ...
```

Add abstraction only when:
- multiple implementations already exist
- duplication is real and recurring
- the common contract is stable

Otherwise, abstraction hides code before it helps.

---

### 3.7 Avoid mixed-level files

One of the biggest readability killers is mixing multiple levels of concern in one file.

Examples of mixed levels:
- business rules + HTTP formatting
- database writes + UI text
- agent-exposed wrapper + internal algorithm
- domain logic + retry infrastructure

Wrong:
```python
def validate_resume():
    ...

@tool("Validate Resume")
def validate_resume_tool():
    ...

def connect_to_database():
    ...
```

Right:
- one file for internal validation logic
- one file for wrapper/exposed tool
- one file for persistence infrastructure

The file should not force the reader to constantly change mental level.

---

### 3.8 Make the top of the file do the teaching

A reader should learn most of what matters from:
- the module docstring
- the function names
- the function signatures
- the short comments above important steps

If a reader must dive into every line before understanding the point, the surface is weak.

Good surface:
```python
def audit_keyword_coverage(resume_text: str, required_keywords: list[str]) -> ReviewResult:
    """Measure which required keywords appear in the resume text."""
```

Bad surface:
```python
def run(data):
    ...
```

You should not need detective work to discover the contract.

---

### 3.9 Use comments to explain decisions, not obvious syntax

Comments should answer:
- why this exists
- why this algorithm was chosen
- what tradeoff is being made
- what a reader might otherwise misunderstand

Comments should not narrate the obvious.

Wrong:
```python
# Assign value to count
count = len(items)
```

Right:
```python
# Count short lines because multi-column PDFs often break into many short fragments
short_line_count = sum(1 for line in non_empty_lines if len(line) < 30)
```

Best use of comments:
- where the reasoning matters more than the syntax

---

### 3.10 Use step comments to show flow, not to decorate code

Step comments are useful when a function has a sequence a reader should follow.

Good:
```python
####################################################
# STEP 1: TURN RAW INPUT INTO A CLEAN SHAPE#
####################################################

####################################################
# STEP 2: RUN THE CORE CHECKS#
####################################################

####################################################
# STEP 3: BUILD THE FINAL RESULT#
####################################################
```

This helps when:
- the function coordinates several operations
- the order matters
- the code is easier to read as a small pipeline

Do not add step comments to trivial one-liners. Use them where they improve flow.

---

### 3.11 Keep helper functions only when they clarify meaning

Not every repeated line deserves a helper.

Good helper:
- it gives a meaningful name to an idea
- it removes noise
- it improves scanability

Bad helper:
- it just wraps one obvious line
- it forces the reader to jump away for no gain
- it fragments the logic into tiny meaningless pieces

Wrong:
```python
def _get_name(user):
    return user.name
```

Right:
```python
def build_resume_identifier(company_name: str, role_title: str, start_date: date) -> str:
    ...
```

The test:

> If inlining the code would make the caller easier to understand, do not extract it.

---

### 3.12 Reduce private-function clutter

A leading underscore is not evil, but overuse makes a file noisy.

Use a private helper when it is:
- substantial
- clearly internal
- worth naming

Do not break one function into six underscored fragments just to appear “modular”.

Wrong:
```python
def process():
    data = _load()
    cleaned = _clean(data)
    checked = _check(cleaned)
    result = _build(checked)
    return _finish(result)
```

This is often just over-fragmentation.

Better:
```python
def process_invoice(invoice_text: str) -> InvoiceResult:
    parsed_invoice = parse_invoice(invoice_text)
    validated_invoice = validate_invoice(parsed_invoice)
    return build_invoice_result(validated_invoice)
```

Keep helpers where they genuinely isolate a reusable or complex idea.

---

### 3.13 Prefer direct data flow over hidden flow

A reader should be able to follow:
- what comes in
- what transforms it
- what comes out

Good code passes data openly.

Bad code hides flow in:
- side effects
- global registries
- magically mutated state
- multi-step hidden wrappers

Wrong:
```python
registry.register(data)
manager.process()
builder.finalize()
```

Right:
```python
validated_data = validate_data(raw_data)
processed_data = process_data(validated_data)
final_result = build_result(processed_data)
```

Open data flow is easier to debug and easier to trust.

---

### 3.14 Keep contracts stable and small

Shared contracts should be:
- obvious
- small
- meaningful
- validated when worth validating

Do not create a large model hierarchy just to look “enterprise”.

A good contract:
- solves a real boundary problem
- keeps callers honest
- is stable enough to be worth sharing

Wrong:
```python
class BaseResponse(BaseModel): ...
class ExtendedResponse(BaseResponse): ...
class FinalExtendedResponse(ExtendedResponse): ...
```

Right:
```python
class ReviewResult(BaseModel):
    comments: list[ReviewComment]
    summary: str = ""
    score: float | None = None
```

Small contracts are easier to carry across the codebase.

---

### 3.15 Keep configuration for real runtime decisions, not every local constant

Not every threshold belongs in global config.

Put something in config when:
- the business may tune it
- environments may differ
- operators need to control it

Keep it local when:
- it is an implementation detail
- it only supports one algorithm
- moving it to config would add indirection but not clarity

Wrong:
- exporting every local constant into YAML

Right:
- keep local heuristics local
- lift true runtime policy into configuration

This keeps configuration meaningful instead of becoming a dumping ground.

---

### 3.16 Refactor for the next reader, not for aesthetic symmetry

Symmetry is nice, but readability matters more.

Do not split code just because:
- every folder “should” have the same number of files
- every concept “should” have a matching class
- every public function “should” have an interface

Refactoring should make reading easier, not just make diagrams prettier.

The question is always:

> Did this make the system easier to understand and safer to change?

If the answer is no, the refactor is ornamental.

---

## 4. A Practical Refactoring Checklist

Before refactoring, ask:

1. What does this file/package actually own?
2. What should remain public?
3. What is internal implementation?
4. Where are two or more concerns mixed together?
5. Which names are hiding meaning?
6. Which helpers are clarifying, and which are noise?
7. Which abstractions are earned, and which are decorative?
8. Which constants are true policy, and which are local detail?
9. Can a new developer follow the flow in 2-3 jumps?
10. After the change, will the structure teach itself?

---

## 5. Right vs Wrong Patterns

### Pattern: Mixed wrapper and internal logic

Wrong:
```python
@tool("Validate Invoice")
def validate_invoice_tool(raw_text: str) -> str:
    ...

def validate_invoice(raw_text: str) -> ValidationResult:
    ...
```

in the same file as database helpers and formatting helpers.

Right:
```python
# agent_tools/invoice_tools.py
@tool("Validate Invoice")
def validate_invoice_tool(raw_text: str) -> str:
    review_result = validate_invoice(raw_text)
    return render_review_result(review_result)

# validation/invoice_validation.py
def validate_invoice(raw_text: str) -> ValidationResult:
    ...
```

---

### Pattern: Generic helper buckets

Wrong:
```text
utils/
helpers/
common/
misc/
```

Right:
```text
keyword_matching/
section_validation/
resume_rendering/
request_parsing/
```

---

### Pattern: Premature base classes

Wrong:
```python
class BaseValidator:
    def validate(self, value):
        raise NotImplementedError
```

when there is one validator.

Right:
```python
def validate_email_address(email_address: str) -> ValidationResult:
    ...
```

---

### Pattern: Tiny private helper explosion

Wrong:
```python
def run():
    user = _get_user()
    name = _get_name(user)
    email = _get_email(user)
    return _build_result(name, email)
```

Right:
```python
def build_user_contact_summary(user: User) -> str:
    return f"{user.full_name} <{user.email_address}>"
```

---

### Pattern: Invisible contract

Wrong:
```python
def run(data):
    ...
```

Right:
```python
def analyze_keyword_coverage(
    resume_text: str,
    required_keywords: list[str],
) -> ReviewResult:
    """Measure which required keywords appear in the resume text."""
```

---

### Pattern: Comments that explain syntax instead of decisions

Wrong:
```python
# Loop through users
for user in users:
    ...
```

Right:
```python
# Keep the first matching user because the upstream contract guarantees uniqueness
for user in users:
    ...
```

---

## 6. The Simplicity Tests

Use these tests on any refactor.

### Test 1: The tab test

If someone sees the filename in an editor tab, do they know what it probably owns?

If not, rename it.

### Test 2: The first-screen test

If someone reads only the first screen of the file, do they understand:
- what the file is for
- what the main function does
- what level of responsibility it owns

If not, improve the surface.

### Test 3: The jump-count test

How many files must a new developer open to understand one behavior?

If the answer is “too many for a simple feature,” collapse structure.

### Test 4: The no-explanation test

Would this code still be understandable if you were not there to explain it?

If not, the code depends too much on author memory.

### Test 5: The fear test

Would a careful developer be afraid to change it because the structure is too hidden,
too magical, or too indirect?

If yes, the code is not elegant yet.

---

## 7. What Elegant Code Feels Like

When code is elegant:
- you stop scrolling to understand the point
- names carry weight
- functions read like actions
- files have identity
- comments explain decisions
- boundaries are visible
- data flow feels linear
- there are fewer surprises

Elegant code reduces cognitive load.

That is the real goal.

---

## 8. Final Rule

If you remember only one thing, remember this:

> Refactor toward code that teaches itself.

If a developer must repeatedly ask the author what a file is for, why a layer exists,
or where the real logic lives, the refactor is incomplete.

Readable code is not just easier to maintain.

Readable code is easier to trust.

---

## 9. How To Use This In Other Projects

If you want to reuse this document as an instruction for future refactors, use it in
three ways.

### 9.1 Use it as a review standard

When reviewing a module, ask:
- does each file own one clear thing
- are public entry points separate from internal logic
- are names based on domain meaning
- is there any abstraction that exists only for style
- does the code explain its decisions without a verbal tour

If several answers are "no", the module likely needs refactoring.

---

### 9.2 Use it as a refactoring brief for coding agents

You can give a coding agent a short instruction like this:

> Refactor this module for readability and simplicity. Keep behavior intact unless the
> current design is clearly hacky or misleading. Separate public entry points from
> internal logic. Rename files, functions, variables, and intermediate values so they
> explain domain meaning clearly. Remove decorative abstractions. Keep functions small
> and linear. Add comments only where they explain a decision, a tradeoff, or a step
> in the flow. Prefer code that a new developer can understand without extra
> explanation.

This works better than vague instructions such as:
- "clean this up"
- "make it better"
- "improve architecture"

Those phrases are too open and often lead to unnecessary abstraction.

---

### 9.3 Use it as a self-check before merging

Before merging a refactor, ask:
- did readability improve immediately on first read
- did the structure become more obvious
- did the number of mental jumps go down
- did comments become more useful instead of more numerous
- did we remove accidental complexity instead of moving it around

If the refactor made the code more layered but not more understandable, it is not done.
