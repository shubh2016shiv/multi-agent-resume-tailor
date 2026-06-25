# The Practical Guide to LLM Response Caching in Resume Tailor
## What `src/core/llm_cache.py` does, how LiteLLM caching actually works, and how to apply the same pattern in any project

> **Scope:** `src/core/llm_cache.py` and the two call sites that use it  
> **Audience:** Developers with little or no prior knowledge of LiteLLM, callbacks, or cache architecture  
> **Goal:** Give you a clear mental model of what is happening and why one small configuration call is enough

---

## Table of Contents

1. [What Problem This Solves](#1-what-problem-this-solves)
2. [The One-Sentence Mental Model](#2-the-one-sentence-mental-model)
3. [What “Caching” Means Here](#3-what-caching-means-here)
4. [Why `configure_llm_cache()` Looks Too Small](#4-why-configure_llm_cache-looks-too-small)
5. [What a Callback Means, in Plain English](#5-what-a-callback-means-in-plain-english)
6. [The Mechanism: What LiteLLM Actually Does](#6-the-mechanism-what-litellm-actually-does)
7. [How It Is Wired in This Project](#7-how-it-is-wired-in-this-project)
8. [What Gets Cached and How LiteLLM Knows Two Requests Match](#8-what-gets-cached-and-how-litellm-knows-two-requests-match)
9. [What Happens on a Cache Hit vs a Cache Miss](#9-what-happens-on-a-cache-hit-vs-a-cache-miss)
10. [Why This Design Is Reasonable](#10-why-this-design-is-reasonable)
11. [Common Misunderstandings](#11-common-misunderstandings)
12. [How To Use This Pattern in Any Other Project](#12-how-to-use-this-pattern-in-any-other-project)
13. [What `src/core/llm_cache.py` Owns and What It Does Not Own](#13-what-srccorellm_cachepy-owns-and-what-it-does-not-own)

---

## 1. What Problem This Solves

LLM calls cost money and time.

During development, the same prompts often get sent again and again:

- rerunning the same test
- retrying a pipeline after a non-LLM bug fix
- reloading a tool with identical inputs
- repeating a task after changing unrelated code

Without caching, every identical call goes back to the provider again.

That means:

- you pay again
- you wait again
- nothing new is learned from the second call

The point of `src/core/llm_cache.py` is simple:

> if we send the exact same LLM request again, reuse the previous answer instead of paying for another provider call

---

## 2. The One-Sentence Mental Model

`src/core/llm_cache.py` turns on LiteLLM’s global response cache once, and every later LiteLLM-backed call automatically uses that cache.

That is the most important sentence in this document.

---

## 3. What “Caching” Means Here

In this project, caching means:

- an LLM request is made once
- the response is saved locally on disk
- if the exact same request happens again, the saved response is returned
- the provider is not called again

This is **response caching**.

It is not:

- token counting
- prompt compression
- vector search
- retrieval
- storing conversation memory

It is much simpler than that.

Think of it like this:

```text
FIRST TIME:
    request -> provider -> response -> save response locally

SECOND TIME, SAME REQUEST:
    request -> local cache -> same saved response
```

---

## 4. Why `configure_llm_cache()` Looks Too Small

At first glance, the code can feel suspiciously small:

```python
configure_llm_cache()
```

That looks like “just configuration,” not “real caching.”

That is a fair reaction.

The reason it works is:

- our code does **not** manually cache each individual response
- our code installs LiteLLM’s cache system once
- LiteLLM’s internal completion path then reads and writes the cache automatically

So `configure_llm_cache()` is small because it is the **switch**, not the **whole machine**.

---

## 5. What a Callback Means, in Plain English

The word **callback** sounds more complicated than it is.

Plain English definition:

> A callback is just “some code the library promises to run for you at a certain moment.”

That is all.

Examples:

- “before the request is sent, run this”
- “after the request succeeds, run this”
- “after the async request succeeds, run this”

You can think of it like an event hook.

Example in plain language:

```text
Library says:
    "When I am about to send an LLM request, I will also run the code registered
     for the 'before request' moment."

We say:
    "Great. One of those hooks should be the cache system."
```

So when this document says LiteLLM “registers cache callbacks,” it means:

> LiteLLM adds cache logic to the moments before and after real LLM calls happen.

---

## 6. The Mechanism: What LiteLLM Actually Does

Here is the actual sequence.

### Step 1: our code creates a LiteLLM cache object

`src/core/llm_cache.py` does this when caching is enabled:

```python
litellm.cache = litellm.Cache(type="disk", disk_cache_dir=".litellm_cache")
```

This does two things:

1. creates a disk-backed cache object
2. registers LiteLLM’s internal cache hooks

That second part is the part many developers miss.

### Step 2: LiteLLM registers its cache hooks

Inside LiteLLM’s `Cache(...)` constructor, it adds `"cache"` into LiteLLM’s internal hook lists.

In plain English:

- before or around a completion call, LiteLLM now knows the cache system should participate
- after a successful call, LiteLLM now knows the cache system should store the result

So the cache object is not passive configuration.

It actively wires itself into LiteLLM’s request lifecycle.

### Step 3: later completion calls automatically consult the cache

When a real LiteLLM-backed completion happens later, LiteLLM:

- builds a cache key from the request parameters
- checks whether that key already exists in the cache
- returns the saved result if found
- otherwise calls the provider and then stores the response

That is why the caching behavior does not have to be manually written at each call site.

---

## 7. How It Is Wired in This Project

This project intentionally turns caching on only at the two real LLM entry points.

### A. Agent-side LLM path

File:

[`src/orchestration/crew_task_execution.py`](/home/shubham_singh/Projects/resume_tailor/src/orchestration/crew_task_execution.py:1)

Function:

`run_agent_task(...)`

Important line:

```python
configure_llm_cache()
```

Meaning:

- before a CrewAI agent kickoff uses the LLM
- make sure LiteLLM’s process-wide cache state matches the current feature flag

### B. Tool-side structured-output path

File:

[`src/tools/llm_gateway/structured_output.py`](/home/shubham_singh/Projects/resume_tailor/src/tools/llm_gateway/structured_output.py:1)

Function:

`request_structured_output(...)`

Important line:

```python
configure_llm_cache()
```

Meaning:

- before a tool-side structured LLM call happens
- make sure the same process-wide cache is configured

### Why only these two places?

Because they are the two real LLM choke points.

That is good design.

It means:

- one place for cache policy
- no duplicated “should we cache here?” logic
- all LiteLLM-backed calls pass through a known boundary

---

## 8. What Gets Cached and How LiteLLM Knows Two Requests Match

LiteLLM does not guess.

It builds a **cache key** from the request parameters.

That includes things like:

- model name
- messages
- generation parameters
- other supported request arguments

Then it hashes that information into a stable key.

So these two requests are treated as the same:

```text
same model
same messages
same relevant parameters
```

But these are treated as different:

- same prompt, different model
- same model, different messages
- same prompt, different temperature or other relevant params

That is how LiteLLM knows what is safe to reuse.

---

## 9. What Happens on a Cache Hit vs a Cache Miss

### Cache hit

A cache hit means:

> LiteLLM found a saved response for this exact request key.

Flow:

```text
request arrives
    -> cache key is built
    -> saved response is found
    -> saved response is returned
    -> provider call is skipped
```

Result:

- faster
- cheaper
- deterministic for that identical request

### Cache miss

A cache miss means:

> LiteLLM did not find a saved response for this request key.

Flow:

```text
request arrives
    -> cache key is built
    -> no saved response exists
    -> provider is called
    -> response comes back
    -> response is written to cache
    -> response is returned
```

Result:

- first call pays the normal cost
- later identical calls can reuse it

---

## 10. Why This Design Is Reasonable

This design is good for three reasons.

### Reason 1: the caching concern stays centralized

`src/core/llm_cache.py` owns:

- whether caching is enabled
- what cache backend is used
- where the cache directory lives
- how process-wide cache state is synchronized

That is much cleaner than sprinkling caching decisions through many agent or tool modules.

### Reason 2: callers stay simple

The caller only needs:

```python
configure_llm_cache()
```

Then it can continue making normal LLM calls.

That keeps LLM entrypoints readable.

### Reason 3: the third-party library owns the low-level cache mechanics

We do not need to re-implement:

- cache key generation
- cache lookup
- cache write-back
- provider bypass on hit

LiteLLM already does that.

We only need to turn it on at the right place.

---

## 11. Common Misunderstandings

### “This function only configures. It does not cache.”

Partly true, but incomplete.

Our function does only configure.

But that configuration installs the LiteLLM machinery that performs the actual caching later.

So the correct sentence is:

> `configure_llm_cache()` does not itself save a response, but it turns on the LiteLLM system that will save and reuse responses automatically.

### “If I do not see explicit read/write cache code here, nothing is happening.”

Not true.

This is one of those cases where the library owns the behavior after setup.

You do not see the read/write logic in our file because LiteLLM handles it internally.

### “Does CrewAI use this?”

In this project, yes.

The agent and structured-output paths both pass through LiteLLM-backed LLM usage, so configuring LiteLLM’s global cache matters.

### “Is this caching every possible thing in the app?”

No.

It caches LiteLLM-supported call types, not arbitrary business logic.

---

## 12. How To Use This Pattern in Any Other Project

If you want to reuse this design elsewhere, the pattern is simple.

### Step 1: choose one cache module

Create one module that owns:

- cache enable/disable policy
- backend choice
- cache location
- any logging around cache state

Do not spread this across many files.

### Step 2: put cache setup at real LLM choke points

Find the few places where the application truly enters the LLM layer.

Examples:

- “call one agent”
- “request one structured output”
- “send one provider completion”

Call your cache configurator there.

Do **not** call it randomly in unrelated business code.

### Step 3: let the library own the low-level cache lifecycle

If the library already knows how to:

- build cache keys
- look up responses
- store responses
- bypass provider calls on cache hits

then do not rewrite that behavior yourself.

Use the library’s mechanism cleanly.

### Step 4: test the policy, not the library internals

Your tests should prove:

- enabling the flag turns cache on
- disabling the flag turns cache off
- changing the flag updates process state correctly

You do not need to re-test LiteLLM’s whole internal cache engine.

That is why this project’s unit tests focus on configuration behavior, not provider-level replay semantics.

---

## 13. What `src/core/llm_cache.py` Owns and What It Does Not Own

This module owns:

- process-wide LiteLLM cache configuration
- disk cache enable/disable behavior
- synchronizing cache state with `feature_flags.enable_cache`

This module does **not** own:

- token budgeting
- retry logic
- agent orchestration
- structured-output validation
- provider selection
- cache key algorithm internals

Those belong elsewhere.

That separation is intentional.

---

## Final Summary

If you want the shortest accurate explanation:

1. `configure_llm_cache()` installs LiteLLM’s disk cache for the process.
2. LiteLLM’s cache object registers internal hooks for request-time cache handling.
3. Later LiteLLM-backed completion calls automatically:
   - build a request key
   - look for a saved response
   - return it on a hit
   - or call the provider and save the new response on a miss
4. This project calls `configure_llm_cache()` only at the two true LLM entry points.

So yes, the small setup call really is enough, because the actual caching work happens later inside LiteLLM’s own completion path.
