---
name: review-pr
description: Intelligent PR code review with dynamic agent allocation based on change types
allowed-tools: Read, Grep, Glob, Bash, Task
---

# PR Code Review

Intelligent code review for the current branch's Pull Request.
Tailored for a **staff engineer reviewer**: focus on architecture, design, correctness,
and cross-cutting concerns. Nits are collected but flagged for AI auto-fix.

## Arguments

`$ARGUMENTS`

- No arguments: Review PR for current branch
- PR number: Review specific PR (e.g., `/review-pr 123`)
- `--quick`: Quick mode, only Phase 1 + high-level summary

## Quick Start

1. Get current branch PR: `gh pr view --json number,title,state,isDraft`
1. If PR doesn't exist or is closed, stop and explain
1. Execute Phases 1-4 in order

## Workflow Overview

```
Phase 1: PR Analysis & Risk Detection
    |- 1.0 PR Status Check
    |- 1.1 PR Summary
    +- 1.2 Change Classification (auto-detect from codebase)
    |
Phase 2: Review Planning
    |
Phase 3: Execute Review (Parallel Agents)
    |
Phase 4: Staff Engineer Report
```

______________________________________________________________________

## Phase 1: PR Analysis & Risk Detection

### 1.0 PR Status Check

- Closed? -> Stop
- Draft? -> Note but continue
- Bot-generated? -> Skip

### 1.1 PR Summary

Get basic PR info: title, description, modified files, stats.

### 1.2 Change Classification

Auto-detect change types by analyzing the **actual diff and repo structure**.
Do NOT use hardcoded framework lists — infer from what you see.

**Risk Levels:**

| Level      | Heuristics                                                             |
| ---------- | ---------------------------------------------------------------------- |
| CRITICAL   | Core abstractions, distributed/parallel logic, data pipeline, security |
| HIGH       | Public API changes, cross-module refactors, state management           |
| MEDIUM     | Internal logic, new features with tests, config changes with effects   |
| LOW        | Tests only, docs only, comments, formatting, dependency bumps          |

**Classification approach:**
1. Read the diff
2. Identify which subsystems/modules are touched
3. Check if changes cross module boundaries
4. Flag any changes to public interfaces, error handling, or concurrency
5. Note the test coverage of changed code

______________________________________________________________________

## Phase 2: Review Planning

### 2.1 Planning Principles

1. **One task per risk area** — each high-risk subsystem gets a dedicated review
2. **Merge related changes** — files in the same module can share a task
3. **Scale to PR size** — 1 task for trivial PRs, up to 5 for large ones
4. **Minimum coverage** — every PR gets at least 1 review task

### 2.2 Review Focus Areas (pick what applies)

| Focus Area               | When to Use                                   |
| ------------------------ | --------------------------------------------- |
| Architecture & Design    | Cross-module changes, new abstractions         |
| Correctness & Logic      | Algorithm changes, state transitions, edge cases|
| Concurrency & Safety     | Threading, async, locks, distributed ops       |
| API & Contract           | Public interface changes, backwards compat     |
| Error Handling           | New error paths, exception handling, fallbacks |
| Performance              | Hot paths, memory allocation, I/O patterns     |
| Security                 | Auth, input validation, secrets, permissions   |
| Test Quality             | Test coverage, assertion quality, edge cases   |

### 2.3 Output Review Plan

```
REVIEW_PLAN:
1. [CRITICAL] Task Name
   - Focus: Architecture & Design
   - Files: [...]
   - Why: ...

2. [HIGH] Task Name
   - Focus: Correctness & Logic
   - Files: [...]
   - Why: ...
```

______________________________________________________________________

## Phase 3: Execute Review (Parallel Agents)

### 3.1 Execution Rules

- Execute all tasks **in parallel** using Agent tool
- Each agent reviews independently
- Agent prompt must include: focus area, file list, specific concerns from Phase 2

### 3.2 Finding Categories

Every finding must be classified:

| Category                  | Description                                 | Action         |
| ------------------------- | ------------------------------------------- | -------------- |
| **🔴 Architecture**       | Design issues, abstraction leaks, coupling  | Discuss w/ author |
| **🟠 Correctness**        | Bugs, logic errors, race conditions         | Must fix       |
| **🟡 Design suggestion**  | Better patterns, simplification ideas       | Author decides |
| **🔵 Nit**                | Style, naming, formatting, minor cleanup    | AI auto-fix    |

### 3.3 Agent Output Format

```
REVIEW_RESULT:
findings:
  - category: Architecture | Correctness | Design suggestion | Nit
    severity: CRITICAL | HIGH | MEDIUM | LOW
    file: "path/to/file:123"
    issue: "What's wrong"
    reason: "Why it matters"
    suggestion: "How to fix"
```

______________________________________________________________________

## Phase 4: Staff Engineer Report

### 4.1 Report Structure

The report is optimized for a staff engineer who reviews at the **architecture level**
and delegates nits to AI.

```markdown
# PR Review: <title>

## TL;DR
<2-3 sentences: what this PR does, overall assessment, key concern if any>

## Architecture & Design
<High-level observations about the approach, trade-offs, alternatives.
 Write in plain English — no code-speak. Explain like you're talking
 to a smart person who hasn't read the diff.>

## Must-Address Issues
<Only CRITICAL and HIGH findings. Each with:>
- **[file:line]** Issue title
  Why: explanation
  Suggest: fix direction

## Worth Discussing
<MEDIUM findings — design suggestions the author should consider.
 Brief, opinionated. "Consider X because Y.">

## Nits (AI auto-fixable)
<LOW findings — list them briefly. These are for AI to fix, not
 for the author to manually address.>
- `file:line` — description

## Verdict
- [ ] **Approve** — ready to merge
- [ ] **Approve with nits** — merge after AI fixes nits
- [ ] **Request changes** — must-address items need resolution
- [ ] **Needs discussion** — architecture concerns to resolve first
```

### 4.2 Writing Style

- **Plain English** — avoid jargon, implementation details, code-speak
- **Opinionated** — staff engineers have opinions, express them
- **Concise** — if the PR is simple, the review should be short
- **Honest** — if the approach is wrong, say so directly but respectfully

### 4.3 Output Integrity

The Phase 4 report is the **FINAL DELIVERABLE**.

- Output the COMPLETE report as specified in 4.1
- Do NOT abbreviate findings
- If the PR is trivial, the report should be short — that's fine
- The orchestrating agent MUST present it **VERBATIM** to the user

______________________________________________________________________

## Confidence Scoring

Apply to each finding before including in report:

| Score   | Meaning                               | Include? |
| ------- | ------------------------------------- | -------- |
| **0**   | False positive or pre-existing issue  | Drop     |
| **25**  | May be real, cannot verify            | Drop     |
| **50**  | Real but minor or rare                | Nit only |
| **75**  | Very likely real, important           | Yes      |
| **100** | Confirmed real, will frequently occur | Yes      |

Only findings with confidence >= 50 appear in the report.

______________________________________________________________________

## False Positive Guide (Rate Confidence 0)

- Pre-existing issues (not introduced by this PR)
- Intentionally designed code that looks like a bug
- Issues linter/compiler would catch
- Issues on lines user didn't modify
- Explicitly disabled issues (lint ignore comments)

______________________________________________________________________

## Important Notes

- **Do NOT** check build signals or try to build/type-check
- Use `gh` to interact with GitHub, not web fetch
- **Do NOT** automatically post comments to PR
- Must provide file path and line number when referencing issues
- **Repo-agnostic** — do not assume any specific framework or domain;
  detect everything from the actual codebase
