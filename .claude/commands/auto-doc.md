---
name: auto-doc
description: Document the current discussion into a concise report saved to disk for future reference. Invoke with /auto-doc.
---

# Document Discussion

Save the current conversation as a concise, structured report for future reference.

## Usage

```
/auto-doc [path_to_save]
```

**Arguments:**

- `[path_to_save]`: Directory to save the report. Defaults to `<project_root>/findings/`.

## Workflow

### Step 1: Analyze the Discussion

Review the full conversation and identify:

- **Primary goal**: What was the user trying to accomplish?
- **Key decisions**: What choices were made and why?
- **Discoveries**: What was learned about the codebase, tools, or domain?
- **Actions taken**: What files were created, modified, or commands run?
- **Open items**: Anything unresolved or deferred?

### Step 2: Generate the Report

Write a markdown report with this structure:

```markdown
# <Descriptive Title>

**Date:** YYYY-MM-DD
**Context:** <1-sentence summary of what prompted the discussion>

## Summary

<2-4 sentence overview of the discussion and outcome>

## Key Findings

- Finding 1
- Finding 2

## Decisions Made

| Decision | Rationale |
| -------- | --------- |
| ...      | ...       |

## Changes Made (if any)

- `path/to/file` — what changed and why

## Open Items (if any)

- [ ] Item 1
- [ ] Item 2
```

**Rules:**

- Keep the report **under 100 lines** — favor conciseness over completeness
- Use bullet points, not paragraphs
- Include file paths and command snippets where relevant
- Omit sections that have no content (e.g., skip "Open Items" if none exist)
- Write for a reader who was **not** part of the conversation

### Step 3: Save the Report

1. Determine the save path:
   - If `[path_to_save]` is provided, use it
   - Otherwise, use `<project_root>/findings/`

2. Create the directory if it doesn't exist:

```bash
mkdir -p <save_path>
```

3. Generate the filename using the pattern:

```
report_<slug>_<YYYYMMDD_HHMMSS>.md
```

Where `<slug>` is a short (2-4 word) kebab-case summary of the topic, e.g.:

- `report_api-refactoring-plan_20260323_043500.md`
- `report_auth-token-bug_20260323_120000.md`

4. Write the file and confirm:

```
Saved: <full_path_to_report>
```

## Examples

**Architecture exploration session:**

```
Saved: findings/report_dispatch-layer-analysis_20260323_043500.md
```

**Bug investigation:**

```
Saved: findings/report_cache-invalidation-fix_20260323_091200.md
```

______________________________________________________________________

<!--
================================================================================
                            MAINTAINER GUIDE
================================================================================

Location: commands/auto-doc.md
Invocation: /auto-doc

## Design Philosophy

- Captures discussion context that would otherwise be lost between sessions
- Structured format makes reports scannable and searchable
- Slug-based filenames enable quick identification without opening files

## How to Update

### Changing Report Structure
Update the markdown template in "Step 2: Generate the Report".

### Changing File Naming
Update the filename pattern in "Step 3: Save the Report".

### Adding New Sections
Add to the template and update the rules accordingly.

================================================================================
-->
