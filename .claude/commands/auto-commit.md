---
name: auto-commit
description: Generate intelligent commit messages based on staged changes. Invoke with /auto-commit.
---

# Generate Commit Message

Generate a well-formatted commit message based on staged changes.

## Usage

```
/auto-commit [--push <remote_branch_name>]
```

**Arguments:**

- `--push <remote_branch_name>`: Push to the specified remote branch remote_branch_name; otherwise, push to the default remote branch.

## Workflow

### Step 1: Analyze Changes

```bash
# Check staged files
git diff --cached --name-only

# Check staged content
git diff --cached

# Check recent commit style
git log --oneline -5
```

### Step 2: Categorize Changes

| Type       | When to Use                     |
| ---------- | ------------------------------- |
| `feat`     | New feature or capability       |
| `fix`      | Bug fix                         |
| `docs`     | Documentation only              |
| `refactor` | Code change without feature/fix |
| `test`     | Adding or fixing tests          |
| `chore`    | Build, deps, config changes     |
| `perf`     | Performance improvement         |

Group related changes into small, focused commits.


### Step 3: Generate Message

Group related changes into small, focused commits,
Each commit message should be descriptive and conventional, but keep CONCISE (ideally under 24 characters for the subject line). Use the format:

**Format:**

```
<type>: <subject>

<body>

[Optional sections:]
Key changes:
- change 1
- change 2

Refs: #123, #456
```

Use `git commit -s` for automatic Signed-off-by.

**Rules:**

- Subject: imperative mood, ~16-32 chars, no period
- Body: explain "why" not "what", wrap at 32 chars
- Key changes: bullet list of main modifications (for complex commits)
- Refs: reference issues/PRs if applicable
- Keep concise as much as possible

### Step 4: Commit

Show preview:

```
─────────────────────────────────────
feat: add retry logic to client

Add exponential backoff for transient failures.
Configurable max retries via environment variable.
─────────────────────────────────────
```

Then execute:

```bash
git commit -m "$(cat <<'EOF'
<message>
EOF
)"
```

## Examples

**Single file fix:**

```
fix: handle null input in parser

Return empty result instead of raising exception
when input string is empty after extraction.
```

**Docs only:**

```
docs: update API comparison table

Add new methods to the API reference
with usage examples.
```

______________________________________________________________________

<!--
================================================================================
                            MAINTAINER GUIDE
================================================================================

Location: .claude/commands/gen-commit-msg.md
Invocation: /gen-commit-msg

## Design Philosophy

- Automates commit message generation following Conventional Commits format
- Matches repository's existing style
- Requires user confirmation before commit

## How to Update

### Adding New Scopes
Update "Determine Scope" section with new file path mappings.

### Changing Format
Update "Generate Message" format template and rules.

================================================================================

<!-- 
**Multi-file feature:**

```
feat(engine): add CPU offload support

Enable memory saver for model offloading during
processing to reduce GPU memory pressure.

Key changes:
- Add offload/onload methods to Engine
- Integrate with update flow
- Handle platform compatibility
``` -->