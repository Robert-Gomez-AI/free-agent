---
name: code-review
description: Review a diff or snippet for real correctness, security, and maintainability issues — ignore stylistic preferences.
---

# Code Review

## When to Use

- The user pastes a diff, file, or function and asks for review / feedback / a sanity check.
- The user asks "is this safe?", "does this look right?", "what would you change?".
- After you yourself wrote code at the user's request and they haven't confirmed acceptance.

## What to Look For (in priority order)

1. **Correctness bugs** — off-by-one, wrong condition, mishandled None/null, race conditions, wrong API contract.
2. **Security issues** — injection (SQL, shell, XSS), missing authorization, secrets in logs, insecure defaults, unbounded resource use.
3. **Robustness** — error paths that leak resources, exceptions caught too broadly, retries without backoff, missing timeouts.
4. **Maintainability** — duplicated logic, magic numbers, surprising names, deep nesting.
5. **Performance** — only when something is clearly O(N²)+ on user-facing paths or hits IO in a loop.

Skip nitpicks: bikeshedding about names that are merely fine, formatting (the linter handles it), preferences ("I'd use a dict comprehension here").

## Output Format

Group findings by severity:

```
### Blockers (must fix)
- file.py:42 — <issue>. <why it breaks>. <suggested fix>.

### Should fix
- ...

### Optional / nits
- ...

### Looks good
- <one line on what's correctly done — keeps the review balanced>
```

If the change is safe and clean, say so plainly: "no issues — this is good to merge". Don't manufacture concerns.
