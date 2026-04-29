# Sample tools

Drop any of these into `./free_agent_tools/` next to where you run `free-agent`
and they will be auto-discovered at boot. They demonstrate three common
patterns you can adapt:

| File | Pattern |
|---|---|
| [`http_get.py`](http_get.py) | I/O against an external service (HTTP), with timeouts and bounded output |
| [`calculator.py`](calculator.py) | Pure computation with input safety (AST walker, no `eval`) |
| [`file_search.py`](file_search.py) | Filesystem access scoped to the cwd |

## The `@tool` contract

Every tool is just a Python function decorated with `@tool` from
`langchain_core.tools`:

```python
from langchain_core.tools import tool

@tool
def my_tool(arg1: str, arg2: int = 5) -> str:
    """One-line summary the agent reads to decide when to call this.

    Optional longer description for humans reading the source.
    """
    ...
    return "string result"
```

Hard rules:

- **Return a string.** The agent reads tool output as text. If you need to
  return structured data, JSON-encode it.
- **Use stdlib types in the signature** (`str`, `int`, `float`, `bool`,
  `list[str]`). LangChain auto-derives a JSON schema from the type hints.
- **First docstring line is the selector.** It's literally what the
  orchestrator (and any subagent that can call this tool) sees when deciding
  whether to invoke. Be specific and action-oriented.
- **Catch foreseeable errors and return a human-readable string.** Don't let
  exceptions bubble up to the agent — it can't recover gracefully.
- **Be idempotent and bounded.** No infinite loops, no unbounded memory, no
  silent state mutation that the user can't see.

## Adding extra dependencies

Stdlib-only tools are zero-friction. If your tool needs a third-party package
(e.g. `requests`, `duckduckgo-search`, `feedparser`), add the dep to your
project's `pyproject.toml` (or run `uv add <pkg>` in the project where you'll
use the CLI). The tool file itself just imports normally.

## Generating a tool with the LLM

Inside the chat, run `/tool new` to walk through a wizard that uses the
configured model to draft the source for you. The wizard restricts the LLM
to stdlib-only and string returns by default — the resulting file lands in
the same `./free_agent_tools/` directory.
