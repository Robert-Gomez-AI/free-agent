# free-agent

A terminal-first base for building local agent and multi-agent systems on top
of LangChain's [deepagents](https://docs.langchain.com/oss/python/deepagents/overview).
Bring your own LLM (Ollama, Anthropic, ...), your own tools (Python `@tool`
functions), and your own agents (declared in YAML or built interactively from
the chat).

```
░▒▓ neural shell :: deepagents runtime :: zero leash ▓▒░
 NET online   MODEL qwen3.5:9b   LINK ollama
```

## What you get

- **Local-first.** Runs against any local Ollama model with tool calling
  (Qwen, Llama 3.1+, Mistral, GLM, ...) or Anthropic's API. Pick per project.
- **Streaming chat REPL** with cyberpunk styling, slash commands, history,
  Ctrl+C cancel, Ctrl+D quit.
- **Subagents you can author or generate.** Declare them in
  `free-agent.yaml` or run `/sub new` and the LLM drafts the system prompt.
- **Tools you can author or generate.** Drop a `@tool` function into a
  folder, or run `/tool new` and the LLM writes the Python.
- **Single-file YAML config**, hot-reloaded from disk with backup.
- **Three-layer architecture** (CLI · agent · infrastructure) so you can swap
  the rendering, the model backend, or the tool registry without touching the
  rest.

## Install

### As a CLI you use across projects

```bash
uv tool install --from git+https://github.com/Robert-Gomez-AI/free-agent free-agent
free-agent          # available globally — run from any project directory
```

The CLI reads its config **relative to the cwd** where you launch it:

| File | Purpose |
|---|---|
| `.env` | provider + API keys (see `.env.example`) |
| `free-agent.yaml` | optional — main agent prompt, tool subset, subagents |
| `free_agent_tools/*.py` | optional — your custom tools, auto-loaded |

So a typical workflow is `cd ~/projects/some-project && free-agent` and that
project's config + tools come along automatically.

To upgrade later: `uv tool upgrade free-agent`. To remove:
`uv tool uninstall free-agent`.

### As a base you fork

Use GitHub's "Use this template" button, or:

```bash
git clone https://github.com/Robert-Gomez-AI/free-agent
cd free-agent
uv sync
cp .env.example .env          # set your provider
uv run free-agent
```

If you fork, customizing the package itself is fair game — rename it in
`pyproject.toml`, swap the cyberpunk banner in `cli/console.py`, replace the
default system prompt in `agent/prompts.py`. The architecture is small enough
to read end-to-end in an afternoon.

## Letting the agent touch the filesystem

By default, the agent's `read_file` / `write_file` / `ls` / `edit_file` tools
operate on an **in-memory virtual filesystem** that exists only for the
session. To let the agent actually modify files in your working directory,
pass `--writable`:

```bash
free-agent --writable          # or -w
# or persist via env: FREE_AGENT_WRITABLE=1
```

When enabled:

- The agent's filesystem tools route through `FilesystemBackend` rooted at the
  cwd, with `virtual_mode=True` — paths cannot escape via `..`, `~`, or
  absolute paths outside the root.
- The boot banner shows a red `[ WRT ] WRITE MODE — agent can modify files
  under <cwd>` line so you don't forget.
- `/agent` shows `filesystem: WRITABLE → <path>` in the profile panel.

> ⚠️ **What you're agreeing to.** With `--writable`, the agent (and any
> subagent that has access to the filesystem tools) can create, modify, and
> delete files inside the cwd. Run only in directories you're comfortable
> letting it touch — typically your project's working tree, with git so you
> can review the diff. For higher-risk environments, leave it off; the
> in-memory backend is enough for most chat use cases.

## Pick a model provider

Set `FREE_AGENT_PROVIDER` in `.env`. Defaults to `ollama`.

### Ollama (local, open-source) — default

> **Heads up:** `uv tool install free-agent` installs the Python CLI, **not
> Ollama itself**. Ollama is a separate native daemon that runs on your
> machine. Install it once from [ollama.com](https://ollama.com) (or `brew
> install ollama` on macOS, the install script on Linux).

1. Install Ollama and make sure the daemon is running (`ollama serve` or via
   the desktop app / systemd).
2. Pull a **tool-capable** model. deepagents requires the model to support
   function/tool calling — most modern open-source models do:
   ```bash
   ollama pull qwen3.5:9b      # solid default, ~6 GB
   # also good: llama3.1:8b · mistral-nemo · hermes3 · command-r
   ```
   …or pull from inside the chat with `/model pull qwen3.5:9b` (see below).
3. In `.env`:
   ```
   FREE_AGENT_PROVIDER=ollama
   FREE_AGENT_OLLAMA_MODEL=qwen3.5:9b
   ```

The boot sequence does a preflight against Ollama and lists what you have if
the configured model isn't pulled — so a misconfigured model fails fast with
a clear message, not a buried 404.

### Manage models from inside the chat

The `/model` commands wrap the Ollama HTTP API so you don't have to leave the
chat to pull or switch models:

```
/model                  ← list models on this host (▶ marks the active one)
/model browse           ← scrape ollama.com/library and show pullable models
/model browse all       ← include non-tool-calling models too
/model browse qwen      ← filter by substring (name + description + sizes + caps)
/model browse refresh   ← bypass the 6h cache and re-scrape
/model pull qwen3:14b   ← download with live progress; offers to switch on success
/model use qwen3:14b    ← switch active model in-session (rebuilds the agent)
/model rm gemma2:27b    ← delete a local model (refuses if it's the active one)
```

`/model browse` hits **ollama.com/library** live and parses out every
publicly-listed model (~225 entries as of writing). Default view filters to
**tool-capable** models (deepagents requires tool calling — non-capable
models are hidden but counted). Result is cached at
`~/.cache/free-agent/library.json` for 6h to avoid hammering ollama.com on
every browse.

| Marker | Meaning |
|---|---|
| `▶ active` | the base model you're chatting with right now |
| `✓ pulled` | downloaded locally (any tag of this base) — `/model use <name>:<size>` to switch |
| `○`        | available on ollama.com — `/model pull <name>:<size>` to fetch |

| Capability badge | Meaning |
|---|---|
| `tools`  | function/tool calling — required by deepagents |
| `think`  | reasoning / chain-of-thought (deepseek-r1, qwen3, etc.) |
| `vis`    | vision (images in input) |
| `emb`    | embedding model (not chat) |

The scrape is the source of truth. If the network or page is unavailable,
the command falls back to a small curated catalog at
`src/free_agent/agent/ollama_catalog.py` so you still get something to pick
from offline.

`/model use` checks that the model is pulled — if not, it offers to pull it
first. Switching rebuilds the deepagents graph in place; conversation history
survives. Reverts cleanly if the new model fails preflight.

These commands only apply to Ollama. With `FREE_AGENT_PROVIDER=anthropic`,
only `/model use <name>` works (cloud models are always available).

### Anthropic (cloud)

```
FREE_AGENT_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-...
FREE_AGENT_ANTHROPIC_MODEL=claude-sonnet-4-6
```

## Bring your own tools

A tool is a Python function decorated with `@tool` from
`langchain_core.tools`. Tools are auto-discovered from two locations:

| Location | Scope | Use it for |
|---|---|---|
| `./free_agent_tools/*.py` | **project** — only loaded when running `free-agent` from this dir | tools with project-specific assumptions, versioned with the repo |
| `~/.config/free-agent/tools/*.py` | **global** — loaded from anywhere | tools you want available always (web search, weather, your personal helpers) |

If a tool with the same name exists in both, **project wins**. The shipped
`current_time` tool is overridable too.

Drop a file in either directory:

```python
# ./free_agent_tools/weather.py
import urllib.request
from langchain_core.tools import tool

@tool
def weather(city: str) -> str:
    """Get the current weather for a city. Use this when the user asks about weather."""
    try:
        with urllib.request.urlopen(f"https://wttr.in/{city}?format=3", timeout=5) as r:
            return r.read().decode().strip()
    except Exception as e:
        return f"weather lookup failed: {e}"
```

Run `free-agent` (or `/tool reload` from inside the chat) and it's live.
Verify with `/tools`.

**Three sample tools** are checked in at [`examples/tools/`](examples/tools/)
showing the patterns you'll use most:

- **HTTP fetch** with timeouts and bounded output
- **AST-based calculator** (no `eval`) for safe input
- **Filesystem search** scoped to the cwd

Copy them into `./free_agent_tools/` to enable.

### The `@tool` contract

| Rule | Why |
|---|---|
| Return a `str` | The agent reads tool output as text — JSON-encode if you need structure |
| Use stdlib types in the signature | LangChain derives the JSON schema from your hints |
| First docstring line = selector | It's what the orchestrator sees when deciding whether to call |
| Catch foreseeable errors and return a string | Bare exceptions don't help the agent recover |

See [`examples/tools/README.md`](examples/tools/README.md) for the full guide.

### Generate a tool with the LLM

If you don't want to write boilerplate, run `/tool new` from the chat:

```
▓▒░ tool name ▶ random_pick
▓▒░ description ▶ pick one option at random from a comma-separated list
▓▒░ arguments ▶ options:str
▓▒░ goal ▶ split, strip, return random.choice; empty input → say so

  ╭─ DRAFT // tool source ──────────────────────────╮
  │ 1 │ import random                               │
  │ 2 │ from langchain_core.tools import tool       │
  │ 3 │ ...                                         │
  ╰─────────────────────────────────────────────────╯

▓▒░ accept this draft? [Y]es / [r]egenerate / [n]o ▶ y

» where should this tool live?
   Project  → ./free_agent_tools          (only this directory)
   Global   → ~/.config/free-agent/tools  (every directory)
▓▒░ scope [P/g] ▶ g
» tool random_pick registered → ~/.config/free-agent/tools/random_pick.py
```

The wizard validates the draft contains an `@tool` decorator and a function
matching the chosen name before saving, then hot-reloads the registry.

> ⚠️ **The LLM writes Python that runs in your Python process.** Always read
> the draft before accepting. The wizard restricts the model to stdlib-only
> and string returns, but you're still the reviewer.

| sub-command | effect |
|---|---|
| `/tool new`        | wizard: name → description → args → goal → LLM draft → save |
| `/tool rm <name>`  | delete a user tool file (built-ins refuse) and rebuild |
| `/tool reload`     | re-scan `./free_agent_tools/` after editing files by hand |
| `/tool list` / `/tools` | inventory grouped by **PKG** (built-in) and **USER** (file-loaded) |

## Bring your own skills

A *skill* is deepagents' progressive-disclosure playbook — a folder
containing a `SKILL.md` file with YAML frontmatter and Markdown
instructions. The agent sees only the `description` initially; when it
decides the skill is relevant it expands the body and follows the steps.

```
my-skill/
├── SKILL.md          # YAML frontmatter + body
└── helpers/          # optional supporting files
```

`SKILL.md`:

```markdown
---
name: web-research
description: Multi-step research — decompose, search, triangulate, cite.
---

# Web Research

## When to Use
- ...

## Steps
1. ...

## Output Format
...
```

Two scopes (mirroring the tools layout):

| Location | Scope |
|---|---|
| `./free_agent_skills/<name>/SKILL.md` | **project** — only when running from this dir |
| `~/.config/free-agent/skills/<name>/SKILL.md` | **global** — every directory |

Project takes precedence on name collision. Two ready-to-use samples ship at
[`examples/skills/`](examples/skills/) — `code-review` and `web-research`.

```bash
cp -r examples/skills/web-research ~/.config/free-agent/skills/
free-agent
# inside:
/skill list
```

### Generate a skill with the LLM

`/skill new` walks you through a wizard that uses the chat model to draft
the SKILL.md body:

```
▓▒░ skill name ▶ code-review
▓▒░ description ▶ review a diff for real bugs, ignoring style nitpicks
▓▒░ goal ▶ thorough reviewer that prioritizes correctness over aesthetics

  drafting SKILL.md...
  ╭─ DRAFT // skill ─────────────────────────────╮
  │ ---                                          │
  │ name: code-review                            │
  │ description: Review a diff for real bugs...  │
  │ ---                                          │
  │                                              │
  │ # Code Review                                │
  │ ## When to Use                               │
  │ ...                                          │  ← streamed live
  ╰──────────────────────────────────────────────╯

▓▒░ accept this draft? [Y]es / [r]egenerate / [n]o ▶ y
▓▒░ scope [P/g] ▶ g
» skill code-review online.
```

| sub-command | effect |
|---|---|
| `/skill list`        | inventory grouped by PROJECT vs GLOBAL |
| `/skill new`         | wizard: name → description → goal → LLM-drafted SKILL.md → save |
| `/skill rm <name>`   | delete the skill folder (with confirmation) and rebuild |
| `/skill reload`      | rebuild the agent so it sees changes after manual edits |
| `/skill open [scope]` | open the skills folder in your file manager |
| `/skill dir`         | print the skill folder paths |

Behind the scenes, free-agent registers a `SkillsMiddleware` with its own
`FilesystemBackend` rooted at `/`, so the global folder works even when the
main agent is in safe (read-only / cwd-scoped) mode. The middleware is
read-only — skills give the agent **knowledge**, not new filesystem powers.

## Define your agents

Drop a `free-agent.yaml` next to the project. Every section is optional —
omit a key to use the default.

```yaml
system_prompt: |
  You are my agent. You decide whether to handle requests yourself or
  delegate to a subagent via the `task()` tool.

tools:                # which tools the main agent sees
  - current_time      # omit to expose every registered tool
  - weather

subagents:
  - name: researcher
    description: Use for open-ended research — multi-step lookups + synthesis.
    system_prompt: |
      You are a research specialist. Cite sources. Self-contained: you only
      see the task description, not the chat history.
    tools: [weather, current_time]

  - name: scribe
    description: Use to draft, edit, or restructure prose. Pure text work.
    system_prompt: You are a careful editor.
    tools: []         # no user tools — pure reasoning
```

Tool subset semantics, per agent (main + each subagent):

| YAML | Effect |
|---|---|
| `tools:` omitted   | inherits parent's tools (subagent) / all registered tools (main) |
| `tools: []`         | no user tools — deepagents builtins (filesystem + todo) still apply |
| `tools: [a, b]`    | only `a` and `b` |

A starter file ships at [`free-agent.example.yaml`](free-agent.example.yaml).

### Build subagents from inside the chat

`/sub new` walks you through a wizard. The chat model itself drafts the
system prompt — you review, regenerate as needed, and pick which tools the
subagent should have:

```
▓▒░ name ▶ code-reviewer
▓▒░ description ▶ reviews staged diffs for real bugs
▓▒░ goal ▶ thorough reviewer that prioritizes correctness over style nitpicks

  drafting system prompt... [streaming live]

▓▒░ accept this draft? [Y]es / [r]egenerate / [n]o ▶ y
▓▒░ tools ▶ current_time

» subagent code-reviewer online (1 total).
  save to ./free-agent.yaml? [Y/n] y
» persisted → ./free-agent.yaml
```

| sub-command | effect |
|---|---|
| `/sub new`         | wizard: name → description → goal → LLM-drafted prompt → tool selection → save |
| `/sub rm <name>`   | remove a subagent and rebuild the agent in place |
| `/sub list`        | alias for `/agent` |

Persistence rewrites `free-agent.yaml` from scratch (the previous file is
backed up to `.yaml.bak` first). Comments in your edited YAML are not
preserved — if you've heavily commented the file, decline the save prompt
and edit by hand instead.

## CLI flags

```
free-agent [-w] [-c PATH] [--version]

  -w, --writable        let the agent modify files in cwd (scoped, see above)
  -c, --config PATH     use a specific YAML profile (default: ./free-agent.yaml)
  --version             print version and exit
  -h, --help            show this help
```

## Force planning mode

deepagents includes a `write_todos` tool that the agent calls when it judges
the task complex enough (3+ steps). For tasks where you want planning even
when the agent would otherwise skip it, use `/plan <task>`:

```
▓▒░ you ▶ /plan refactor the auth middleware to use JWT validation

» planning mode — write_todos forced for this turn.

◆ ai ▶ I'll lay out a plan first.

  ▦ PLAN  0/5 done
    ○ research the current auth middleware
    ○ design JWT validation approach
    ○ implement replacement
    ○ add unit tests
    ○ run integration suite

  ✓ research the current auth middleware
  ▶ design JWT validation approach
  (...continues, updating todos as it goes)
```

Each call to `write_todos` re-renders the plan inline with status icons:

| Glyph | Status | Style |
|---|---|---|
| `✓` | completed | green + strikethrough |
| `▶` | in_progress | bright yellow (active focus) |
| `○` | pending | dim grey |

Without `/plan`, the agent decides for itself whether to plan. The slash form
prepends a directive to your message instructing the model to call
`write_todos` first with at least 3 todos, then update statuses as it
progresses.

## Slash commands

```
/help              show all commands
/plan <task>       force the agent to write_todos before acting on the task
/tools             tool inventory (PKG vs USER)
/agent             current agent profile (main + subagents)
/sub new           wizard — create a subagent
/sub rm <name>     remove a subagent
/tool new          wizard — create a tool
/tool rm <name>    delete a user tool
/tool reload       re-scan tool folders
/tool open [scope] open the tools folder in your file manager
/tool dir          print tool folder paths
/skill list        inventory of loaded SKILL.md folders
/skill new         wizard — generate a SKILL.md
/skill rm <name>   delete a skill folder
/skill reload      rebuild after manual edits
/skill open [s]    open the skills folder in your file manager
/skill dir         print skill folder paths
/model list        list local Ollama models
/model browse [q]  curated catalog of pullable models (filter optional)
/model pull <name> download a model (live progress)
/model use <name>  switch active model
/model rm <name>   remove a local model
/clear             wipe the conversation buffer (keeps the agent loaded)
/history           dump the session as markdown
/save [path]       export to file
/retry             drop last AI turn and re-fire the prior prompt
/exit              quit (Ctrl+D works too)
```

`Ctrl+C` aborts the current turn without quitting.

## Architecture

Three-layer design with one-directional dependencies. The CLI never imports
LangChain types; the agent layer is the only place that touches `deepagents`.

```
src/free_agent/
├── __main__.py           # entrypoint: load .env, start the async loop
├── config.py             # pydantic-settings — provider, keys, model
├── agent/
│   ├── builder.py        # the only file that imports deepagents
│   ├── profile.py        # AgentProfile / SubAgentProfile (Pydantic)
│   ├── loader.py         # free-agent.yaml → AgentProfile (with save)
│   └── prompts.py        # default system prompt fallback
├── tools/
│   ├── basic.py          # BUILTIN_TOOLS — what ships with the package
│   ├── registry.py       # TOOLS (mutable) + reload from ./free_agent_tools/
│   └── __init__.py       # public surface
├── session/
│   └── history.py        # plain-dataclass conversation state
└── cli/
    ├── app.py            # async event loop
    ├── console.py        # cyberpunk theme + glitch boot animation
    ├── commands.py       # slash dispatch
    ├── wizard.py         # /sub new + /tool new (LLM-driven)
    └── context.py        # SessionContext bundle
```

## Customize the look

The cyberpunk theme is opinionated by design — it's centralized in
[`src/free_agent/cli/console.py`](src/free_agent/cli/console.py). To swap
colors, edit the `NEON` `Theme` dict at the top. To change the banner, edit
`_ASCII`. To kill the boot glitch, replace the call to `render_banner_glitch`
in `cli/app.py` with `render_banner`.

## Contributing

Issues and PRs welcome. The codebase is small (~1.5k LOC of source) and
intentionally easy to read end-to-end. Run the dev tools:

```bash
uv sync --group dev
uv run ruff check .
uv run mypy src/
uv run pytest
```

## License

[MIT](LICENSE) — do whatever you want, no warranty.
