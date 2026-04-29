# Sample skills

Each subfolder here is a self-contained skill ready to drop into one of the
auto-discovered locations:

- `./free_agent_skills/<skill-name>/` — project-only
- `~/.config/free-agent/skills/<skill-name>/` — global, available everywhere

Copy a folder to enable:

```bash
cp -r examples/skills/web-research ~/.config/free-agent/skills/
free-agent
# inside the chat:
/skill list
```

## How skills work

Skills are deepagents' **progressive-disclosure playbooks**. The agent sees
only the YAML `description` initially. When it determines a skill applies to
the current task, it reads the full body and follows the instructions.

A skill is a directory containing at least `SKILL.md`:

```
my-skill/
├── SKILL.md          # required: YAML frontmatter + markdown body
└── helpers/          # optional: any supporting files the skill body refers to
    └── checklist.md
```

`SKILL.md` format:

```markdown
---
name: my-skill
description: One line — what this skill does and when the agent should apply it.
---

# My Skill

## When to Use
- Triggers...

## Steps
1. Do X
2. Then Y

## Output Format
How the agent should structure its response.
```

The first paragraph (`description`) is the only thing the agent sees until it
chooses to "expand" the skill. Make it crisp and trigger-oriented — the agent
uses it like a tool description.

## Authoring a skill via the LLM

Inside the chat:

```
/skill new
```

The wizard asks name, description, and intent — then the configured model
drafts the SKILL.md body for you. You can regenerate, then save to either
the project or global folder.
