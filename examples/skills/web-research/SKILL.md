---
name: web-research
description: Structured approach for multi-step research questions — break the topic into sub-questions, gather evidence, synthesize findings with citations.
---

# Web Research

## When to Use

- The user asks a question that requires gathering information from multiple sources.
- The answer is non-obvious and benefits from comparing perspectives.
- The user explicitly asks for "research", "deep dive", "compare", or "what do people say about".

Do NOT apply this skill for simple factual lookups (use a single tool call instead) or for questions about the user's local files.

## Steps

1. **Decompose** the question into 2–4 concrete sub-questions. Write them down in your todo list before searching.
2. **Search** for each sub-question independently. Prefer primary sources (official docs, papers, project repos) over aggregators.
3. **Triangulate** — when sources disagree, note the disagreement and what evidence each side cites. Don't pick a side without reason.
4. **Note dates.** If you cite a benchmark, statistic, or "X happened" claim, include when it was reported.
5. **Synthesize** — write a coherent answer that flows from setup → findings → caveats, NOT a list of "Source 1 says X, Source 2 says Y".

## Output Format

End every research answer with a `## Sources` section listing the URLs or file paths you actually used (not everything you searched). Each source gets one line with what you took from it.

If a key sub-question had no good source, say so explicitly: "I could not find an authoritative answer on X; this part is my inference based on Y."
