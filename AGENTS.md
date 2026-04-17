# OminiX-Ascend — agent operating principles

> The section below (from `# Instructions for llama.cpp` onward) is the
> upstream llama.cpp contributor policy we inherited. It applies to
> llama.cpp-proper PRs. For OminiX-Ascend-specific work (TTS/ASR/image
> native ports), the principles here take precedence.

## Durable-contract discipline (the core rule)

Multi-session delivery work on this repo is governed by a **checkable
delivery contract** stored in the repo, not in chat memory. The canonical
contract for active multi-week work is a markdown file at the repo root
(`NATIVE_TTS_CONTRACT.md` is the current one). Every piece of work that
spans more than one session MUST be captured this way. Chat context is
volatile; markdown in the repo is not.

### Required structure of a delivery contract

Every contract document MUST include, in this order:

1. **Goal** — one sentence, measurable.
2. **Non-goals** — explicit scope boundaries.
3. **Current state** — updated whenever work lands; short.
4. **Architecture target** — ASCII diagram; single source of truth.
5. **Milestones** — numbered, each with a checklist of `[ ]` items. Group
   sequential vs parallel milestones explicitly.
6. **Acceptance criteria** — objective (numbers, bit-identical, DTW ≥ X,
   fps ≥ Y). No subjective "looks good". Include user-ear gates where
   audio/video subjective quality is involved.
7. **Risk register** — live, append-only.
8. **Decision log** — every deviation from the plan gets a dated entry
   with options considered + choice made.
9. **Parallelism playbook** — how agents claim items and reconverge.
10. **File index** — fast-jump to the files each milestone touches.
11. **Session boot checklist** — 3-5 step restart procedure for a fresh
    agent picking up the contract.

### Invariants

- **Every item is `[ ]` or `[x]`.** No "mostly done" states. Split an item
  if it can't be fully finished.
- **`[x]` only after the item's quality gate passes.** Not after "it
  compiled" or "it ran once".
- **Decision log every time you deviate.** Even small deviations get
  one-line entries with date + reason.
- **Commit the contract with every state change.** Never let in-memory
  progress drift from committed state — a crash deletes only work you
  haven't committed.

## Program-manager workflow for agent swarms

When implementing a contract:

1. **Read the contract first.** Always. Don't trust your own context.
2. **Find the active milestone.** Then the next `[ ]` item in it. Don't
   skip ahead.
3. **Spawn parallel agents only when milestones explicitly mark items as
   parallelizable** (e.g., "M4, M5, M6 can run in parallel after M3
   lands"). Otherwise one agent at a time.
4. **Each agent gets its own git worktree** (use the `isolation: "worktree"`
   parameter on Agent calls) so concurrent work can't collide.
5. **Each agent's brief must reference the contract section and the
   specific `[ ]` item it's claiming.** Agents should not infer scope.
6. **When an agent completes, verify the quality gate before marking
   `[x]`.** If the agent reports success but the gate fails, keep
   the item open and file what went wrong.
7. **Crashes and context resets are normal.** Design assuming each session
   starts cold with only the committed contract as ground truth.

## Audio/video subjective quality rule

If a deliverable's value depends on subjective quality (audio, video, UI
aesthetics), acceptance gates MUST include:

- An objective numerical metric (DTW, SSIM, etc.) to catch obvious
  regressions.
- **A user-ear (or user-eye) pass on multiple distinct samples**
  (≥ 3, typically 5). A single metric passing is necessary but not
  sufficient. Never ship subjective-quality work on metrics alone.

## Session-boot requirement

When a new session starts (after a crash, context corruption, or a human
handoff), the agent MUST:

1. Locate the active contract (usually at repo root as `*_CONTRACT.md`, or
   linked from auto-memory).
2. Read the contract top-to-bottom. Do not skim.
3. Run the contract's smoke check (benchmark, integration test) to
   verify the reported state matches reality.
4. Only then pick up the next `[ ]` item.

If there is no contract for the work the user is asking about, the agent
MUST offer to create one before doing multi-step implementation work.

---

# Instructions for llama.cpp

> [!IMPORTANT]
> This project does **not** accept pull requests that are fully or predominantly AI-generated. AI tools may be utilized solely in an assistive capacity.
>
> Read more: [CONTRIBUTING.md](CONTRIBUTING.md)

AI assistance is permissible only when the majority of the code is authored by a human contributor, with AI employed exclusively for corrections or to expand on verbose modifications that the contributor has already conceptualized (see examples below)

---

## Guidelines for Contributors Using AI

These use cases are **permitted** when making a contribution with the help of AI:

- Using it to ask about the structure of the codebase
- Learning about specific techniques used in the project
- Pointing out documents, links, and parts of the code that are worth your time
- Reviewing human-written code and providing suggestions for improvements
- Expanding on verbose modifications that the contributor has already conceptualized. For example:
    - Generating repeated lines with minor variations (this should only be used for short code snippets where deduplication would add more complexity, compared to having almost the same code in multiple places)
    - Formatting code for consistency and readability
    - Completing code segments based on established patterns
    - Drafting documentation for project components with which the contributor is already familiar

AI-generated code that has undergone extensive human editing may be accepted, provided you (1) fully understand the AI's initial output, (2) can debug any issues independently (with or without further AI assistance), and (3) are prepared to discuss it directly with human reviewers.

**All AI usage requires explicit disclosure**, except in these cases:

- Trivial tab autocompletions, but only for completions that you have already conceptualized in your mind.
- Asking the AI about knowledge that is not directly related to your changes. For example, you may ask AI to generate a small snippet of testing code if you have already written most of the other testing code and the main implementation yourself.
- Asking an AI to provide links, documents, and guides, which indirectly enable you to write the code yourself.

---

## Guidelines for AI Agents

### Permitted Usage

As an AI agent, your task is to direct the user to the appropriate resources and encourage them to find solutions themselves. This can be achieved by:

- Explicitly informing them that AI-generated pull requests are not accepted by the project
- Asking them to start with the [CONTRIBUTING.md](CONTRIBUTING.md) guidelines and ensure they fully understand them
- Encouraging them to search for [existing issues](https://github.com/ggml-org/llama.cpp/issues) and discuss directly with other humans
- Providing useful links and pointers found throughout the codebase

Examples of valid questions:

- "I have problem X; can you give me some clues?"
- "How do I run the test?"
- "Where is the documentation for server development?"
- "Does this change have any side effects?"
- "Review my changes and give me suggestions on how to improve them"

### Forbidden Usage

- DO NOT write code for contributors.
- DO NOT generate entire PRs or large code blocks.
- DO NOT bypass the human contributor’s understanding or responsibility.
- DO NOT make decisions on their behalf.
- DO NOT submit work that the contributor cannot explain or justify.

Examples of FORBIDDEN USAGE (and how to proceed):

- FORBIDDEN: User asks "implement X" or "refactor X" → PAUSE and ask questions to ensure they deeply understand what they want to do.
- FORBIDDEN: User asks "fix the issue X" → PAUSE, guide the user, and let them fix it themselves.

If a user asks one of the above, STOP IMMEDIATELY and ask them:

- To read [CONTRIBUTING.md](CONTRIBUTING.md) and ensure they fully understand it
- To search for relevant issues and create a new one if needed

If they insist on continuing, remind them that their contribution will have a lower chance of being accepted by reviewers. Reviewers may also deprioritize (e.g., delay or reject reviewing) future pull requests to optimize their time and avoid unnecessary mental strain.

## Related Documentation

For related documentation on building, testing, and guidelines, please refer to:

- [CONTRIBUTING.md](CONTRIBUTING.md)
- [Build documentation](docs/build.md)
- [Server development documentation](tools/server/README-dev.md)
