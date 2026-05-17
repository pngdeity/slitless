---
name: ai-agent-orchestration
description: >-
  AI agent orchestration design patterns and production workflows. Use when
  designing multi-agent systems, choosing orchestration strategies (sequential,
  concurrent, group chat, handoff, magentic), refactoring monolithic agents into
  coordinated sub-agents, implementing production hardening (structured outputs,
  RAG pipelines, observability, circuit breakers), or planning multi-agent
  refactors with verification gates. Not for single-agent tool selection or
  prompt engineering within one agent.
allowed-tools: Read, Bash, Grep, Glob, Write, Edit
metadata:
  tags: "orchestration multi-agent coordination refactoring patterns production design observability rag structured-outputs"
compatibility: Generic — no framework restrictions. Patterns apply to any agent framework (ADK, Semantic Kernel, Agent Framework, LangGraph, CrewAI, custom).
---

# AI Agent Orchestration

Design patterns and production workflows for multi-agent systems. Sources: [Microsoft Azure Architecture Center](https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/ai-agent-design-patterns), [Google AI Agent Clinic](https://developers.googleblog.com/production-ready-ai-agents-5-lessons-from-refactoring-a-monolith/), [Cyrus orchestration case study](https://www.atcyrus.com/stories/ai-agent-orchestration-complex-refactoring).

## Complexity Spectrum

Before adopting multi-agent orchestration, evaluate whether the task requires it. Each level adds coordination overhead, latency, and cost.

| Level | When to Use | Escalation Trigger |
|-------|-------------|-------------------|
| **Direct model call** | Single-step tasks (classification, summarization, translation) | Task requires tool access or multi-step reasoning |
| **Single agent with tools** | Varied queries within one domain, dynamic tool use | Cross-domain problems, security boundaries, tool overload, prompt complexity |
| **Multiagent orchestration** | Cross-functional problems, distinct security per agent, tasks benefiting from parallel specialization | — |

Use the lowest complexity level that reliably meets requirements.

## Five Orchestration Patterns

### Decision Matrix

| Pattern | Coordination | Routing | Best For | Risk |
|---------|-------------|---------|----------|------|
| **Sequential** | Linear pipeline, output flows A→B→C | Deterministic, predefined order | Step-by-step refinement, clear dependencies | Early-stage failures cascade; no parallelism |
| **Concurrent** | Parallel, agents work on same input independently | Deterministic or dynamic selection | Multiple perspectives, latency-sensitive tasks | Conflicting results need resolution; resource-intensive |
| **Group Chat** | Shared conversation thread, chat manager controls turns | Manager-mediated turn order | Consensus-building, brainstorming, maker-checker validation | Infinite conversation loops; hard to control with >3 agents |
| **Handoff** | One active agent at a time, dynamic delegation | Agents self-route based on context | Tasks where optimal specialist emerges during processing | Infinite handoff loops; unpredictable routing paths |
| **Magentic** | Manager builds dynamic task ledger, iterations refine plan | Manager assigns/reorders tasks adaptively | Open-ended problems with no predetermined solution path | Slow to converge; stalls on ambiguous goals |

### Sequential Orchestration

**When to use:** Multistage processes with clear linear dependencies (draft→review→polish), data transformation pipelines, progressive refinement.

**When to avoid:** Stages can be parallelized, only 1-2 stages exist, early stages might fail with no recovery path, backtracking or iteration needed.

**Example:** Contract generation pipeline — template selection → clause customization → regulatory compliance → risk assessment. Each agent builds on the previous output.

### Concurrent Orchestration

**When to use:** Tasks benefiting from multiple independent perspectives (technical, business, creative), time-sensitive parallel processing, ensemble reasoning or voting-based decisions.

**When to avoid:** Agents need cumulative context in sequence, no clear conflict resolution strategy, resource constraints make parallelism impossible.

**Example:** Stock analysis — fundamental analysis, technical analysis, sentiment analysis, and ESG evaluation run in parallel, then results are aggregated into a recommendation.

### Group Chat Orchestration

**When to use:** Creative brainstorming requiring cross-functional dialogue, structured maker-checker quality gates, human-in-the-loop scenarios, compliance validation needing multiple expert perspectives.

**When to avoid:** Basic task delegation suffices, real-time processing required, no objective way to determine task completion, more than 3 agents (control degrades).

**Maker-checker sub-pattern:** One agent creates (maker), another evaluates against criteria (checker). Checker rejects with specific feedback → maker revises → cycle repeats until approval or iteration cap. Requires clear acceptance criteria and an iteration limit with defined fallback behavior.

### Handoff Orchestration

**When to use:** Specialized knowledge or tools needed but the right agent isn't known upfront, expertise requirements emerge during processing, multi-domain problems needing different specialists sequentially.

**When to avoid:** Appropriate agent is identifiable from initial input (use deterministic routing instead), suboptimal routing could frustrate users, infinite handoff loops are hard to prevent.

**Example:** Customer support — triage agent handles common issues, hands off network problems to infrastructure agent, billing disputes to financial resolution agent. Agents can further hand off or escalate to human support.

### Magentic Orchestration

**When to use:** Complex open-ended problems with no predetermined solution, requirement to generate a plan for human review before execution, agents equipped with tools that modify external systems.

**When to avoid:** Deterministic solution path exists, no plan ledger needed, low complexity, time-sensitive (this pattern optimizes for correctness not speed).

**Example:** SRE incident response — manager agent builds initial task ledger (restore service, identify root cause), consults diagnostics/infrastructure/rollback/communication agents, continuously refines the plan as new information emerges.

### Implementation Guardrails

- **Context management:** Compact context between agents via summarization; persist shared state externally for long-running tasks.
- **Reliability:** Timeout and retry mechanisms, circuit breakers, validate agent output before forwarding, isolate agents from shared failure points.
- **Security:** Least privilege per agent, authenticate inter-agent communication, apply content safety guardrails at user input, tool calls, tool responses, and final output.
- **Cost:** Assign smaller/cheaper models to classification/extraction/formatting agents; monitor per-agent token consumption; compact context between agents.
- **Observability:** Instrument all agent operations and handoffs; use scoring rubrics or LLM-as-judge evaluations for testing nondeterministic agent outputs.

## Five Production Hardening Lessons

From Google's AI Agent Clinic refactoring of "Titanium" — a sales research agent rebuilt from prototype to production.

### 1. Ditch the Monolith for Orchestrated Sub-Agents

| Anti-Pattern | Fix |
|-------------|-----|
| Monolithic script with linear `for` loop — one failure stalls everything | Decompose into specialized agents in a `SequentialAgent` pipeline |
| Single LLM executing massive multi-step prompts | Narrow agents with single responsibilities |

**Action:** When a single agent's prompt exceeds ~200 lines or touches 3+ distinct domains, decompose into orchestrated sub-agents.

### 2. Force Structured Outputs with Schema Contracts

| Anti-Pattern | Fix |
|-------------|-----|
| JSON format instructions embedded in prompt strings | Inject native schema objects (Pydantic, Zod, etc.) as explicit contracts |
| Brittle string parsing of model outputs | Framework enforces structural adherence at runtime |

**Action:** Replace any prompt string that says "return JSON in this format" with a typed schema. Let the framework handle serialization.

### 3. Replace Hardcoded State with Dynamic RAG Pipelines

| Anti-Pattern | Fix |
|-------------|-----|
| Hardcoded data in source files (e.g., 12 case studies in Python) | Autonomous crawling pipeline feeding a vector search index |
| Manual code changes to update agent knowledge | Hybrid search (semantic + keyword) over indexed corpus |

**Action:** Any data list embedded in code that grows over time should be replaced with a RAG pipeline — crawl, embed, index, query.

### 4. Observability is Non-Negotiable

| Anti-Pattern | Fix |
|-------------|-----|
| Black-box failures with no component-level diagnostics | OpenTelemetry distributed traces for full execution flows |
| No visibility into which agent caused a break | Live telemetry dashboard capturing model requests, tokens, and tool executions |

**Action:** Instrument with OpenTelemetry before deploying to production. Trace every agent transition, model call, and tool invocation.

### 5. Tame Token Burn with Circuit Breakers

| Anti-Pattern | Fix |
|-------------|-----|
| Agent retries prompts without bounds on errors | Exponential backoff, timeout boundaries, configurable retry limits |
| Custom try-catch retry logic in application code | Let the orchestration framework handle graceful failures natively |

**Action:** Every agent invocation should have a max-retry ceiling, a timeout, and a defined fallback. Never allow unbounded retry loops.

## Practical Orchestration Workflow

From the Cyrus case study: refactoring a complex webhook system (5,554 lines deprecated, 3 competing implementations, 59 files) in 3.5 hours with zero human intervention.

### The 5-Step Loop

**1. Atomic Decomposition**
- Create precise, verifiable sub-tasks — not vague instructions.
- Bad: "Clean up the webhook code." Good: "Delete packages/ndjson-client/ directory. Remove all NdjsonClient imports from EdgeWorker. Verify 99 tests passing."
- Each sub-task must have concrete, binary acceptance criteria.

**2. Sequential Dependency Management**
- Identify and respect dependency order. Don't over-parallelize.
- Can't convert to a new framework until deprecated code is removed. Can't integrate until all components are refactored.
- Build a dependency graph and execute in dependency order.

**3. Independent Verification Before Integration**
- After each sub-agent completes: pull its output, run tests/typecheck/build independently.
- Do not trust self-reported success. Verify before merging into the main line.
- A bad merge in hour 1 cascades through hours 2–4.

**4. Feedback Loops with Specific Diagnosis**
- When verification fails, provide: specific failure count, affected files, root cause diagnosis, actionable fix list.
- Allow the sub-agent to correct and re-submit.
- Result: agent learns and delivers working code without human escalation.

**5. Isolated Workspaces**
- Each sub-agent works in a separate git worktree or sandbox.
- Prevents merge conflicts, corrupts nothing on failure, enables clean rollback.
- Orchestrator pulls completed work only after verification passes.

### When to Use Orchestration vs. Not

| Use Orchestration | Don't Use Orchestration |
|------------------|------------------------|
| Task has clear sub-problems with concrete acceptance criteria | Task is simple and linear (single-file changes) |
| Dependencies between sub-tasks exist | No dependencies to manage |
| Each sub-task can be independently verified (tests pass/fail is binary) | Requirements are ambiguous, need human judgment |
| Human coordination cost exceeds orchestration setup cost (tasks taking days+) | Task takes less than 1 hour of human time |
| You want a documented, reproducible process | You need creative problem-solving (novel algorithms, UX design) |

## Verification

Verify this skill activates correctly:
1. Present a task: "Refactor this monolithic agent into orchestrated sub-agents for our document processing pipeline."
2. Confirm the agent identifies the complexity spectrum, selects an appropriate orchestration pattern, and applies the production hardening checklist.
3. Present a task: "We need to choose between concurrent and group chat orchestration for our compliance review system."
4. Confirm the agent uses the decision matrix to evaluate trade-offs and recommends a pattern with justification.
