# MailFind — Project Documentation

This folder contains the planning and architecture documentation for MailFind.

Read in order if new:

1. [Vision](00-vision.md) — what we're building and why
2. [Product Requirements](01-prd.md) — what gets built, by phase
3. [Architecture](02-architecture.md) — system design with diagrams
4. [Roadmap](03-roadmap.md) — phase ordering and dependencies
5. [Risks](04-risks.md) — what could kill the project + mitigations
6. [Marketing Plan](05-marketing-plan.md) — distribution strategy, launch playbook
7. [Open Questions](06-open-questions.md) — what's not decided yet

## Decision records

Architectural decisions with rationale live in [decisions/](decisions/). Add a new ADR whenever a non-obvious technical choice is made. See [decisions/0001-tauri-over-electron.md](decisions/0001-tauri-over-electron.md) for format.

## Diagrams

Architecture diagrams are embedded inline in the architecture doc using Mermaid. Larger or standalone diagrams live in [diagrams/](diagrams/) as `.md` files with Mermaid source.

## Living documents

These docs evolve with the project. Out-of-date docs are worse than no docs — when something changes, update the relevant file. The PRD, roadmap, and open-questions docs especially should reflect current reality, not the original plan.

## Personal notes

Higher-level reflections and conversational context live in [`../personal_notes.md`](../personal_notes.md) (not in this folder, since it's more of a personal scratchpad than project documentation).
