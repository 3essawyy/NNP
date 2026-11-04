# ADR-011: Git flow in cloud sessions

- **Status:** Accepted
- **Date:** 2026-11-04
- **Session:** S001
- **Related:** CLAUDE.md

## Context
Cloud sessions must push to a designated `claude/*` branch. The repository started empty (no `main`).

## Decision
- Bootstrap `main` with an initial commit, then deliver work as PRs from the designated branch, squash-merged by Claude; after each merge the branch is reset onto the new `main`.
- A PR may close several tasks of one milestone; the squash title lists them (`T-003..T-008: …`).
- Local (non-cloud) work keeps GATP's `<type>/T-NNN-slug` branches.
- Tag `submission` on `main` when the zip is handed in.
