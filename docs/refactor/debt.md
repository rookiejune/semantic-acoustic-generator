# Refactor Debt

This file tracks runnable but poorly designed code in `semantic-acoustic-generator`.

Keep this as an index of unfinished refactor debt. Move completed design notes into nearby `docs/` files or leave them to git history instead of keeping done items here.

## Scope

Use this file for:

- Interface debt: public APIs, module contracts, config surfaces, or caller expectations that should change.
- Boundary debt: logic living in the wrong layer, duplicated runtime constraints, or scripts carrying reusable rules.
- Implementation debt: ugly internal code that is stable enough to defer.

Do not use this file for experiment logs, benchmark results, private paths, remote training notes, checkpoints, or completed history.

## Priority

| Priority | Meaning |
|---|---|
| P0 | Currently blocks development, correctness, or result trust. |
| P1 | Likely to be touched soon or spreading into other modules. |
| P2 | Runnable and stable, but should be fixed when this area is next changed. |
| P3 | Mostly aesthetic; do not schedule proactively. |

## Debt Index

There is no open refactor debt currently tracked for this repository.

## Item Template

Use a separate `docs/refactor/R###-short-name.md` file only when an item needs more detail.

```md
# R### Short title

## Current Behavior

Describe what works today.

## Design Problem

Explain why the current shape is costly or risky.

## Why Not Now

State why this remains deferred instead of being fixed immediately.

## Target Contract

Define the desired public interface, module boundary, or invariant.

## Smallest Safe Path

1. Add or identify a behavior-preserving check.
2. Move one responsibility to the target owner.
3. Replace callers.
4. Remove the old path.

## Acceptance

- List the exact lint, type, test, smoke, or training command that proves the change.
```
