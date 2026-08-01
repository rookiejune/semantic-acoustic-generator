# Refactor Debt

This file tracks runnable but poorly designed code in `semantic-acoustic-codec`.

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

| ID | Type | Area | Problem | Risk | Priority | Status | Next Action | Notes |
|---|---|---|---|---|---|---|---|---|
| R001 | Interface debt | `SemanticCodecBatch` reference/mask access | `acoustic_mask` is typed optional but filled inside `__post_init__`, while reference tensors remain optional with no public required-accessor contract; callers re-create private `_reference_*` / `_acoustic_mask` helpers. | Optional-reference logic can diverge across training, sampling, smoke, and artifact eval; future fixed-eval work may copy more mask/error behavior. | P1 | pending | Add batch-level required accessors or make masks required, then replace duplicated private helpers one caller at a time. | Evidence: `src/semantic_acoustic_codec/types.py:73-87,157-165`, `src/semantic_acoustic_codec/callback/sample.py:377-408`, `scripts/eval_artifact.py:269-294`, `scripts/smoke.py:394-419`. Acceptance: `pytest tests/test_scripts.py tests/test_runtime_reference.py tests/test_fixed_layout.py && ruff check . && basedpyright`. |
| R002 | Boundary debt | `scripts/train.py` composition | The Hydra entry script builds device/output paths, data config, support config, feature stats, callbacks, checkpoints, trainer, and dataloading branches directly. | Training rules are hard to reuse from jobs/tests and new experiment variants must edit an entry script instead of a service/factory boundary. | P1 | pending | Move behavior-preserving train component construction into `src/semantic_acoustic_codec/` and leave `scripts/train.py` as the thin Hydra entry. | Evidence: `scripts/train.py:40-177` plus manual config conversion in `scripts/train.py:180-285`; README declares `scripts/train.py` as the production entry. Acceptance: `pytest tests/test_config.py tests/test_training.py && python scripts/smoke.py && ruff check . && basedpyright`. |
| R003 | Boundary debt | datamodule source routing | `datamodule/longcat.py` owns generic `DataConfig`, Lightning `DataModule`, Qwen grid loading, WMT LongCat fallback, source dispatch, length planning, collation, and metadata assembly. | Adding a backend-native source or changing cross-text rules requires editing one mixed file with multiple source-string branches. | P1 | pending | Split source adapters/registry from the shared batching/collation owner; keep `DataModule` consuming a source adapter contract. | Evidence: `src/semantic_acoustic_codec/datamodule/longcat.py:59-141,166-251,444-499,569-618`; Qwen-specific dataset is already isolated in `datamodule/qwen.py:40-178`. Acceptance: `pytest tests/test_qwen_fixed_source.py tests/test_structured_data.py tests/test_data.py && python scripts/smoke.py --skip-routes && ruff check . && basedpyright`. |
| R005 | Implementation debt | eval/sample feature metrics | Artifact eval and sample logging both compute masked codec features, feature MSE, seeded with/without-reference paths, and audio summaries with local helper code. | Metric semantics can drift between training callbacks and offline artifact evaluation, especially while optional-reference fixed eval is still pending. | P2 | pending | Extract a small evaluation helper module for codec features, feature MSE, seeded generators, and paired sample summaries; migrate one caller first. | Evidence: `src/semantic_acoustic_codec/callback/sample.py:110-175,263-284,309-339`, `scripts/eval_artifact.py:113-190,246-266`, TODO still requires fixed eval in `docs/experiments/todo.md`. Acceptance: `pytest tests/test_scripts.py tests/test_fixed_layout.py && python scripts/smoke.py && ruff check . && basedpyright`. |

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
