# R001 Generator Naming Migration

## Legacy Behavior

The project and distribution are named `semantic-acoustic-generator`, but the Python package remains
`semantic_acoustic_codec`. Existing environment variables, job event labels, output directories, checkpoint metadata,
artifact filename `codec.json`, and downstream configuration field `semantic_codec_artifact` also retain the old name.

## Design Problem

These identifiers imply that this repository owns a codec. The actual boundary is narrower: AnyCodec owns codec
encoding, decoding, codebooks, layouts, backend weights, and loading; this repository owns semantic-to-acoustic
generation, training, evaluation, artifacts, and thin runtime composition.

## Migration Status

Completed on 2026-08-05. The repository, distribution, Python package, local checkout, GitHub remote, artifact writer,
checkpoint writer, environment variables, job identifiers, and downstream configuration now use generator-owned
names. Compatibility is intentionally limited to explicit readers for schema-7 `codec.json`, the legacy checkpoint
metadata key, and the downstream legacy runtime field.

## Target Contract

- Use `semantic_acoustic_generator` as the Python package.
- Replace `SEMANTIC_ACOUSTIC_CODEC_TRAIN_ROOT` and new job/output identifiers with generator-owned names.
- Version the artifact migration from `codec.json` and codec-owned metadata keys to generator-owned identifiers.
- Update downstream `speech-to-speech` configuration with an explicit compatibility migration.
- Keep codec interface type names only where they refer to AnyCodec-owned contracts.

## Completed Sequence

1. Add the new package and configuration names with explicit, tested compatibility readers.
2. Migrate internal and downstream callers, job wrappers, and documentation.
3. Version the artifact schema and remove legacy writers.
4. Rename the GitHub repository, origin, and local checkout after both repositories pass validation.

## Acceptance

- Existing schema-7 artifacts still load through the documented compatibility path.
- New artifacts and jobs contain no project-owned `codec` naming outside references to AnyCodec contracts.
- `python -m pytest`, `ruff check .`, and `basedpyright` pass with the new import path.
- A real `speech-to-speech` semantic-only decode loads the migrated artifact and produces a finite waveform.
