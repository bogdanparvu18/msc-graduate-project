# Phase 3 — Step 1: verified benchmark inputs

This step connects the published S8 preparation to a reproducible input loader. It reads existing metadata and images, selects comparable visual previews, and separates model inputs from evaluator evidence and provenance. It does not regenerate images, extract new video frames, author clinical answers, assign splits, or run model scoring.

## Run

Python 3.10 or newer:

```bash
python -m pip install -e '.[test]'
python -m mmvqa.phase3.cli --metadata-only --checkout .
```

The first command installs the package. The second resolves `main` through GitHub, then reads metadata from that immutable commit using local Git objects. Omit `--checkout .` to download the same commit's files directly from GitHub. A partial clone may fetch missing objects. `GITHUB_TOKEN` is optional for authenticated GitHub access; tokens are never written to reports.

Edit `configs/phase3/bootstrap.json` to point to the existing storage mount, then run the image check:

```bash
python -m mmvqa.phase3.cli --checkout .
```

The [notebook](../../notebooks/phase3/phase3_01_benchmark_bootstrap.ipynb) provides the same workflow with actual image display and Colab Drive mounting. Before the PR is merged, set its `CODE_REF` to `codex/phase3-benchmark-bootstrap`; after merge, use `main`. `CODE_REF` selects the implementation; `DATA_REF` independently selects the published metadata. The notebook records both resolved identities.

## Which source provides which data?

All paths below are relative to the configured `MMVQA_Clinical` storage root unless noted otherwise. No basename search or silent fallback selects a different file.

| Data | Authority and location | Use |
| --- | --- | --- |
| Storage index | GitHub: `outputs/phase2/sector8/configs/sector8_github_storage.json` | Describes logical metadata files, exact sizes, hashes and direct/part storage. |
| Active S8 catalog | GitHub logical path `configs/sector8_archive_state.json` | Selects the completed processing run, table descriptors and archive manifest. |
| Archive manifest | GitHub logical path selected by `archive_delivery.manifest_relative_path` | Binds TAR archives and individual JPEG members to the logical output catalog. |
| Ordered input recipes | GitHub logical path `metadata/<selected-signature>/benchmark/model_inputs.jsonl.gz` | Variant, order, image roles, source hashes and provenance identities. |
| Annotation evidence and source links | GitHub tables selected by the catalog | Evaluator-only evidence and source/product/annotation/example joins. |
| Original annotated JPEGs | Mount: `data/raw/kvasir_capsule/osfstorage/...` | Full-frame input, including the original half of `fullcrop`. |
| Previously extracted temporal JPEGs | Mount: `curated/phase2/temporal_frames/<video>/frame_....jpg` | Existing context frames selected by the recipe. A sequence can also include an original annotated frame. |
| Crops | Mount: `outputs/phase2/sector8/archives/roi_crop.tar` | Existing `products/roi_crop/...jpg` members. |
| Overlays | Mount: `outputs/phase2/sector8/archives/bbox_overlay.tar` | Existing `products/bbox_overlay/...jpg` members. |
| Temporal collages | Mount: `outputs/phase2/sector8/archives/temporal_collage.tar` | Existing `products/temporal_collage/...jpg` members. |
| Optional materialized derivatives | Explicit `derived_roots` | Existing S8 product trees; each selected image still must match the archive/member metadata. |

`fullcrop` loads two separate images: original first, crop second. It does not require `evidence_panel.tar`; that archive belongs to the separate `fullcrop_panel` variant. `temporal_multi` preserves the declared chronological frame sequence. Its ordering is not inferred from filenames.

GitHub contains metadata, not the S8 image TARs. Some large metadata exist **only as ordered `.bin` parts**. The loader joins those bytes into a local cache, checking every part's ordinal, offset, length and SHA-256, then the final file's length and SHA-256. This is lossless file assembly, not regeneration of a dataset or visual product. Ordinary files are read directly. Only requested logical files are loaded; recovery snapshots and unused visual tables are not loaded.

## What “current” and “verified” mean

1. The configured branch/tag is resolved through the GitHub commits API at run start. The result must be a full immutable commit SHA. No S8 upstream commit or historical publication SHA is used as the default data revision.
2. All index, catalog and artifact reads use that resolved commit. A branch movement during the run cannot mix two snapshots.
3. The storage index authenticates the **archive catalog** with `source_catalog_sha256`. The archive manifest's same-named field instead refers to the **semantic catalog** (`configs/sector8_state.json`). These identities are checked separately.
4. Consumed metadata must agree across index, catalog and table descriptors. Parquet row counts and columns must match the verified schema companions. Cached bytes are verified before reuse.
5. Mounted originals must match the selected recipe's SHA-256 and image dimensions. Source byte counts are checked when S8 provides them; the current source table has missing byte counts for some originals, so no size is invented.
6. Mounted TARs must match their declared size/hash. The resolver checks exact safe membership and the selected image's size/hash/dimensions. It reads regular members without extracting the archive. It never overwrites source data.

A newer file on the mount is not automatically accepted. If its bytes differ from the GitHub publication, the check fails. A new S8 run must publish its corresponding metadata before Phase 3 can select it. File modification dates alone do not establish dataset identity.

The first selected image from a TAR requires verifying that archive; in the inspected publication `temporal_collage.tar` is about 3 GB. This can take time over a mounted drive. An unchanged archive is verified once per resolver instance; each consumed image is still checked. Other unused archives are not opened.

## Input contract

| Destination | Fields and boundary |
| --- | --- |
| Model adapter | `question`, decoded RGB `images`, neutral `image_roles`, explicitly allowlisted `context` (empty by default). No paths, labels, annotation IDs, ground truth or provenance objects. EXIF and image metadata are cleared. |
| Evaluator | Source annotation evidence, QA validation status and `reference_answer`. The preview has no validated reference answer; `reference_answer` remains `None`. Source labels are not converted to answers. |
| Provenance report | Example, subject and ground-truth references; ordered artifact and source links; hashes, physical frame identity, annotation links, annotation-assistance flag, source-sharing group, pending split state and resolved GitHub snapshot. |

The default sample selects one subject that has all six requested variants. Different pixel availability does not certify equal question/task difficulty. Annotation-assisted crops/overlays must remain identifiable in later comparisons.

```python
from mmvqa.phase3.storage import S8Store
from mmvqa.phase3.benchmark import S8Benchmark
from mmvqa.phase3.images import ImageResolver

store = S8Store(ref="main", cache_dir=".cache/mmvqa-phase3")
benchmark = S8Benchmark(store)
examples = benchmark.sample()
resolver = ImageResolver(
    original_roots=["/content/drive/MyDrive/MMVQA_Clinical"],
    archive_root="/content/drive/MyDrive/MMVQA_Clinical/outputs/phase2/sector8",
    archive_manifest=benchmark.archive_manifest,
    source_records=benchmark.source_records(examples),
)
preview = benchmark.prepare(examples[0], resolver, question="Describe the visible image content.")
model_payload = preview["model"]  # Only this projection goes to a model.
```

Explicitly allowlisted task context is supplied separately; the loader never copies context from a raw S8 row. The caller must ensure the allowed text is available to the model for the intended task, rather than a paraphrase of a reference answer.

## Reports and readiness

Reports default to `outputs/phase3/local/bootstrap/` (ignored by Git):

- `source_snapshot.json`: requested ref, actual resolved commit, index identity and consumed logical artifact hashes/sizes.
- `sample_report.json`: selected examples, assistance flags and whether image pixels were checked in this run.
- `evaluator_records.json`: source annotation evidence kept outside model inputs.
- `provenance.json`: source → product → annotation → example lineage.

`metadata_verified_images_not_checked` only confirms metadata. `image_smoke_passed` requires all selected mounted images to load and verify. Missing files or integrity errors produce `status="failed"` and a nonzero CLI exit. A failed run cannot reuse a prior success report.

Every preview remains `preview_only=true` and `training_ready=false`. The next benchmark steps are question/answer authoring and validation, source-group-safe split assignment, then model and evaluator runners. This loader does not claim those steps are complete.

For reproducibility, set `ref` to the exact `resolved_commit` saved in a previous source snapshot. An explicit commit is also checked through GitHub rather than silently assumed to exist.

## Verification

```bash
python -m pytest
```

Tests use small synthetic JPEGs, archives and metadata to exercise corruption, source identity, incomplete provenance, input order, context isolation, metadata stripping and missing-file handling. They do not require Drive, GPU access or downloading the medical dataset. Real metadata validation and its explicit pixel-check limitation are recorded under `outputs/phase3/bootstrap_verification/`.
