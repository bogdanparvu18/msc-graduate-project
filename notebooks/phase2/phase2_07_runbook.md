# Phase 2, Sector 7 — deferred-split QC runbook

## Purpose and migration

Run `phase2_07_image_quality_control.ipynb` after publishing a complete S6 bundle using the contract introduced in PR #13. S7 verifies the published source chain and image material, computes technical image metrics, and preserves all annotations and temporal references. It does not create final dataset splits or authorize training.

The previous S7 contract rejected shared images across historical splits, required S6 exclusion fields, and fitted thresholds on the inherited `train` assignment. Those rules are incompatible with deferred S6 material. The revised contract keeps historical assignments as provenance and exports:

| Field | Meaning in this S7 revision |
| --- | --- |
| `upstream_split` / `upstream_splits_json` | Historical S4/S5 assignment(s), never a QC filter or calibration set. |
| `physical_frame_id` | Canonical `[video_key, opencv_frame_index]` identity in the QC inventory/reference tables. Original S5/S6 identifiers remain intact in enriched manifests. |
| `final_split` | Null. No automatic copy from the historical split. |
| `split_assignment_status` | `pending`. |
| `training_ready` | `false`. |
| `image_technically_usable` | The file decoded successfully; this does not certify clinical adequacy or training eligibility. |
| `image_usable_for_modelling` | Retained as a compatibility safeguard; always `false`. |
| `qc_assessment_status` | `metrics_only_final_split_pending`. |
| `threshold_calibration_status` | `deferred_final_split_pending`; numeric quantile thresholds and outlier flags are null. |

Absolute pixel settings still define underexposure, overexposure and specular fractions. They do not fit statistics on a dataset split. Statistical calibration must be implemented later against a validated final training partition.

## Before starting

1. Publish the matching S1–S6 metadata chain to GitHub `main`. S6 must declare deferred/pending/not-training-ready material, have a complete catalog, and have no unresolved materialization failures.
2. Mount the persistent data at `/content/drive/MyDrive/MMVQA_Clinical`, or change `BOOTSTRAP.storage_root` to the correct root. Official images, extracted temporal images and extraction `*.provenance.json` sidecars must be present at the committed portable paths.
3. Use a Colab runtime with sufficient SSD space for the image collection and checkpoints. The default SSD root is `/content/mmvqa_sector7`, separate from Drive. A GPU is preferred; CPU fallback is supported and reported. S7 does not need to decode raw videos.
4. Keep `SECTOR7_SETTINGS['push']=False` for the initial run. The optional publication step requires the `GITHUB_TOKEN` Colab secret with repository write access.

Do not point metadata settings at an uncommitted Drive catalog to bypass a failed source check. Fix and publish the upstream producer, then load the new current-main snapshot.

## Run order

| Notebook section | Action | Expected result |
| --- | --- | --- |
| 1–2 | Install dependencies, import libraries and set roots/options. | CPU dependencies available; optional GPU failure is visible in the install log and execution summary. |
| 3–6 | Define input, QC, persistence and orchestration functions. | No dataset processing or publication yet. |
| 7 | Mount storage and load the published chain. | One pinned `main` commit; verified producer catalogs, configurations, schemas, SHA-256 values, annotation identities and deferred temporal contract. |
| 8 | Verify/stage media, then compute or resume QC. | Each unique image file measured once; source checks cover every S5 labelled image and S6 temporal image; every original reference receives QC enrichment. |
| 9 | Persist the complete bundle. | Schema 3 S7 catalog installed last; saved tables read back and verified; readiness remains false. |
| 10 | Optionally publish metadata. | Disabled by default. Enabled publication rechecks consumed upstream files against current `main`. |

The default `qc_workers=16` controls concurrent media I/O. `qc_decode_batch_size=64` bounds each GPU/CPU decoder batch. `qc_checkpoint_batch_size=1000` controls reusable metric parts; it is independent of decoder batch size. The default source-image size bound is 64 MiB and the SSD free-space reserve is 2 GiB.

The first metadata load may take several minutes: Git fetches certified upstream blobs into a local cache. Subsequent loads reuse those bytes after checking their identities. Keep the local workspace when resuming a session.

For publication, set `SECTOR7_SETTINGS['push']=True` and rerun sections 7–10. This resolves configuration explicitly while retaining valid source and metric caches. An unrelated change on `main` is allowed; a change to any consumed upstream artifact stops publication and requires a new compatible input load. This publishes S7 metadata only, never images, sidecars or notebooks.

## Integrity and provenance checks

- Metadata comes exclusively from a fixed GitHub commit. Catalog-selected artifacts must match committed hashes, typed schemas and table fingerprints. S6's saved consumed-input identities must agree with the same current S1–S5 producer chain.
- Every labelled image, including files not referenced by S6, uses the source hash inherited into S5. S6 temporal references, image audit, extraction requests and extraction results must agree on file identity, physical frame and SHA-256.
- Existing SSD files are read and hashed before reuse. Missing/corrupt SSD files are recopied from Drive into temporary files; only verified copies are installed. There is no separate full Drive hashing pass before copying.
- Extracted images additionally require a sidecar matching the published image hash, source-video hash, frame/index offset, decoder and encoding provenance. A sidecar supplies corroborating provenance; it cannot redefine the expected image hash.
- QC hashes the exact encoded bytes passed to the decoder. Persistence validates the SSD snapshot that was processed. A later change to the Drive source is not certified by this run; downstream consumers must verify bytes when they stage them again.
- A file with correct bytes that fails decoding remains a QC failure row. Missing files, source-hash mismatches and invalid provenance stop completion; S7 does not silently adopt replacement hashes.

## Recovery after interruption

| Situation | Recovery procedure |
| --- | --- |
| Source copy interrupted | Rerun section 8. Complete SSD copies are hash-checked and reused; incomplete temporary copies are not accepted as inputs. |
| SSD copy has the wrong hash | Section 8 recopies that file from Drive. If the replacement still disagrees with the published hash, repair or republish the upstream material first. |
| Extracted-image sidecar missing or inconsistent | Restore the correct S6 image/sidecar pair or rerun and publish S6. Rerun section 8 afterward. |
| QC interrupted between batches | Rerun section 8. Complete metric parts under `.qc_checkpoints` are reused only when their checksums, table fingerprints, exact image coverage and processing identity agree. |
| One metric checkpoint is corrupt | Only the invalid part is recomputed. Other valid parts remain reusable. |
| S7 output catalog/table is missing, corrupt or schema 2 | Recompute from verified inputs. Existing output files stay in place until replacement is verified; overwritten invalid-cache files are retained in a recovery backup. |
| Process stops during final installation | Rerun section 8 or 9. The transaction journal restores the preceding state if installation was incomplete, or accepts a fully installed and verified new state. |
| Colab runtime is replaced | Rerun from section 1. `/content` caches are ephemeral; the persistent S7 bundle can still be reused after source verification if its processing identity matches. |
| GitHub push fails due to credentials/network | Correct the secret/network and rerun section 10 with the validated `SECTOR7_PERSISTED` handle. |
| Consumed upstream data changed on main | Rerun section 7, then sections 8–10 against a complete compatible chain. Do not bypass the upstream guard. |

A failed load, computation or persistence cell clears its dependent result/publication variables. It cannot silently publish a stale successful run left in memory. Recovery never converts an upstream/source integrity failure into cache reuse.

## Acceptance checks

The exact counts come from the consumed snapshot and are not hard-coded gates. For the deferred S6 bundle reviewed on 21 October 2026, the expected metadata coverage is:

| Measure | Count |
| --- | ---: |
| S5 annotation rows preserved | 47,239 |
| S6 temporal reference rows preserved | 110,666 |
| Temporal windows / retained targets | 22,220 |
| Distinct physical frames in the combined inventory | 66,118 |
| Distinct image files requiring QC | 66,127 |
| Image files referenced under multiple historical splits | 30,100 |

The nine-file difference between physical frames and image files reflects multiple official files for some frames. A repeated temporal reference is not another physical image and must not trigger another decode.

Before publishing, inspect `image_qc_summary`, `image_read_failure_report` and `missing_image_reference_report`. All source hashes must pass, all source rows must remain, final assignments must be null/pending, and all model-readiness flags must be false. Quantile thresholds being null is expected. A completed material/QC run does not mean every image is medically useful.

The five result tables and five report tables are authoritative typed Parquet artifacts with JSON schemas. Only the compact QC summary needs an additional CSV view; large row-level reports do not need duplicate CSV exports. Follow `configs/sector7_state.json` to select the active configuration, reports and tables instead of choosing timestamped files by name.

## Validation and limitations

All 90 regression cases passed against the final notebook. Checks cover the new deferred contract, multiple historical split references, preserved annotations, complete expected-hash coverage, corrupt/stale inputs, sidecar identity, CPU JPEG/PNG processing, interrupted batch resume, corrupted checkpoints, inventory-code cache invalidation, typed persistence, transaction recovery after forced process exit, and publication against changed upstream data. Tests are delivered separately and kept out of GitHub in accordance with the repository owner's preference.

The complete metadata reader passed against main commit `6790082b006e2aaad49e54c288217387f232fb02`, checking 133 published paths and 196 metadata hash bindings. The configured Drive root was empty and remained untouched. The cold load took approximately 11 minutes in the validation environment, including first-time Git blob downloads and full cross-sector checks.

The metadata contract and inventory can be tested against the real published manifests without reading image binaries. Full pixel processing of the actual image collection, NVIDIA runtime compatibility/performance and Google Drive throughput must be verified in Colab. Saved notebook execution outputs are cleared so they cannot be mistaken for a completed run of this revision.

## Downstream changes required

Consume the new S7 contract explicitly. Treat the old labelled-manifest `split` column as retained source provenance, and use the new `upstream_split` alias for clarity. Never infer a final assignment from it. Require a separately validated final split policy before training or fitting dataset-dependent thresholds.

Preserve physical-frame and annotation identities across future crops, collages and QA examples. Validate inherited image hashes during staging, and bind newly derived artifacts to their source hashes. The earlier illustrative 332-group allocation is not implemented or published by this QC revision.
