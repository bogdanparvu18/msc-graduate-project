# Phase 3 notebooks

This directory holds the Phase 3 notebooks, starting with `phase3_01_benchmark_bootstrap_colab.ipynb`. Upload the tested standalone notebook here after its Colab check. Creating this README establishes the directory; it does not publish notebook execution results.

## Phase 3.01 — Verified input loading

The first notebook resolves the current GitHub source ref to an immutable commit, reads verified S8 metadata, selects visual input samples, and loads existing originals/derived images from mounted storage. It keeps model inputs, evaluator evidence and provenance separate.

It does not regenerate images, author validated QA, assign final splits, train a model or produce benchmark scores. `training_ready=false` is expected at this step.

## Outputs already produced by the notebook

The current notebook writes these five JSON files into one run directory:

| File | Contents | Proposed GitHub destination |
| --- | --- | --- |
| `configuration.json` | Run settings, source ref, mount paths, variants and preview question. Credentials are not included. | `outputs/phase3/sector1/configs/configuration.json` |
| `source_snapshot.json` | Actual resolved source commit, storage index and consumed metadata sizes/hashes. | `outputs/phase3/sector1/configs/source_snapshot.json` |
| `sample_report.json` | Run status, environment, selected examples, image counts and pixel-verification outcome. | `outputs/phase3/sector1/reports/sample_report.json` |
| `evaluator_records.json` | Source annotation evidence for the selected samples; no validated reference answers. | `outputs/phase3/sector1/results/evaluator_records.json` |
| `provenance.json` | Example, subject, annotation, source and visual-product links, with hashes and image-resolution evidence when checked. | `outputs/phase3/sector1/results/provenance.json` |

Current local save directory:

```text
/content/mmvqa_phase3_start/reports/<RUN_ID>/
```

With the default `save_reports_to_drive=true`, the final successful cell also saves them to:

```text
/content/drive/MyDrive/MMVQA_Clinical/outputs/phase3/bootstrap_reports/<RUN_ID>_<commit12>/
```

The `sector1/configs`, `sector1/reports` and `sector1/results` GitHub locations above are a publication convention, not directories currently created by the notebook. All five files must come from the same run.

## Publish after the Colab test

1. Complete the notebook. For a full image-loading check, confirm `sample_report.json` has `status="image_smoke_passed"` and every sample has `pixels_verified_now=true`. A metadata-only run may be published with its honest `metadata_verified_images_not_checked` status.
2. Save the exact notebook used for that run to `notebooks/phase3/phase3_01_benchmark_bootstrap_colab.ipynb`.
3. Upload the five JSON files from that run to the destinations listed above, preferably together in one commit.
4. Add `outputs/phase3/sector1/README.md` describing the purpose, source commit, run ID, tested variants, observed results and limitations. Describe the results as input-loading verification, not model/QA benchmark performance.

There is currently no automatic GitHub publication cell. Saving reports to Drive does not publish them to GitHub.

No new `data/` directory is needed for this step. Keep original images, S8 TAR archives and metadata cache in their existing storage locations. Phase 3.01 exports references and sample-level verification evidence rather than another copy of the dataset.

If several runs must coexist, use a later `runs/<RUN_ID>/` publication convention and record which run is active. The simple paths above can represent the selected completed run while Git retains earlier committed versions.
