# Executive report: Phase 3, Sector 1 — Verified benchmark input loading

**Source:** [`phase3_01_benchmark_bootstrap.ipynb`](https://github.com/bogdanparvu18/msc-graduate-project/blob/f567456c9ab7260ad1fec39ca6d4f4ea1f1b286d/notebooks/phase3/phase3_01_benchmark_bootstrap.ipynb)  
**Review date:** 4 November 2026  
**Basis:** Notebook code and its saved Colab execution outputs at repository commit `f567456c9ab7260ad1fec39ca6d4f4ea1f1b286d`, together with the repository's published file inventory. The dataset and Drive report files were not rerun or independently revalidated for this report.  
**Saved run reviewed:** `20261104T183559_825320Z` (UTC; 13:35:59 on 4 November 2026 in `America/Toronto`).

## Executive summary

This notebook establishes the first Phase 3 interface to the visual evidence already prepared in Phase 2, Sector 8. It resolves the configured GitHub source ref to its actual commit, reads and verifies the selected S8 catalog and metadata, selects a small set of visual-input recipes, and loads their existing image bytes from mounted Google Drive storage. Original frames remain source references, while crops, overlays and collages are read from the existing S8 TAR archives. The notebook does not regenerate these images.

The saved execution records **`image_smoke_passed` for six variants**, comprising **11 image input slots and eight distinct image paths**. All six variants were selected for the same annotation-based subject. The input-slot count includes repeated use of the target frame and crop; it is not the number of newly created images. The run verifies three selected image archives and displays the loaded images. It then saves five JSON artifacts containing configuration, source identity, sample verification, evaluator evidence and provenance.

The actual data revision resolved during that run was [`8b91a1b92b01870242a09a8f98c1fc4f7f5e6abe`](https://github.com/bogdanparvu18/msc-graduate-project/commit/8b91a1b92b01870242a09a8f98c1fc4f7f5e6abe). This is distinct from the later commit containing the saved notebook. The result establishes successful loading of the selected sample, not a complete-dataset audit or a validated QA benchmark. The saved output explicitly retains **`training_ready=False`**, **`qa_validation_status="not_authored"`** and a **`None` reference answer**.

## Workflow

| Stage | What the notebook does |
| --- | --- |
| Configuration and dependencies | Installs only missing or below-minimum Pillow/PyArrow packages, centralizes imports, defines the GitHub ref and storage roots, and obtains an optional `GITHUB_TOKEN` without including it in saved configuration. The implementation is contained in the notebook. |
| Source resolution and metadata loading | Resolves `main` through GitHub once per run and uses the resulting immutable commit for subsequent reads. Follows the S8 storage index and selected archive catalog; verifies direct metadata and, where necessary, assembles the already-published exact-byte parts with offset, size and SHA-256 checks. |
| Catalog and provenance checks | Binds storage-index entries, transport/logical catalog descriptors, archive membership and consumed table schemas. Checks Parquet row counts and columns, source hashes, annotation links, input order and temporal frame identities. Cached files are verified before reuse. |
| Sample selection | Selects one subject with all six requested variants: `full`, `crop`, `overlay`, `fullcrop`, `temporal_multi` and `temporal_collage`. Retains annotation-assistance flags and pending split information. Shared-subject sampling does not by itself validate clinical task equivalence. |
| Mounted-image loading | Reads originals from their exact portable paths and derivatives from declared regular TAR members. Checks archive and consumed-image hashes, available byte counts and image dimensions. Reads members without extracting the full archive or overwriting source data. |
| Input separation and display | Creates a model projection containing the question, RGB images, neutral image roles and explicitly allowed context. Keeps source annotation evidence and provenance separate, clears image metadata, and displays the verified inputs. |
| Persistence | Writes five JSON artifacts to a local run directory and, after successful completion with the default configuration, to a separate Drive run directory. The reviewed notebook has no GitHub publication cell. |

## Results visible in the saved notebook

| Measure | Saved result | Interpretation |
| --- | ---: | --- |
| Final input-loading status | `image_smoke_passed` | All selected sample inputs completed the notebook's image-verification path. |
| Selected variants / annotation-based subjects | 6 / 1 | One recipe per requested variant for the same subject. |
| Image input slots / distinct image paths | 11 / 8 | Calculated from the saved ordered paths; repeated target/crop use explains the difference. |
| Distinct original/context images / derived images | 5 / 3 | Five existing frames, plus one crop, one overlay and one temporal collage. |
| Image archives used | 3 | `roi_crop.tar`, `bbox_overlay.tar` and `temporal_collage.tar`. |
| Logical metadata files verified | 9, plus the storage index | Archive catalog, archive manifest, ordered recipes, and three Parquet/schema pairs for source evidence, annotation evidence and visual-source provenance. |
| Physical metadata file reads shown | 14 | Five direct logical files, eight parts for four logical files, and the index; commit resolution is a separate API call. |
| Model-facing fields | 4 | `question`, `images`, `image_roles` and `context`. |
| Reference answer / QA validation | `None` / `not_authored` | Source annotation labels were not converted into answers to the preview question. |
| Saved report artifacts | 5 JSON files | Configuration, source snapshot, sample report, evaluator records and provenance. |
| Training readiness | `False` | Validated QA and final source-group-safe split assignment remain downstream requirements. |

The displayed inputs have the following sizes:

| Variant | Input images | Displayed dimensions |
| --- | ---: | --- |
| `full` | 1 | 336 × 336 |
| `crop` | 1 | 336 × 283 |
| `overlay` | 1 | 336 × 336 |
| `fullcrop` | 2 | 336 × 336, followed by 336 × 283 |
| `temporal_multi` | 5 | Each 336 × 336 |
| `temporal_collage` | 1 | 1680 × 366 |
| **Total** | **11** | Image slots, including reused inputs. |

The target is frame **8112** of video **`8885668afb844852`**. The temporal sequence contains frames **8052, 8082, 8112, 8142 and 8172**, with declared offsets **−2, −1, 0, +1 and +2 seconds**. `fullcrop` uses two separate images; it does not create another panel or require the `evidence_panel.tar` archive. The crop, overlay and fullcrop recipes are marked annotation-assisted.

The saved archive checks report approximately **32.5 MB** for `roi_crop.tar`, **109.3 MB** for `bbox_overlay.tar` and **2991.2 MB** for `temporal_collage.tar`. The selected archive is hashed on first use; unchanged archive verification is reused within the resolver instance, while consumed image members are checked individually.

The recorded environment is **Python 3.13.18**, **Pillow 11.3.0** and **PyArrow 23.0.1**, running in Colab. The synthetic self-test passed, all 13 code cells have sequential saved execution counts, and there are no saved error outputs. These observations describe the preserved execution; they are not a fresh inspection of the current Drive contents.

## Outputs and scope

The saved local report directory is:

```text
/content/mmvqa_phase3_start/reports/20261104T183559_825320Z/
```

The final cell reports the corresponding persistent Drive directory:

```text
/content/drive/MyDrive/MMVQA_Clinical/outputs/phase3/bootstrap_reports/20261104T183559_825320Z_8b91a1b92b01/
```

The notebook's `save_reports_to_drive=True` setting controls that Drive copy. It does not publish files to GitHub. All five files are written together in the run directory; the subdirectories below describe their intended GitHub organization under `outputs/phase3/sector1/`.

| Artifact | Contents | GitHub destination |
| --- | --- | --- |
| `configuration.json` | Run settings, requested ref, mount locations, variants and preview question; no credentials. | `configs/configuration.json` |
| `source_snapshot.json` | Actual resolved source commit, storage-index identity and consumed metadata sizes/hashes. | `configs/source_snapshot.json` |
| `sample_report.json` | Run ID, timestamps, environment, selected examples, assistance flags, image sizes and verification status. | `reports/sample_report.json` |
| `evaluator_records.json` | Source annotation evidence for the selected recipes, with no validated reference answers. | `results/evaluator_records.json` |
| `provenance.json` | Example, subject, annotation, source and product links, hashes, pending split fields and verified image-resolution records. | `results/provenance.json` |

At the reviewed repository snapshot, these five report files had not been published to GitHub; the saved notebook records their Drive persistence. Publishing them requires uploading the five files from the same completed run or adding a dedicated publication cell. The intended publication should preserve their exact bytes and commit the selected set together. For a full image-loading result, the report must retain `status="image_smoke_passed"` with every selected sample's `pixels_verified_now=true`. A metadata-only run can be published separately with its original `metadata_verified_images_not_checked` status rather than being presented as pixel verification.

No new `data/` directory or duplicate S8 dataset is produced. Original images, temporal frames, image TARs and the metadata cache remain in their existing storage locations. The two `results/` files are sample-level annotation/provenance exports, not predictions or evaluation scores.

The source revision is resolved dynamically at the beginning of a run; the S8 upstream commit is retained as lineage rather than substituted for that revision. Mounted image bytes must match the selected GitHub metadata. If storage has changed independently, a mismatch fails loading rather than mixing new pixels with old references. This sample check does not establish all-dataset availability, authored clinical QA, split safety across a completed benchmark, model performance or training readiness.
