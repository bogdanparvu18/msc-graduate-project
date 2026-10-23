# Executive report: Phase 2, Sector 7 — Image quality control with deferred splits

**Source:** [`phase2_07_image_quality_control.ipynb`](../../../notebooks/phase2/phase2_07_image_quality_control.ipynb)  
**Review date:** 23 October 2026  
**Basis:** The Colab notebook and published S7 bundle in GitHub `main` at commit [`cb4ac8d8f94caeb83dcddfccf495cdd4188bf9f0`](https://github.com/bogdanparvu18/msc-graduate-project/commit/cb4ac8d8f94caeb83dcddfccf495cdd4188bf9f0). The active [catalog](configs/sector7_state.json) selects run `20261023T005643407465Z`. Counts below were checked against its summary, typed schemas and image inventory; the inventory's SHA-256 matches the catalog. Execution and cloud-delivery outcomes are those recorded in the saved notebook.

## Executive summary

Sector 7 measures the technical quality of every distinct image file required by the S5 labelled-frame manifest or the S6 temporal manifest. It consumes the checked S1–S6 metadata chain from a pinned GitHub commit, copies and verifies source images on the local Colab SSD, and computes image dimensions, readability, brightness, contrast, blur, exposure and specular-reflection metrics. One file receives one QC record, which is attached to every annotation or temporal reference that uses it.

The current bundle covers **66,127 image files representing 66,118 physical video frames**. All 66,127 files passed source-integrity verification and were readable. The saved run used NVIDIA image decoding and GPU pixel processing for all images, with no CPU or hybrid fallback. It preserves **47,239 labelled annotation rows** and **110,666 temporal-reference rows**, providing QC coverage for all **157,905 source references**.

**Final split assignment and statistical QC thresholds remain deferred.** S7 preserves upstream split information as provenance, leaves `final_split` missing and records `split_assignment_status="pending"`. Blur, contrast and brightness outlier thresholds have not been fitted. Accordingly, `training_ready=false` and `image_usable_for_modelling=false` remain explicit even for readable images. The zero review-required count is the outcome of the current technical-readability rules; it does not mean that all images have passed a calibrated visual-quality assessment.

Generated tables, reports and configuration records are saved and validated on the SSD. They are uploaded directly from there through the Google Drive API, and completion requires the API to report the expected byte count and SHA-256 for every selected file. GitHub publication also uses the SSD copy. The saved execution reports **26 of 26 cloud files confirmed** and successful publication to `main`.

## Workflow

| Stage | What the notebook does |
| --- | --- |
| Configuration and source authority | Pins GitHub `main` and follows the explicit S1–S6 catalogs, configurations, table descriptors and schemas. Checks consumed hashes, producer identities, annotation consistency and S6's deferred-split contract. Active artifacts are selected through their catalogs, not by choosing the newest filename. |
| Complete image inventory | Takes the union of all S5 labelled-image paths and every S6 temporal-image reference. Deduplicates by image file while retaining physical frame identities and upstream split provenance. Shared files are analysed once; distinct files representing the same frame retain separate QC records. |
| SSD staging and integrity | Uses 32 staging workers to copy required images from mounted Drive into `/content/mmvqa_sector7`. Hashes bytes during transfer and separately verifies the SSD copy against the committed SHA-256. Already valid SSD images can be reused. Extracted-frame sidecars are checked alongside their images. |
| Technical image QC | Reads compressed image bytes from the SSD with 16 local-read workers and decodes batches of 64 through NVIDIA nvImageCodec. CuPy computes pixel metrics on the GPU. The configured `gpu_preferred` policy permits hybrid decoding and OpenCV/NumPy fallback, with the actual backend and fallback reason recorded per image. |
| Resumable computation | Stores metric checkpoints in batches of 1,000 images. Reuses a checkpoint only after its identity, file hash, inventory coverage and source-image hashes pass validation. A complete cached bundle must also match the current processing identity. |
| Deferred assessment | Records technical readability and metrics without fitting quantile thresholds to inherited splits. Statistical outlier flags remain null, thresholds remain uncalibrated, and modelling eligibility remains false. Missing references and unreadable images remain explicit report entries. |
| Manifest enrichment | Joins QC results back to every original labelled and temporal row. Preserves annotation IDs, bounding boxes, finding-segment membership and temporal offsets. Adds QC-prefixed measurements where necessary to preserve existing upstream audit fields. |
| Local persistence | Rechecks source images on the SSD, validates table contracts, writes Parquet tables and typed schemas, and verifies exact serialized readback. Installs the local catalog last as part of the validated output transaction. |
| Verified cloud delivery | Uploads generated artifacts directly from the SSD through the Drive API. Fresh paginated directory inventories compare cloud size and `sha256Checksum` with the local catalog. Retries only missing or mismatched files; uploads the catalog after the other artifacts match; performs a final inventory including the catalog before returning `complete`. |
| GitHub publication | Requires cloud confirmation bound to the current SSD snapshot for an actual push. Reads the selected metadata from the SSD, rechecks the consumed upstream GitHub inputs and publishes under `outputs/phase2/sector7`. Unchanged artifacts produce no Git diff, and unrelated repository files, including this README, are preserved. |

The image-to-annotation relationship is inherited from the upstream manifests. S7 validates and carries that relationship forward; it does not reconstruct clinical annotations from pixels. A physical frame is identified by its video and decoder frame index, whereas an image file is identified by its portable path and verified bytes.

## Results visible in the saved notebook

The active [published summary](reports/summary_20261023T005643407465Z.json), [QC summary CSV](reports/image_qc_summary_20261023T005643407465Z.csv) and matching schemas record the following completed run. Section 8 of the saved notebook reports `computed`, with 66,127 newly computed images and no reused QC checkpoints. Section 9 records cloud verification; section 10 records GitHub publication.

| Measure | Published result |
| --- | ---: |
| Distinct physical video frames | 66,118 |
| Distinct image files / QC records | 66,127 |
| Distinct S5 labelled-image files | 47,238 |
| Distinct image files referenced by S6 | 61,329 |
| Image files shared by S5 and S6 | 42,440 |
| S5 image files outside the S6 temporal references | 4,798 |
| S6-only image files | 18,889 |
| Preserved labelled annotation rows | 47,239 |
| Preserved temporal-reference rows | 110,666 |
| Total source references covered | 157,905 |
| Readable / technically usable image files | 66,127 / 66,127 |
| Unreadable images / unresolved image references | 0 / 0 |
| Images decoded by NVIDIA GPU / processed with GPU QC | 66,127 / 66,127 |
| Hybrid-decoded / CPU-decoded / fallback images | 0 / 0 / 0 |
| Review-required images under current rules | 0 |
| Threshold-calibration candidates | 0 |
| Images marked usable for modelling | 0 |
| Final split assignment / training readiness | Pending / false |
| Selected metadata files confirmed in cloud | 26 / 26 |

The file counts reconcile as **47,238 + 61,329 − 42,440 = 66,127**. Equivalently, **61,329 + 4,798 = 66,127**. The additional 4,798 files are labelled images that do not occur among S6's selected temporal-image references. S6 builds context around segment representatives at selected offsets; its temporal manifest is not a complete copy of S5's labelled-image list. S7 therefore uses both inputs so these labelled files also receive QC.

Annotation rows, image files and physical frames are different units. Several annotation rows can refer to one file, and more than one image file can represent the same physical frame. The difference between 66,127 files and 66,118 frames does not indicate that nine frames are missing. Likewise, the 47,239 labelled rows must not be interpreted as 47,239 distinct labelled files.

The saved runtime identifies an NVIDIA L4 GPU. Source staging reports 32 workers, 66,127 copied images, zero reused SSD images and approximately 33.41 images per second. These are observations from this run, not guaranteed throughput for another Colab session.

The four statistical thresholds are missing in the saved threshold table, and the four corresponding outlier flags are unassessed. Thus `review_required_images=0` must be read together with `qc_assessment_status="metrics_only_final_split_pending"`. The bundle's `material_validation_passed=true` establishes the implemented material and reference checks; it does not establish clinical quality, final split safety or readiness for training.

## Outputs and scope

The local output root is `/content/mmvqa_sector7/outputs/phase2/sector7`. Its verified cloud destination is under `/content/drive/MyDrive/MMVQA_Clinical/outputs/phase2/sector7`, and selected metadata is published at the same project-relative directory in GitHub. Image access paths retain persistent-storage references, while portable relative paths and hashes support staging in another environment. The original image binaries remain outside this GitHub output bundle.

| Artifact | Purpose |
| --- | --- |
| `results/image_qc_inventory.parquet` | One row per distinct image file, with physical frame identity, source-manifest membership, upstream split provenance and pending final split fields. |
| `results/image_qc_manifest.parquet` | Per-file source verification, image hash, dimensions, readability, technical metrics, actual decode/compute backend, deferred outlier flags and eligibility status. |
| `results/image_qc_thresholds.parquet` | One row recording the uncalibrated threshold values, deferred fitting status and zero calibration candidates. |
| `results/labelled_frame_manifest_with_qc.parquet` | All 47,239 S5 annotation rows, including their original annotations and bounding boxes, enriched with QC and explicit deferred split fields. |
| `results/temporal_manifest_with_qc.parquet` | All 110,666 S6 temporal references, preserving target/context identities, offsets, annotation semantics, extraction provenance and upstream image-audit fields, with S7 QC attached. Contains metadata and image references, not embedded JPEG bytes. |
| `reports/image_qc_source_audit_<timestamp>.parquet` | Expected and observed hashes and byte counts, source verification status, SSD access paths and extraction-sidecar evidence for all 66,127 files. |
| `reports/qc_reference_report_<timestamp>.parquet` | Coverage and QC status for every labelled or temporal source row, retaining the source manifest, row and reference identity. |
| `reports/image_read_failure_report_<timestamp>.parquet` | Unreadable-image diagnostics; zero rows in the current run. |
| `reports/missing_image_reference_report_<timestamp>.parquet` | Unresolved image-reference diagnostics; zero rows in the current run. |
| `reports/image_qc_summary_<timestamp>.parquet` and `.csv` | One aggregate QC row with readability, backend, deferred-calibration and eligibility counts. Only this small summary has a CSV companion. |
| Matching `*_schema.json` files | Typed readback, column order and definitions, null handling, row counts and fingerprints for all ten tables. |
| `configs/sector7_config_<timestamp>.json`, `configs/sector7_lineage.json` | Effective configuration, checked input identities, source lineage and active processing-code identity. |
| `configs/sector7_state.json` | Active artifact selection, configuration and summary descriptors, processing identity, deferred execution contract, byte counts and SHA-256 values. |
| `reports/summary_<timestamp>.json`, `reports/executive_report_<timestamp>.md` | Machine-readable outcome and concise narrative for the active bundle. |
| `_transfer_state/sector7_cloud_status.json` on the SSD | Operational delivery status, verification counts, pending reasons and copy errors. This local transfer journal is outside the published artifact catalog. |

Here, **“schema 3” means version 3 of the S7 state/catalog contract**. It does not mean that S7 produces three catalogs. The active catalog selects 25 generated artifacts; together with `configs/sector7_state.json` itself, these form the 26-file delivery verified in the saved run. Table-schema and summary-format version numbers are independent. Consumers should follow the active catalog rather than combine timestamped files from different runs.

Pixel metrics use a centred circular field of view with radius `0.95 × min(height, width) / 2`, without modifying the source image. Brightness is mean grayscale intensity divided by 255; contrast is grayscale standard deviation divided by 255; blur is the variance of a four-neighbour Laplacian evaluated inside an eroded field-of-view mask. Exposure fractions use grayscale values below 20 or above 240. The specular proxy uses value above 240 and saturation below 30. These fixed pixel criteria are distinct from the deferred dataset-level quantile thresholds.

Extraction sidecars retain source-video, frame, encoding and image-hash provenance. Differences in `opencv_version`, `decoder_signature` and `nvdec_runtime` are treated as informational during S7 sidecar validation and do not produce a mismatch warning or invalidate matching image bytes. Source/frame/encoding identity and SHA-256 checks remain enforced. Runtime information can still contribute to computation or cache identity without redefining the expected image SHA-256.

Cloud delivery verifies the uploaded object through fresh Google Drive API metadata, rather than treating a successful write to the DriveFS mount as cloud confirmation. It compares both size and server-reported SHA-256 with the SSD record, without downloading the cloud contents for another hashing pass. A missing or mismatched object remains pending and is eligible for retry. Missing cloud checksum metadata also prevents completion; ambiguous destination identities or changes to the SSD source block delivery. The saved `complete` result describes the verified state at that time, not protection against later replacement.

The current saved notebook explicitly sets `CONFIG["sector7"]["push"]=True` and `dry_run=False`; the recorded publication commit is [`ef44f481ea6ed653246adae876e043df04fd0dca`](https://github.com/bogdanparvu18/msc-graduate-project/commit/ef44f481ea6ed653246adae876e043df04fd0dca). Publication reads newly generated Parquet files directly from the SSD and requires the matching cloud-delivery proof. It does not upload them to Drive and then download them again as the GitHub source.

S7 supplies technical measurements and preserves evidence for later preparation and final split decisions. It does not remove source rows, repair annotations, assign new clinical labels, interpret unlabelled context as normal, extract new temporal frames, generate crops or collages, allocate final train/validation/test membership, or train a model.
