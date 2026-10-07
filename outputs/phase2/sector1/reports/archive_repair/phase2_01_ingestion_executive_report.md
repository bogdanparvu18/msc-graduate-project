# Executive report: Phase 2, Sector 1 — Kvasir-Capsule ingestion

**Source:** [`phase2_01_ingestion.ipynb`](https://github.com/bogdanparvu18/msc-graduate-project/blob/0f35cbeb75ebcfa0c10631d44aa79e7e61d7d6f8/notebooks/phase2/phase2_01_ingestion.ipynb)  
**Review date:** 6 October 2026  
**Basis:** Notebook code, its saved execution outputs, and published dataset-audit files at repository commit `0f35cbeb75ebcfa0c10631d44aa79e7e61d7d6f8`; the dataset was not rerun for this report.  
**Saved run reviewed:** `20261006_200455` (`America/Toronto`).

## Executive summary

This notebook establishes the starting point for Phase 2 of the MMVQA Clinical data pipeline. It defines a reproducible run configuration in Google Colab, mounts persistent Google Drive storage, and prepares directories for raw data, curated data, and reports. It checks the Kvasir-Capsule files, downloads or repairs missing source material when needed, reads the annotation metadata, standardizes its fields and labels, removes exact duplicate annotation rows, and checks that the remaining annotations match the physical image files and class folders. The resulting clean annotation table and audit reports are saved for later stages. A separate, confirmation-gated step synchronizes selected configuration and report files to GitHub using source–destination SHA-256 comparisons.

The saved execution shows a complete dataset inventory and a passing audit: **47,239 clean annotation rows**, **47,238 physical labelled image files**, **47,229 distinct labelled frames**, and **22 of 22 dataset consistency rules passed**, including **zero invalid bounding-box annotation rows**. The three counts measure different things; they should not be used interchangeably. This run reused existing raw files and a previously saved audit, so its displayed audit results were checked and loaded from storage rather than rebuilt from the raw dataset during that execution. The current GitHub validation report also contains all 22 rules, and the export catalog describes all nine persisted audit tables.

## Workflow

| Stage | What the notebook does |
| --- | --- |
| Configuration and storage | Defines the literal runtime `CONFIG` before initialization, synchronizes a clean local checkout of the project repository, prints the checkout commit and adds a Toronto-timezone run ID to `CONFIG`, sets seed 42, mounts Google Drive, and creates Phase 2 directories. |
| Provisioning | Reuses complete existing Kvasir-Capsule storage. When required source material is missing, it downloads the configured source folder or completes missing/zero-byte metadata and video files. Missing labelled-image classes or incomplete class counts trigger archive recovery and, when necessary, replacement archive download. It produces inventory and archive-repair reports. |
| Raw-data checks | Verifies `metadata.csv`, a minimum video-file count, the 14 expected labelled-image classes, and their expected image counts; then reads the semicolon-delimited metadata. These checks establish file availability and inventory consistency. |
| Normalization | Canonicalizes metadata column names, standardizes finding labels into configured clinical groups, derives `image_key` and `video_key` from normalized filenames/IDs, and adds bounding-box flags and geometry while retaining source coordinates. |
| Cleaning and audit | Removes only exact duplicate full annotation rows, reconciles metadata with physical image paths and class folders, preserves multiclass and distinct repeated-annotation cases, and evaluates 22 consistency rules. Each physical path must identify its folder class unambiguously; a logical frame may have several classes. Complete saved audit sets can be verified and reloaded. |
| Persistence and publication | Saves clean metadata as Parquet and other audit tables as CSV or JSON on Drive, with a SHA-256 export catalog. After an explicit `PUSH` confirmation, it commits new or changed eligible files under `outputs/phase2/sector1/`; identical files are skipped. Changed audit reports and the catalog are committed together. The clean Parquet and full physical-image inventory remain on Drive and are excluded from that GitHub publication. |

## Results visible in the saved notebook

| Measure | Saved result | Interpretation |
| --- | ---: | --- |
| Raw metadata rows | 47,248 | Original annotation records read from `metadata.csv`. |
| Exact duplicate rows removed | 9 | Surplus full-row duplicates removed; the diagnostic retains all 18 rows belonging to the nine duplicate groups. |
| Clean metadata rows (`df_clean`) | 47,239 | Distinct full annotation records available to later cells. |
| Physical labelled image files | 47,238 | File instances counted in labelled class folders. |
| Distinct labelled frames / unique `image_key` values | 47,229 | Unique logical frame identities; several annotation rows may share one image key. |
| Multiclass frames | 9 | Each has two classes in this snapshot and contributes one additional class-specific physical image instance. |
| Same-frame, same-class extra annotation | 1 | One frame has two retained `erosion` annotations, explaining the extra clean annotation row above the physical-image count. |
| Invalid bounding-box annotation rows | 0 | `valid_bounding_boxes_when_present` passed; `invalid_bounding_boxes.csv` contains its column headers and zero data rows. |
| Labelled classes / discovered video files | 14 / 117 | Dataset provisioning inventory; the audit separately reports 43 videos represented in labelled annotations. |
| Dataset audit rules | 22 / 22 passed | Internal path, identity, class, bounding-box, count, and annotation consistency checks passed. |

The count relationships are **47,248 = 47,239 + 9** (raw annotations = clean annotations + exact duplicates removed), **47,238 = 47,229 + 9** (physical image instances = distinct frames + extra class memberships), and **47,239 = 47,238 + 1** (clean annotations = physical image instances + an extra same-class annotation). Thus the ten annotation rows above the distinct-frame count consist of nine extra class annotations and one extra annotation within the same class. The latter is frame **2380** of video **`eb0203196e284797`**, with `image_key=eb0203196e2847972380` and two retained `erosion` records, as recorded in [`multi_annotation_same_class_groups.csv`](../../dataset_audit/multi_annotation_same_class_groups.csv). These two records were not identical full rows; that grouping report alone does not identify which field differs.

The bounding-box rule was added at position **15**, bringing the total from 21 to 22. It accepts completely absent localization, but rejects populated annotations with incomplete, nonnumeric or nonfinite coordinates, nonfinite derived geometry, or nonpositive width/height. This metadata check does not test whether a box lies inside the decoded image boundaries. Its diagnostic, `invalid_bounding_boxes.csv`, is the ninth table in `AUDIT_STORAGE_SPEC`; the export catalog is a separate file.

The saved run reported `existing_storage` and `already_present`, with no download, archive recovery or archive redownload. Its audit status was `loaded_from_files`; the notebook verified the saved audit set and displayed its contents. The final publication output recorded a push of **9 files** to `main` in commit [`90d560e`](https://github.com/bogdanparvu18/msc-graduate-project/commit/90d560e2e1e250d04348a584e7af505788ccd09d): one run configuration, six provisioning reports, and two updated dataset-audit files (`dataset_validation.csv` and `audit_export_report.json`). The other six eligible dataset-audit files had identical source and destination SHA-256 hashes and were skipped.

## Outputs and scope

The main reusable data artifact is `data/curated/phase2/dataset_audit/metadata_clean.parquet` under the configured Google Drive root. The full saved audit consists of **nine table files plus `audit_export_report.json`**: clean metadata, the physical-image inventory, dataset characteristics, validation results, exact-duplicate diagnostics, multiclass frames, same-class multiple-annotation groups, the image-key definition, and invalid bounding-box diagnostics. The catalog records formats, dimensions, column/dtype declarations, byte sizes and SHA-256 checksums.

`image_key` is a deterministic matching key derived from `filename`: it removes the path and extension, keeps only ASCII letters and digits, and uses lowercase. It identifies a logical frame in this audited snapshot, rather than an individual annotation row. Several classes or distinct annotations for one frame intentionally share the same key. It is not an image-content hash; the audit separately checks for key collisions and agreement with video/frame identities.

Provisioning reports and run configurations remain under the shared Drive `outputs/phase2/` directories. Their GitHub destination is **`outputs/phase2/sector1/`**, with only `configs/phase2_01_ingestion_*_config.json`, `reports/archive_repair/**`, and the selected `dataset_audit/` files eligible. The GitHub audit subset is **seven table files plus the catalog**; `metadata_clean.parquet` and `physical_image_inventory.csv` are excluded. Because the catalog describes all nine local tables, the selected GitHub folder alone cannot restore the complete saved audit.

Before publication, every eligible audit source must exist and match its catalog checksum. The publisher refreshes a clean repository checkout, compares source and destination SHA-256 content hashes even when a destination already exists, and classifies files as `new`, `updated` or `unchanged`. It rechecks the sources and verifies the actual staged audit-file bytes before committing, so Git text conversion cannot silently invalidate the catalog checksums. This is output synchronization; it does not establish raw-source freshness.

The cached audit verifies its saved files and validation results, but it does **not** automatically detect changes to the raw dataset; after changing source files or audit rules, the audit must be explicitly rebuilt with `force_rebuild=True`, then ordinary reuse can return to `False`. Its filesystem inventory counts paths and reconciles metadata; it does not decode images, compare their pixels or prove image-to-video alignment. The notebook prepares directories for later work but does not create data splits or temporal windows, train a model, or perform medical question answering.
