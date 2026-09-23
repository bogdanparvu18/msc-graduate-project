# Executive report: Phase 2, Sector 1 — Kvasir-Capsule ingestion

**Source:** `phase2_01_ingestion.ipynb`  
**Review date:** 23 September 2026  
**Basis:** Notebook code and its saved execution outputs; the dataset was not rerun for this report.

## Executive summary

This notebook establishes the starting point for Phase 2 of the MMVQA Clinical data pipeline. It sets a reproducible run configuration in Google Colab, mounts persistent Google Drive storage, and prepares directories for raw data, curated data, and reports. It then checks the Kvasir-Capsule files, downloads or repairs missing source material when needed, reads the annotation metadata, standardizes its fields and labels, removes exact duplicate annotation rows, and checks that the remaining annotations match the physical image files. The resulting clean annotation table and audit reports are saved for later stages. A separate, confirmation-gated step publishes selected configuration and report files to GitHub.

The saved execution shows a complete dataset inventory and a passing audit: **47,239 clean annotation rows**, **47,238 physical labelled image files**, **47,229 distinct labelled frames**, and **21 of 21 dataset consistency rules passed**. The three counts measure different things; they should not be used interchangeably. This run reused existing raw files and a previously saved audit, so its displayed audit results were checked and loaded from storage rather than rebuilt from the raw dataset during that execution.

## Workflow

| Stage | What the notebook does |
| --- | --- |
| Configuration and storage | Synchronizes a clean local checkout of the project repository, loads the tracked notebook's literal `CONFIG`, records a run ID and source commit, sets seed 42, mounts Google Drive, and creates Phase 2 directories. |
| Provisioning | Reuses existing Kvasir-Capsule files or downloads the dataset when absent. It can recover missing labelled images from archives and produces inventory and archive-repair reports. |
| Raw-data checks | Verifies `metadata.csv`, video files, the 14 expected labelled-image classes, and their expected image counts; then reads the semicolon-delimited metadata. |
| Normalization | Canonicalizes metadata column names, standardizes finding labels into configured clinical groups, derives frame/video matching keys, and adds validated bounding-box fields while retaining source coordinates. |
| Cleaning and audit | Removes only exact duplicate annotation rows, reconciles metadata with physical image paths and class folders, records multiclass and repeated-annotation cases, and evaluates 21 consistency rules. Complete saved audit sets can be reloaded. |
| Persistence and publication | Saves clean metadata as Parquet and audit tables as CSV or JSON on Drive. After an explicit `PUSH` confirmation, it can commit eligible configs and reports to GitHub; the clean Parquet and full physical-image inventory are excluded from that publication. |

## Results visible in the saved notebook

| Measure | Saved result | Interpretation |
| --- | ---: | --- |
| Raw metadata rows | 47,248 | Original annotation records read from `metadata.csv`. |
| Exact duplicate rows removed | 9 | Duplicate full rows removed by the audit. |
| Clean metadata rows (`df_clean`) | 47,239 | Annotation records available to later cells. |
| Physical labelled image files | 47,238 | File instances counted in labelled class folders. |
| Distinct labelled frames | 47,229 | Unique frame identities represented by those files. |
| Multiclass frames | 9 | Each accounts for an additional class-specific physical image instance. |
| Same-frame, same-class extra annotation | 1 | Explains why clean annotation rows exceed physical image instances by one. |
| Labelled classes / discovered video files | 14 / 117 | Dataset provisioning inventory; the audit separately reports 43 videos represented in labelled annotations. |
| Dataset audit rules | 21 / 21 passed | Internal path, identity, class, count, and annotation consistency checks passed. |

The saved run reported `existing_storage`, with no download or archive recovery. Its audit status was `loaded_from_files`; the notebook validated the saved audit set and displayed its contents. The final publication output recorded a push of 69 eligible files to the repository's `main` branch, excluding `metadata_clean.parquet` and `physical_image_inventory.csv` from the dataset-audit publication.

## Outputs and scope

The main reusable data artifact is `data/curated/phase2/dataset_audit/metadata_clean.parquet` under the configured Google Drive root. The same directory contains a physical-image inventory, dataset characteristics, validation results, duplicate and multiclass diagnostics, and an export catalog in CSV or JSON. Provisioning reports and the run configuration are stored under `outputs/phase2/`.

The cached audit checks its saved files, but it does **not** automatically detect changes to the raw dataset; after changing source files or audit rules, the audit must be explicitly rebuilt with `force_rebuild=True`. Its filesystem inventory counts file paths and reconciles metadata; it does not decode images or compare their pixels. The notebook prepares directories for later work but does not create data splits or temporal windows, train a model, or perform medical question answering.
