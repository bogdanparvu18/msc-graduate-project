# Executive report: Phase 2, Sector 8 — Visual input preparation with verified archive delivery

**Source:** [`phase2_08_visual_input_preparation.ipynb`](../../../notebooks/phase2/phase2_08_visual_input_preparation.ipynb)  
**Review date:** 26 October 2026  
**Basis:** The Colab notebook and published S8 bundle in GitHub `main` at commit [`5129a3a65df483ab0b96e699135f9258fad234c0`](https://github.com/bogdanparvu18/msc-graduate-project/commit/5129a3a65df483ab0b96e699135f9258fad234c0). The [GitHub storage index](configs/sector8_github_storage.json) selects the metadata for processing signature `5d2f2d49883814ea4c8226b409c6139a125b95bd02cf3bbe29bcb5f83f80c865`. Counts below were checked against the published summary, table schemas, storage index, delivery report and saved notebook outputs. Cloud-delivery and publication outcomes are those recorded in that snapshot; this documentation review did not perform a new live Drive audit.

## Executive summary

Sector 8 turns the checked image references, annotations, bounding boxes, temporal windows and quality measurements from S7 into reproducible visual inputs. It preserves original images by reference and creates only new JPEG derivatives: region crops, bounding-box overlays, paired evidence panels, temporal collages, observed-anchor progression panels, quality panels, reference-comparison panels and combined panels. Each product retains its source identities, source SHA-256 values, construction parameters and output hash.

The completed bundle contains **118,201 derived JPEGs across eight product types**, alongside **66,127 original-image references**. Together these form **184,328 visual-manifest records**, connected to their sources through **478,701 provenance rows**. S8 prepares **216,808 benchmark input recipes** for **47,239 annotation-based subjects**. Here, a subject identifies an annotation and its source image; it is not a patient count. Recipes can reuse the same original or derived image, so their count is not the number of JPEGs generated.

**The bundle is preparation-complete, not training-ready.** All 216,808 recipes have available pixels in the completed preparation snapshot, but final split assignment remains pending and validated question–answer pairs have not been authored. The source annotations are preserved as evidence for later tasks, not converted into automatic answers to arbitrary questions. The current summary explicitly records `training_ready=false`.

Delivery packages the new images into **eight uncompressed TAR archives, one per product folder**, and retains their image-level hashes and membership in metadata. The saved execution confirms **30 of 30 Drive objects: eight archives and 22 metadata files, with zero pending**. After a fresh Drive metadata fetch and checksum verification, GitHub receives **23 logical metadata files**, represented by **41 physical bundle files** because eight large logical files are stored in parts. The additional logical file is the Drive-verification report generated for GitHub. Image binaries and TAR payloads remain in Drive.

## Workflow

| Stage | What the notebook does |
| --- | --- |
| Configuration and source authority | Loads S7 from the pinned GitHub commit `cb4ac8d8f94caeb83dcddfccf495cdd4188bf9f0`. Follows its catalog and required configurations, lineage, summary, schemas and QC-enriched tables; checks the consumed hashes and records the exact upstream snapshot. |
| Source inventory and provenance | Builds a common source-evidence inventory from the labelled and temporal manifests. Retains portable image paths, byte hashes, physical video/frame identities, annotation and segment links, S7 QC measurements and inherited split provenance. Original images become reference views rather than newly copied S8 products. |
| Deterministic product planning | Selects eligible crops and panels from available verified sources. Records every product's ordered source references, parameters, annotation-assistance status and recipe identity. Shared products are reused across subjects. Unavailable requests receive explicit reasons instead of fabricated imagery or silently removed source rows. |
| SSD staging and integrity | Plans required paths in memory, verifies reusable SSD sources and stages only sources needed for pending products from mounted Drive into `/content`. Parallel staging checks bytes against the S7 SHA-256. Rendering reads the SSD copy; it does not repeatedly decode images through the Drive mount. |
| GPU decoding and visual construction | Checks NVIDIA JPEG decoding before dataset processing. Uses nvImageCodec and CuPy for supported device operations; the renderer also uses CPU-side composition, annotation drawing and JPEG encoding. GPU decoding does not imply that every rendering operation runs on the GPU. The captured processing configuration uses JPEG quality 95, a 20% crop margin and a panel image size of 336 pixels. |
| Parallel execution and recovery | Supports bounded source batches, concurrent batch scheduling and parallel product construction. Validates cached product hashes and construction identities before reuse. Missing or mismatched products remain eligible for regeneration; completed products survive an interrupted cell. Runtime and implementation fingerprints are calculated from the active implementation. |
| Benchmark input preparation | Builds ordered model-input recipes, a separate annotation-evidence table and source-sharing groups. A multi-image recipe can reference several existing images without creating another JPEG. Split assignment, question authoring and QA validation remain explicit downstream requirements. |
| Local persistence | Verifies completed products, writes seven Parquet tables with schema companions, ordered compressed JSONL recipes, configuration, summary and recovery metadata. The semantic catalog records the completed logical output set, byte counts and SHA-256 values. |
| Archive creation and cloud delivery | Builds deterministic TARs containing exactly the catalogued product images. Checks image bytes during packing and verifies the resulting archive hash. The saved export config uses four packing workers, eight upload workers and 32 MiB resumable upload chunks. Fresh Drive API inventories compare object size and server SHA-256; incomplete or mismatched deliveries remain recoverable. |
| GitHub publication | Requires a complete archive receipt bound to the current logical run. Rechecks all Drive objects, fetches the 22 metadata files and verifies their actual bytes, then repeats the Drive inventory. Creates exact-byte parts for large metadata and publishes changed objects in a non-forced commit. Reads committed blobs and verifies content hashes before declaring publication complete. |

The model-facing adapter returns `question`, `images` and `image_roles`. For example, `fullcrop` loads two actual images in order: the original frame and its crop. In the saved preview these were a 336 × 336 full frame and a 110 × 116 crop. `fullcrop_panel` instead supplies one JPEG containing the paired views. Likewise, `temporal_multi` supplies separate ordered frames, whereas `temporal_collage` supplies one assembled image. Source labels and answer evidence remain separate from the adapter's model-facing fields.

Bounding-box derivatives are marked `annotation_assisted`: a supplied crop or rectangle already indicates a region of interest. Comparisons against unassisted full-frame inputs must therefore preserve that distinction. S8 does not infer the boxes from image pixels; it uses the upstream annotation coordinates.

## Results visible in the saved notebook

The active [published summary](metadata/5d2f2d49883814ea4c8226b409c6139a125b95bd02cf3bbe29bcb5f83f80c865/reports/summary.json), matching schema files and [archive-delivery report](reports/sector8_archive_delivery.json) record the following completed bundle. Section 9 shows a resumed verification of existing products, section 10 demonstrates model-input loading, section 11 confirms archive delivery and section 12 records successful GitHub publication.

| Measure | Published result |
| --- | ---: |
| Source image files / original reference views | 66,127 |
| Annotation-based subjects | 47,239 |
| New derived JPEG products | 118,201 |
| Visual-manifest records | 184,328 |
| Visual-to-source provenance rows | 478,701 |
| Benchmark input recipes / recipes with pixels available | 216,808 / 216,808 |
| Benchmark input variants | 11 |
| Subjects with all six declared same-cohort variants available | 2,183 |
| Shared-source groups for future split assignment | 237 |
| Unavailable product requests recorded as skipped | 80,475 |
| Products reused / restored / newly generated in the saved Section 9 execution | 118,201 / 0 / 0 |
| Final split assignment / training readiness | Pending / false |
| Question–answer authoring status | Not authored |
| Image archives / archived image members | 8 / 118,201 |
| Drive objects verified / pending | 30 / 0 |
| Metadata files fetched and verified from Drive | 22 |
| Logical metadata files published to GitHub | 23 |
| Logical metadata files stored in parts | 8 |
| Physical files in the published GitHub data bundle | 41 |

The visual count reconciles as **66,127 original references + 118,201 derived JPEGs = 184,328 manifest records**. The benchmark count is the sum of the available `(subject, variant)` recipes:

| Input variant | What the model receives | Recipes |
| --- | --- | ---: |
| `full` | One original source frame. | 47,239 |
| `crop` | One bounding-box crop with the configured margin. | 4,127 |
| `fullcrop` | Two separate images: full frame, then crop. | 4,127 |
| `fullcrop_panel` | One paired full-frame-and-crop JPEG. | 4,127 |
| `overlay` | One full-frame JPEG with the available annotation rectangles drawn on it. | 4,127 |
| `temporal_multi` | Separate source frames in chronological order, retaining target and temporal-offset metadata. | 22,220 |
| `temporal_collage` | One collage of the selected temporal frames. | 22,220 |
| `progression` | One panel of observed verified anchors from the subject's finding segment. | 33,861 |
| `qc` | One panel showing the target and selected existing nearby frames. | 47,239 |
| `comparison` | One panel pairing a target with an explicitly annotated normal reference. | 3,357 |
| `combined` | One panel combining the available full-frame, local and temporal evidence. | 24,164 |
| **Total** | | **216,808** |

Recipe counts can exceed unique product counts. The 4,126 overlay files support 4,127 overlay recipes; the 47,238 quality panels support 47,239 recipes; and 8,842 progression panels support 33,861 recipes. This is reuse through the manifests, not missing output. The same-cohort count of 2,183 applies only to `full`, `crop`, `fullcrop`, `overlay`, `temporal_multi` and `temporal_collage`. It establishes input availability for those six variants, not validated task comparability or eligibility for all eleven variants.

The saved skipped summary records **43,112 annotations without a spatially usable bounding box**, **23,075 combined-panel requests without additional available evidence**, **13,378 segments with fewer than two verified anchors**, and **910 comparison requests without a verified normal reference in the same video**. These sum to 80,475 decisions across different product scopes. They do not mean that 80,475 original images were deleted or that the delivered archives are incomplete. For a one-anchor segment, only the multi-anchor progression panel is unavailable; its original evidence remains referenced.

The saved GPU preflight identifies an **NVIDIA L4**, Python 3.13.16, `cupy-cuda12x` 14.2.0, `nvidia-nvimgcodec-cu12` 0.9.0.20 and `nvidia-nvjpeg-cu12` 12.4.0.76. It confirms `nvidia_gpu` decoding using `HW_GPU_ONLY` and `GPU_ONLY` backends. However, the saved Section 9 run had no pending products and executed zero rendering batches: it verified and reused all 118,201 existing derivatives. It therefore does not measure fresh-generation throughput or demonstrate 50 simultaneous GPU batches.

The recorded processing configuration uses `batch_size=1000`; the implementation permits batches up to 5,000 source files. The saved call requests `render_workers=8` and `batch_workers=50`. These are execution settings, not evidence that this completed snapshot was generated with 5,000-file batches. Export and publication have their own configuration overlays. In particular, `github_publish=false` in the preserved processing configuration does not negate the separate publication cell that executed successfully.

The saved publication commit is [`30386545aa0ee96484d3ac2640897d5cb1d25938`](https://github.com/bogdanparvu18/msc-graduate-project/commit/30386545aa0ee96484d3ac2640897d5cb1d25938). Its output records 41 changed files, 18 newly uploaded Git objects, 23 verified existing objects reused, zero objects recovered after an ambiguous response in that invocation, and 21 blob POST attempts including retries. Reused objects were included in the commit; they were not omitted from publication.

## Outputs and scope

The local workspace root is `/content/mmvqa_sector8_v2`, with completed products under `runs/<processing_signature>`. The verified Drive destination is under `/content/drive/MyDrive/MMVQA_Clinical/outputs/phase2/sector8`, and selected metadata is published under `outputs/phase2/sector8` in GitHub. New image paths use `products/<kind>/<filename>.jpg`; each corresponding TAR retains those paths as member names. The eight folders describe representation types, not clinical classes.

| Product | New JPEGs | Purpose |
| --- | ---: | --- |
| `roi_crop` | 4,127 | Crops the annotated region with a 20% margin on each side, clipped to image boundaries. The crop retains its local pixel extent rather than being resized into a full-frame replacement. |
| `bbox_overlay` | 4,126 | Draws the usable annotation rectangles over a copy of the full image. The original JPEG and the original coordinate annotations remain unchanged. |
| `evidence_panel` | 4,127 | Places the full frame and its crop together in one image for the `fullcrop_panel` variant. |
| `temporal_collage` | 22,220 | Arranges up to five existing frames selected from the S7 temporal window by actual available offsets, in chronological order. |
| `progression_panel` | 8,842 | Shows up to five observed verified segment anchors, including endpoints when available. Requires at least two distinct verified frames. |
| `quality_panel` | 47,238 | Shows the target plus up to two distinct same-video neighbours within a 30-frame gap, prioritizing variation in available raw blur, contrast and brightness metrics. |
| `comparison_panel` | 3,357 | Pairs an eligible target with a distinct same-video reference explicitly annotated as normal, without contradictory source labels. A deterministic ordering avoids always placing the normal reference in one fixed position. |
| `combined_panel` | 24,164 | Combines the full image with available crop, overlay and temporal evidence; requires additional evidence beyond the original alone. |
| **Total** | **118,201** | |

Panel image areas use the configured 336-pixel size with aspect-ratio preservation and caption strips. Captions use neutral panel letters or frame indices, offsets and target markers rather than drawing clinical labels or QC scores into the pixels. A temporal panel does not verify clinical boundaries or continuous visibility. A progression panel depicts observed anchor appearance, not established disease progression. Quality panels use uncalibrated technical proxies, and a normal-reference comparison is not a clinically validated matched pair. Unlabelled context is never treated as normal merely because it lacks an annotation.

The table below lists **logical artifact paths**. Run metadata is exported beneath `metadata/<processing_signature>/`. Large logical files may exist in GitHub only as indexed parts; consumers must follow [`configs/sector8_github_storage.json`](configs/sector8_github_storage.json) instead of assuming that every logical filename exists directly in the repository.

| Artifact | Purpose |
| --- | --- |
| `metadata/<signature>/results/source_evidence_manifest.parquet` | The 66,127 source-file records, including physical identity, image hashes, annotation/segment membership, QC, preparation eligibility and pending split fields. |
| `metadata/<signature>/results/visual_manifest.parquet` | The common visual manifest: 184,328 original-reference and derived-product records, with kind, dimensions, materialization status, parameters, construction signatures, hashes and source references. |
| `metadata/<signature>/results/visual_source_manifest.parquet` | The 478,701 ordered product-to-source links, retaining source SHA-256, video/frame identity, annotation IDs, panel position, target role and temporal offsets. |
| `metadata/<signature>/results/skipped_products.parquet` | The 80,475 unavailable product requests, with reason, scope, affected subject/segment and available versus required anchor counts where applicable. |
| `metadata/<signature>/results/benchmark_input_manifest.parquet` | The 216,808 ordered model-input recipes, variant identity, pixel availability, annotation-assistance flag, same-cohort information, source-sharing group and preparation-only status. |
| `metadata/<signature>/results/benchmark_ground_truth_manifest.parquet` | Source annotation evidence for 47,239 subjects, kept separate from model inputs. Question and answer fields remain empty and `qa_validation_status="not_authored"`; the filename does not imply that a validated QA benchmark exists. |
| `metadata/<signature>/results/shared_source_groups.parquet` | The 237 connected groups built from physical-frame identities, shared source hashes and multi-image usage. Members and their variants must remain together during later final split assignment. |
| Matching `*_schema.json` files | Column names, recorded dtypes and row counts for all seven tables. These describe serialized tables; clinical validity and final split safety require separate checks. |
| `metadata/<signature>/benchmark/model_inputs.jsonl.gz` | Compressed ordered benchmark recipes for later loading. These are input specifications, not model predictions or authored answers. |
| `metadata/<signature>/configs/artifact_records.json.gz` | Recovery snapshot of visual records and product identities used to verify and resume preparation. |
| `metadata/<signature>/configs/sector8_config.json`, `metadata/<signature>/configs/upstream_snapshot.json` | Preserved processing configuration and consumed S7 commit, catalog identity and artifact SHA-256 values. |
| `metadata/<signature>/reports/summary.json` | Completed preparation counts, per-kind product totals, benchmark coverage, deferred status and limitations. |
| `configs/sector8_state.json` | Original semantic catalog describing the logical S8 output set and its hashes. Archive delivery preserves its bytes. |
| `configs/sector8_archives.json` | Archive manifest with each TAR's kind, SHA-256, size and exact image-member paths, byte sizes and SHA-256 values. |
| `configs/sector8_archive_state.json` | Transport catalog selecting archives and metadata while retaining the original logical artifact set and run identity. |
| `archives/<kind>.tar` in Drive | The eight image payload archives, containing exactly the 118,201 catalogued derivatives. TARs and JPEG bytes are not published to GitHub. |
| `reports/sector8_archive_delivery.json` in GitHub | Stable verification evidence: archive/member counts, the 30 confirmed cloud object identities by path/hash/size, zero pending and confirmation that metadata bytes were fetched from Drive before publication. |
| `configs/sector8_github_storage.json`, `metadata/_github_chunks/<sha256>.bin` in GitHub | Mapping from logical metadata to direct files or ordered exact-byte parts. Records original and part SHA-256 values, sizes and offsets, allowing lossless reconstruction of large metadata. |
| Local `checkpoints/` journals | Operational preparation, resumable-upload, metadata-fetch and publication state. These journals are outside the GitHub data bundle; tokens and private resumable session state are not published. |

Storage counts reconcile as **8 archives + 22 metadata files = 30 Drive objects**. GitHub contains those **22 logical metadata files + 1 generated verification report = 23 logical files**. Of these, 15 are stored directly and eight are represented by 25 parts; adding the storage index gives **15 + 25 + 1 = 41 physical data-bundle files**. This README is separate documentation and is not included in those publication counts. The GitHub-only report, index and parts are not additional missing Drive uploads.

Archive delivery compares local archive SHA-256 and byte counts with fresh Google Drive API checksum metadata, not merely the contents of the DriveFS mount. Missing or mismatched files remain pending; resumable sessions and later reruns retry incomplete delivery. Metadata publication additionally downloads the selected metadata bytes and hashes them before GitHub packaging. It does not download or extract the image TARs. GitHub parts preserve the exact original metadata bytes, including compressed files, and are verified against both their individual hashes and the logical mapping. Consumers should pin one commit and use its storage index and archive manifest together rather than mixing files from different snapshots.

S8 provides visual preparation, provenance, input recipes and verified delivery. It does not duplicate S7 originals into the archive bundle, extract new video frames, create clinical diagnoses, author validated QA pairs, infer disease progression, calibrate clinical quality thresholds, assign final train/validation/test membership, train a model or restore the image archives for a training session. A successful export establishes integrity and coverage for the selected catalog at verification time; it does not establish those downstream scientific requirements or prevent later remote replacement.
