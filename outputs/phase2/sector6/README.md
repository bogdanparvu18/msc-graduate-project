# Executive report: Phase 2, Sector 6 — Temporal context images and evidence metadata

**Source:** [`phase2_06_temporal_context_images.ipynb`](../../../notebooks/phase2/phase2_06_temporal_context_images.ipynb)  
**Review date:** 17 October 2026  
**Basis:** This revision builds on the Colab notebook saved in GitHub `main` at commit `6bc824c425bda001e3238f075f28e75785398144`. Its published summary records the previous extraction run. The new split-independent planning results below use the same S3–S5 Parquet bytes, checked against that commit's Git blob identities. Full NVDEC extraction and Google Drive persistence must be rerun in Colab for this revision.

## Executive summary

Sector 6 prepares temporal image material around the representative verified frame of each eligible S5 finding segment. It consumes the checked S1–S5 chain from a pinned GitHub `main` commit, copies required files to the local Colab SSD, and verifies their bytes against the upstream SHA-256 references committed in GitHub. It reuses official labelled images, extracts missing context frames from local videos through NVIDIA NVDEC, and audits all referenced images before persistence.

**Material creation is independent of the inherited train/validation/test assignments.** With `minimum_verified_segment_frames=1` and `temporal_context_split_policy="deferred"`, all 22,220 current segments, including 13,378 singletons, retain their verified targets. The revised plan contains **110,666 in-bounds temporal references** and **18,889 unique video-frame extraction requests**. No reference is suppressed because its current split conflicts with another reference. Out-of-bounds offsets remain reported and cannot be extracted.

Final splits and the model's eventual use of targets or temporal context are undecided. S6 preserves upstream assignments as provenance, exports `final_split` as missing with `split_assignment_status="pending"`, and reports `training_ready=false`. It imposes no video-level grouping. A future file/frame-level assignment can use physical frame identities and file hashes to resolve shared frames across the examples actually selected for the model.

Pure table transformations remain separate from I/O. The saved Colab settings are retained: 32 shared input-copy workers, 1,000 local SHA-256 workers, 10 concurrent NVDEC video workers, and a 100 GB decimal SSD budget. Final Drive copying uses 32 workers in batches of 1,000 files and verifies each new copy against its SSD source. Unlabelled context means no ground truth; it does not mean normal or continuous visibility of the target finding.

## Workflow

| Stage | What the notebook does |
| --- | --- |
| Configuration and source authority | Pins current GitHub `main`; follows explicit catalog entries for S1–S5 configurations, schemas, manifests and reports. Checks producer identities, consumed hashes, source references and S5 target annotation collections. No filename-timestamp selection is used. |
| Local staging and verification | Copies images, required videos and producer metadata through one shared 32-worker queue into `/content/mmvqa_sector6`. Local SHA-256 checks use the committed upstream references and 1,000 hashing workers. Only missing or mismatched inputs are recopied. Drive is read to copy source files, without a separate source-media hashing pass. |
| Temporal candidates | Uses the exact S5 representative and configured time offsets, audited FPS and verified decoder index offsets. Retains every in-bounds segment/offset reference, including singleton windows. Official images take priority and annotation identities remain intact. |
| Split provenance | Checks S4/S5 source consistency and target ownership. Preserves `upstream_split` and `context_frame_upstream_split`, records current split conflicts without filtering images, and supplies a deterministic `context_frame_id` for each `(video_key, opencv_frame_index)`. Final split assignment remains pending. |
| Extraction planning and NVDEC | Deduplicates all missing-image requests by physical video/frame identity, including frames referenced from multiple upstream splits. Decodes local SSD videos with up to 10 workers. Selected frames move to host memory for JPEG/PNG encoding; video decoding has no CPU fallback. |
| Complete physical audit | Audits every referenced image, including references previously suppressed by split isolation. Checks source hashes, readability and extracted-image provenance. Shared image paths are audited once. Reports include all missing or unreadable references; no split flag exempts a reference from validation. |
| Local result validation | Builds the seven result/report tables and verifies serialized values, schemas and hashes. The processing identity binds configuration, consumed inputs and active code. A changed split policy cannot reuse an old result bundle as if it were newly computed. Compatible image caches remain reusable after verification. |
| Drive persistence | Validates local images and sidecars with 1,000 workers in batches of 5,000. Writes current S6 outputs using 32 workers in batches of 1,000. Hashes SSD source bytes during copying and rereads the newly written Drive temporary file to compare SHA-256 before replacing the destination. Existing Drive file contents are not compared. Installs the catalog last. |
| GitHub publication | Rechecks consumed upstream blobs against current `main` and publishes selected metadata under `outputs/phase2/sector6`. Matching Git blobs are skipped. Image binaries and provenance sidecars stay on persistent storage. Unrelated repository files and this maintained README are preserved. |

## Results visible in the saved notebook

The base Colab revision and its [published summary](https://github.com/bogdanparvu18/msc-graduate-project/blob/6bc824c425bda001e3238f075f28e75785398144/outputs/phase2/sector6/reports/summary.json) record a complete run under the previous split-exclusion policy: 16,964 extraction results, 51,396 audited image files and zero failure references. Those results do not certify the newly restored references. Execution outputs are cleared in this PR so the previous run is not presented as an execution of the new code.

| Measure | Previous published run | Revised local plan |
| --- | ---: | ---: |
| Eligible finding segments / retained targets | 22,220 / 22,220 | 22,220 / 22,220 |
| Singleton / multi-frame segments | 13,378 / 8,842 | 13,378 / 8,842 |
| In-bounds temporal references | 110,666 | 110,666 |
| References suppressed by split policy | 43,760 | 0 |
| References to official images | 48,920 | 88,764 |
| References requiring video-derived images | 17,986 | 21,902 |
| Unique video-frame extraction requests | 16,964 | 18,889 |
| Distinct physical frames referenced | 61,323 | 61,323 |
| Final split assignment | Inherited S4/S5 eligibility | Pending |
| Full extraction and persistence | Complete for previous policy | Requires a new Colab run |

The restored 43,760 references comprise 39,844 references to existing official images and 3,916 references to 1,925 additional unique unlabelled frames. Thus **16,964 + 1,925 = 18,889** unique extraction requests. Multiple references can share one physical frame or image; these counts are not interchangeable with image-file counts. With five requested offsets per segment, 111,100 theoretical references become 110,666 in-bounds references; 434 boundary offsets remain unavailable.

Local verification covers cross-split official-image reuse, shared unlabelled-frame deduplication, preserved target identities, deferred split fields, all-reference failure reporting, and audit/persistence coverage. The real upstream planning check uses committed Parquet inputs. It does not measure hardware NVDEC throughput or Drive performance, and it does not regenerate published manifests/results in this PR.

## Outputs and scope

Runtime processing and serialization use the local SSD workspace. Persistent metadata is saved under `outputs/phase2/sector6` relative to the mounted storage root, and GitHub uses the same project-relative directory. Saved tables rebase access paths to persistent storage and retain portable relative references. Consumers must follow the active catalog and typed schemas.

| Artifact | Purpose |
| --- | --- |
| `results/temporal_manifest.parquet` | Every retained segment/offset reference, official or extracted image metadata, source annotation identities, physical frame identity, upstream split provenance, deferred final split fields, and image audit results. |
| `results/temporal_extraction_requests.parquet` | One deterministic extraction request for every unique video/frame identity lacking an official image, irrespective of upstream split. |
| `results/extracted_temporal_frames.parquet` | Extraction status, error, source-video SHA-256, output-image SHA-256 and decoder provenance for each request. |
| `reports/temporal_window_report.csv` | Requested, in-bounds and readable context counts, completeness flags and diagnostic upstream split-conflict counts per segment. Readability does not certify model eligibility. |
| `reports/temporal_extraction_failures.csv` | Every missing/unreadable context reference when extraction is enabled. |
| `reports/temporal_image_audit.csv` | Unique image paths with SHA-256, dimensions, readability and validation errors. |
| `reports/temporal_status_report.csv` | Aggregate annotation, image-source, upstream split-provenance and readability statuses. |
| Matching `*_schema.json` files | Typed readback, null restoration, column definitions, row counts and fingerprints for all seven tables. |
| `configs/sector6_config.json`, `configs/sector6_lineage.json`, `configs/sector6_state.json` | Effective settings, source identities, processing identity, selected artifacts and their SHA-256 values. The catalog is installed last. |
| `reports/summary.json`, `reports/executive_report.md` | Materialization outcome, all-reference validation counts, deferred split policy and `training_ready=false`. |
| Image files and adjacent `*.provenance.json` files | Derived images and cache-verification records on Drive; excluded from Git commits. |

The new manifest deliberately replaces the ambiguous `split` output column with `upstream_split`. `context_frame_upstream_split` records the known source assignment, while `context_split_provenance_status`, `context_requested_upstream_split_count` and `context_upstream_split_conflict` describe overlaps without suppressing evidence. `context_frame_id` is a canonical JSON pair `[video_key, opencv_frame_index]`. Repeated references retain their own segment and offset, while the frame identity permits later matching. `final_split` stays missing and `split_assignment_status` stays `pending`.

This is an explicit output-contract change. The existing S7 validator requires the old split-isolation fields and must be adapted in a subsequent change before consuming the revised bundle. This PR does not modify S7, allocate final splits, select a training regime, or declare the generated collection safe for direct model training. At final assignment, shared file/frame identities must be considered across whatever complete examples are chosen; the mechanism need not group entire videos.

A sidecar is a separate file beside an image. For example, `frame_00005582.jpg.provenance.json` accompanies `frame_00005582.jpg` and records the source-video hash, frame index, verified index offset, decoder identity, encoding settings and image hash. A local cached image is reused only when those identities and the verified bytes match the request. Rerunning can therefore reuse valid images while extracting the additional requests.

Final Drive persistence overwrites current selected outputs. Each worker checks the SSD hash against the expected local record, verifies the new Drive temporary file has the same SHA-256 and byte count, then installs that file at the final path. Progress advances only after successful verification and installation. A failed copy stops final catalog installation. Rerunning persistence rewrites selected destinations; it does not perform the old comparison against existing Drive content. Verification covers bytes accessible through the mounted filesystem and cannot prevent a later writer from replacing them.

Git publication remains selective by blob identity. Failed or planning-only runs retain local diagnostics and do not finalize a Drive/GitHub bundle. A successful materialization certifies image integrity and provenance; it does not certify final model splits. The existing SHA-256 authority, SSD capacity checks, source consistency checks and annotation semantics remain in force.

GPU decoding uses PyNvVideoCodec's high-level decoder API and DLPack exchange. JPEG/PNG encoding uses host memory; physical image auditing may use nvImageCodec or OpenCV. The 1,000 hash workers are a concurrency limit, not 1,000 physical CPU cores.

- [PyNvVideoCodec Decoder API](https://docs.nvidia.com/video-technologies/pynvvideocodec/pynvc-api-reference/decoder.html)
- [NVIDIA API programming guide and interoperability](https://docs.nvidia.com/video-technologies/pynvvideocodec/pynvc-api-prog-guide/using_pynvvideocodec_apis.html)
- [NVIDIA parallel decoding samples](https://docs.nvidia.com/video-technologies/pynvvideocodec/read-me/running-samples.html#decode-perf-py)

Singleton eligibility provides one verified anchor without upgrading its inherited supervision level. Neighbouring unlabelled frames receive no invented annotations. Nominal FPS-based offsets are not verified clinical onset/offset or exact variable-frame-rate timestamps; `verified_temporal_boundary_available` remains false. S6 does not track lesions, repair source annotations, generate crops/collages, train models or perform medical question answering.
