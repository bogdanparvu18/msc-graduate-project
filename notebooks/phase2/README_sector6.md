# Sector 6: local SSD temporal evidence workflow

Run `phase2_06_temporal_context_images.ipynb` from top to bottom in a Colab GPU runtime. The bootstrap pins the current GitHub `main` commit before loading S1–S5. That commit is the authority for expected file identities; a newer Drive file is not automatically an accepted upstream input.

## Execution order

1. Fetch the current `main` snapshot and validate its S1–S5 catalogs, schemas, configurations, raw-reference certificates and consumed-input hashes. Use the catalog-selected files, never a filename timestamp or an older commit as fallback.
2. Prepare the required manifests/reports and raw files on the local SSD. Images and videos share one copy queue with **32 workers**. Only videos required by S6's verified segments need to be staged; the full video inventory remains available as metadata.
3. Hash the local copies using a CPU thread pool configured for **1,000 concurrent hash tasks**, with streaming reads. Compare both the exact requested file set and every SHA-256 against the pinned `main` references. A thread count is a concurrency limit, not a claim of 1,000 physical CPU cores or guaranteed speedup.
4. Recopy only missing or mismatched files, then verify their local replacements. Drive is read to copy a source file; there is no separate source-media hashing pass on Drive. Exhausted repair attempts stop processing and leave diagnostics locally.
5. Build temporal windows with `minimum_verified_segment_frames=1`, then extract requested frames from SSD videos through NVDEC. Audit image readability and content on the SSD.
6. After local validation, reconcile completed outputs to persistent storage and GitHub. Compare destination content hashes, skip identical files and replace only missing or different files. Write the artifact catalog last. GitHub receives metadata artifacts, while extracted image binaries and their provenance JSON files go to mounted storage.

All processing paths are rebased from committed storage-relative references. For example, `raw/videos/video.mp4` becomes `/content/mmvqa_sector6/raw/videos/video.mp4` during execution. Published absolute access paths are rebased to the mounted storage root only for the saved artifact copies.

## Declarative controls

| Control | Default | Purpose |
|---|---:|---|
| `BOOTSTRAP.branch` | `main` | Authoritative upstream branch |
| `BOOTSTRAP.local_workspace_root` | `/content/mmvqa_sector6` | Local SSD workspace |
| `BOOTSTRAP.local_storage_budget_bytes` | `100_000_000_000` | 100 GB decimal task storage budget |
| `BOOTSTRAP.local_output_reserve_bytes` | `10_000_000_000` | Initial headroom reserved for outputs and temporary work |
| `BOOTSTRAP.copy_workers` | `32` | Shared image/video copy workers |
| `BOOTSTRAP.hash_workers` | `1000` | Local SHA-256 worker limit |
| `BOOTSTRAP.staging_max_attempts` | `3` | Bounded selective repair attempts |
| `SECTOR6_SETTINGS.minimum_verified_segment_frames` | `1` | Include singleton verified anchors |
| `SECTOR6_SETTINGS.temporal_context_workers` | `10` | Concurrent NVDEC video decoders |

Capacity checks account for required inputs and local work/output space. S6 stops with a capacity error if the declared budget or actual free space is insufficient; it does not silently process files from Drive instead. The local Colab disk is temporary: resuming within the same runtime reuses verified staged files. After runtime deletion, inputs must be staged again.

## Annotation and split semantics

S5's `temporal_grounding_candidate` flag describes multi-frame supervision. It does not decide S6 eligibility. S6 accepts a segment with one verified frame and retains its `single_frame_anchor_only` supervision and `verified_temporal_boundary_available=False` status. An unlabelled neighbour means no ground-truth annotation is available; it does not mean a normal finding.

S4 remains authoritative for physical-frame splits. Verified targets retain their assigned split. Context crossing a labelled split boundary, or unlabelled context requested by multiple splits, is excluded from model evidence and retained only as diagnostic metadata.

The loader rejects stale or mixed upstream generations by comparing the consumed upstream hashes saved by each producing sector with the corresponding files in the pinned current `main` snapshot. Updating S2 alone is insufficient if S3–S5 still certify previous S2 bytes. There are no hardcoded dataset hashes. If `main` advances before publication, S6 verifies that the consumed upstream files still match; unrelated changes can be preserved, but changed upstream inputs require a fresh load.

## GPU decoding and provenance

Video decoding uses NVIDIA NVDEC through PyNvVideoCodec's high-level `SimpleDecoder` API, with an independent decoder and ordinal cursor for each video. Selected GPU frames are exchanged through DLPack and copied to host memory for JPEG/PNG encoding. Image auditing can use nvImageCodec for supported images and OpenCV CPU decoding for its fallback; this does not enable CPU video decoding.

A **sidecar** is a separate metadata file stored next to an image. For example, an extracted `frame_00000120.jpg` has a neighbouring `frame_00000120.jpg.provenance.json` file recording the source-video SHA-256, frame index, verified index offset, decoder identity, encoding settings and output-image SHA-256. A readable existing image alone is not sufficient evidence for cache reuse.

- [PyNvVideoCodec Decoder API](https://docs.nvidia.com/video-technologies/pynvvideocodec/pynvc-api-reference/decoder.html)
- [NVIDIA API programming guide and interoperability](https://docs.nvidia.com/video-technologies/pynvvideocodec/pynvc-api-prog-guide/using_pynvvideocodec_apis.html)
- [NVIDIA parallel decoding samples](https://docs.nvidia.com/video-technologies/pynvvideocodec/read-me/running-samples.html#decode-perf-py)

## Persistence and retry

GitHub metadata belongs under `outputs/phase2/sector6`: Parquet manifests/results, CSV/JSON reports, schemas, configuration, lineage and the catalog. Image/video binaries are excluded. Destination reconciliation uses the freshly calculated local result hashes, so existing different bytes are replaced with this run's verified outputs rather than selected by modification time.

An interrupted final copy or Git publication can be retried. Matching destination files remain untouched; only missing or mismatched artifacts are installed. Publication uses an isolated temporary Git index on SSD, preserves unrelated repository files and performs ordinary pushes without force. Existing remote blobs are compared with the locally generated Git blob identities; matching blobs are neither downloaded nor materialized. The SHA-256 artifact catalog certifies the output bundle. Source credentials are referenced by secret name and are never written into saved artifacts.

The saved lineage retains the commit that certified the bundle. If main advances without changing any consumed upstream bytes, an unchanged bundle keeps that provenance and its existing hashes. The local execution receipt records the newly fetched commit and this run's copy/reuse counters without creating another Git commit solely for operational statistics.
