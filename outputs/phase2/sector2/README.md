# Phase 2 — Sector 2: Video Inventory, Decode Auditing, and CUDA Frame Alignment

## Executive summary

Sector 2 establishes the technical connection between the physical Kvasir-Capsule videos, the cleaned annotation metadata produced in Sector 1, and the individual video frames that later temporal processing will use. It turns that connection into persistent, inspectable artifacts rather than relying on notebook variables or unverified filename assumptions. The implementation covers physical inventory, a cached video manifest, complete video decoding and frame counting, image-to-video-frame alignment, conservative eligibility decisions, and a controlled GitHub publication step. **It is a data-integrity and preparation stage, not a trained VQA model or a validation of medical diagnoses.** ([N1](#source-n1), [G1](#source-g1))

The processing is intentionally separated into three measurements. The **manifest** records container properties and whether the first frame can be read. The **decode audit** counts frames actually returned by a decoder over a complete video. The **alignment audit** compares each eligible annotated image with candidate frames from its own source video. These measurements answer different questions and are not interchangeable. A readable first frame does not establish complete readability; a complete frame count does not establish image-to-frame correspondence; high visual similarity does not establish clinical correctness. ([N1](#source-n1), cells 22–28)

The reviewed notebook contains evidence of **117 physical videos**, divided into **43 videos with annotations** and **74 without annotations in the loaded metadata**. All 43 annotated video identifiers were matched to physical filenames. Its saved decode-audit output reports **4,765,114 successfully decoded frames**, with all 117 records attributed to `nvdec`, all 117 marked fully processed, and zero records flagged for review by the implemented decode rules. The same output demonstrates reuse of 117 persistent checkpoints with no new decoding in that particular run. ([N1](#source-n1), outputs of cells 20 and 25)

An important discrepancy remains visible: the configured published reference is **4,741,504 frames**, while the observed decoded total is **23,610 frames higher**. The entire difference belongs to the 43 partially labelled videos: **1,979,285 observed versus 1,955,675 referenced**. The 74 fully unlabelled videos total **2,785,829** in both the observed audit and its configured reference. The workflow preserves these values separately. It does not replace the reference, silently tolerate the discrepancy as a success, or force an observed total to match a publication. ([N1](#source-n1), outputs of cells 25–26)

Alignment evolved from sampled OpenCV comparisons to an exhaustive, full-resolution, colour comparison pipeline. The latest supplied alignment replacement uses **TorchCodec CUDA/NVDEC for video frames**, **nvImageCodec with only the nvJPEG `HW_GPU_ONLY` backend for the original JPEG references**, and **batched PyTorch SSIM on CUDA**. Independent annotated-video tasks are submitted concurrently; each active video task has its own decoder and CUDA work stream, and reference-image workers have their own hardware JPEG decoder contexts. This removes OpenCV pixel decoding from **alignment**, not automatically from the separate manifest probe. Filesystem access, codestream/header parsing, container indexing, coordination, and audit decisions still use CPU work. ([A2](#source-a2); the earlier integrated version is [N1](#source-n1), cell 28)

Each annotated mapping is evaluated against a neighbourhood of candidate indices. With permitted offsets `[-1, 0, 1]`, immediate neighbouring controls extend the evaluated set to `[-2, -1, 0, 1, 2]`. The workload of **236,190 candidate tests** corresponds to **47,238 eligible unique mappings multiplied by five offsets**, not 236,190 annotated images. The latest working SSIM threshold is **`0.92`**, with a **`0.005` best-versus-runner-up margin**; these are project acceptance settings, not validated medical probabilities. The latest decision-rule edit retains **all planned candidate rows**, but requires finite scores only for candidates inside the video's valid index interval, with at least two scored alternatives in total. It retains permitted-offset, video-consistency, and related-annotation safeguards. The original `df_clean` remains intact, and accepted annotation rows are saved separately in `frame_alignment_confirmed.parquet`. ([A2](#source-a2), [P1](#source-p1), [H2](#source-h2))

Persistence is a central design decision. Manifest reuse checks input metadata fingerprints and the saved CSV hash. Decode and alignment work are checkpointed per video. Run timestamps are not used to invalidate otherwise reusable expensive computations. Exported reports carry checksums, and the final GitHub cell publishes selected metadata and audit artifacts—not the raw videos, raw images, or disposable runtime caches. It fetches the configured branch, allows only fast-forward integration, stages an explicit file list, and uses a normal commit and push. ([N1](#source-n1), cells 23, 25, 28; [G1](#source-g1))

**Evidence boundary and revision:** this documentation update, dated **2026-10-01**, combines the original notebook snapshots, the subsequently supplied strict nvJPEG alignment replacement, the explicit boundary-policy edit, and the later user-supplied results. Original notebook cell numbers describe their frozen snapshots, not an automatically updated notebook. Later runtime observations are labelled as such. An earlier v3 GitHub bundle was inspected during development; that does not establish that the latest nvJPEG or boundary-aware results have been published. No new dataset execution, raw-storage inspection, or live-repository verification was performed for this documentation-only revision. ([N1](#source-n1), [N2](#source-n2), [A2](#source-a2), [P1](#source-p1), [H2](#source-h2))

---

## Contents

1. [Scope and evidence](#1-scope-and-evidence)
2. [Position in the project](#2-position-in-the-project)
3. [End-to-end processing flow](#3-end-to-end-processing-flow)
4. [Terminology and units of observation](#4-terminology-and-units-of-observation)
5. [Environment and dependency ordering](#5-environment-and-dependency-ordering)
6. [Configuration, run identity, and storage](#6-configuration-run-identity-and-storage)
7. [Reusing Sector 1 outputs](#7-reusing-sector-1-outputs)
8. [Physical video inventory and identifier resolution](#8-physical-video-inventory-and-identifier-resolution)
9. [Video manifest and lightweight probing](#9-video-manifest-and-lightweight-probing)
10. [Complete decode audit](#10-complete-decode-audit)
11. [Manifest validation and the reference discrepancy](#11-manifest-validation-and-the-reference-discrepancy)
12. [Frame-alignment input preparation and exact retrieval](#12-frame-alignment-input-preparation-and-exact-retrieval)
13. [SSIM definition and acceptance decisions](#13-ssim-definition-and-acceptance-decisions)
14. [Parallel execution and resource management](#14-parallel-execution-and-resource-management)
15. [Persistence, hashes, and restart behaviour](#15-persistence-hashes-and-restart-behaviour)
16. [Artifact catalogue and data contracts](#16-artifact-catalogue-and-data-contracts)
17. [Observed results and what they establish](#17-observed-results-and-what-they-establish)
18. [Technical decision record](#18-technical-decision-record)
19. [Operating the notebook](#19-operating-the-notebook)
20. [Using the outputs in later phases](#20-using-the-outputs-in-later-phases)
21. [GitHub publication](#21-github-publication)
22. [Troubleshooting and recovery](#22-troubleshooting-and-recovery)
23. [Known limitations and remaining integration work](#23-known-limitations-and-remaining-integration-work)
24. [Maintaining this README during export](#24-maintaining-this-readme-during-export)
25. [Function reference](#25-function-reference)
26. [Source map and external references](#26-source-map-and-external-references)

## 1. Scope and evidence

### 1.1 Primary source of truth

The original integrated implementation is `phase2_02_video_frame_validation-2.ipynb`, a 28-cell snapshot (**N1**). Its extended copy (**N2**) retains those cells and adds the final GitHub publication heading and implementation from `sector2_github_export_cell.py` (**G1**). They remain the evidence for inventory, manifest, full-video counting, storage, and publication mechanics.

For **current alignment**, the later supplied `frame_alignment_cuda_nvjpeg_strict_cell.py` (**A2**) supersedes N1 cell 28. The boundary-aware change to `decide_alignment()` was supplied later as an inline code edit (**P1**); the A2 file available for this review still contains the preceding all-candidates-scored condition. This README documents **A2 plus P1 as the intended latest alignment implementation**, not a claim that the mounted A2 file or original notebook was silently modified. Runtime results are drawn from the explicitly described later observations (**H2**).

Cell numbers count **every notebook cell, including Markdown**, starting at one. Function locations in sections 25.6–25.9 refer to the supplied **A2 file before P1**; adding the patch moves subsequent line numbers. Unchanged notebook function locations retain their original N1 references.

The relevant versions are: sampled OpenCV alignment; exhaustive sequential OpenCV alignment; CPU TorchCodec plus GPU SSIM; concurrent CUDA video plus OpenCV reference loading (v3); and concurrent CUDA video plus hardware nvJPEG reference decoding (v4), followed by the boundary-aware decision edit. These versions are not interchangeable, and old results retain their original method and threshold provenance.

### 1.2 Meaning of status statements

| Status used here | Meaning |
|---|---|
| Implemented | The relevant code is present in the reviewed snapshot. |
| Observed | A supplied notebook output or explicitly identified development screenshot shows the result. |
| Configured | The value appears in configuration; it is not necessarily a measurement. |
| Recommended follow-up | An identified improvement or evaluation, not a feature claimed as already implemented. |

Embedded outputs and subsequent screenshots are useful provenance, but are not an independent re-execution. No raw Drive data was accessed for this README revision. The later reports establish the displayed results only for the runs they describe; they do not certify the latest source edit. A saved execution count, prior published state, or progress bar cannot substitute for the final JSON/CSV/Parquet generated after the boundary-policy rerun.

### 1.3 Boundaries of this sector

Sector 2 does not train a model, create a knowledge graph, execute a neuro-symbolic reasoning engine, produce clinical recommendations, or demonstrate generalization to another dataset. It prepares evidence-linked data that such later components can consume. It also does not assign new train/validation/test splits: the alignment publisher explicitly preserves any existing split column and performs no new split assignment. [A2, `_alignment_publish`]

The directories reserved for `temporal_frames` and `splits` do not demonstrate that temporal extraction or split generation has occurred. Directory creation is infrastructure preparation, not execution of those later stages.

### 1.4 What changed in this revision

| Area | Previous README | This revision |
|---|---|---|
| Reference-image decoding | OpenCV on CPU | A2 hardware-only nvJPEG reference path; OpenCV remains a separate manifest-probe concern. |
| Working SSIM threshold | `0.99` described as current | `0.92` as the latest working policy, with `0.99` and `0.96` retained as historical settings. |
| Candidate completeness | Every planned offset required a score | Every planned offset still requires an audit row; only in-range candidates require a score. |
| Boundary handling | Nonexistent neighbours prevented acceptance | P1 exempts only `negative_target`/`target_out_of_range` from score requirements; at least two in-range scored candidates remain necessary. |
| Results | Final alignment outcome unavailable | Historical zero acceptance, 3,782 acceptance at `0.96`, and the latest reported 47,239 result are distinguished. Post-P1 result remains unreported. |
| Cache identity | v3 alignment directory | v4 nvJPEG directory, with v3 retained as historical; acceptance changes do not masquerade as scoring changes. |
| Execution controls | 8/8/8 batches and image workers, 4 copies | Latest supplied working configuration is 16/16/16 and 6 copies; not a measured optimal setting. |
| Runtime launch | No define-only explanation | `frame_alignment_define_only` and the separate smoke test are documented. |
| Temporal use | Broad downstream description | Explicit anchor/context separation, no repeated offset, no inherited neighbour labels, and no alignment threshold applied to context frames. |

The sections below integrate these changes; this table is not a substitute for updating the code, rerunning decisions, or publishing the resulting artifacts.

## 2. Position in the project

### 2.1 Relationship to Sector 1

Sector 1 handles dataset provisioning, metadata normalization, physical-image inventory, duplicate auditing, and creation of `metadata_clean.parquet`. Sector 2 consumes those outputs instead of rebuilding the entire ingestion pipeline. It additionally inventories the videos, verifies decoding outcomes, and establishes which frame indices can be used for image-linked temporal work. ([I1](#source-i1); [N1](#source-n1), cells 18–28)

The distinction is deliberate. A saved Parquet file preserves the data columns and their values; it does not preserve a Python function such as `normalize_id`, nor a live list such as `video_files`. Sector 2 therefore reloads the persistent data and reconstructs runtime objects where needed.

### 2.2 Relationship to image-only and temporal VQA

For an experiment that reads only the supplied annotated still images, proving each image's position in a video is not inherently required. The alignment audit was chosen because this project intends to use video context and neighbouring frames. A wrong index would otherwise attach a valid annotation to the wrong temporal evidence.

The audit checks the pipeline's interpretation as well as the dataset's declared relationships: identifier joins, path resolution, zero-based indexing, offset handling, decoding, and image comparison. That is a defensible technical quality-control objective. It is not a claim that the dataset authors' clinical annotations are unreliable.

### 2.3 Relationship to later reasoning

SSIM is used here to compare **two images**: a supplied annotated image and a retrieved video candidate. It is not a confidence score for a diagnosis or an automatic prerequisite for reasoning about every new frame. In particular, SSIM `0.92` does not mean 92% of pixels match, 92% classification accuracy, or a 92% probability of bleeding. Accepted fraction, alignment precision, classifier scores, and model performance are different quantities.

A later unannotated video does not automatically have a reference image against which to calculate this alignment SSIM. Such a video can still be decoded and analysed. Any later use of SSIM between neighbouring frames for redundancy detection would be a different operation with a different purpose and validation policy.

## 3. End-to-end processing flow

```text
Sector 1 persistent artifacts
  metadata_clean.parquet + physical_image_inventory.csv
                       |
                       v
Dependency setup --> CONFIG / run_id --> Drive mount --> DIRS
                       |
                       v
Physical video inventory and verified video-key matching
                       |
                       v
Cached video_manifest.csv
  paths + sizes + category + container/first-frame probe
                       |
             +---------+-------------------+
             |                             |
             v                             v
Complete decode audit                Annotated mappings only
  all physical videos                  image + video + frame_number
  PyNvVideoCodec/NVDEC                         |
  optional explicit PyAV fallback             v
  per-video checkpoints               Concurrent CUDA alignment
             |                          exact CUDA video candidates
             |                          HW-only nvJPEG references
             |                          full-resolution CUDA SSIM
             v                                 |
Observed counts + references                    v
  retain discrepancy                     acceptance rules
             |                                 |
             +----------------+----------------+
                              v
Persistent audits + confirmed Parquet + excluded annotations
                              |
                              v
Verified export bundle --> Git fetch / fast-forward --> commit --> push
```

The long operations are not coupled unnecessarily. A valid manifest does not have to be reprobed just because an alignment threshold changes. A new run timestamp does not require decoding unchanged videos. Publication does not rerun SSIM. This separation makes failures easier to isolate and results easier to reuse. ([N1](#source-n1); [G1](#source-g1))

## 4. Terminology and units of observation

| Term | Meaning in this implementation |
|---|---|
| Physical video | One file found by the configured extension filter and assigned a unique `video_key`. |
| Annotation row | One row of `df_clean`; multiple rows may refer to the same source frame. |
| Physical annotated image | One image file, potentially stored under a class-specific directory. |
| Alignment mapping | A unique valid `(video_key, frame_number, image_path)` unit, with explicit records also retained for invalid inputs. |
| Candidate pair | One annotated image paired with one candidate video frame at one evaluated offset. |
| Container-reported count | A property read through a video library, not a complete count of decoded outputs. |
| Decoded-frame count | Number of frame objects actually returned and accepted by the decode loop. |
| Published reference | A configured number taken from dataset documentation; not measured from the local files. |
| Confirmed alignment | Operational acceptance under the configured SSIM, ambiguity, and consistency rules. |
| Complete checkpoint | A committed terminal computation record; it does not necessarily mean the content passed every quality rule. |
| Cache fingerprint | SHA-256 over a defined description of inputs and computation. Its coverage depends on the stage. |

### 4.1 Why the counts differ in kind

The supplied Sector 1 output reports **47,239 cleaned annotation rows** and **47,238 physical image-inventory rows**. It also contains nine multi-class frame groups and one same-class multi-annotation group. These are different units, so equal counts should not be imposed merely because both are described informally as “frames.” ([I1](#source-i1), output of cell 25)

The alignment workload separately reports 47,238 eligible unique mappings in the development run. With five candidate offsets, that gives:

```text
47,238 mappings × 5 evaluated offsets = 236,190 candidate tests
```

The latter is neither the number of supplied images nor the number of distinct video frames decoded. Candidate indices can overlap between nearby annotations. A candidate can also finish with a recorded issue rather than a numeric SSIM. ([N1](#source-n1), cell 28; [H1](#source-h1))

### 4.2 Category names

`partially_labelled` means the physical video's key is present in the current cleaned annotation metadata. `fully_unlabelled` means it is absent from that metadata. The category names are processing labels; the code does not independently inspect whether every possible finding in a video has been annotated. They must not be interpreted as positive/negative clinical classes. ([N1](#source-n1), cells 23 and 25)

## 5. Environment and dependency ordering

### 5.1 Libraries have different responsibilities

| Component | Role in the reviewed implementation |
|---|---|
| Pandas / NumPy | Tables, validation, counts, eligibility masks, and summaries. |
| OpenCV | Lightweight manifest probing in the unchanged notebook. No annotated-reference pixel decoding in A2 alignment. |
| nvImageCodec / nvJPEG | Decode supported original JPEG reference images with the explicitly restricted `HW_GPU_ONLY` backend. |
| PyNvVideoCodec | Hardware decode of complete videos for the frame-count audit. |
| PyAV | Header inspection in the decode audit and optional software fallback. |
| TorchCodec | Exact-index candidate retrieval on CUDA for alignment. |
| PyTorch | CUDA tensors, work streams, and batched SSIM arithmetic. |
| scikit-image | Reference SSIM implementation used in the numerical startup check. |
| PyArrow | Parquet output and export-time Parquet validation. |
| `concurrent.futures` / `threading` | Video workers, image workers, completion collection, copy gates, cancellation, and memory reservations. |
| Git CLI | Repository synchronization and controlled artifact publication. |

NVIDIA **NVDEC** is the decoding mechanism used for accelerated video reading. **NVENC** refers to encoding; a library initialization log mentioning an NVENC SDK version is not evidence that this workflow re-encodes videos. The source files are never recoded by these audit cells. ([N1](#source-n1), cells 25 and 28)

### 5.2 Required notebook ordering

The reviewed notebook separates general imports from TorchCodec imports:

```text
General dependency installation
    -> general imports, including existing torch
    -> compatible CUDA TorchCodec and nvImageCodec/nvJPEG setup
    -> import torchcodec, VideoDecoder, set_cuda_backend, pyarrow
    -> from nvidia import nvimgcodec
    -> configuration and data setup
    -> processing
```

This ordering addresses the earlier failure where an already-imported native TorchCodec package was replaced on disk. The installer now checks for loaded `torchcodec` modules **before** making a replacement and asks for a fresh session if necessary. Deleting the raised exception would bypass that safety check without fixing the in-memory/native-library mismatch. ([N1](#source-n1), cells 4–6)

### 5.3 Recorded environment, not a universal compatibility promise

The saved outputs show the following combination:

| Component | Saved evidence |
|---|---|
| PyTorch | `2.11.0+cu128` |
| PyTorch CUDA build | `12.8` |
| TorchCodec before setup | `0.11.0+cu128` |
| TorchCodec selected and imported | `0.11.1+cu128` |
| PyArrow | `23.0.1` |
| CUDA available | `True` |
| PyNvVideoCodec install pin | `2.2.3` |
| PyAV install pin | `18.1.0` |

The last two entries are package pins in the installation cell, not version values independently recovered from every audit record. `select_cuda_wheel()` contains an explicit compatibility selection table; it is not a live compatibility discovery service. The README records the notebook's environment without declaring these versions universally appropriate for a different future runtime. ([N1](#source-n1), cells 3, 5–6)

The installer compares the full TorchCodec version, including the CUDA suffix. A CPU wheel and a CUDA wheel with similar public version numbers are not treated as interchangeable. The code keeps the installed PyTorch rather than changing the entire environment merely to install a decoder.

The later A2 replacement adds nvImageCodec/nvJPEG. Its supplied setup instructions use the following for the recorded CUDA 12.x environment, **before importing the native image package**:

```python
%pip install "nvidia-nvimgcodec-cu12[nvjpeg]"
```

This line comes from the supplied replacement instructions, not a fresh compatibility test in this README revision. The actual installed image-library versions and native binary hashes are recorded by A2; they were not present in the original N1 environment table and must not be invented here. Retain the already working CUDA TorchCodec installation. Replacing an already imported native package still requires a fresh Python session. ([A2](#source-a2), [A3](#source-a3))

### 5.4 Import success versus hardware success

Importing a CUDA-capable package only shows that the import worked. Alignment separately retrieves a warm-up frame and verifies tensor dtype, shape, device, and `cpu_fallback` status. Unknown or detected fallback is rejected. Full-file counting has its own CUDA-driver preflight, including `libcuda` and `libnvcuvid`. ([N1](#source-n1), cells 25 and 28)

The APIs and backend names are version-sensitive. For example, the earlier recorded alignment run uses TorchCodec's `beta` backend. External documentation is useful for understanding the API, but the selected installed version and the actual runtime checks govern this notebook; backend names from newer documentation must not be substituted silently. ([R2](#source-r2), [R3](#source-r3))

### 5.5 Hardware-only JPEG references are a separate contract

A2 registers only `nvimgcodec.BackendKind.HW_GPU_ONLY` for references. It does not register CPU or hybrid backends. It verifies a device-resident image buffer and a CUDA `uint8` tensor, converts HWC to CHW, and hands the tensor to the video worker through an explicitly completed image-worker stream. A GPU output alone would not establish how it was decoded; backend restriction is an additional requirement. ([A2](#source-a2), `_alignment_reference_context`, `_alignment_reference`)

The format guard accepts `.jpg` and `.jpeg` and refuses other extensions; header parsing and the backend then determine actual JPEG support. An unsupported format/bitstream or missing hardware support stops the run rather than reclassifying the reference as a bad annotation. The file is not converted to another format, nor decoded on CPU and uploaded as though it were hardware-decoded.

This GPU-only claim is deliberately limited to A2's **pixel-decoding paths**. Host file I/O, header parsing, indexing and table operations remain. N1's manifest `probe_video()` still performs one OpenCV frame read when rebuilding a manifest, and the count audit still permits its separately configured CPU fallback. Those are unchanged stages, not hidden changes to A2.

## 6. Configuration, run identity, and storage

### 6.1 Declarative configuration with procedural execution

The design is **data-oriented and declaratively configured**, not purely declarative. `CONFIG` describes choices, schemas describe records, and functions operate on explicit inputs and tables. Reading video packets, synchronizing CUDA work, copying files, and committing Git changes necessarily use procedural control flow.

The intended arrangement is one initial configuration cell, followed by processing cells that read it. The CUDA alignment settings use `.get()` fallbacks without mutating `CONFIG`; the decode settings are mapped directly from `CONFIG`. The effective alignment settings are additionally saved in the final state so resolved values such as worker count and backend remain visible. ([N1](#source-n1), cells 10, 25, 28)

### 6.2 Run identity and configuration snapshots

The sector-specific literal dictionary must be created before adding the timestamp:

```python
CONFIG = {
    # Dataset, storage, validation, decode, and alignment parameters.
}

CONFIG.setdefault("timezone", "America/Toronto")
CONFIG["run_id"] = datetime.now(
    ZoneInfo(CONFIG["timezone"])
).strftime("%Y%m%d_%H%M%S")
```

Setting `run_id` and then replacing `CONFIG` loses the key. That caused the earlier `KeyError: 'run_id'`. A fresh run gets a new identifier; rerunning only the save step uses the current identifier.

The snapshot is written as:

```text
outputs/phase2/configs/
  phase2_02_video_frame_validation_<run_id>_config.json
```

The chosen policy is one small JSON per run, not a hash-deduplication system for configuration files. By contrast, expensive computation caches deliberately do not use run timestamps as evidence that source pixels changed. `allow_nan=False` prevents non-standard NaN values from entering the configuration JSON. ([N1](#source-n1), cells 10 and 14)

The recorded Sector 2 run is `20260925_211052`. The initial Git bootstrap also creates a run identifier, but the later Sector 2 dictionary replaces `CONFIG` and assigns another one. The two happen to match in the supplied output; this does not guarantee they always will. The later `CONFIG['run_id']` is the value actually used by Sector 2 saving and export. See the remaining-integration notes in Section 23.

### 6.3 Storage layers

| Layer | Purpose | Survival after runtime loss |
|---|---|---|
| Mounted Drive under `/content/drive/MyDrive/MMVQA_Clinical` | Raw dataset, curated metadata, manifests, checkpoints, configurations, reports. | Files already committed to Drive persist. |
| Local Colab directories under `/content/..._work` | Temporary compressed-video copies used for processing. | Disposable; not persistent project data. |
| RAM and VRAM | DataFrames, decoded candidate tensors, reference tensors, SSIM intermediates. | Lost with the runtime. |
| GitHub artifact bundle | Versioned selected outputs, configuration and provenance; no raw pixels. | Persists only after a successful push. |

A mounted Drive path looks like a local filesystem path but is not the same as a separate local copy. This motivated temporary staging for repeated video access. ([N1](#source-n1), cells 12, 25, 28; [R4](#source-r4))

### 6.4 Directory layout

The current `prepare_phase2_dirs()` resolves main relative paths against `storage_root` and creates writable working directories. Dataset subdirectories are returned as references rather than manufactured as if the dataset existed.

```text
MMVQA_Clinical/
├── data/
│   ├── raw/kvasir_capsule/
│   │   └── osfstorage/
│   │       ├── metadata.csv
│   │       ├── labelled_images/
│   │       ├── labelled_videos/
│   │       └── unlabelled_videos/
│   ├── interim/phase2/
│   │   └── temporal_frames/                 # reserved, not an extraction result
│   └── curated/phase2/
│       ├── dataset_audit/                   # Sector 1 outputs
│       ├── manifests/                       # Sector 2 inventory and manifest
│       └── splits/                          # reserved, not split creation
└── outputs/phase2/
    ├── configs/
    ├── results/
    │   ├── decode_audit_cache_v2/
    │   ├── frame_alignment_cache_cuda_v3/          # older reference method
    │   └── frame_alignment_cache_cuda_nvjpeg_v4/    # current reference method
    └── reports/
```

A leading `/` in a supposedly relative configuration value changes its interpretation. `outputs/phase2/results` is relative; `/outputs/phase2/results` is absolute. Also, the current helper derives `DIRS['results_dir']` from `output_dir / 'results'`; it does **not** independently consume `CONFIG['results_dir']`. The saved configuration should not be assumed to override a path the helper does not read. ([N1](#source-n1), cell 14)

### 6.5 Current execution controls

The latest supplied working configuration is shown below. It must be distinguished from the original N1 snapshot and from the `.get()` fallback values in the supplied A2 file. **A2 still falls back to `min_ssim=0.99` and 8/8/8/4 when the corresponding keys are absent.** Documentation does not change those defaults: declare the desired settings explicitly in the initial `CONFIG` and inspect the completed state's `effective_settings`.

| Key or group | Latest working value / scope | Effect |
|---|---:|---|
| `seed` | `42` inherited | Declared reproducibility setting; see Section 23 about actual application. |
| `minimum_video_files` | `117` inherited | Minimum physical inventory size, not an equality test. |
| `expected_export_container_fps` | `30.0` inherited | Manifest reference, not a recovered clinical clock. |
| `frame_index_offset_candidates` | `[-1, 0, 1]` | Permitted alignment corrections; controls expand evaluation to five offsets. |
| `frame_alignment_min_ssim` | **`0.92`** | Latest working threshold; not a calibrated accuracy or clinical probability. |
| `frame_alignment_min_margin` | `0.005` | Minimum best-minus-second-best separation. |
| `frame_alignment_cuda_device` | `cuda:0` | GPU for video targets, reference pixels and SSIM. |
| `frame_alignment_cuda_backend` | `auto` | Resolve a supported installed TorchCodec backend; actual value goes into state. |
| `frame_alignment_reference_decoder` | `nvjpeg_hw_only` | A2's only permitted reference decoding mode. |
| `frame_alignment_reference_fancy_upsampling` | `True` | Explicit JPEG reconstruction option included in method identity. |
| `frame_alignment_reference_apply_exif_orientation` | `True` | Explicit orientation policy included in method identity. |
| `frame_alignment_define_only` | **`False` for normal runs** | `True` defines helpers only; it neither starts the full audit nor runs the smoke test automatically. |
| `frame_alignment_video_workers` | `all` | Submit all eligible annotated-video jobs; not a guarantee that all kernels run at once. |
| `frame_alignment_mapping_batch_size` | **`16`** | Up to 16 distinct mappings prepared per video's iteration, subject to the memory guard. |
| `frame_alignment_ssim_batch_size` | **`16`** | Up to 16 image/target pairs per SSIM call, per video worker. |
| `frame_alignment_image_workers` | **`16`** | Shared reference task pool, not 16 workers per video; each worker owns a JPEG decoder context. |
| `frame_alignment_copy_workers` | **`6`** | Concurrent compressed-video copies, not a cap on all staged files. |
| `frame_alignment_vram_budget_fraction` | `0.55` | Estimated temporary-tensor reservation budget. |
| `frame_alignment_local_disk_reserve_gib` | `2.0` | Disk headroom checked before copying. |
| `frame_alignment_hash_source_bytes` | `False` | File-metadata source signatures by default. |
| `frame_alignment_force_rescore` | `False` | Reuse compatible per-video score caches. |
| `decode_audit_backend` | `nvdec`, unchanged baseline | Full-file counting implementation. |
| `decode_audit_allow_cpu_fallback` | `True`, unchanged baseline | Explicit fallback in the count audit only; A2 has no such fallback. |
| `decode_audit_cpu_threads` | `2`, unchanged baseline | Software fallback threads, not concurrent videos. |
| `decode_audit_force_redecode` | `False` | Reuse compatible count checkpoints. |
| `decode_audit_retry_failed` | `False` | Retain terminal failed attempts unless deliberately retried. |
| `decode_audit_retry_video_keys` | `[]` | Selected-video recount control. |

The 16/16/16/6 values are the later configuration discussed and supplied by the user; **no controlled speedup or exact per-run provenance for these four settings is inferred from the screenshots**. The original startup captures use 8/8/8/4. Both versions are retained in the history rather than retroactively relabelling an earlier run. ([N1](#source-n1), [A2](#source-a2), [H2](#source-h2))

For clarity, 16 mappings with five offsets produce up to 80 candidate rows. If all are comparable, SSIM batches of 16 require five calls. Requested video indices are deduplicated before retrieval, so 80 candidate rows do not necessarily require 80 distinct decoded frames. Image workers are a shared producer pool, while copy slots are released immediately after each file transfer.

Both audits enable local staging in the supplied workflow. Inherited acquisition flags and the clinical taxonomy remain in `CONFIG`; their presence does not execute the Sector 1 provisioning/normalization pipeline again.

## 7. Reusing Sector 1 outputs

### 7.1 Load the authoritative cleaned metadata

```python
metadata_path = (
    DIRS["curated_data_dir"]
    / "dataset_audit"
    / "metadata_clean.parquet"
)

df_clean = pd.read_parquet(metadata_path)
```

The actual cell checks that the file exists first and previews `video_id`, `video_key`, and `frame_number`. The full Parquet is loaded, not merely a CSV preview. ([N1](#source-n1), cell 18)

`df_clean` remains the complete cleaned source. Alignment creates new columns and a new accepted subset; it does not overwrite `metadata_clean.parquet` or remove source rows from the original file.

### 7.2 Preserve metadata and clinical taxonomy rather than rebuild them

The earlier `normalize_id` helper was not automatically available in a fresh notebook. Instead of inventing a second normalization rule, the workflow reused `video_key` and checked it against physical filename stems. A zero missing-key result established that the direct stem rule was usable for this dataset snapshot. It was not assumed solely from example filenames. ([N1](#source-n1), cell 20)

The inherited `finding_class_normalized` and `clinical_group` fields remain available to downstream processing. SSIM is not used to relabel them. The taxonomy mapping in `CONFIG` is a carried-forward processing ontology, not something inferred by the alignment code.

### 7.3 Recover image paths without losing legitimate class-specific copies

`prepare_alignment_inputs()` prefers an existing `image_path`. Where a path is unavailable, it can load:

```text
data/curated/phase2/dataset_audit/physical_image_inventory.csv
```

The lookup uses the existing image identifier plus a normalized class key, rather than a filename-only dictionary that would reject legitimate class-folder duplicates. It checks that the inventory match is unique and its `class_match_count` equals one. [A2, `prepare_alignment_inputs`]

The physical image inventory is not a second annotation generator. Its purpose is to locate the image file supporting an already-existing annotation.

### 7.4 File formats in the working distribution

The user explicitly confirmed **`.jpg` annotated images and `.mp4` videos in both the labelled and unlabelled directories**. These are the working distribution's formats. Broader extension allowlists in the inventory, and differently worded external descriptions, do not establish that PNG or AVI files are used in this run. No PNG-to-JPEG conversion or AVI-to-MP4 recoding by this pipeline has been demonstrated. ([H2](#source-h2))

The two pixel inputs can represent the same source scene without being byte-identical after decoding. The cause of the observed approximately 0.953 SSIM distribution has not been established from the scores alone. The README does not invent an encoding/conversion history to explain it.

## 8. Physical video inventory and identifier resolution

### 8.1 Initial inspection

The inspection cell verifies the dataset root and raw metadata file, prints a bounded top-level directory preview, reads 30 metadata rows, and recursively finds configured video extensions. It does not decode videos. The raw CSV preview is for schema/layout inspection; the downstream metadata source is still `metadata_clean.parquet`. ([N1](#source-n1), cell 16)

The extension test is applied before `is_file()` in the list comprehension. This avoids file-status checks for paths whose suffix cannot qualify, although recursive enumeration of the directory tree still has a cost.

### 8.2 Identifier matching

The identifier is:

```python
video_key = video_path.stem
```

`stem` is a `Path` property, not a function call. The inventory rejects duplicate stems before expensive processing. Distinct directories do not make two identical video keys acceptable: a downstream join must still identify exactly one physical source video.

The matching cell computes the set of annotated keys not found in the physical index. In the saved output it reports 43 annotated video IDs and zero missing matches. It also writes `video_inventory.csv`, containing a filename-derived identifier, relative path, filename, and size. ([N1](#source-n1), cell 20)

### 8.3 Why the later inventory is repeated

The cached-manifest and decode cells construct fresh inventories instead of trusting an old `video_files` variable. This repeated enumeration is intentional: a stale Python list cannot reveal that someone added, removed, renamed, or replaced a file after the list was created.

It is much cheaper in principle than repeatedly decoding pixels, but it is not free—especially on a Drive mount with many image files. Inventory freshness and pixel recomputation are separate costs.

## 9. Video manifest and lightweight probing

### 9.1 One record per physical video

`build_video_record()` combines physical identity and technical probe results. `build_video_manifest()` assembles these records, sorts by key, and restores stable dtypes. The schema includes:

| Field | Meaning |
|---|---|
| `video_key` | Verified filename stem. |
| `video_filename` | Physical filename including extension. |
| `video_path` | Absolute path in the recorded environment. |
| `video_relpath` | Path relative to `CONFIG['storage_root']`. |
| `video_annotation_type` | Membership in the current annotated-key set. |
| `video_size_bytes` | Encoded file size. |
| `container_opened` | Whether OpenCV opened the source in the probe. |
| `first_frame_readable` | Whether the probe returned its first frame. |
| `container_fps` | Reported positive finite FPS. |
| `container_reported_frame_count` | Positive integral count obtained from the container interface. |
| `width`, `height` | Positive integral reported dimensions. |
| `estimated_container_duration_seconds` | Reported frame count divided by reported FPS. |

### 9.2 What `probe_video()` actually does

The function opens `cv2.VideoCapture`, reads `CAP_PROP_FPS`, `CAP_PROP_FRAME_COUNT`, width, and height, then calls `cap.read()` once. A `finally` block releases the capture object. Invalid numerical properties become missing values rather than invented measurements. If the container cannot be opened, `empty_video_probe()` returns the same field structure with false flags and missing numerical values. ([N1](#source-n1), cell 22)

The original generic names `frame_count` and `container_duration_seconds` were renamed to make their provenance explicit. The manifest does **not** claim to contain an independently measured number of all decodable frames or the original clinical capture duration.

### 9.3 Schema composition

The technical columns are defined once in `VIDEO_PROBE_COLUMNS` and expanded into `VIDEO_MANIFEST_COLUMNS`:

```python
VIDEO_MANIFEST_COLUMNS = [
    "video_key",
    "video_filename",
    "video_path",
    "video_relpath",
    "video_annotation_type",
    "video_size_bytes",
    *VIDEO_PROBE_COLUMNS,
]
```

This also prevents maintaining two inconsistent technical-column lists. An earlier missing comma between adjacent strings silently created a combined column name, causing `KeyError: 'container_opened'`. The current implementation checks schema uniqueness and verifies that the probe dictionary matches the expected probe columns. ([N1](#source-n1), cell 23; [H1](#source-h1))

### 9.4 Manifest cache policy

The input fingerprint includes the current inventory, category membership, paths, sizes, nanosecond modification times, schema/cache version, and OpenCV version/build information. The manifest CSV's actual bytes are independently hashed and compared with `csv_sha256` in `video_manifest_state.json`.

If these checks and the row identities agree, the saved CSV is reloaded and typed. **`build_video_manifest()`, `build_video_record()`, and `probe_video()` are not called.** If any check fails, the current implementation rebuilds the entire manifest, not only one row. The expensive full decode audit has a different, per-video policy. ([N1](#source-n1), cell 23)

## 10. Complete decode audit

### 10.1 Purpose and scope

The count audit processes all physical videos, both annotation categories. It counts the actual frame objects returned by the decoder and records whether processing reached its defined terminal state. It never uses the header frame count as the loop bound. This prevents the audit from merely reproducing the number it is supposed to check. ([N1](#source-n1), cell 25)

The currently reviewed count-audit driver iterates videos **one at a time**. Its GPU acceleration must not be confused with the separate alignment cell's all-video concurrency. GPU execution and concurrent-video scheduling are distinct design dimensions.

### 10.2 Header inspection is separate from decoding

`_decode_header_probe()` uses PyAV to inspect the container header without decoding pixels. With one unambiguous video stream it records `stream.frames` as a diagnostic and sets `header_source='pyav_container_header'`. Unknown or unsuitable header values remain missing.

Consequently there are even two library-level sources for container counts: the manifest's earlier OpenCV probe and this audit's PyAV header probe. Both are distinct from the number counted through NVDEC. The field name and `header_source` preserve that distinction.

### 10.3 NVDEC counting and end-of-stream handling

The reviewed implementation creates a PyNvVideoCodec demuxer and decoder with device memory enabled. It iterates compressed packets and counts every returned valid frame. It records packet counts, decoded dimensions, and frames returned during final draining. [N1, cell 25, `_decode_with_nvdec`]

The code targets the observed `PyNvVideoCodec==2.2.3` binding. In this implementation the demux iterator supplies a terminal packet with `bsl == 0`; that packet is passed to `Decode()` to drain delayed frames, followed by stream synchronization. The code specifically does **not** call the `Flush()` method that failed in the earlier runtime.

This behaviour is a pinned implementation assumption and is explicitly checked. An iterator that ends without the expected EOS packet causes `DecodeBackendAPIError`, not a silently accepted truncated count. Generic documentation examples are not a substitute for testing the actual installed API.

A packet is not equivalent to a frame. The loop counts decoded frame objects, not packet iterations; otherwise reordering, buffering, or multiple returned frames could corrupt the count.

### 10.4 Explicit software fallback

The count audit permits PyAV CPU fallback when configured and a per-video NVDEC attempt cannot complete usefully. The software attempt restarts the whole file. **A partial GPU count is never added to the CPU count.** The returned record identifies `pyav_cpu`, preserves the fallback reason, and can include GPU-attempt and CPU-attempt timings. [N1, cell 25, `decode_video_gpu`]

Missing GPU driver libraries, missing methods, incompatible call signatures, and similar environment/API failures are not treated as bad content. They stop the appropriate path rather than converting every source into an apparently valid CPU fallback.

Alignment uses a stricter policy: its CUDA implementation rejects CPU fallback altogether. That difference is intentional and must remain visible in documentation and provenance.

### 10.5 Status and review fields

| Field/status | Interpretation |
|---|---|
| `frames_read_before_stop` | Actual returned-frame count, retained for compatibility with earlier audit tables. |
| `read_completed` | The attempt completed its implemented read/drain sequence and returned frames. |
| `no_readable_frames` | Processing terminated but produced zero useful frames. |
| `open_failed` | Opening failed. |
| `decode_error` | Processing failed after opening or during decoding/draining/synchronization. |
| `decoding_completed` | Whether the implemented decoding path reached a complete terminal traversal. |
| `needs_review` | Derived flag for incomplete processing, zero frames, missing/different reported counts, or recorded errors. |

The earlier `eof_or_read_failure` label belonged to an OpenCV loop in which `read()==False` could not distinguish an ordinary end from a failed read. The current GPU implementation has more explicit terminal handling, but even its successful completion is not a proof that every conceivable codec anomaly would be detected rather than concealed.

### 10.6 Per-video checkpointing

A completed attempt—successful or explicitly failed—is written to:

```text
results/decode_audit_cache_v2/<sha256-of-video-key>.json
```

The JSON contains the source identity, decoder signature, result, input fingerprint, record hash, payload hash, and completion marker. These are independent of the global report's state. An interrupted run can therefore resume from committed individual videos even if the aggregate publication was never completed. ([N1](#source-n1), cell 25)

Failed attempts are reusable records by default, avoiding an endless re-decode loop on every notebook run. `retry_failed`, `retry_video_keys`, and `force_redecode` provide deliberate retry controls. A completed checkpoint means an attempt was recorded consistently; its quality fields still determine whether its result needs review.

### 10.7 Category and reference changes do not necessarily require decoding

Per-video count signatures exclude annotation category and run ID. If the annotation metadata changes only the labelled/unlabelled membership, the cached count can be reused and the category refreshed. The aggregate report is then rebuilt.

Likewise, a change to a published reference requires new comparisons in the summary, not necessarily another pass over identical video pixels. Source files and computation identity determine pixel reuse; category and reference values determine report interpretation. [N1, cell 25, `decode_source_signature`, `read_decode_checkpoint`, `run_gpu_decode_audit`]

## 11. Manifest validation and the reference discrepancy

### 11.1 What the validator checks

`validate_video_manifest()` checks the table's schema and non-emptiness, nonblank unique video IDs, nonblank existing unique absolute paths, relative-path validity/uniqueness, configured minimum video count, category values, counts derived from the annotated-key set, technical probe values, expected exported FPS, and exact correspondence between labelled metadata keys and labelled manifest keys. It returns the same DataFrame; it does not decode video again. ([N1](#source-n1), cell 26)

The FPS check uses `rtol=0` and `atol=1e-3`. This accommodates the saved maximum of `30.00003` while preserving the configured 30 FPS expectation.

The validator reads the manifest, not `decode_audit.csv`. Full decode-record validation is performed by the count-audit code. This is why the manifest validator does not need a GPU conversion merely because the counting stage uses NVDEC.

### 11.2 Why exact equality was changed to a warning

The original validator raised an error when the observed frame total differed from the reference. After full decoding confirmed the larger observed count, the workflow retained the discrepancy as an audit warning rather than blocking all subsequent work.

Changing the rule to `observed >= expected` would answer a different question: whether a minimum was reached. The published frame number is used here as a reference total, not as an established lower bound. A larger result is not automatically evidence that the dataset is correct.

The current warning records the reference, observed total, signed difference, and unresolved status. The code does not explain the cause, modify the dataset, or certify equivalence to the source publication. ([N1](#source-n1), cell 26)

### 11.3 Measured values must retain their provenance

```text
Published/configured reference      4,741,504
Container-reported manifest total   4,765,114
Completed decode-audit total        4,765,114
Difference from reference              23,610
```

Agreement between the two local observations strengthens the conclusion that the larger value is not merely an untested container property. It does not establish why the referenced dataset count differs. The group references used by the notebook are also stated in the dataset's labelled/unlabelled descriptions. ([N1](#source-n1), outputs of cells 25–26; [R1](#source-r1))

`expected_total_extractable_frames` is a legacy configuration name still present in the snapshot. In this workflow it is the published-reference value. Renaming a displayed total to `total_container_reported_frames` made the measured side clearer without changing the reference's meaning.

## 12. Frame-alignment input preparation and exact retrieval

### 12.1 Resolve the relationship before comparing pixels

`prepare_alignment_inputs()` takes cleaned metadata, the video manifest, an optional physical-image inventory, and the resolved directories. It requires `image_key`, `video_key`, and `frame_number`. It does not derive clinical labels from visual similarity.

For each row it verifies that the identifiers are present; the frame number is a finite nonnegative integer; the image identifier does not point to contradictory video/frame identities; the manifest resolves the video uniquely; the image path is supplied or uniquely recoverable from the image/class inventory; and the resolved files stay within the declared roots. ([A2](#source-a2))

Unusable rows do not disappear. They receive explicit input statuses such as `invalid_frame_number`, `missing_video`, `ambiguous_video_key`, `missing_image`, `ambiguous_image_path`, or an invalid-path status. They remain traceable in the final exclusion/audit outputs.

### 12.2 Mapping IDs and multi-annotation preservation

For valid inputs the logical work item is a unique `(video_key, frame_number, physical image path)`. Duplicate metadata rows for that same mapping need not trigger duplicate pixel calculations. Invalid cases also receive stable work records so that the original annotation count is not silently reduced.

The annotations store an `alignment_mapping_id` which joins them back to mapping-level decisions. The final merge is validated as many-to-one, and the code checks that every original annotation row has a decision.

Within each video's cache, mappings are sorted by frame number and image path and assigned local IDs. This avoids invalidating one video's work merely because an unrelated video changed the global order of metadata rows. The original IDs are restored when results return to the main process. [A2, `run_full_frame_alignment_cuda`, `_alignment_video_job`]

### 12.3 Candidate indices

The candidate formula is:

```python
target_frame_index = int(frame_number) + int(offset)
```

The code does not assume that the metadata and decoder start counting at the same point. It explicitly tests offsets. With permitted offsets `[-1, 0, 1]`, the evaluated set is the union of each permitted offset and its immediate neighbours:

```python
allowed = [-1, 0, 1]
evaluated = sorted({offset + neighbour
                    for offset in allowed
                    for neighbour in (-1, 0, 1)})
# [-2, -1, 0, 1, 2]
```

`-2` and `+2` are controls in this configuration, not automatically permitted final corrections. A strongest match outside the permitted set is reported rather than silently expanding the accepted offset range. ([A2](#source-a2))

For example, an annotation with `frame_number=1000` produces requests for indices 998–1002. These numbers are decoder frame indices, not seconds. The accepted index is retained as `aligned_frame_index`; the original metadata frame number is preserved.

### 12.4 Exact retrieval is not the same as “no scanning”

The active constructor is conceptually:

```python
with set_cuda_backend(selected_backend):
    decoder = VideoDecoder(
        str(local_video_path),
        device="cuda:0",
        seek_mode="exact",
        dimension_order="NCHW",
        num_ffmpeg_threads=1,
    )
```

The implementation requests sets of target indices with `get_frames_at(indices=...)`. It deduplicates and sorts indices within each batch. It no longer uses a Python loop that calls `cap.read()` from zero up to the last target for every video's alignment work. [A2, `_alignment_score_cuda`, `_alignment_targets_cuda`]

However, TorchCodec's exact mode scans the file when building its frame/keyframe index; that scan is not full pixel decoding. Retrieving compressed-video targets may additionally require dependency frames around a preceding keyframe. Therefore the correct claim is **exact indexed requests with accelerated decoding**, not constant-time arbitrary access or “the file is never parsed.” ([R2](#source-r2))

The current code does not persist TorchCodec's internal index as a separately reusable project artifact. If a video's score cache is invalidated, decoder construction may incur its indexing cost again.

### 12.5 Hardware verification and per-candidate failures

A warm-up frame is requested before scoring. `_alignment_check_gpu_decode()` requires a `uint8` NCHW tensor on the requested CUDA device and a known non-fallback status. A GPU infrastructure error or unknown decoding provenance stops the relevant execution rather than converting thousands of untested images into low-score exclusions.

When a native batch request fails in a recoverable way, `_alignment_targets_cuda()` divides the request recursively to isolate individual target failures. Those targets remain explicit candidate records. The code does not replace their scores with zero or compute offset means only from whichever targets happened to succeed. ([A2](#source-a2))

## 13. SSIM definition and acceptance decisions

### 13.1 What is compared

The current reference is the **original annotated JPEG** decoded by nvImageCodec using only nvJPEG's `HW_GPU_ONLY` backend. `_alignment_reference_context()` requests an SRGB colour specification and explicit orientation/upsampling settings. `_alignment_reference()` checks that the decoded buffer is on-device, imports it into PyTorch through DLPack, and returns a CUDA CHW `uint8` tensor. The video candidate is the RGB CUDA tensor returned by TorchCodec. This replaces the historical OpenCV `imread` → BGR-to-RGB → CPU tensor path. ([A2](#source-a2), `_alignment_reference_context`, `_alignment_reference`)

Both inputs must have identical dimensions and at least seven pixels in each spatial dimension. The comparison uses original resolution, without 128×128 resizing, grayscale conversion, geometric registration, arbitrary content cropping, or comparison of dimensions alone. Unsupported JPEG decoding is an infrastructure/capability error, not an automatic annotation exclusion.

Matching device and channel layout is necessary, but does not prove that different compressed representations reconstruct identical pixels. Changing from OpenCV reference loading to nvJPEG shifted the reported mean best SSIM only from **0.953065 to 0.953224**. It did not make the scores approach 1; the remaining cause has not been established. Do not attribute acceptance gains to the decoder when the threshold changed at the same time. ([H2](#source-h2), Section 17.3)

### 13.2 Metric used by the implementation

The implementation preserves a particular reference SSIM definition rather than silently choosing another library's defaults: ([A2](#source-a2); [R6](#source-r6))

```text
colour input, uint8, data_range = 255
uniform 7 × 7 local window
sample covariance correction = 49 / 48
valid spatial windows (the outer 3-pixel border is excluded)
spatial mean per colour channel, then equal mean across channels
```

For local image windows `x` and `y`, the principal calculation is:

```text
SSIM(x, y) =
  ((2 μx μy + C1) (2 σxy + C2))
  ---------------------------------
  ((μx² + μy² + C1) (σx² + σy² + C2))

C1 = (0.01 × 255)²
C2 = (0.03 × 255)²
```

The code computes integer-valued window sums using float32 tensors and converts the moments and final metric arithmetic to float64. It uses `torch.inference_mode()` and disables autocast for this calculation. It is not a learned loss, and no gradient graph is required. [A2, `_alignment_ssim_tensor_batch`]

This matters because a Gaussian-window SSIM implementation, a resized grayscale comparison, and an approximate low-precision implementation need not produce the same scores around strict thresholds. The optimization sought to accelerate the existing comparison definition, not change it unnoticed.

### 13.3 Numerical startup check

`validate_torch_ssim_backend()` constructs deterministic test pairs—including identical images, small noise, shifted images, contrasting intensities, and near-uniform cases—and compares the CUDA arithmetic with the installed scikit-image reference. It stops if the maximum absolute difference exceeds its configured numerical tolerance. The default tolerance is `1e-10`. ([A2](#source-a2))

This self-check validates selected arithmetic cases. It does **not** prove correct frame indexing, validate all real dataset inputs, establish clinical truth, benchmark performance, or guarantee that CPU and GPU video decoding return bit-identical pixels. Those are separate questions.

### 13.4 Scores are calculated in batches but decisions remain individual

`torch_ssim_uint8_pairs()` returns one score per pair in the original order. Different image sizes are grouped rather than resized. Each score remains attached to its mapping, offset, and requested target index.

The pipeline does not compare every frame in every video with every image. Concurrency means independent per-video tasks overlap; it does not change the pairing rule. Every reference image is tested only against candidates from its resolved source video.

### 13.5 Acceptance rules

`decide_alignment()` sorts successfully scored candidates, records the best offset/index/score, the second-best score and their margin, then applies the decision rules. **The latest P1 edit changes score completeness at video boundaries, not candidate-record completeness or the SSIM formula.**

| Rule | Latest policy and reason |
|---|---|
| All planned candidate rows exist and offsets are not duplicated. | Retain the full expected offset set even when some requests are out of range. Missing audit rows are still an error. |
| Every candidate inside the valid video interval has a finite score with status `scored`. | A real failed comparison cannot be disguised as an unavailable neighbour. |
| Out-of-range candidates stay in the audit without invented scores. | Only `negative_target` and `target_out_of_range` are exempt from score requirements. |
| At least two in-range candidates have been scored. | A best-versus-runner-up comparison still requires alternatives. This is a project rule, not proof of exact identity. |
| Best SSIM is at least **`0.92`**. | Latest working threshold; the previous `0.99`/`0.96` values remain historical experiments. |
| Best-minus-second-best is at least `0.005`. | Retain the ambiguity check on the available in-range alternatives. |
| Best offset belongs to the permitted set. | The extra controls do not silently become allowed corrections. |
| Accepted offsets within a video are consistent. | Keep the existing video-level consistency rule. |
| Related image/class copies remain jointly eligible. | Preserve the complete annotated label set for the source frame under the existing related-record policy. |

The boundary decision trusts the statuses produced by the scorer, which checks the target index against zero and the decoder's indexed frame count before retrieval. It does **not** turn `decode_error`, unreadability, dimension mismatch, or a missing candidate record into an excused boundary. With only one or no scored in-range alternatives, it still rejects the mapping.

The threshold `0.92` was adopted as an exploratory working policy after examining the reported score distribution; it is **not independently calibrated as a universal alignment threshold**. The minimum reported best score was 0.923936, but choosing a threshold below that observed minimum does not itself measure alignment precision. Report the policy, sample questionable pairs, and assess any effect on the class/video distribution. ([A2](#source-a2), [P1](#source-p1), [H2](#source-h2))

### 13.6 Why high SSIM alone is not enough

Consider illustrative scores for one annotated image:

| Candidate | Score |
|---|---:|
| Best | 0.998 |
| Second best | 0.997 |

The margin is `0.001`, so the current rule marks the result ambiguous despite excellent similarity. Nearly identical adjacent frames may make exact identity impossible to distinguish using this test alone.

In contrast, best `0.998` and second `0.990` yield margin `0.008`, passing the configured margin rule if every other condition also passes. These are examples of the rule, not observed dataset statistics.

### 13.7 The earlier unequal-pair averaging problem

Suppose reference A scores at offset 0 but cannot be read at offset −1, while reference B scores at both. Comparing the offset-0 mean over A+B with the offset-−1 mean over only B changes the comparison population.

The code retains every attempted candidate, including out-of-range records and true failures. Under P1, a mapping without every **in-range** score is not accepted. Separately, `_alignment_publish()` still calculates offset means and medians over the **common mapping set scored at all evaluated offsets**. Boundary-aware acceptance does not broaden that common-set population. Therefore a boundary mapping can become accepted while remaining outside the common-pairs summary. Do not average each offset over its independently available rows. Acceptance remains individual, not “pick the offset with the highest global mean.” ([A2](#source-a2), `_alignment_publish`; [P1](#source-p1))

### 13.8 Consistency rules have strong consequences

When more than one best offset exists among otherwise acceptable mappings in the same video, the implementation changes those mappings to `inconsistent_video_offsets` and removes their eligibility. It does not choose a majority offset or average the offsets.

A further rule groups records by `(video_key, frame_number)`. If one related image/class copy is excluded, other copies that would otherwise pass are also withheld as `related_annotation_excluded`. This preserves the complete annotation set for that source frame rather than training on only the convenient labels.

These choices can exclude technically usable individual images. The audit preserves the reasons so the policy can be evaluated, not hidden.

### 13.9 Meaning of “confirmed” and “excluded”

`alignment_confirmed` and `alignment_training_eligible` mean accepted by this operational policy. They do not mean that a human verified the medical finding or that the original historical capture identity has been proved beyond doubt.

Likewise, exclusion means “not accepted for the current frame-linked subset.” It does not establish that an original still image is invalid for every image-only task. A2 treats unsupported JPEG capability as a run failure, not as an exclusion list. Genuine weak/ambiguous matches, unresolved in-range decode errors, invalid inputs, and consistency conflicts remain distinct causes. In P1, an absent boundary neighbour alone no longer requires exclusion when the available candidates satisfy the other rules.

### 13.10 Boundary-aware completeness: the precise code change

The reported remaining cases have 36 candidate rows with `target_out_of_range`. The visible pattern is the final two annotated positions in affected videos: the penultimate position lacks `+2`, and the last position lacks `+1` and `+2`. No `decode_error` appears in the displayed missing-score reason summary. This is evidence about the supplied diagnostic, not an assertion that every later run has the same failures. ([H2](#source-h2))

For a video of 38,620 frames, valid indices are 0–38,619. At index 38,619, candidate indices 38,620 and 38,621 do not exist, while 38,617, 38,618 and 38,619 can still be compared. Five planned rows are required, but only three scores are required.

The P1 insertion belongs **at the same indentation level as** `if len(scored) >= 2:`, outside that block and before the `if`/`elif` decision chain:

```python
# This runs for every mapping with a candidate group, even with < 2 scores.
outside_video_mask = group["status"].isin(
    ["negative_target", "target_out_of_range"]
)
required_score_count = int((~outside_video_mask).sum())

result["in_range_candidates"] = required_score_count
result["out_of_range_candidates"] = int(outside_video_mask.sum())
result["candidate_window_truncated"] = bool(outside_video_mask.any())

if group["offset"].duplicated().any():
    result["alignment_status"] = "duplicate_candidates"
elif len(group) != expected_count or set(group["offset"]) != expected_offsets:
    result["alignment_status"] = "incomplete_candidate_set"
elif len(scored) != required_score_count:
    result["alignment_status"] = "incomplete_candidate_scores"
elif required_score_count < 2:
    result["alignment_status"] = "insufficient_offset_candidates"
elif result["best_ssim"] < min_ssim:
    result["alignment_status"] = "low_ssim"
elif result["ssim_margin"] < min_margin:
    result["alignment_status"] = "ambiguous_offset"
elif result["best_offset"] not in allowed_offsets:
    result["alignment_status"] = "outside_candidate_offsets"
else:
    result["alignment_status"] = "accepted_ssim"
    result["training_eligible"] = True

# Append exactly once; preserve the later video-consistency checks.
decisions.append(result)
```

This excerpt assumes `best_ssim`, the runner-up and margin have already been computed by the surrounding function. It replaces the decision-chain portion, not the whole function. **Do not replace `expected_count` in the candidate-set check.** It counts all planned audit rows, whereas `required_score_count` counts only rows requiring a real score.

P1 adds three mapping-level diagnostics: `in_range_candidates`, `out_of_range_candidates`, and `candidate_window_truncated`. They are defined when the function reaches a candidate group; early no-candidate cases can have missing diagnostic values unless defaults are explicitly added. The last flag refers to the **alignment search neighbourhood**, not to a temporal context window that has already been extracted.

Only decisions change. Candidate caches retain their full offset coverage and SSIM values; all in-range failures remain visible. No result after rerunning P1 has been supplied yet, so the potential recovery of the 24 mappings is not documented as an observed complete acceptance of the dataset.

## 14. Parallel execution and resource management

### 14.1 What “all videos in parallel” means here

`run_full_frame_alignment_cuda()` builds all ready annotated-video jobs and submits them to a `ThreadPoolExecutor` before collecting completed futures. With `video_workers='all'`, the current workload creates 43 video workers. Each active job owns its decoder and its PyTorch CUDA stream. ([A2](#source-a2))

This removes the former policy of finishing one entire video's comparisons before beginning the next. It does **not** guarantee that 43 decoders or 43 GPU kernels execute simultaneously at every instant. The hardware, stream dependencies, file reading, image pool, and memory budget still constrain actual overlap.

Within a video, mappings advance through bounded batches. Within an SSIM call, pair operations are vectorized. Across videos, independent CUDA streams permit overlapping work. Final Pandas decision aggregation remains on CPU.

### 14.2 Separate controls for separate bottlenecks

| Control | Governs | Does not govern |
|---|---|---|
| `video_workers` | Number of concurrently submitted video-processing tasks. | Guaranteed physical GPU parallelism. |
| `copy_workers` | Simultaneous Drive-to-local video copies. | Total staged-video count or SSIM work. |
| `image_workers` | Shared JPEG reference tasks, including HW-only nvJPEG decoding and GPU tensor preparation. | Video decoding sessions or a physical count of JPEG hardware engines. |
| `mapping_batch_size` | Mappings prepared per video's iteration. | Number of videos in the dataset. |
| `ssim_batch_size` | Pairs per SSIM call. | Global number of pairs active across all workers. |
| `vram_budget_fraction` | Accounting budget for estimated transient batch tensors. | A strict cap on every decoder or CUDA allocation. |

`copy_workers=6` does not mean only six local videos can exist. Copies release the copy slot after transfer and remain on disk while their own video's computation continues. The original 4-copy configuration obeyed the same rule. The shared 16-image-worker pool serves all video jobs; it is not multiplied by the video-worker count.

### 14.3 Disk usage

`_alignment_local_video()` copies the **compressed video file**, not a directory of extracted frames, into a per-task temporary folder. On normal exit and handled exceptions, cleanup removes that temporary folder; the original Drive source is not modified.

With many active video jobs, many compressed copies can coexist. The code checks free space under a lock and accounts for in-progress copy bytes. Insufficient space raises an error; it does not implement a waiting queue for disk capacity. A two-GiB reserve is a configured safety margin, not a guarantee that all videos fit.

A hard process or VM termination may bypass Python cleanup. Leftover local temporary files can therefore require inspection after an abnormal interruption, although a deleted VM removes its ephemeral disk. Persistent audit files and local staging directories must never be confused during cleanup. ([A2](#source-a2); [R4](#source-r4))

### 14.4 GPU memory

The decode-count audit generally needs limited buffers because it counts native frames and releases them. Alignment needs more: candidate tensors, reference tensors, SSIM moment maps, and multiple active decoder contexts. More VRAM usage during alignment is therefore consistent with a different workload, not evidence that the earlier audit failed to use NVDEC.

The displayed transient budget is calculated from a fraction of initially free GPU memory. A weighted reservation gate limits estimated batch work. It excludes some native video-decoder and nvJPEG allocations and cannot guarantee that out-of-memory errors are impossible. Reference pixels now also reside on GPU, and nvJPEG can allocate memory outside the PyTorch allocator. Workers release reservations after their batch work finishes; a low/high VRAM graph alone does not quantify useful GPU throughput. ([A2](#source-a2), [A3](#source-a3))

PyTorch may retain freed tensor storage in its allocator cache. Memory shown as reserved by the framework or occupied in an external monitor need not all belong to live tensors. This is a reason not to infer compute utilization or a memory leak merely from the VRAM graph. ([R5](#source-r5))

### 14.5 Progress bars and timings

The candidate progress total is:

```text
number of ready mappings × number of evaluated offsets
```

It advances for attempted candidate records, including recorded failures, and for reused cached candidates. It is not a count of accepted annotations or necessarily a count of newly computed SSIM scores.

The video bar tracks completed futures. Because the collector also advances it after a failed future, the label `Video checkpoints` alone is not proof that every counted task produced a reusable success. Final state and per-video cache records remain authoritative. ([A2](#source-a2))

Different videos contain different numbers of targets. A candidate bar at 42% and a video bar at 19% can therefore be perfectly consistent. Early estimated remaining times are extrapolations, not benchmarks.

Per-video timing fields include `video_copy_seconds`, `decoder_open_seconds`, `decode_seconds`, `reference_read_seconds`, `ssim_seconds`, and `memory_wait_seconds`. The copy timing includes waiting for a copy slot. Reference reading overlaps decoding, so its recorded interval is the residual wait when results are collected. The SSIM helper brings scores back to the CPU, which waits for the relevant batch. With concurrency these timings overlap and must not be summed as though they were serial wall-clock time.

### 14.6 Tuning decisions versus measured improvements

The later supplied working controls are **16 mappings, 16 SSIM pairs, 16 shared image workers and 6 concurrent copies**. The original N1 snapshot and early screenshots instead used 8/8/8/4. This change is a tuning choice, not a demonstrated optimum or a controlled speedup result. ([H2](#source-h2))

With 16 mappings and five evaluated offsets, a video-worker group produces up to 80 candidate rows. SSIM batches of 16 require up to five calls when all are comparable. Targets shared between mappings are deduplicated for retrieval. Larger batches can reduce call overhead but increase reservations and reduce how many other workers are admitted simultaneously; more image workers or transfers can increase storage contention. The useful measure is completed wall-clock work on the same uncached workload, not merely memory occupancy.

The per-video score signature does not include the copy count or these batch sizes. Changing them alone does not force a recalculation of compatible scored videos. Identify cache reuse separately when measuring performance. The JPEG decoder and its reconstruction options **are** included in the engine signature, unlike pure scheduling settings.

## 15. Persistence, hashes, and restart behaviour

### 15.1 Two fundamentally different hash questions

**Input fingerprint:** has the source identity or computation definition changed enough to invalidate a result?

**Output hash:** are the saved result bytes still the bytes that were committed?

A SHA-256 algorithm only answers questions about its input bytes. Hashing a JSON description of paths, sizes, and modification times is not the same as hashing the entire video's bytes. The default source fingerprints use that cheaper file-metadata description. If a replacement preserves the same identity, size, and modification time, it may go undetected.

Alignment offers `frame_alignment_hash_source_bytes=True` to include hashes of source file contents. This is stronger for change detection but requires reading the video and image bytes during signature construction, which can be expensive on Drive. The reviewed default is `False`. The manifest and count-audit code do not expose the same optional source-byte switch. ([N1](#source-n1), cells 23, 25, 28)

### 15.2 Cache granularity and provenance coverage

| Stage | Reuse unit | Input identity includes | Saved output verification |
|---|---|---|---|
| Manifest | Entire table | Inventory, category set, roots, schema/cache version, OpenCV version/build. | CSV SHA-256 plus schema/types and exact identity rows. |
| Decode | One video | Original path/key/size/mtime, count-method version, selected decoder package/settings signature. | JSON payload SHA-256 and record SHA-256 plus semantic record validation. |
| Alignment | One annotated video | Canonical mappings, offsets, source metadata, engine/library/native-binary/GPU provenance, optional source-byte hashes. | Candidate CSV SHA-256 plus exact mapping-by-offset coverage and valid terminal statuses. |
| Export | Selected output bundle | Committed audit states, configuration, publication policy, selected files. | Original/source hashes, exported hashes, decompression verification, and staged Git blob checks. |

The fingerprint coverage is deliberately not described as identical everywhere. In particular, the decode signature contains the method, package versions, requested backend, GPU ID where relevant, and fallback policy. It does **not** include the full GPU model/driver provenance that the alignment engine signature includes. The latter records GPU properties, driver string, TorchCodec native-library hashes, PyTorch/CUDA versions, and the SSIM implementation identity. ([N1](#source-n1), cells 25 and 28)

A decoder change may invalidate a cache because the computation identity changed, not because the old result was proved false. Installing an unrelated library cannot change a fingerprint unless that library or its effects are included in the signature inputs.

For A2, the alignment engine additionally records `reference_decoder`, `reference_backend`, the no-CPU-fallback policy, output layout, JPEG upsampling/orientation options, nvImageCodec/nvJPEG package versions and native binary hashes. Therefore replacing OpenCV reference decoding with nvJPEG creates a new scoring identity even when video decoding is unchanged.

```python
ALIGNMENT_SCORING_VERSION = 4
ALIGNMENT_BACKEND_ID = (
    "torchcodec_cuda_nvjpeg_hw_reference_ssim_uniform7_v4"
)
# Cache directory is an explicit path, not computed from the integer above:
# frame_alignment_cache_cuda_nvjpeg_v4/
```

Setting only `ALIGNMENT_SCORING_VERSION=3` does not restore the v3 method, change the directory, or recover the v3 fingerprint. It changes one fingerprint input and can invalidate already completed v4 work. Do not falsify the method label to force reuse. These caches contain **scores and attempts**, not a permanent store of decoded video pixels; a changed JPEG method can require repeating video target retrieval because no separate persistent pixel cache exists. ([A2](#source-a2))

### 15.3 What can change without repeating pixels

A new `run_id` does not by itself require manifest probing, full decoding, or SSIM recomputation. Alignment thresholds and acceptance decisions are also separated from candidate scoring: changing the minimum SSIM or margin reapplies decisions to compatible saved scores.

Changing the evaluated offsets does require new candidate coverage. Changing source paths, file sizes, mtimes, relevant versions, or alignment engine identity can require recomputation. The code treats these as reasons to distrust compatibility, even where a human might determine that the actual video content is unchanged.

The final alignment tables are regenerated after decisions are reapplied, even when every score is reused. The count-audit publisher can keep unchanged CSV bytes when its report fingerprint and output hashes still match. Therefore “no decoding” does not necessarily mean “no JSON or CSV writes.”

The P1 boundary edit also changes only acceptance. Keep the nvJPEG scoring version/backend and cache directory unchanged when applying P1. Preserve the previous reports, then reapply decisions using compatible scores and republish the CSV/Parquet/state as a consistent set. Do not manually edit totals or checksums to make a prior report appear current.

**Remaining provenance improvement:** A2 saves thresholds in `effective_settings`, but its `confirmation_rule` string is still `ssim_threshold_margin_consistent_video_offset`; it does not explicitly encode P1's boundary behaviour. A separate decision-policy identity such as `in_range_candidates_v1` should be recorded with the decision outputs and summary in a future small code edit. That field is a recommendation, **not something the current P1 patch already writes**. It belongs to decision provenance, not the cached SSIM engine signature.

### 15.4 Complete is not synonymous with passed

The implementation distinguishes an attempt that reached a terminal record from a content result that passes quality criteria. This fixes the earlier behaviour in which any imperfect result could mark the global audit incomplete and cause all videos to be decoded again.

Decode checkpoints may contain errors and still be complete/reusable. Alignment caches may contain unscored terminal candidate statuses and still constitute a complete record of attempted comparisons. Acceptance and `needs_review` are separate interpretations.

Infrastructure failures are not treated as completed image-quality judgements. An interruption, GPU error, or incompatible decoder can leave a video's cache incomplete while other videos' committed checkpoints remain reusable.

### 15.5 Interrupted runs

For full decoding, completed per-video JSON checkpoints survive. The current video's incomplete work is repeated. Aggregate CSVs can be reconstructed from the checkpoints even if the previous global export was interrupted.

For concurrent alignment, completed video CSV/JSON pairs survive. **Several videos may be in flight when interruption occurs**, so more than one video's work may need to be restarted. The earlier sequential intuition of “only the sixth video is lost after five complete videos” no longer describes the 43-worker implementation.

The alignment orchestrator uses cooperative cancellation, checks the stop event between operations, cancels pending futures, and waits for workers to stop before returning. An interactive interrupt may therefore take time to unwind safely. A hard VM termination cannot run those cleanup handlers, but already committed Drive checkpoints remain separate from runtime memory.

### 15.6 Publication order and consistency

Writers use temporary sibling files and replacement operations. CSV/Parquet data is written before the JSON marks it complete. Readers check the completion flag and hashes before trusting saved results.

This is a defensive application-level commit pattern, not an assertion that mounted Drive behaves like a distributed transactional database. Files still need to be fully available and readable at reuse time. SHA-256 detects byte changes but is not a signature proving authorship, authenticity, or clinical validity.

### 15.7 GPU requirements on cached runs

The full decode driver delays GPU initialization until a video actually requires decoding; compatible completed checkpoints can be read without executing NVDEC again.

The current CUDA alignment orchestration is stricter: it resolves the CUDA device/backend, reads environment identity, and runs its numerical check before per-video cache decisions. Thus a fully cached alignment rerun still expects a compatible GPU-enabled environment, even though the dataset's frames need not be decoded again. The notebook's dependency setup also explicitly requires CUDA. This is current behaviour, not a claim that reading CSV scores intrinsically requires a GPU. ([N1](#source-n1), cells 5, 25, 28)

## 16. Artifact catalogue and data contracts

### 16.1 Persistent inputs and manifest artifacts

| Artifact | Location relative to project storage | Role |
|---|---|---|
| `metadata_clean.parquet` | `data/curated/phase2/dataset_audit/` | Complete cleaned Sector 1 annotations. |
| `physical_image_inventory.csv` | Same directory | Image/class/path lookup for alignment. |
| `video_inventory.csv` | `data/curated/phase2/manifests/` | Basic physical video inventory. |
| `video_manifest.csv` | Same directory | One row per physical video with technical probe metadata. |
| `video_manifest_state.json` | Same directory | Manifest cache identity, completion and CSV checksum. |

### 16.2 Full-decode artifacts

All the following are under `outputs/phase2/results/`:

| Artifact | Role |
|---|---|
| `decode_audit.csv` | One final record per physical video, including counts and decoder provenance. |
| `labelled_video_decode_audit.csv` | The `partially_labelled` subset of the combined audit. |
| `unlabelled_video_decode_audit.csv` | The `fully_unlabelled` subset. |
| `decode_audit_summary.csv` | Group and overall totals, reference comparisons and review counts. |
| `decode_audit_state.json` | Committed aggregate state, report fingerprint and output checksums. |
| `decode_audit_run_state.json` | Current invocation progress and completion state. |
| `decode_audit_cache_v2/*.json` | Independent per-video checkpoint records. |
| `decode_audit_legacy_backup/` | Previous report copies where the legacy-preservation path applies. |

Important columns include source identity, `container_reported_frames`, `frames_read_before_stop`, `read_minus_reported`, `decoder_backend`, `fallback_reason`, `decoding_completed`, `header_source`, `packets_processed`, and `frames_returned_during_flush`. Some backend-specific fields may be missing or null; absence is not replaced with a fabricated measurement.

`needs_review` is computed from the implemented count/status rules. The published-reference mismatch is displayed in the group summary but does not automatically force every video into per-file review when its own counts and status are consistent.

### 16.3 Alignment artifacts

| Artifact | Grain and intended use |
|---|---|
| `frame_alignment_candidates.csv` | Every mapping/offset attempt and its score or explicit problem. |
| `frame_alignment_audit.csv` | One decision record per alignment mapping. |
| `frame_alignment_offset_summary.csv` | Common-set score summaries and confirmed counts by offset. |
| `frame_alignment_confirmed.csv` | Original annotation rows whose mappings satisfy the acceptance policy. |
| `frame_alignment_confirmed.parquet` | Typed machine-readable accepted subset for later phases. |
| `frame_alignment_excluded.csv` | Original annotation rows withheld by the current policy, with reasons. |
| `frame_alignment_performance.csv` | Per-video provenance, reuse/recompute source, and timing information. |
| `frame_alignment_state.json` | Completion, run ID, engine, effective settings, totals and file checksums. |
| `frame_alignment_run_errors.csv` | Run-level failures when present; removed after successful completion. |
| `frame_alignment_cache_cuda_nvjpeg_v4/*.csv` and `*.json` | Current hardware-JPEG method's per-video candidate scores and commit metadata. |
| `frame_alignment_cache_cuda_v3/*.csv` and `*.json` | Historical OpenCV-reference method; not silently reused by A2. |

The seven-column candidate record is:

```text
mapping_id
video_key
offset
target_frame_index
ssim
status
error_message
```

The seven columns above define the cache/scoring core. The final combined candidate CSV additionally carries `score_source`, `decoder_backend` and `ssim_backend` provenance where a video result is available. Invalid-input rows can have missing per-video provenance.

The candidate schema permits `scored`, `negative_target`, `target_out_of_range`, `decode_error`, `image_unreadable`, `dimension_mismatch`, and `image_too_small`. Not every permitted status is necessarily emitted by the strict JPEG path: hardware JPEG failures raise rather than silently producing an image-quality judgement. Final decision states include `low_ssim`, `ambiguous_offset`, `incomplete_candidate_scores`, `outside_candidate_offsets`, `inconsistent_video_offsets`, and invalid-input reasons.

P1 additionally puts the following in the **mapping-level `frame_alignment_audit.csv`**:

| Field | Meaning |
|---|---|
| `in_range_candidates` | Number of candidate rows for which a finite successful score is required. |
| `out_of_range_candidates` | Number of rows marked `negative_target` or `target_out_of_range`. |
| `candidate_window_truncated` | Whether the alignment candidate neighbourhood crosses a video boundary. |

`candidate_issues` may legitimately still contain `target_out_of_range` for a mapping accepted under P1. Such retained evidence is not a contradictory failure. The explicit projection in A2's `_alignment_publish()` does **not automatically copy these three new fields into the confirmed Parquet**. To access them downstream, join the mapping audit through `alignment_mapping_id` ↔ `mapping_id`, or explicitly extend the projection in a separate code change. They are not new members of the seven-column cached candidate schema.

### 16.4 Downstream accepted-subset fields

In addition to the original annotation columns, consumers can use:

| Field | Purpose |
|---|---|
| `alignment_mapping_id` | Link back to mapping-level evidence. |
| `alignment_image_path` | Resolved reference-image location in the recorded runtime. |
| `alignment_video_path` | Resolved source-video location. |
| `alignment_input_status`, `alignment_input_error` | Input-resolution evidence. |
| `alignment_offset` | Accepted best correction relative to metadata frame number. |
| `aligned_frame_index` | Decoder's accepted zero-based target index. |
| `alignment_ssim` | Best comparison score. |
| `alignment_ssim_margin` | Separation from the second-best candidate. |
| `alignment_status` | Decision reason. |
| `alignment_training_eligible` | Eligibility under the configured policy. |

The invariant is:

```text
confirmed annotation rows + excluded annotation rows = original metadata rows
```

Mapping counts are reported separately and must not replace annotation counts in this invariant. The export step checks the saved Parquet and exclusion CSV against the completed state. [A2, `_alignment_publish`; G1, `_s2_summary`]

### 16.5 Relative-path conventions are not uniform

| File | Relative field | Base |
|---|---|---|
| `video_manifest.csv` | `video_relpath` | `CONFIG['storage_root']` |
| `decode_audit.csv` | `video_relpath` | `DIRS['dataset_root_dir']` |
| `video_inventory.csv` | `relative_path` | Dataset root used by the inventory cell. |
| `physical_image_inventory.csv` | `relative_path` | `DIRS['raw_data_dir']` from Sector 1. |

For example, the manifest may store `data/raw/kvasir_capsule/osfstorage/labelled_videos/<key>.mp4`, while the decode audit stores `labelled_videos/<key>.mp4`. They can refer to the same file. Applying one base indiscriminately to every relative column is a downstream bug, not an inconsistency in the video itself.

## 17. Observed results and what they establish

### 17.1 Inventory and manifest

The saved notebook reports 117 discovered videos, 43 annotated metadata IDs, and zero unmatched annotated video IDs. Its manifest cache was successfully reloaded without `probe_video()` being called. The displayed technical examples are 336×336 videos, and the output reports no container/first-frame issues. The validator's observed FPS range is `30.0` to `30.00003`. ([N1](#source-n1), outputs of cells 16, 20, 23 and 26)

The example dimensions are not asserted here as a verified universal dimension for every source merely because five preview rows show them. The complete manifest is the correct source for such a claim.

### 17.2 Completed decode results

| Group | Videos | Frames returned | Reference in the audit | Difference | Fully processed | Review flags |
|---|---:|---:|---:|---:|---:|---:|
| `partially_labelled` | 43 | 1,979,285 | 1,955,675 | +23,610 | 43 | 0 |
| `fully_unlabelled` | 74 | 2,785,829 | 2,785,829 | 0 | 74 | 0 |
| `all_videos` | 117 | 4,765,114 | 4,741,504 | +23,610 | 117 | 0 |

The output explicitly records `{'nvdec': 117}`, `Videos decoded this run: 0`, and `checkpoints reused: 117`. Thus this saved invocation demonstrates persistence/reuse of an earlier NVDEC audit rather than measuring fresh NVDEC throughput. The runtime did not decode 4.76 million frames during that cached invocation. ([N1](#source-n1), output of cell 25)

Zero review flags means zero records violated the implemented review predicates. It does not mean the larger reference discrepancy was solved or that hidden codec problems and clinical errors are impossible.

### 17.3 Alignment evidence available at documentation time

The original N1 notebook had no final alignment counts. Subsequent development results now provide the following history. These are distinct runs/policies; they must not be collapsed into one benchmark. ([H2](#source-h2))

| Observation | Reference decoder | Minimum SSIM | Boundary score rule | Accepted annotation rows | Excluded annotation rows |
|---|---|---:|---|---:|---:|
| Earlier completed CUDA-video run | OpenCV CPU | `0.99` | All evaluated offsets need scores | 0 | 47,239 |
| Later hardware-JPEG run | nvJPEG hardware | `0.96` | All evaluated offsets need scores | 3,782 | 43,457 |
| Latest reported completed threshold reevaluation | nvJPEG hardware | `0.92` working setting | All evaluated offsets need scores | **47,239** |
| Boundary-aware decision edit P1 | nvJPEG hardware | `0.92` working setting | All planned rows; all in-range scores; at least two | **Not yet supplied** | **Not yet supplied** |

The latest reported completed result partitions the original 47,239 annotation rows exactly. Its approximately **99.95% acceptance/retention** is not measured alignment precision, classification accuracy, or clinical sensitivity. Mapping counts and annotation counts remain distinct: there are 47,238 mappings in the displayed diagnostics. No value is invented for a completed P1 rerun.

The displayed missing-score diagnostic reports **36 `target_out_of_range` candidate rows**. The visible examples are consistent with requests after the end of videos. P1 addresses that decision-policy issue while retaining those rows and their missing scores. It does not fabricate extra source frames.

#### Reference-decoder comparison

Values below are transcribed from the two supplied diagnostic screenshots, to the displayed precision:

| Statistic | Earlier OpenCV-reference best SSIM | nvJPEG-reference best SSIM |
|---|---:|---:|
| Count | 47,238 | 47,238 |
| Mean | 0.953065 | 0.953224 |
| Standard deviation | 0.004670 | 0.004665 |
| Minimum | 0.923801 | 0.923936 |
| 1st percentile | 0.943877 | 0.944179 |
| 5th percentile | 0.946063 | 0.946312 |
| Median | 0.952658 | 0.952786 |
| 95th percentile | 0.961154 | 0.961355 |
| 99th percentile | 0.966211 | 0.966434 |
| Maximum | 0.974506 | 0.974663 |
| Mean best-minus-second margin | 0.243477 | 0.243470 |
| Minimum best-minus-second margin | 0.025150 | 0.025380 |

The displayed mean shift is only **+0.000159**. Aggregates do not establish that each pair improved. Since the acceptance threshold also changed from `0.99` to `0.96`, the move to 3,782 acceptances cannot be attributed to nvJPEG alone. Neither decoder's reported maximum reaches `0.99`.

The earlier common-set offset report favoured offset `0` in aggregate (approximately 0.953 mean, versus approximately 0.668–0.691 at the tested neighbours). This supports investigating that mapping convention; it is not a new independent proof of a universally valid zero correction. Consumers should use each accepted row's `aligned_frame_index`. Do not claim all offsets are verified from an aggregate mean alone.

The referenced historical v3 summary was inspected earlier in the development conversation. This documentation revision did not refetch the remote and does not claim that its current content matches any later nvJPEG run. Read each final state's method, threshold, run ID and output hashes before comparing results.

### 17.4 No fabricated performance or test claims

The source conversation includes development descriptions of local tests, but the reviewed package does not supply their complete test suite and machine-readable results. This README therefore does not turn those descriptions into a reproducible benchmark or a certification claim.

Likewise, early sequential timing and CUDA progress estimates are not a controlled speedup study: cache reuse, disk state, dataset target distribution, library versions, and concurrent workloads differ. A defensible performance comparison would fix these conditions and report actual completed elapsed time and work counts.

## 18. Technical decision record

The following decisions summarize the development history and its implementation consequences. They are not additional features beyond the code described above. ([N1](#source-n1), [G1](#source-g1), [H1](#source-h1))

| Decision | Reason and alternative considered | Consequence or trade-off |
|---|---|---|
| Reuse Sector 1 Parquet. | Avoid re-normalizing an already-audited dataset in each notebook. | Runtime variables must still be rebuilt; source rows remain preserved. |
| Verify stems before using them as keys. | Avoid duplicating or drifting from the original normalization function. | Direct `Path.stem` is appropriate only after the observed join check, not as a universal dataset rule. |
| Reject duplicate video identities before probing. | Ambiguous joins should fail before expensive file processing. | No silent “take the first match.” |
| Keep inventory, probe, full decoding and alignment separate. | They validate different properties. | More distinct artifacts, but clearer provenance and reuse. |
| Retain reported counts as diagnostics. | They are cheap and useful even when complete decoding is available. | They must not be called measured extractable-frame totals. |
| Warn on published-total mismatch instead of enforcing equality or `>=`. | A measured discrepancy should be preserved without forcing a result. | The discrepancy remains unresolved and visible. |
| Move full decoding to NVDEC. | Reduce CPU pixel-decoding cost for all-video counting. | Requires compatible libraries and explicit end-of-stream handling. |
| Distinguish API failures from content failures. | A missing method such as `Flush()` is not evidence of corrupt videos. | Some errors stop the run instead of triggering repeated CPU fallback. |
| Save checkpoints per video. | Avoid repeating all prior work after an interrupted global audit. | In-flight videos restart, and checkpoint verification becomes part of the pipeline. |
| Separate scoring from acceptance. | Threshold changes need new decisions, not new video pixels. | Candidate evidence remains stored even for exclusions. |
| Expand from sampling to all eligible mappings. | Temporal extraction benefits from evidence for every retained image/video link. | Much greater work and a need for batching, concurrency and persistence. |
| Preserve a common set for offset summary means. | Unequal missing-read populations bias aggregate comparisons. | Common-set summaries stay separate from boundary-aware individual acceptance. |
| Excuse only out-of-range score requirements at boundaries. | A nonexistent neighbour is not a failed decode of an existing frame. | Keep every planned row; score every in-range candidate and require at least two. Post-edit counts must still be measured. |
| Preserve colour and original resolution. | Avoid changing the meaning of a strict similarity threshold through resizing or grayscale conversion. | Greater arithmetic and memory cost. |
| Add a best-versus-second margin. | A near-identical neighbouring frame may make identity ambiguous. | Correct but indistinguishable matches can be withheld. |
| Require consistent accepted offsets within a video. | Prevent contradictory indexing conventions from entering temporal extraction. | The policy may exclude many otherwise-high-scoring mappings. |
| Preserve related annotation sets. | Avoid partially dropping labels attached to the same frame. | Eligibility propagates across related image/class copies. |
| Use exact CUDA target retrieval rather than a Python frame-by-frame loop. | Request the necessary targets while preserving indexing semantics. | Indexing and dependency decoding still occur internally. |
| Submit all annotated-video jobs concurrently. | Remove the old whole-video sequential bottleneck. | Hardware, disk, memory and I/O limits prevent unlimited useful parallelism. |
| Stage compressed videos locally. | Reduce repeated accesses through the Drive mount. | More ephemeral disk use; image references are still loaded separately. |
| Require CUDA provenance for video targets and HW-only nvJPEG reference decoding. | The selected pipeline explicitly excludes CPU pixel decoding in alignment. | Host I/O remains; unsupported JPEG hardware/variants stop rather than fall back. |
| Lower the working threshold after inspecting the observed distribution. | `0.99` rejected every scored mapping; `0.96` retained only a small fraction. | `0.92` is a documented working choice, not independent proof of accuracy; originals and all scores remain available. |
| Separate scoring version from acceptance-policy changes. | JPEG reconstruction changes pixels; thresholds and boundary rules reinterpret saved scores. | v4 must not be relabelled v3 to force cache reuse; decision provenance still deserves its own explicit version. |
| Keep typed Parquet plus inspectable CSV/JSON. | Later phases need reusable tables and humans need auditable evidence. | Some logical data is available in multiple formats for different consumers. |
| Snapshot CONFIG once per run. | Small-file duplication is simpler than hash-based configuration deduplication. | Timestamp snapshots do not prove full environment reproducibility by themselves. |
| Publish a selected bundle, not the entire storage root. | Preserve downstream data and evidence without raw-pixel/cache bloat. | GitHub alone cannot reconstruct the raw video/image dataset. |
| Use scoped Git staging and a normal push. | Avoid accidental unrelated commits and destructive synchronization. | Dirty/diverged repositories require deliberate user resolution. |

## 19. Operating the notebook

### 19.1 First run or a new runtime

Run the notebook in its documented top-to-bottom order. Install general dependencies, retain the existing PyTorch, prepare compatible CUDA TorchCodec and nvImageCodec/nvJPEG, then import the alignment dependencies. Do not first import a native package and subsequently replace it in the same session. Run Git bootstrap, define the complete Sector 2 configuration, mount Drive, resolve paths, and save the configuration snapshot. For normal execution explicitly set `frame_alignment_min_ssim=0.92` and `frame_alignment_define_only=False`; this avoids accidentally relying on A2's older fallback threshold.

Next inspect the physical layout, reload Sector 1 metadata, check identifier coverage, and build or reuse the manifest. Run the count audit, validate the manifest, then run alignment. Execute publication only after the completed alignment state and outputs exist.

The notebook should fail early for missing prerequisites rather than manufacture missing dataset directories or placeholder audit results.

#### Define-only mode and the short capability test

`frame_alignment_define_only=True` imports dependencies and defines functions but skips the final `run_full_frame_alignment_cuda(...)` call. It does not automatically run a test, validate all JPEGs, or write a completed audit. The supplied A2 file has no `else` message for this branch, so finishing without output can be expected.

For a capability check, load A2 in define-only mode and separately run `frame_alignment_nvjpeg_smoke_test.py`. It compares a few existing references/candidates and prints diagnostics without changing saved audits. After the test succeeds, set `CONFIG["frame_alignment_define_only"] = False`, save the intended configuration without regenerating `run_id`, and reexecute the alignment cell. A configuration assignment alone does not restart a previously executed cell. ([A2](#source-a2), [A3](#source-a3))

### 19.2 Normal rerun

Keep `force_redecode=False` and `force_rescore=False`. Fresh inventories and signatures are still checked. Compatible manifest data, decode checkpoints, and alignment score caches are reused. Threshold-based alignment decisions and final reports may still be recreated.

A new run's configuration file is expected. The goal is to avoid expensive redundant pixel work, not to eliminate every small write.

### 19.3 Resume after interruption

Recreate the runtime environment, mount the same persistent storage, load the same inputs, and rerun the relevant cell. The current cache files—not a progress bar remembered from the previous session—determine which videos can be reused.

For concurrent alignment, any number of video tasks may have been in flight. Completed compatible checkpoints survive; unfinished videos must be processed again. Do not delete all caches as a routine response to one failed video.

### 19.4 Deliberate retry and recomputation

For a full decode recount, use `decode_audit_force_redecode=True`. For selected recounts, populate `decode_audit_retry_video_keys`. To retry terminal failures selectively under the implemented predicate, enable `decode_audit_retry_failed`.

For alignment, use `frame_alignment_force_rescore=True` only when deliberately discarding reuse. A genuine change to scoring inputs or arithmetic requires an appropriate engine/cache identity change, not a new timestamp. The OpenCV-to-nvJPEG transition is such a change. By contrast, changing `0.96` to `0.92` or applying P1's in-range score requirement is a decision-only change: keep the v4 scoring identity, preserve the preceding final reports, and rerun with reuse enabled to generate current decisions and hashes.

The current alignment cache has no dedicated retry-only-failed-candidates switch: its unit of reuse is a video's complete candidate table. That table can legitimately include recorded unreadable candidates. A deliberate rescore or source/signature change is needed to retry those attempts.

### 19.5 Before consuming or publishing a result

Read the final state and check `complete`, the effective settings and expected output hashes. Review decode discrepancies and alignment status counts. Check that confirmed and excluded annotation rows partition the original metadata. Treat an empty confirmed subset as a real result requiring investigation, not as a reason to fall back silently to all rows.

This is especially important after an interruption: old final files can still exist while the current state says incomplete. Their existence alone does not make them the result of the current run.

## 20. Using the outputs in later phases

### 20.1 Load from persistent storage

The following is a downstream usage example, not another audit:

```python
from pathlib import Path
import json
import pandas as pd

results = Path(DIRS["results_dir"])
state = json.loads((results / "frame_alignment_state.json").read_text())
if state.get("complete") is not True:
    raise RuntimeError("The alignment export is incomplete.")

df_clean = pd.read_parquet(
    Path(DIRS["curated_data_dir"]) / "dataset_audit" / "metadata_clean.parquet"
)
df_alignment_confirmed = pd.read_parquet(
    results / "frame_alignment_confirmed.parquet"
)
video_manifest = pd.read_csv(
    Path(DIRS["manifests_dir"]) / "video_manifest.csv",
    dtype={"video_key": "string"},
)
```

For production consumption, also compare file hashes to the state before trusting the tables. Checking `complete` alone does not replace the checksum validation implemented in the pipeline.

### 20.2 Which table to use

`df_clean` is the complete cleaned source. `df_alignment_confirmed` is the selected frame-linked subset. A temporal workflow adopting the current filtering policy should consume the latter and use `aligned_frame_index` without subtracting another one or applying a global offset.

Exclusions remain available for audit, manual inspection, policy evaluation, or a differently defined image-only experiment. Such alternative use must be explicitly documented; the present cell does not automatically reintroduce excluded rows into training.

The saved subset is metadata, not a file containing every decoded image. Later extraction still needs the original image/video bytes or an independently produced frame store.

### 20.3 Path rebasing

Recorded absolute paths are provenance, not portable locations. In a different environment, use the saved configuration's root and the appropriate relative-path convention to build new paths.

For an accepted annotation path that lies under the original storage root:

```python
old_root = Path(saved_config["storage_root"])
new_root = Path("/your/new/MMVQA_Clinical")

old_path = Path(accepted_row["alignment_video_path"])
new_path = new_root / old_path.relative_to(old_root)
```

The example deliberately fails if the old path is outside the expected root, rather than performing an unrestricted string replacement. Apply the distinct base conventions in Section 16 for manifest, decode, and image-inventory relative fields.

### 20.4 Time and frame indices

The manifest duration is computed from container-reported count/FPS. It describes the exported video timeline under that estimate. It is not automatically a clinically meaningful elapsed capture time.

Sector 2 does not reconstruct original physiological timing or validate a conversion from metadata index to the original capsule acquisition clock. Later temporal analyses must state which timeline they use. An exact decoder index and a clinically correct timestamp are not the same guarantee.

### 20.5 Split construction remains downstream

The alignment code does not use `MultilabelStratifiedShuffleSplit` merely because that class is imported. No new split is generated here. The grouping strategy, leakage controls, split version, and model-evaluation protocol remain downstream work.

The agreed downstream plan is to keep all artifacts derived from the same source video together and, where identifiers permit, also keep patient-related sources together. It remains a plan, not a split-generation feature implemented by this audit. Training uses its training partition only; validation and test sources do not become training data merely because their alignment was checked.

### 20.6 Constructing temporal context from an accepted anchor

The accepted correspondence supplies a **central anchor**, not a label for every surrounding frame. For a row with `aligned_frame_index=k`, the first proposed context experiment uses:

```python
# Proposed downstream setting, not consumed by the current alignment cell.
CONFIG["temporal_context_offsets_frames"] = [-2, -1, 0, 1, 2]
```

Its indices are `k + context_offset`. The alignment correction has already been applied when producing `aligned_frame_index`; do not apply it again. Context offsets and alignment-search offsets may have the same numbers but answer different questions.

For example, anchor 1000 yields `[998, 999, 1000, 1001, 1002]`. The `0.92` threshold concerns the **JPEG-to-anchor association only**. Neighbouring frames do not need SSIM above `0.92` relative to the anchor: useful context can include motion, changes of view, or changing visibility. Using the same threshold to retain only similar neighbours would implement a different selection policy.

Plan one context per unique `(video_key, aligned_frame_index)` rather than creating duplicate pixel windows for multiple labels. Preserve all labels in a related table. A neighbour receives its own label only when an annotation exists for that neighbour; the anchor's finding and bounding box must not be copied to all context positions. An anchor-labelled window does not by itself support a ground-truth claim that a finding persists across the complete clip.

At video boundaries, retain available neighbours and mark missing positions. Do not cross into another video or fabricate frames to make the window appear complete. Padding for a model's fixed-size input, if later implemented, must carry a validity mask. `candidate_window_truncated` from P1 refers to the alignment search; compute temporal-window validity independently for the selected context offsets.

For efficient extraction, collect the unique required indices **per video**, decode them in bounded CUDA batches and reuse returned frames across overlapping windows. A suggested later `temporal_context_manifest.parquet` would track a context ID, anchor mapping/index, video key, context offset, actual frame index, container PTS, role, annotation availability, extraction status and split. This is a proposed downstream artifact, not an output already created by Sector 2.

### 20.7 New videos and supervised model use

On an unannotated new video, preserve `video_key`, frame index and container timestamp as frames are extracted. There is normally no known reference JPEG for this audit, so decoding and model inference do not require comparisons with all Kvasir reference images. An offset observed for existing source metadata is not a universal setting for arbitrary new inputs.

For supervised adaptation or evaluation on decoded Kvasir frames, alignment reduces the risk of pairing the wrong visual input with a valid existing target label. The model learns through its later training objective, **not through SSIM or the alignment routine**. In an image-recognition experiment, the true finding belongs in the target/evaluation fields, not in an answer-revealing input prompt. Original JPEG-only experiments can retain `df_clean` even where a temporal link remains unconfirmed. These choices describe the downstream use of the evidence layer; they do not claim that model training has already run.

## 21. GitHub publication

### 21.1 Purpose and execution status

The final export cell packages the important saved metadata and completed audits for later phases and reproducible review. It does not decode videos, compute SSIM, install packages, reset branches, or force-push.

The supplied N2 snapshot has no executed publication cell. Later in development, the user supplied a GitHub summary and that historical v3 bundle was read; it recorded a completed run at `min_ssim=0.99` with zero confirmed rows. That observation establishes an earlier publication, not publication of the latest nvJPEG/boundary-aware result. No fresh remote check was made for this documentation-only update. Reexecute G1 after a successful current decision rerun to publish the matching results, state, configuration and this README. ([N2](#source-n2), [G1](#source-g1), [H2](#source-h2))

### 21.2 Inputs and selection policy

`SECTOR2_EXPORT` derives repository identity, branch, local checkout, author, and Colab secret name from `GIT_CONFIG`. The stage-specific publication policy uses:

```python
# Policy excerpt; repository/authorship values are provided by GIT_CONFIG.
publication_policy = {
    "repo_output_dir": "outputs/phase2/sector2",
    "confirm_push": True,
    "push": True,
    "require_current_alignment_run": True,
    "include_physical_image_inventory": True,
    "gzip_csv_threshold_mib": 10,
    "always_gzip_csv": ["frame_alignment_candidates.csv"],
    "max_file_mib": 45,
    "max_export_mib": 250,
}
```

These are application limits, not a claim about universal GitHub repository policies. The exporter does not automatically install or configure Git LFS if a limit is exceeded.

It selects the original cleaned Parquet, physical image inventory, manifest and state, optional basic inventory, decode/alignment state files, and every required output listed by the completed audits' `output_sha256` mappings. It also considers existing top-level `*summary*.json` files under results. Raw videos/images, per-video caches, temporary copies, partial reports, and legacy backups are not recursively swept into the repository. ([G1](#source-g1))

### 21.3 Validation before publication

`_s2_validate_states()` requires complete manifest, decode, and alignment states. It verifies saved output hashes, the current alignment run ID when required, thresholds and offsets against the current `CONFIG`, and the decode run state's completion when that file exists.

`_s2_summary()` reads the saved artifacts rather than relying on possibly stale notebook globals. It checks matching video-key sets, counts against states, the confirmed/excluded partition, accepted eligibility values, and nonnegative integral aligned indices. An empty confirmed subset is reported as a warning; it is not silently repaired with unfiltered data.

The exporter checks committed report consistency. It does **not** rescan and revalidate all raw video bytes before publication. The preceding audit stages are responsible for their source checks. Nor does matching row counts prove byte identity between the original Sector 1 metadata and every carried-through value: that stronger verification is not implemented here.

A2 keeps the public output filenames unchanged, so G1's core artifact selection does not need to be replaced merely because references now use nvJPEG or decisions include boundary diagnostics. Rerun decisions first, however: a hash-valid old report can still describe the old policy. The exporter verifies saved thresholds/offsets and current run identity, but has no independent check that P1 was applied. The recommended separate decision-policy identifier would make this provenance stronger; it is not already enforced by G1.

### 21.4 Bundle structure

```text
outputs/phase2/sector2/
├── README.md
├── artifact_catalog.json
├── configs/
│   ├── phase2_02_video_frame_validation_<run_id>_config.json
│   └── github_export_settings.json
├── data/
│   ├── metadata_clean.parquet
│   └── physical_image_inventory.csv[.gz]
├── manifests/
│   ├── video_manifest.csv
│   ├── video_manifest_state.json
│   └── video_inventory.csv                  # when available
├── results/
│   ├── decode_audit.csv
│   ├── decode_audit_summary.csv
│   ├── labelled_video_decode_audit.csv
│   ├── unlabelled_video_decode_audit.csv
│   ├── decode_audit_state.json
│   ├── decode_audit_run_state.json          # when available
│   ├── frame_alignment_audit.csv[.gz]
│   ├── frame_alignment_candidates.csv.gz
│   ├── frame_alignment_confirmed.csv[.gz]
│   ├── frame_alignment_confirmed.parquet
│   ├── frame_alignment_excluded.csv[.gz]
│   ├── frame_alignment_offset_summary.csv
│   ├── frame_alignment_performance.csv
│   └── frame_alignment_state.json
└── reports/
    ├── sector2_summary.json
    └── sector2_summary.md
```

The `[.gz]` notation means optional compression based on policy; it is not part of an actual filename. The catalogue records the exact exported path for each logical path.

### 21.5 Compression and two sets of checksums

Large CSVs are compressed with gzip. Candidate scores are always compressed. Gzip headers omit timestamps and filenames so identical input bytes produce deterministic output under the implementation's compression settings.

Compression is verified by decompressing and comparing against the original SHA-256. The catalogue records both the source hash and the hash of the exported, possibly compressed bytes. Original audit state JSONs are copied unchanged, so their hashes still refer to **uncompressed original report bytes**.

Therefore:

```text
hash of exported .csv.gz       -> compare with artifact_catalog.json
hash of decompressed CSV      -> compare with original audit output_sha256
```

The exporter is not deleting rows or rounding scores to meet a file-size limit. It stops if a selected artifact remains too large rather than omitting important data invisibly.

### 21.6 Git safety sequence

The publication function stages the bundle in a temporary directory, validates it, and refreshes the checkout before copying artifacts into the repository. It verifies an HTTPS GitHub repository identity without embedded credentials, checks both fetch and push destinations, verifies the branch, and requires a clean checkout.

It then fetches the configured branch. If the local branch is only behind, `merge --ff-only` advances it. If histories diverge, it stops. If the local branch is ahead, only unpublished commits carrying this exporter's marker and confined to its publication prefix qualify for its retry path. [G1, `_s2_refresh`]

After displaying changed files, sizes, and key dataset counts, the default policy requires the user to type `PUSH`. Source hashes are checked again after this review window. The code copies and stages only approved explicit paths, verifies the Git staged content against the intended bytes, creates a scoped commit, and pushes normally.

`git add --force` in this implementation only permits the explicit approved paths to bypass ignore rules. It is **not** `git push --force`. The exporter uses neither `git add .` nor an automatic reset/rebase/stash to resolve unrelated changes.

### 21.7 Credentials and disclosures

Authentication uses a Colab secret and an ephemeral child environment/askpass mechanism. The token is not embedded in the repository URL or the command-line arguments. Configuration credential-key checks and byte-pattern scans provide additional safeguards, but they are convenience checks rather than a proof that arbitrary compressed or structured data contains no sensitive information.

Absolute paths, configuration values, and author information may be present in the selected artifacts. Review the displayed selection and the repository's visibility. Source attribution and applicable source terms remain necessary; this project export is not a new official Kvasir-Capsule release. ([G1](#source-g1))

### 21.8 Receipts and no-op behaviour

The function records `pushed`, `committed_locally`, or `unchanged`, together with commit, run ID, repository, branch, bundle prefix, catalogue hash and file counts. It writes a Drive receipt under:

```text
outputs/phase2/reports/
  phase2_02_github_publish_<run_id>.json
```

It also saves `sector2_github_artifact_catalog.json` and `sector2_summary.json` under reports. The receipt is deliberately not folded into the next bundle, avoiding a self-changing publication cycle.

When selected bytes are unchanged and no owned commit is waiting to be pushed, no empty commit is created. A new run ID or regenerated provenance files can legitimately make a later bundle different even where the underlying pixel results were reused.

If push fails, the local commit is preserved and the code raises an error instead of claiming remote success. Unselected older files in the publication directory are not pruned automatically; `artifact_catalog.json` describes the active current bundle.

### 21.9 Loading the GitHub bundle

```python
from pathlib import Path
import json
import pandas as pd

bundle = Path("outputs/phase2/sector2")  # relative to repository root
catalog = json.loads((bundle / "artifact_catalog.json").read_text())
paths = {item["logical_path"]: bundle / item["path"]
         for item in catalog["artifacts"]}

original = pd.read_parquet(paths["data/metadata_clean.parquet"])
accepted = pd.read_parquet(paths["results/frame_alignment_confirmed.parquet"])
candidates = pd.read_csv(
    paths["results/frame_alignment_candidates.csv"],
    dtype={"video_key": "string"},
)
```

The catalogue avoids guessing which CSVs were compressed. Source pixels remain external to this bundle. Cache files are deliberately not included, so copying state JSONs back from GitHub does not recreate the per-video working cache on Drive.

## 22. Troubleshooting and recovery

| Symptom | Interpretation and appropriate action |
|---|---|
| `KeyError: 'run_id'` | The dictionary may have been replaced after the key was set. Generate the run ID after defining the final Sector 2 `CONFIG`, then save. |
| `NameError: normalize_id` | A helper from Sector 1 is not automatically present in a new runtime. Use the already-verified stem/key relationship or import the exact shared helper deliberately. |
| `Path.stem(...)` error | `stem` is a property. Use `path.stem`. |
| `NameError: probe_video` | Execute the probe definitions before the manifest builder. A Parquet file does not contain Python function definitions. |
| `KeyError: container_opened` | Inspect the manifest schema and stale DataFrame. In the development version, a missing comma joined adjacent string literals. Rebuild/reload after correcting the schema. |
| Missing expected-count config keys | Use the actual current configuration schema; do not insert unexplained constants merely to silence validation. |
| Reference mismatch of +23,610 | A documented unresolved comparison, not a reason to alter the published reference or decode forever. |
| `TorchCodec was replaced after import` | Restart the Python session and install/verify the compatible CUDA wheel before importing TorchCodec. |
| CUDA available but CPU decoding used | Inspect actual fallback provenance; CUDA availability alone is insufficient. Alignment intentionally rejects fallback. |
| `PyNvDecoder` has no `Flush` | An API incompatibility in the earlier code. The reviewed pinned implementation uses the demuxer's checked EOS packet. |
| Existing CSV still triggers work | Existence alone is insufficient: inspect state completion, schema, hashes, input/engine fingerprint and retry settings. |
| A globally incomplete audit after some finished videos | Reuse verified individual checkpoints; the new drivers do not infer that all work is lost. |
| SSIM result near one is excluded | Inspect margin, genuinely missing in-range scores, permitted offsets, video consistency, and related-annotation rules. |
| Zero accepted at `0.99` with maximum near `0.975` | Expected consequence of the threshold, not failure to decode all frames. Use the documented working policy and inspect real pairs; do not invent accuracy. |
| Alignment cell finishes silently | Check `frame_alignment_define_only`; `True` defines functions only and does not auto-run the smoke test. |
| 24 mappings excluded with 36 `target_out_of_range` candidates | Inspect true boundaries and apply P1's score-completeness rule; retain all candidate rows and all real in-range failure checks. |
| New boundary columns absent from confirmed Parquet | P1 adds them to the mapping audit; join by mapping ID or explicitly extend the publisher projection. |
| nvJPEG format/hardware error | Strict A2 forbids CPU/hybrid fallback and conversion. Treat as capability/infrastructure failure, not proof of bad labels. |
| Changing scoring version to 3 does not load old results | Version is a fingerprint input, not a backend/cache selector; restore the correct v4 identity. |
| Candidate and video progress percentages disagree | Videos have different target counts; candidate attempts and completed futures are different units. |
| GPU memory remains occupied | Inspect live versus reserved allocations and native decoder resources; memory graphs alone do not prove a leak. |
| Local disk fills | Concurrent compressed-video copies accumulate. Reduce active video workers or deliberately alter staging policy; do not delete raw Drive data. |
| GPU or memory infrastructure error | The run is incomplete. Do not classify unprocessed annotations as rejected medical data. |
| Export rejects current run ID | Rerun alignment with the intended configuration; compatible scores may be reused to generate a current final state. |
| Git checkout dirty or histories diverged | Resolve deliberately. The exporter does not reset, stash, rebase, or force-push to hide the problem. |
| Export rejects a hash | Identify whether the original report changed, the state is stale, or the compared object is compressed. Do not disable checks to publish inconsistent data. |

The most important debugging principle is to distinguish **source-content issues**, **environment/API issues**, **cache-compatibility issues**, and **publication issues**. They have different remedies and should not all lead to deleting the dataset or rerunning every expensive stage.

## 23. Known limitations and remaining integration work

### 23.1 The technical audit is not medical validation

The pipeline does not independently validate class labels, bounding-box medical correctness, patient histories, diagnosis, or treatment decisions. Similarity evidence concerns image/video correspondence. Future reports must not describe this as clinician validation or a deployable clinical safety system.

### 23.2 The reference mismatch remains unresolved

The local counts are internally consistent under the reported checks, but the project has not established the cause of the +23,610 difference. Aggregate agreement is not proof that every distributed file is identical to a publication's original release. File-level authoritative references or a documented release comparison would be needed for a stronger explanation.

### 23.3 The strict filter can change the dataset

The accepted subset may have a different class/video distribution from `df_clean`. The current code preserves exclusions and reasons, but it does not perform a threshold-sensitivity study, a visual adjudication exercise, or a statistical assessment of selection bias.

These are recommended methodological follow-ups before presenting experiments on the subset as directly comparable to experiments on the entire original image dataset. The newer `0.92` policy has a much higher observed retention than `0.96`, but no independently verified alignment precision has been supplied. Threshold sensitivity, manual examination, hard-negative comparisons and class/video distribution checks remain useful. The approximately 0.953 SSIM distribution persists after switching JPEG decoders; its physical cause is still unresolved. In particular, ambiguous near-duplicate neighbours should not automatically be described as annotation errors.

### 23.4 Reproducibility is improved, not absolute

The code saves configuration, versions, fingerprints and source identities. It does not establish bitwise-identical behaviour across arbitrary future software and hardware. Not every library is pinned by the general installation cell, and not all possible dependencies are included in every cache signature.

The current Sector 2 configuration declares `seed=42`, but this snapshot does not contain a corresponding general cell applying that seed to Python, NumPy and PyTorch globally. The SSIM self-check uses its own fixed local generator. Because the main alignment is exhaustive rather than sampled, the seed is not its frame-selection mechanism. A claim that all random generators or all GPU operations were made deterministic would exceed the code.

### 23.5 Configuration cleanup still worth doing

The initial Git synchronization cell imports a literal Sector 1 `CONFIG` and records a source commit, then the Sector 2 configuration cell replaces the dictionary. That is workable, but it is not automatic inheritance of every loaded field. Source-commit provenance remains in separate variables/output unless explicitly carried into the saved configuration.

`CONFIG['results_dir']` is currently not independently used by `prepare_phase2_dirs()`, which derives results from `output_dir`. The manifest's force/cache-version controls also remain cell-level constants rather than all residing in `CONFIG`. These details should be documented or consolidated in a future cleanup, not silently described as already centralized.

The legacy key `expected_unlabelled_frames=4_694_266` is also retained in the configuration. It is not the reference total for the 74 `fully_unlabelled` videos: that group uses `2_785_829` in `decode_audit_published_reference_frames`. Numerically, the legacy value equals `4_741_504 - 47_238`; this arithmetic does not redefine the group-level count or independently establish what every unannotated frame represents. The current decode comparison uses the explicit category reference dictionary.

The original opening Markdown in the submitted notebook still says full decoding “is being added,” although the code and saved output already contain the completed NVDEC count audit. This README reflects the implemented state and flags that heading as stale text.

### 23.6 GPU and API limitations

Exact indexing is a library service with a startup scan. Hardware support depends on the installed driver, native libraries, codec and selected backend. Bounded batches reduce memory demand, but the reservation estimate is not a hard total-device limit. No optimal worker/batch configuration has been demonstrated by a controlled benchmark in the supplied sources.

The PyNvVideoCodec EOS behaviour is tied to the checked binding used by this code. Alignment verifies CPU-fallback reporting and rejects a package that cannot supply it. A2 additionally requires a supported hardware JPEG path and rejects reference formats outside `.jpg`/`.jpeg`; a filename guard alone is not a guarantee that a JPEG's internal encoding is supported. Future upgrades require representative native tests and appropriate scoring identity updates, not merely an import-success check.

The small supplied nvJPEG smoke test and development screenshots do not establish performance, correctness, or support for every future GPU and JPEG variant. The current notebook's OpenCV manifest probe and optional count-audit software fallback remain distinct from the strict alignment decoding requirement.

### 23.7 Caches and storage do not provide cryptographic authenticity

Metadata fingerprints can miss same-size/same-mtime source substitutions. Output hashes can detect changes relative to stored hash values, but a party changing both data and its state can create a new matching pair. These files are integrity/reproducibility aids, not signed trusted attestations.

Local source staging and Python cleanup are not durable storage guarantees. Permanent outputs belong on Drive or in the verified Git publication; working copies and tensors are disposable.

### 23.8 Later phases still need their own validation

Temporal window construction, split generation, leakage control, training, cross-dataset testing, retrieval, model reasoning, contradiction checks and guarded decision support are not delivered by these cells. Sector 2 provides inputs and traceability for that work. It does not make those later components correct by association.


## 24. Maintaining this README during export

### 24.1 Important exporter interaction

The current export script generates the bundle's `README.md` from the short string `_S2_BUNDLE_README`. Manually replacing the GitHub README once is therefore insufficient: a later run can overwrite it with the short built-in version. [G1, `_s2_prepare_bundle`]

To publish this full document consistently, save it on Drive as:

```text
DIRS["reports_dir"]/README_Phase2_Sector2.md
```

Then add the following **after** the built-in `_S2_BUNDLE_README` declaration and **before** the final `publish_sector2_github(...)` call:

```python
comprehensive_readme_path = (
    Path(DIRS["reports_dir"]) / "README_Phase2_Sector2.md"
)
_S2_BUNDLE_README = comprehensive_readme_path.read_text(encoding="utf-8")
```

This is a proposed integration edit, not a modification already applied to the reviewed exporter. The existing export code will then write the supplied text as `README.md`, include it in the bundle checksums, and stage it with the other selected artifacts. The source document must exist before running publication; do not silently fall back to a shorter README when the requested full document is missing.


## 25. Function reference

This reference explains the purpose, important inputs, outputs, and side effects of the main functions and helpers in the reviewed implementation. It is a navigation guide, not a replacement for their source. A helper being defined does not mean it is executed in every run: a valid cache bypasses the expensive construction or scoring path.

Locations below use one-based lines inside the specified notebook cell for the unchanged N1 functions. **Sections 25.6–25.9 use standalone A2 file line numbers before P1**; G1 numbers refer to the export script. The boundary insertion shifts lines and changes `decide_alignment()` behaviour without rewriting the attached A2 baseline. These are navigation references, not claims that a notebook has been replaced automatically. Private helpers remain implementation details; downstream consumers should prefer the saved data contracts.

### 25.1 Environment, configuration, and directories

| Function | Location | Responsibility and behaviour |
|---|---|---|
| `select_cuda_wheel` | N1, cell 5, lines 17–36 | Selects a TorchCodec release from the installer's declared PyTorch compatibility mapping and searches the appropriate CUDA wheel index. It is an installation-time compatibility decision, not a test that an actual video decodes on the GPU. |
| `mount_storage` | N1, cell 12, lines 1–37 | Examines the configured storage backend and mounts Google Drive when running the supported Colab/Drive configuration. Returns the storage context needed by later cells. It neither copies videos into local storage nor downloads a missing dataset. |
| `prepare_phase2_dirs` | N1, cell 14, lines 1–162 | Resolves the persistent root, declared dataset directories, derived output directories and raw-data references. Creates the required directory skeleton and returns the `DIRS` dictionary of `Path` objects. Its internal `resolve_path` helper interprets relative paths against `storage_root`; absolute paths are preserved. It does not prove that the referenced input files exist. |

Run-ID creation and configuration JSON saving are cell-level statements, not separate processing functions. Their placement matters: the final Sector 2 dictionary is created first, `timezone` and `run_id` are added, then the complete dictionary is serialized. There is no need to infer runtime settings from an earlier saved configuration when the current effective configuration is available.

### 25.2 Git bootstrap and inherited publication helpers

The initial repository cell has two responsibilities that should not be confused. Repository preparation and literal configuration loading are executed there. Several older publication helpers are also defined, but the final Sector 2 publication is performed by `publish_sector2_github()` from `G1`.

| Function | N1 cell 8 lines | Responsibility and status |
|---|---:|---|
| `run_git` | 25–43 | Runs Git as an argument list rather than through a shell; captures results and controls accepted return codes. It is the bootstrap cell's command wrapper. |
| `prepare_git_repository` | 46–70 | Creates or checks the local clone and fast-forwards a clean matching repository. Produces the local repository path. This operation can contact the remote and update the checkout. |
| `load_config_from_notebook` | 73–108 | Parses the source notebook and extracts a literal `CONFIG` dictionary without executing its code cells. Records the source context used by the bootstrap. It does not run ingestion or reconstruct derived runtime values. |
| `file_sha256` | 111–117 | Streams file bytes into a SHA-256 digest without reading an entire large file into RAM. Used by the inherited publication utilities. |
| `collect_publishable_outputs` | 120–155 | Produces a plan of eligible files from configured output directories. Planning alone does not copy, stage, commit or push files. |
| `build_dataset_audit_github_files` | 158–185 | Derives the inherited dataset-audit publication filenames from the export specification. Avoids independently guessing what Sector 1 wrote. |
| `collect_publishable_dataset_audit` | 188–254 | Plans inherited dataset-audit publication, preserving audit files already present remotely under that older policy. This is different from the current Sector 2 bundle's content-update policy. |
| `authenticated_push` | 257–275 | Supplies authentication through an ephemeral environment rather than a token-bearing remote URL. It is part of the older helper family. |
| `publish_phase2_outputs` | 278–393 | Older general Phase 2 publication routine. Defined in the bootstrap cell, but not the active final export path documented in section 21. Do not run it as a second publication mechanism without deliberately reviewing overlapping destinations. |

The revised exporter uses its own `_s2_*` helpers rather than silently depending on these earlier helper definitions. That separation prevents a change in the inherited publisher's behaviour from redefining the Sector 2 artifact contract.

### 25.3 Inventory, probe, manifest, and manifest validation

| Function | Location | Responsibility and side effects |
|---|---|---|
| `empty_video_probe` | N1, cell 22, lines 16–43 | Returns the complete technical schema with false readability flags and missing numerical properties. It makes an unopened container representable as a record rather than silently dropping it. No file I/O is performed by this helper itself. |
| `probe_video` | N1, cell 22, lines 46–172 | Opens one video with OpenCV, reads reported FPS/count/dimensions, attempts one frame read and releases the capture. Validates numerical properties and computes the estimated container duration. It returns a dictionary, not a full decode audit. |
| `_manifest_walk_error` | N1, cell 23, lines 27–28 | Propagates an inventory traversal error. An unreadable directory must not be interpreted as an empty directory. |
| `_manifest_inventory` | N1, cell 23, lines 31–72 | Re-enumerates configured video files, validates identity/path uniqueness and collects source metadata, including category, size and modification time. It does not open video containers or hash full source contents. Its relative paths use the persistent storage root. |
| `_manifest_sha256` | N1, cell 23, lines 75–77 | Computes a checksum for the saved manifest CSV used by reuse validation. This is a hash of output bytes, distinct from the structured input fingerprint. |
| `_manifest_atomic_write` | N1, cell 23, lines 80–93 | Writes a temporary sibling and then replaces the destination. Supports the manifest's persistence protocol without exposing a partially written new file under the final filename. |
| `_manifest_restore_types` | N1, cell 23, lines 96–114 | Restores manifest strings, booleans and nullable integer/numerical columns after a CSV load. Keeps fresh and cached DataFrames compatible rather than allowing CSV type inference to redefine the schema. |
| `build_video_record` | N1, cell 23, lines 117–142 | Combines one inventory item with its `probe_video()` result. Produces identity, location, size, annotation-category and technical fields for one physical video. This is where the video is opened during a fresh manifest build. |
| `build_video_manifest` | N1, cell 23, lines 145–162 | Calls the record builder over the validated inventory and returns the stable ordered manifest. Called only when the manifest cannot be reused or rebuilding is forced. It is not a decoder for the entire dataset. |
| `validate_video_manifest` | N1, cell 26, lines 1–544 | Checks the completed table's required columns, identifier/path validity, file existence, minimum inventory, category/key consistency and saved technical properties. Compares the reported frame sum with the configured reference using an explicit warning. Returns the original table without decoding frames, changing rows, or saving a new audit. |

The initial `video_index` and `video_inventory` construction in cell 20 is implemented as cell-level statements. Its important output is the verified relationship between metadata `video_key` values and physical filename stems, not a reusable normalization function hidden in memory.

### 25.4 Full-decode identity, validation, and checkpoint helpers

All locations in this subsection refer to **N1, cell 25**.

| Function | Lines | Responsibility and important contract |
|---|---:|---|
| `_decode_canonical_json` | 22–27 | Serializes structured data deterministically for signatures. Provenance timestamps are not mixed into the source identity merely because a run is new. |
| `_decode_digest` | 30–31 | Applies SHA-256 to the canonical structured representation. This helper does not by itself read a video. |
| `_decode_integer` | 34–56 | Validates integer-valued fields rather than silently accepting fractional, missing, nonfinite or inappropriate values. Protects frame counts and checkpoint identity fields. |
| `_decode_source_identity` | 59–73 | Builds a source identity from the persistent source record. Temporary local staging paths are deliberately excluded. |
| `decode_source_signature` | 76–89 | Combines physical source identity and the selected decoder-method signature. Excludes category membership and the run ID because neither changes the video's physical frame count. |
| `validate_decode_record` | 92–173 | Checks the schema, values and consistency of a terminal decode attempt. A terminal record may legitimately report an error; validation is not equivalent to certifying successful decoding. |
| `decode_record_needs_retry` | 176–182 | Applies the explicit retry policy to a validated saved attempt. Prevents an unresolved failed video from silently triggering a full repeat on every run. |
| `_decode_atomic_json` | 185–195 | Writes a JSON checkpoint through a temporary file and replacement. |
| `write_decode_checkpoint` | 198–223 | Persists one terminal record with source/method identity, record checksum and checkpoint-payload checksum. It must not be called to publish a `KeyboardInterrupt` or pending attempt as complete. |
| `read_decode_checkpoint` | 226–256 | Loads a checkpoint, verifies its structure and checksums, validates its record and source signature, then refreshes the current annotation category. A global report's incomplete state does not automatically invalidate a valid individual checkpoint. |
| `decode_json_digest` | 259–260 | Publicly named structured-digest wrapper used by report identity. |
| `decode_file_sha256` | 263–268 | Hashes saved report bytes in chunks. It is used for output integrity, not as an implicit whole-video content hash. |
| `decode_atomic_write` | 271–284 | Publishes CSV or JSON report files through the temporary-write mechanism. |

Two checksums in a checkpoint have distinct jobs. The record checksum detects a changed result record; the checkpoint checksum detects a changed surrounding payload. Both still depend on the stored state being a trustworthy reference. They are not digital signatures from the dataset's authors.

### 25.5 Full-decode execution helpers

All locations in this subsection refer to **N1, cell 25**.

| Function or exception | Lines | Responsibility and important contract |
|---|---:|---|
| `DecodeBackendAPIError` | 290–291 | Distinguishes an incompatible binding or programming/API problem from an unreadable source video. Such an error should not be concealed as an ordinary CPU fallback. |
| `_require_decode_api` | 294–301 | Checks the required binding attributes before submitting compressed packets. |
| `ensure_gpu_available` | 304–368 | Checks CUDA device/driver-library availability and reports the selected GPU. It is a startup check for uncached NVDEC work, not proof that every input codec is supported. |
| `_decode_positive_header_count` | 371–377 | Converts a valid positive container header count into an integer and preserves unknown/zero values as missing. It never substitutes an invented count. |
| `_decode_header_probe` | 380–400 | Reads container metadata with PyAV without decoding video pixels. Supplies the diagnostic reported count and header error independently of the full count. |
| `_decode_result` | 403–421 | Creates the standardized result record and its initial state. Makes backend output comparable without pretending both backends share every internal step. |
| `_decode_progress` | 424–431 | Creates the optional per-video frame-progress display. Display state is not the checkpoint. |
| `_decode_finish` | 434–441 | Finalizes measured duration and terminal result fields for the attempt. |
| `_decode_with_nvdec` | 444–533 | Demuxes packets, submits them to the pinned PyNvVideoCodec decoder, counts returned frames, processes the checked terminal EOS packet and synchronizes before finalizing. Counts the actual decoder outputs rather than relying on a header count or an expected loop length. |
| `_decode_with_pyav` | 536–577 | Performs the explicit software decode path, including delayed outputs produced when decoding drains. Used when selected or when an allowed video-level fallback is required. |
| `decode_video_gpu` | 580–618 | Coordinates the header probe, requested backend and fallback policy for one source. A CPU fallback restarts the entire video and keeps GPU-attempt provenance separate; it does not add partial GPU and CPU totals. |
| `scan_decode_inventory` | 634–664 | Creates a fresh inventory with source metadata and current category membership. Scans files, not frame pixels. |
| `_decode_check_source_unchanged` | 667–670 | Rechecks the source identity after work so an input changed during counting is not committed as if it were stable. |
| `_decode_installed_version` | 673–677 | Retrieves an installed package version for provenance without implying that the package has successfully decoded data. |
| `make_decode_signature` | 680–691 | Describes the counting algorithm version, backend and relevant installed package versions. Does not include a new timestamp or unrelated acceptance thresholds. |
| `local_decode_source` | 695–712 | Context manager that optionally copies the current compressed video into the local working directory, yields that path and cleans up the temporary copy. Persistent source paths remain in the audit identity. |
| `_decode_existing_report_matches` | 715–726 | Checks whether already-published aggregate reports match the current aggregate fingerprint and file hashes. Avoids rewriting identical completed reports. |
| `_decode_preserve_legacy_outputs` | 729–744 | Preserves prior-version report files when the new audit first replaces them. This is not automatic import of their old records into the current checkpoint schema. |
| `run_gpu_decode_audit` | 747–931 | Coordinates inventory, identity, checkpoint reuse, deliberate retries, GPU initialization when needed, per-video processing and final report publication. It walks the full video inventory one video at a time in this snapshot. The concurrent all-video implementation belongs to alignment, not this count driver. |

The central execution relationship is:

```text
run_gpu_decode_audit
    → scan_decode_inventory
    → decode_source_signature + read_decode_checkpoint
        → reusable terminal attempt: use saved count
        → work required:
            ensure_gpu_available, when the NVDEC path needs initialization
            local_decode_source
            decode_video_gpu
                → header probe
                → NVDEC, or permitted complete-file software fallback
            source-stability check
            write_decode_checkpoint
    → aggregate summaries and checked final report publication
```

### 25.6 Alignment environment and cache helpers

All locations in this subsection refer to **A2 before the P1 boundary edit**.

| Function or class | Lines | Responsibility and important contract |
|---|---:|---|
| `AlignmentCancelled` | 91–92 | Signals cooperative cancellation between units of work. It is not an image-quality finding. |
| `AlignmentGPUError` | 94–96 | Represents GPU/backend infrastructure failure. Unprocessed images must not be rejected as bad medical examples because infrastructure failed. |
| `_alignment_digest` | 99–101 | Canonically hashes structured inputs used in alignment identities. |
| `_alignment_file_digest` | 104–109 | Streams output or optional source bytes into SHA-256. |
| `_alignment_atomic_write` | 112–132 | Writes CSV, JSON or Parquet through a temporary sibling. The driver controls the separate final state marker. |
| `_alignment_cancelled` | 135–137 | Checks the shared cancellation event and raises the cancellation exception. |
| `_alignment_settings` | 140–177 | Validates configured offsets, thresholds, devices, worker counts, batch sizes and resource options. Produces the effective settings used by the driver. |
| `_alignment_cuda_backend` | 180–190 | Resolves the configured TorchCodec backend selection. The `auto` policy checks the declared candidate names supported by the installed package; it is not a universal promise that the same backend name exists in future releases. |
| `_alignment_environment` | 193–229 | Records video backend, GPU/device/driver, PyTorch/CUDA/TorchCodec, reference decoder/backend and reconstruction settings, image-library versions/native hashes and scoring arithmetic. JPEG decoding changes are part of the score identity, not mere concurrency changes. |
| `_alignment_signature` | 232–247 | Combines the canonical per-video mapping table, evaluated offsets, environment and source file identities. Optionally hashes source contents. Thresholds, timestamps and concurrency tuning are not included merely because they changed. |
| `_alignment_validate_candidates` | 250–279 | Requires the exact expected set of mapping/offset/index identities, validates field types and terminal statuses, and rejects invalid scored values or incomplete candidate coverage. A low score is valid audit data; a missing attempted record is not. |
| `_alignment_read_cache` | 282–292 | Requires a complete cache state, matching input fingerprint and matching candidate CSV hash; then validates the full candidate table. Returns reusable scores, not an unconditionally accepted training subset. |
| `_alignment_exact_integer` | 769–784 | Validates indices without truncating fractional values and rejects booleans, missing/nonfinite values and int64 overflow. |

A cache hit still requires the decision rules to be applied to the saved scores. This is why changing `min_ssim` or `min_margin` can change accepted rows without decoding pixels again.

### 25.7 Alignment resource, source-read, and CUDA-decode helpers

All locations in this subsection refer to **A2 before the P1 boundary edit**.

| Function or class | Lines | Responsibility and important contract |
|---|---:|---|
| `_AlignmentMemoryBudget` | 295–321 | Shared weighted reservation system for estimated temporary tensors. Its `reserve` context manager waits for enough budget and releases the reservation on exit. It is not a lock around all GPU work and does not account for every native decoder allocation. |
| `_alignment_reference_library_versions` | 330–341 | Collects installed nvImageCodec/nvJPEG module and package versions for the reference-decoder signature; it does not prove successful hardware decoding. |
| `_alignment_reference_native_hashes` | 344–360 | Hashes pip-distributed native JPEG/image-codec library files for provenance. Does not claim to cover every system library. |
| `_alignment_reference_context` | 363–393 | Reuses a thread-local context when device and reconstruction settings match, otherwise builds a HW_GPU_ONLY nvImageCodec decoder, DecodeParams and a dedicated image-worker CUDA stream. |
| `_alignment_reference_format_guard` | 461–472 | Prints reference-extension counts and rejects non-JPEG paths before full scoring. Does not decode pixels and is not proof that every JPEG variant is supported. |
| `_alignment_reference` | 396–458 | Decodes one original JPEG through the worker's HW-only nvJPEG context, verifies an on-device CUDA uint8 RGB buffer, imports via DLPack, produces CHW layout and synchronizes the worker stream for safe handoff. No OpenCV or CPU/hybrid pixel-decoding fallback. Unsupported decoding raises an infrastructure/capability error. |
| `_alignment_local_video` | 476–514 | Acquires a shared copy slot, checks available local disk while accounting for concurrent pending copies, creates a temporary compressed-video copy, releases the copy slot and yields the file for processing. Deletes the copy on normal context exit. Does not copy decoded frames to disk. |
| `_alignment_check_gpu_decode` | 517–526 | Checks output tensor dtype/layout/device and the decoder's fallback-reporting state. Rejects unverifiable provenance or CPU fallback even when an output could be copied to CUDA afterward. |
| `_alignment_targets_cuda` | 529–571 | Requests a sorted deduplicated target-index batch, verifies GPU outputs and associates returned frames with their requested indices. Isolates ordinary batch-read errors by bisection; infrastructure and memory failures propagate. Establishes stream dependencies for safe tensor use. |
| `_alignment_new_cuda_stream` | 575–585 | Creates a worker-local PyTorch stream and synchronizes that stream at cleanup. Does not impose a GPU-wide synchronization barrier between every pair of videos. |
| `_alignment_score_cuda` | 588–710 | Opens a CUDA exact-indexed decoder for one video's mapping group, performs a warm-up provenance check, requests target batches, overlaps reference-image reads, computes SSIM batches, retains every terminal candidate result and records timings. Returns the candidate table and timing dictionary. |
| `_alignment_video_job` | 713–768 | Implements the per-video transaction: compute signature, reuse or mark incomplete, calculate missing results, validate, recheck sources, save the CSV and complete JSON, restore global mapping IDs and return a performance/provenance record. Each worker writes only its own cache files. |

The last two functions are the key distinction between **within-video batching** and **between-video concurrency**. `_alignment_score_cuda()` is a bounded work unit for one video; the driver launches many `_alignment_video_job()` calls so different videos can reach decoding and SSIM concurrently.

### 25.8 Alignment input preparation and SSIM

All locations in this subsection refer to **A2 before the P1 boundary edit**.

| Function | Lines | Responsibility and important contract |
|---|---:|---|
| `prepare_alignment_inputs` | 787–990 | Copies the metadata, validates logical identifiers/frame numbers, resolves a unique physical video through the manifest and a class-aware physical image path through existing metadata or Sector 1 inventory. Returns all original annotations with mapping references and a deduplicated mapping table. Invalid inputs remain represented with reasons. |
| `_alignment_ssim_tensor_batch` | 993–1027 | Computes one SSIM score for every supplied reference/candidate pair using the defined full-resolution colour, uniform 7×7 metric. Uses inference mode, no autocast approximation, float32 window sums and float64 metric arithmetic. Returns the ordered scores to CPU after that batch completes. |
| `torch_ssim_uint8_pairs` | 1030–1088 | Validates image dtypes, layout and matching dimensions; groups compatible image sizes without resizing; divides them into bounded batches; calls the tensor-level metric; returns a score vector in the original pair order. Never reads a video file itself. |
| `_alignment_torch_device` | 1091–1100 | Resolves the metric's execution device and rejects an unavailable requested CUDA device. In the current full-CUDA driver, the selected device must be CUDA. |
| `validate_torch_ssim_backend` | 1103–1162 | Compares the implemented metric against installed scikit-image on ten deterministic synthetic pairs. Returns recorded numerical agreement or stops before dataset scoring. It does not test every real image, video-index identity, medical correctness, or performance. |

The important internal helpers inside `prepare_alignment_inputs()` are:

| Helper | Role |
|---|---|
| `text_value` | Preserves meaningful identifiers and identifies missing/blank values without treating the identifier itself as a number. |
| `frame_value` | Accepts finite nonnegative integral frame numbers and rejects unsupported values rather than rounding them. |
| `class_value` | Applies the same class-match-key spelling rule used by Sector 1 for physical inventory lookup. |
| `checked_path` | Resolves a source path, checks its declared root and regular-file existence, and caches repeated path checks during input preparation. Returns a structured status instead of silently dropping a row. |

These checks establish a usable comparison target. They are not substitutes for the actual visual comparison; a syntactically valid path and index can still identify the wrong image.

### 25.9 Alignment decisions, publication, and driver

| Function | A2 lines before P1 | Responsibility and important contract |
|---|---:|---|
| `decide_alignment` | 1165–1278 | Baseline A2 requires scores for all offsets. With P1, still requires all planned rows, but scores only for in-range targets and at least two alternatives; adds boundary diagnostics and preserves threshold/margin, permitted offsets and video consistency. Returns decisions without modifying source labels/files. See Section 13.10 for the exact edit. |
| `_alignment_publish` | 1280–1382 | Combines candidate tables and invalid-input records, calls the currently defined decision function, applies related-record consistency, keeps common-set offset means, joins selected fields back to annotations and writes outputs/state. P1 diagnostics survive in the mapping audit but require an explicit projection/join for confirmed Parquet. Existing confirmation_rule does not independently encode P1. |
| `run_full_frame_alignment_cuda` | 1385–1507 | Validates prerequisites, establishes the CUDA environment and resource budget, creates canonical per-video jobs, submits all jobs to the executor, collects completed futures without video-order blocking, saves performance/error reports and publishes final results only after successful completion. Coordinates cancellation and waits for workers before returning or propagating a failure. |

The principal call flow is:

```text
run_full_frame_alignment_cuda
    → validate settings, choose CUDA backend, record environment
    → prepare_alignment_inputs
    → build all annotated-video jobs
    → submit all jobs to the bounded executor
        each _alignment_video_job:
            → verify its source fingerprint and cache
            → reuse saved candidates OR:
                mark that cache incomplete
                _alignment_score_cuda
                    local compressed-video staging
                    CUDA exact-indexed decoder and GPU provenance check
                    target batches + shared HW-only nvJPEG reference tasks
                    CUDA SSIM on the worker's stream
                validate candidates and unchanged sources
                publish per-video CSV and complete state
    → collect results; record performance and any failures
    → _alignment_publish
        individual in-range boundary decisions (P1) + consistency rules
        original-annotation join + accepted/excluded partition
        all final artifacts + final complete state
```

A worker's checkpoint can finish well before the entire dataset. That is intentional: restartability is attached to independent physical videos, while the accepted-subset publication requires a consistent complete set of decisions.

### 25.10 Final GitHub export: source validation and bundle construction

All locations in this subsection refer to the **standalone `G1` script**. The same code is appended as cell 30 in `N2`.

| Function | G1 lines | Responsibility and important contract |
|---|---:|---|
| `_s2_sha` | 61–66 | Streams file bytes to compute SHA-256 for source verification and publication cataloguing. |
| `_s2_json` | 69–70 | Produces stable, readable JSON with sorted keys and rejects nonstandard NaN values. |
| `_s2_atomic_text` | 73–83 | Publishes text/JSON reports through a temporary sibling. |
| `_s2_relative` | 86–93 | Rejects absolute or unsafe publication paths, traversal components, `.git`, backslashes and control characters. |
| `_s2_source` | 96–107 | Resolves an approved artifact under its declared source directory and rejects symlink traversal or missing/non-file inputs. |
| `_s2_state` | 110–114 | Loads a dictionary state and requires `complete is True`. Mere file existence is not accepted. |
| `_s2_validate_states` | 117–160 | Verifies manifest/decode/alignment completion, required final output membership, SHA-256 values, current-run requirements and compatible alignment settings. Returns validated state context; never decodes source media. |
| `_s2_summary` | 163–210 | Reads saved audit tables and Parquet metadata, validates row/key consistency and computes compact numerical summaries. Does not trust an unrelated or stale in-memory `df_alignment_confirmed`. |
| `_s2_no_secrets` | 213–225 | Recursively checks configuration values for obvious credential content while allowing secret-name references. A guardrail, not a complete data-loss-prevention system. |
| `_s2_scan_bytes` | 228–237 | Checks selected source bytes for the active token and supported recognizable credential patterns. This cannot establish that every possible sensitive value is absent. |
| `_s2_prepare_bundle` | 240–344 | Selects the approved artifact families, verifies saved outputs, copies or deterministically compresses CSVs, applies file/bundle size limits and generates configuration snapshots, reports, README and the artifact catalogue. Works in a local preparation directory before staging the publication into Git. |

The short `_S2_BUNDLE_README` string is a data declaration between helper groups, not a function. Section 24 explains how to replace that text with this full README so subsequent exports do not restore the shorter built-in description.

### 25.11 Final GitHub export: repository and publication transaction

All locations in this subsection refer to **G1**.

| Function | G1 lines | Responsibility and important contract |
|---|---:|---|
| `_s2_repo_identity` | 405–413 | Parses and validates the configured repository identity used to check the local origin and authentication target. |
| `_s2_auth` | 417–436 | Makes the GitHub token available to child Git processes through an ephemeral environment. Does not store it in repository URLs, command arguments or a token file. |
| `_s2_git` | 439–451 | Runs a Git command with the controlled environment, captures output and raises on unsupported return codes with credential-aware reporting. |
| `_s2_clean` | 454–456 | Requires a clean checkout before the exporter modifies its scoped artifact tree. It does not automatically discard or stash unrelated work. |
| `_s2_refresh` | 459–493 | Verifies clone/origin/branch, fetches the remote, checks branch ancestry and permits only the supported fast-forward or exporter-owned local-ahead state. Stops on divergence instead of resetting history. |
| `publish_sector2_github` | 496–584 | Orchestrates validation, bundle preparation, repository synchronization, a displayed publication plan and optional `PUSH` confirmation. Stages only selected paths, verifies staged content, creates a scoped commit when needed, performs a normal push and writes a receipt for the actual outcome. |

A successful local commit and a successful remote push are distinct outcomes. The exporter must not report a failed push as published simply because the commit object exists locally. Likewise, a current source audit is not inferred from a successful Git operation: scientific artifact validation happens before the Git transaction.

## 26. Source map and external references

### 26.1 How to read the references

Source references such as `[N1, cell 25]` identify implementation evidence, not a claim that this documentation reran the source. The source map deliberately distinguishes submitted notebooks, a generated extension, auxiliary implementation files, and discussion history. Function names and cell numbers make the references useful even when a notebook is moved into a differently named repository directory.

External references provide narrowly scoped explanations of library behaviour. They do not establish this project's measured counts, final filter results, successful publication, or performance. Those claims must come from the project artifacts. Documentation for a different library version is background, not permission to assume the installed binding has an identically named API. Existing external references below were retained from D0; this revision is a comparison of supplied project sources, not a new web verification. The nvJPEG implementation/setup claims are grounded in A2 and its supplied A3 guide.

<a id="source-n1"></a>

**N1 — Original integrated Sector 2 notebook snapshot.** `phase2_02_video_frame_validation-2.ipynb`, 28 cells. Primary evidence for unchanged configuration/storage helpers, manifest, full-decode audit, validator and embedded counts. Cell 28 documents the earlier CUDA-video/OpenCV-reference alignment and is superseded for current alignment by A2 plus P1. The saved notebook itself does not contain the later results.

<a id="source-n2"></a>

**N2 — Sector 2 notebook with the export extension.** `phase2_02_video_frame_validation_with_github_export.ipynb`. First 28 cells match N1; the publication heading and code occupy cells 29–30. The appended publication cell has no saved successful execution result.

<a id="source-i1"></a>

**I1 — Sector 1 ingestion notebook.** `phase2_01_ingestion(1).ipynb`. Used for the upstream data contract, image inventory/class matching, duplicate-handling context, and saved output containing 47,239 cleaned annotation rows and 47,238 physical-image inventory rows. The `(1)` is the supplied file's name, not an additional project phase.

<a id="source-a1"></a>

**A1 — Historical concurrent CUDA alignment cell.** `frame_alignment_cuda_parallel_cell.py`. Documents the v3 method: CUDA video targets and SSIM with OpenCV reference-image loading. Retained for comparison, not the current reference-decoding implementation.

<a id="source-a2"></a>

**A2 — Strict hardware-JPEG alignment replacement.** `frame_alignment_cuda_nvjpeg_strict_cell.py`, 1,531 lines in the reviewed file. Supplies the v4 nvJPEG HW-only reference decoder, versioned cache, concurrency, SSIM, publisher and define-only gate. The attached file still contains the original all-offset score requirement; P1 is a separately supplied subsequent edit. Current alignment function ranges refer to A2 before P1.

<a id="source-a3"></a>

**A3 — Replacement guide and smoke test.** `frame_alignment_cuda_nvjpeg_strict.md` and `frame_alignment_nvjpeg_smoke_test.py`. Supply the GPU-only JPEG intent, setup sequence, capability restrictions and read-only small-test workflow. The guide was written before threshold relaxation and retains historical `0.99` wording; Section 6.5 and H2 document the newer working configuration.

<a id="source-p1"></a>

**P1 — Subsequent inline boundary-policy edit.** The conversation's modification to `decide_alignment()`: calculate `outside_video_mask` and `required_score_count` outside the margin conditional; retain the `expected_count`/offset-set audit check; require scores for in-range candidates only; require at least two; add three boundary diagnostics. The exact decision-chain excerpt is preserved in Section 13.10. It has no independent revised source-file checksum or completed post-edit run result in the supplied materials.

<a id="source-e1"></a>

**E1 — Standalone CUDA dependency installer.** `install_alignment_cuda_dependencies.py`. Supporting explanation of the separate installation/import responsibilities, full CUDA wheel-version comparison and runtime-restart guard. N1 cells 3–6 record the installation sequence used by the reviewed notebook.

<a id="source-g1"></a>

**G1 — Final Sector 2 export cell.** `sector2_github_export_cell.py`. Source for the selective export policy, artifact verification and compression, generated bundle, safe Git synchronization, scoped commit/push and receipts. This is the active publication design, distinct from older helpers still defined in the bootstrap cell.

<a id="source-h1"></a>

### 26.2 Snapshot checksums

These are SHA-256 hashes of the **documentation source files themselves** as reviewed. They are not the dataset source fingerprint, decoder checkpoint hashes, or GitHub artifact checksums. A changed notebook output can change a notebook-file checksum even if its Python source is unchanged.

| Source | SHA-256 |
|---|---|
| N1 | `726f500d9e3e005e962d9b1980c7beb5be03b621bf67a91e06439538a846e508` |
| N2 | `93c755f174436caa0eb7c28b270b170146e3fa74679462e63ba669449e1261b8` |
| I1 | `31dceddb0a400b5127afb0a2534205cf00bb2e1f2d987b24d38fb63c066c2b01` |
| A1 | `a0b5ac1334df1a7a3def24cfe6796bb89adfaaf6ae936a2a37404c40ad3bb23b` |
| E1 | `cdcf1a86091311133712a4d8038519cd40e915dd34c22079e7304dcd731904f9` |
| G1 | `3ee750b8e34ac67bc31c3284affd350619eb875508b1f466aaa8009dd8121a41` |
| A2 | `318d911243508436f21546e15304033d6c37c56ef9f1b068c4bac3d6acb8c8a4` |
| A3 guide | `8ee674432f400eec9fbea7112f88c3e7b31f69bf9343b85c5129113f697e75c3` |
| A3 smoke test | `7ef21330ad6bcd0fc4c6a62c783ee47f4070f64c4abb22cd04622be62c360f7c` |
| D0 | `ac6dad71d7e37714c044513ff4890c7349de8397ddbcfa626a8e4f0013c4389e` |

P1 has no file hash here because it was supplied as an inline edit. These checksum values identify the attached baseline files, not a claim that the latest notebook execution used those exact unmodified bytes.

### 26.3 Library and dataset references

<a id="source-r1"></a>

**R1 — Kvasir-Capsule dataset page.** [Simula: Kvasir-Capsule](https://datasets.simula.no/kvasir-capsule/). Used for the dataset attribution and the published labelled/unlabelled group counts associated with this audit's reference configuration. The page identifies the dataset publication: Smedsrud et al., *Kvasir-Capsule, a video capsule endoscopy dataset*, Scientific Data (2021), DOI `10.1038/s41597-021-00920-z`. The numbers used by code are recorded explicitly in `CONFIG`; this README does not silently replace them with a differently defined number from another source or release.

<a id="source-r2"></a>

**R2 — TorchCodec exact versus approximate indexing.** [TorchCodec 0.10: Exact vs. approximate mode](https://meta-pytorch.org/torchcodec/0.10/generated_examples/decoding/approximate_mode.html). Used for the distinction between the initial index-building scan and actual frame decoding. It does not prove that arbitrary target retrieval has no intermediate compressed-frame dependencies.

<a id="source-r3"></a>

**R3 — TorchCodec CUDA decoding.** [TorchCodec 0.10: CUDA decoding](https://meta-pytorch.org/torchcodec/0.10/generated_examples/decoding/basic_cuda_example.html). Background for device-resident frames, CUDA backend requirements, fallback reporting and potential CPU/GPU pixel differences. The installed snapshot reports TorchCodec 0.11.1+cu128; the code's runtime checks and recorded backend remain authoritative for that run.

<a id="source-r4"></a>

**R4 — Google Colab storage and runtime behaviour.** [Colab FAQ](https://research.google.com/colaboratory/faq.html). Used for the distinction between mounted Drive and temporary VM storage, and the rationale for reducing repeated small Drive operations. The copied filenames, cleanup behaviour and parallel copy limits are established by the project code.

<a id="source-r5"></a>

**R5 — PyTorch CUDA memory and stream semantics.** [PyTorch 2.11: CUDA semantics](https://docs.pytorch.org/docs/2.11/notes/cuda.html). Used for the distinction between allocated and reserved tensor memory and the general behaviour of CUDA execution streams. The custom weighted budget in this project is a separate application-level mechanism.

<a id="source-r6"></a>

**R6 — Reference SSIM implementation.** [scikit-image 0.25: `structural_similarity`](https://scikit-image.org/docs/0.25.x/api/skimage.metrics.html#skimage.metrics.structural_similarity). Defines the reference metric parameters used by the numerical self-check. The project's thresholds, candidate-set completeness requirement, allowed offsets, margin rule and video-consistency decisions are project policies, not guarantees supplied by this API.

### 26.4 Documentation maintenance rule

When the implementation changes, update the affected function descriptions, parameter tables, cache-compatibility explanation and output schemas together. When only results change, update the observed-results section from the completed saved artifacts and retain the configuration/run provenance. When the publication mechanism changes, verify that the exporter still incorporates this README rather than overwriting it with its short built-in template.

**The intended result is a traceable preparation pipeline:** source metadata is preserved; physical files and decoder outputs are checked; image-to-video correspondences are evaluated under explicit rules; incomplete or ambiguous evidence is retained as such; and later phases receive both usable data and the information needed to understand how that data was selected.
