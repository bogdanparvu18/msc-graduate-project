# Phase 2 — Sector 2 artifacts

This folder contains completed video/frame audit outputs and downstream metadata.
It does not contain the raw Kvasir-Capsule images/videos, model weights or runtime caches.
The current artifact set, file hashes and optional gzip encodings are listed in `artifact_catalog.json`.
Git history versions the data; timestamps do not duplicate entire result folders.

## Downstream inputs

- `data/metadata_clean.parquet`: original cleaned Sector 1 annotations; not replaced by filtering.
- `data/physical_image_inventory.csv` (possibly `.gz`): Sector 1 image-path lookup.
- `results/frame_alignment_confirmed.parquet`: aligned subset, including zero-based `aligned_frame_index`.
- `manifests/video_manifest.csv`: source video identifiers and container probe metadata.
- `results/decode_audit.csv`: observed decoded counts and decoder provenance.
- `results/frame_alignment_audit.csv`, candidates, exclusions, offset summary and state JSON: all decisions and their evidence.

```python
from pathlib import Path
import json, pandas as pd

bundle = Path("outputs/phase2/sector2")  # relative to your repository root
catalog = json.loads((bundle / "artifact_catalog.json").read_text())
paths = {r["logical_path"]: bundle / r["path"] for r in catalog["artifacts"]}
df_clean = pd.read_parquet(paths["data/metadata_clean.parquet"])
df_alignment_confirmed = pd.read_parquet(paths["results/frame_alignment_confirmed.parquet"])
video_manifest = pd.read_csv(paths["manifests/video_manifest.csv"], dtype={"video_key": "string"})
candidates = pd.read_csv(paths["results/frame_alignment_candidates.csv"], dtype={"video_key": "string"})
```

Pandas reads `.csv.gz` automatically. The data are not all-to-all matches: each image
is tested only against candidate frames in its own video. Acceptance is a technical
filter under the documented rules, not proof that excluded labels are medically wrong.

## Paths and provenance

Source absolute paths are preserved as audit evidence; they are NOT portable by themselves.
In `video_manifest.csv`, `video_relpath` is relative to CONFIG's `storage_root`.
In `decode_audit.csv`, `video_relpath` is relative to `DIRS["dataset_root_dir"]`.
In the Sector 1 image inventory, `relative_path` is relative to `DIRS["raw_data_dir"]`.
To read actual pixels in another runtime, mount/download the original dataset and rebase
these paths. GitHub alone does not provide raw videos or images.
For confirmed `alignment_image_path` / `alignment_video_path`, replace the saved
CONFIG storage-root prefix with your new storage root, retaining the remaining path.

State JSONs are copied unchanged. Their output hashes refer to ORIGINAL report bytes.
For gzipped reports, decompress before comparing to those hashes; the catalog also
stores the hash of the compressed Git artifact. Do not copy these state JSONs back as
working decoder caches: the per-video caches remain on Drive.

## Source attribution

Kvasir-Capsule dataset: https://datasets.simula.no/kvasir-capsule/
Dataset paper: https://doi.org/10.1038/s41597-021-00920-z
These exports derive from the dataset and this project's processing, not a new official
dataset release. Preserve source attribution and applicable source terms when reusing them.
