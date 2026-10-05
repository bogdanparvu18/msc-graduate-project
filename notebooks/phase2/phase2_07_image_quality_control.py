"""Phase 2 Sector 7. Standalone Colab-compatible readers, pure QC and explicit I/O."""

from pathlib import Path, PurePosixPath
from datetime import datetime, timezone
from types import CodeType, FunctionType
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from urllib.parse import unquote, urlsplit
import copy
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import shutil
import subprocess
import tempfile

import cv2
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from IPython.display import display

try:
    from google.colab import drive, userdata
except ImportError:
    drive, userdata = None, None


BOOTSTRAP = {
    "storage_root": "/content/drive/MyDrive/MMVQA_Clinical",
    "mount_google_drive": True,
    "sector3_dir": None,
    "sector4_dir": None,
    "sector5_dir": None,
    "sector6_dir": None,
    "sector2_results_dir": None,
    "sector2_manifests_dir": None,
    "sector2_configs_dir": None,
}

S7_DEFAULTS = {
    "qc_enabled": True,
    "qc_fov_radius_fraction": 0.95,
    "qc_underexposed_pixel_threshold": 20,
    "qc_overexposed_pixel_threshold": 240,
    "qc_specular_value_threshold": 240,
    "qc_specular_saturation_threshold": 30,
    "qc_blur_quantile": 0.01,
    "qc_contrast_quantile": 0.01,
    "qc_brightness_low_quantile": 0.01,
    "qc_brightness_high_quantile": 0.99,
    "qc_threshold_fit_split": "train",
    "qc_workers": 8,
}

# None inherits the saved effective S6 setting, then uses S7_DEFAULTS if absent.
# The train calibration and enabled QC are explicit Sector 7 choices.
SECTOR7_SETTINGS = {
    **{key: None for key in S7_DEFAULTS},
    "qc_enabled": True,
    "qc_threshold_fit_split": "train",
    "qc_workers": 8,
    "local_output_dir": "outputs/phase2/sector7",
    "export_dir": "outputs/phase2/sector7",
    "push": True,
    "dry_run": False,
    "github_overrides": {"token_secret_name": "GITHUB_TOKEN"},
}


# Checked readers copied from persisted Sector 6 contracts.
# No notebook cell execution, video extraction, writes, or import side effects.

# All imports are supplied by the single Sector 7 imports cell.

SECTOR5_DEFAULTS = {"segment_max_gap_frames": 1, "minimum_verified_segment_frames": 2}

P3_NULL = "\\N"

S3_EXECUTION_METADATA_KEYS = frozenset({
    "run_id", "run_context", "run_datetime", "run_timestamp", "run_date", "run_time",
    "timestamp", "date", "time", "execution_date", "execution_time",
    "created_at", "created_at_utc", "created_utc", "saved_at", "saved_at_utc", "saved_utc",
    "started_at", "started_at_utc", "started_utc", "finished_at", "finished_at_utc", "finished_utc",
    "generated_at", "generated_at_utc", "generated_utc", "updated_at", "updated_at_utc", "updated_utc",
    "modified_at", "modified_at_utc", "modified_utc",
    "source_commit", "source_commit_sha", "source_commit_hash", "source_revision", "config_source_commit",
    "commit", "commit_sha", "head_commit", "current_commit", "repository_commit",
    "publication_receipt", "receipt", "last_publication",
    "source_configs", "config_provenance", "config_conflict_report", "lineage",
    "source_notebooks", "source_config_provenance", "config_sources", "literal_notebooks", "literal_configs",
    "input_sha256", "state_sha256", "upstream_states", "upstream_alignment_run_id",
    "notebook_path", "notebook_sha256", "source_notebook_path", "source_notebook_sha256",
    "source_config_path", "source_config_sha256", "source_snapshot_path", "source_snapshot_sha256",
    "config_snapshot_path", "config_snapshot_sha256", "snapshot_path", "snapshot_sha256",
    "source_config_file", "source_config_hash", "source_snapshot_file", "source_snapshot_hash",
    "config_snapshot_file", "config_snapshot_hash", "snapshot_file", "snapshot_hash",
    "config_inheritance_policy", "taxonomy_validation_policy",
})

def _s3_semantic_json_value(value):
    """Pure conversion to canonical JSON-compatible values; no I/O or clock access."""
    if value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, dict):
        return {str(key): _s3_semantic_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_s3_semantic_json_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        values = [_s3_semantic_json_value(item) for item in value]
        return sorted(values, key=lambda item: json.dumps(item, sort_keys=True))
    if isinstance(value, (Path, PurePosixPath)):
        return str(value)
    if isinstance(value, np.generic):
        return _s3_semantic_json_value(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        raise ValueError("Semantic configuration must not contain NaN or infinite values.")
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f"Unsupported semantic configuration value: {type(value).__name__}")

def _s3_semantic_json(value):
    return json.dumps(_s3_semantic_json_value(value), ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)

def _s3_semantic_digest(value):
    return hashlib.sha256(_s3_semantic_json(value).encode("utf-8")).hexdigest()

def sector3_semantic_config(config):
    """Project authored effective values, excluding execution/source bookkeeping only.

    Full source snapshots and current execution metadata remain available in the
    saved envelope. Authored paths, thresholds, publication settings (including
    push/dry_run), and unknown substantive config keys retain their meaning.
    """
    if not isinstance(config, dict):
        raise TypeError("Sector 3 semantic configuration requires a dictionary.")
    def project(value):
        if isinstance(value, dict):
            return {str(key): project(item) for key, item in value.items()
                    if str(key).lower() not in S3_EXECUTION_METADATA_KEYS}
        if isinstance(value, (list, tuple)):
            return [project(item) for item in value]
        return _s3_semantic_json_value(value)
    return project(config)

def sector3_semantic_config_fingerprint(config):
    """Stable identity of authored configuration, independent of execution time."""
    return _s3_semantic_digest(sector3_semantic_config(config))

def _p3_json_default(value):
    if value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, pd.DataFrame):
        return {"type": "dataframe", "columns": list(value.columns),
                "dtypes": {str(column): str(dtype) for column, dtype in value.dtypes.items()},
                "records": value.astype(object).where(value.notna(), None).to_dict("records")}
    if isinstance(value, (Path, PurePosixPath, datetime, pd.Timestamp)):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (set, tuple)):
        return sorted(value) if isinstance(value, set) else list(value)
    raise TypeError(f"Unsupported Sector 3 JSON value: {type(value).__name__}")

def _p3_json_text(value):
    return json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False,
                      allow_nan=False, default=_p3_json_default) + "\n"

def _p3_json_hash(value):
    return hashlib.sha256(_p3_json_text(value).encode("utf-8")).hexdigest()

def _p3_file_hash(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()

def _p3_relative(value):
    path = PurePosixPath(str(value))
    if path.is_absolute() or ".." in path.parts or len(path.parts) != 2:
        raise ValueError(f"Sector 3 artifacts must be flat relative paths: {value!r}")
    if path.parts[0] not in {"configs", "results", "reports"}:
        raise ValueError(f"Unexpected Sector 3 artifact group: {value!r}")
    return path.as_posix()

def _p3_safe_path(root, relative):
    path = root / _p3_relative(relative)
    if path.is_symlink() or root not in path.resolve().parents or path.parent.is_symlink():
        raise RuntimeError(f"Unsafe Sector 3 artifact path: {relative}")
    return path

def _p3_schema(frame):
    columns = []
    for column, dtype in frame.dtypes.items():
        descriptor = {"name": column, "dtype": str(dtype)}
        if isinstance(dtype, pd.StringDtype):
            descriptor["string_storage"] = dtype.storage
        if str(dtype) == "object":
            descriptor["object_kind"] = pd.api.types.infer_dtype(frame[column], skipna=True)
        if isinstance(dtype, pd.CategoricalDtype):
            descriptor.update({"categories": list(dtype.categories), "ordered": dtype.ordered})
        columns.append(descriptor)
    return {"rows": len(frame), "columns": columns, "column_axis_name": frame.columns.name,
            "csv_null_marker": P3_NULL, "schema_version": 1}

def sector3_table_fingerprint(frame):
    """Hash every value and its dtype schema; no clock or execution metadata is added."""
    if frame.columns.duplicated().any() or not all(isinstance(column, str) for column in frame.columns):
        raise ValueError("Sector 3 tables require unique string column names.")
    normalized = frame.reset_index(drop=True)
    digest = hashlib.sha256(_p3_json_text(_p3_schema(normalized)).encode("utf-8"))
    try:
        # Missing values have one semantic identity, including None/NaN/pd.NA in object columns.
        hash_values = normalized.astype(object).where(normalized.notna(), None)
        values = pd.util.hash_pandas_object(hash_values, index=False, categorize=False)
    except TypeError as error:
        raise TypeError("Sector 3 tables must contain scalar values; canonicalize nested conflict values first.") from error
    digest.update(values.to_numpy(dtype="uint64").astype("<u8", copy=False).tobytes())
    return digest.hexdigest()

def _p3_restore_csv(path, schema):
    frame = pd.read_csv(path, dtype="string", keep_default_na=False)
    expected = [descriptor["name"] for descriptor in schema["columns"]]
    if list(frame.columns) != expected or len(frame) != schema["rows"]:
        raise RuntimeError(f"CSV dimensions differ from saved schema: {path.name}")
    for descriptor in schema["columns"]:
        name, dtype = descriptor["name"], descriptor["dtype"]
        values = frame[name]
        nulls = values.eq(P3_NULL)
        values = values.mask(nulls, pd.NA)
        if dtype == "object":
            kind = descriptor.get("object_kind", "string")
            if kind in {"integer", "floating", "mixed-integer-float", "decimal"}:
                values = pd.to_numeric(values, errors="raise").astype(object)
            elif kind == "boolean":
                values = values.map({"True": True, "False": False}).astype(object)
            elif kind not in {"string", "unicode", "bytes", "empty"}:
                raise RuntimeError(f"Unsupported object CSV dtype {kind!r}: canonicalize this table before saving.")
            else:
                values = values.astype(object)
            frame[name] = values.mask(nulls, None)
        elif dtype in {"bool", "boolean"}:
            if (~nulls & ~values.isin(["True", "False"])).any():
                raise RuntimeError(f"Invalid saved boolean values: {path.name}/{name}")
            frame[name] = values.map({"True": True, "False": False}).astype(dtype)
        elif dtype == "string":
            frame[name] = values.astype(pd.StringDtype(storage=descriptor.get("string_storage", "python")))
        elif dtype == "category":
            category = pd.CategoricalDtype(descriptor["categories"], ordered=descriptor["ordered"])
            if pd.api.types.infer_dtype(descriptor["categories"], skipna=True) in {"integer", "floating", "mixed-integer-float"}:
                values = pd.to_numeric(values, errors="raise")
            frame[name] = values.astype(category)
        elif dtype.startswith("timedelta64"):
            frame[name] = pd.to_timedelta(values).astype(dtype)
        else:
            frame[name] = values.astype(dtype)
    frame.columns.name = schema.get("column_axis_name")
    return frame

def _p3_read_table(root, descriptor):
    path = _p3_safe_path(root, descriptor["relative_path"])
    schema_path = _p3_safe_path(root, descriptor["schema_path"])
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    if schema.get("fingerprint") != descriptor["fingerprint"] or schema.get("artifact") != descriptor["relative_path"]:
        raise RuntimeError("Table schema disagrees with the committed catalog.")
    frame = pd.read_parquet(path) if path.suffix == ".parquet" else _p3_restore_csv(path, schema)
    if sector3_table_fingerprint(frame) != descriptor["fingerprint"]:
        raise RuntimeError(f"Typed table readback fingerprint failed: {path.name}")
    return frame

def _p3_catalog_meaning(state):
    return {key: state[key] for key in (
        "schema_version", "complete", "processing_signature", "semantic_config_fingerprint",
        "input_sha256", "core_source_sha256", "config_snapshot", "tables", "summary", "artifacts", "obsolete_paths")}

S4_EXECUTION_METADATA_KEYS = frozenset({
    'run_id', 'run_context', 'run_datetime', 'run_timestamp', 'run_date', 'run_time',
    'timestamp', 'date', 'time', 'execution_date', 'execution_time',
    'created_at', 'created_at_utc', 'created_utc', 'saved_at', 'saved_at_utc', 'saved_utc',
    'started_at', 'started_at_utc', 'started_utc', 'finished_at', 'finished_at_utc', 'finished_utc',
    'generated_at', 'generated_at_utc', 'generated_utc', 'updated_at', 'updated_at_utc', 'updated_utc',
    'modified_at', 'modified_at_utc', 'modified_utc',
})

def sector4_semantic_config(config):
    """Exclude clock bookkeeping only from the effective authored Sector 4 config."""
    def project(value):
        if isinstance(value, dict):
            return {str(key): project(item) for key, item in value.items()
                    if str(key).lower() not in S4_EXECUTION_METADATA_KEYS}
        if isinstance(value, (tuple, list)):
            return [project(item) for item in value]
        return _s3_semantic_json_value(value)
    return project(config)

def sector4_semantic_config_fingerprint(config):
    return _s3_semantic_digest(sector4_semantic_config(config))

S5_SECTOR3_INPUTS = (
    "labelled_frame_manifest", "video_frame_manifest",
    "video_class_matrix", "video_class_support",
)

S5_SECTOR4_RESULTS = (
    "labelled_frame_manifest_with_split", "video_manifest_with_roles",
    "video_split_manifest", "domain_source_videos", "domain_adaptation_manifest",
)

S5_REQUIRED_LABELLED_COLUMNS = {
    "annotation_id", "annotation_status", "alignment_training_eligible",
    "video_key", "finding_class", "finding_class_normalized",
    "clinical_group", "frame_number", "opencv_frame_index", "frame_index_offset",
    "filename", "image_path", "image_relpath", "has_bbox", "bbox_spatially_usable",
    "bbox_validation_status", "bbox_xmin", "bbox_ymin", "bbox_xmax", "bbox_ymax", "split",
}

def _s5_json(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value

def _s5_checked_hash(path, expected):
    if not isinstance(expected, str) or re.fullmatch(r"[0-9a-f]{64}", expected) is None:
        raise ValueError(f"Missing or invalid committed SHA-256: {path}")
    actual = _p3_file_hash(path)
    if actual != expected:
        raise RuntimeError(f"Artifact changed after its completed audit: {path}")
    return actual

def io_s5_read_catalog(root, sector):
    """Read one current Sector 3/4 catalog, never choose a config by timestamp."""
    root = Path(root).resolve()
    state_path = _p3_safe_path(root, f"configs/sector{sector}_state.json")
    if not state_path.is_file():
        raise FileNotFoundError(f"Run/save Sector {sector} first; current catalog missing: {state_path}")
    state_hash = _p3_file_hash(state_path)
    state = _s5_json(state_path)
    if state.get("schema_version") != 2 or state.get("complete") is not True:
        raise RuntimeError(f"Sector {sector} catalog is incomplete or has an unsupported schema.")
    if state.get("content_id") != _p3_json_hash(_p3_catalog_meaning(state)):
        raise RuntimeError(f"Sector {sector} catalog content fingerprint is invalid.")
    records = {item["relative_path"]: item for item in state["artifacts"]}
    operational = {f"configs/sector{sector}_state.json", f"configs/sector{sector}_publication_receipt.json"}
    if len(records) != len(state["artifacts"]) or set(records) & operational:
        raise RuntimeError(f"Sector {sector} catalog has duplicate or operational artifact entries.")
    for relative, record in records.items():
        path = _p3_safe_path(root, relative)
        if not path.is_file() or path.stat().st_size != record["bytes"]:
            raise RuntimeError(f"Sector {sector} cataloged file missing or size changed: {relative}")
        _s5_checked_hash(path, record["sha256"])
    if state["config_snapshot"] not in records:
        raise RuntimeError(f"Sector {sector} current config snapshot is absent from its catalog.")
    snapshot_path = _p3_safe_path(root, state["config_snapshot"])
    envelope = _s5_json(snapshot_path)
    semantic = sector3_semantic_config_fingerprint if sector == 3 else sector4_semantic_config_fingerprint
    if (envelope.get("semantic_config_fingerprint") != state["semantic_config_fingerprint"]
            or semantic(envelope["effective_config"]) != state["semantic_config_fingerprint"]):
        raise RuntimeError(f"Sector {sector} effective config disagrees with its catalog.")
    for name, descriptor in state["tables"].items():
        if not {descriptor["relative_path"], descriptor["schema_path"]}.issubset(records):
            raise RuntimeError(f"Sector {sector} table/schema absent from catalog: {name}")
        schema = _s5_json(_p3_safe_path(root, descriptor["schema_path"]))
        if schema.get("artifact") != descriptor["relative_path"] or schema.get("fingerprint") != descriptor["fingerprint"]:
            raise RuntimeError(f"Sector {sector} table schema disagrees with its catalog: {name}")
    summary = state["summary"]
    if not {summary["summary_relative_path"], summary["report_relative_path"]}.issubset(records):
        raise RuntimeError(f"Sector {sector} summary/report absent from catalog.")
    if (_p3_json_hash(_s5_json(_p3_safe_path(root, summary["summary_relative_path"]))) != summary["fingerprint"]
            or records[summary["report_relative_path"]]["sha256"] != summary["report_fingerprint"]):
        raise RuntimeError(f"Sector {sector} summary/report disagrees with its catalog.")
    for relative in state.get("obsolete_paths", []):
        if not _p3_relative(relative).startswith("results/") or relative in records:
            raise RuntimeError(f"Sector {sector} catalog has an invalid obsolete-result entry.")
    return {"root": root, "state": state, "config": envelope["effective_config"],
            "state_path": state_path, "state_sha256": state_hash,
            "snapshot_path": snapshot_path, "snapshot_sha256": _p3_file_hash(snapshot_path)}

def _s5_required_tables(context, names):
    missing = sorted(set(names) - set(context["state"]["tables"]))
    if missing:
        raise KeyError(f"Current catalog at {context['root']} lacks required tables: {missing}")
    return {name: _p3_read_table(context["root"], context["state"]["tables"][name]) for name in names}

def _s5_rebase_declared_directory(value, old_storage_root, storage_root):
    path = Path(value).expanduser()
    if not path.is_absolute():
        return (storage_root / path).resolve()
    try:
        relative = path.resolve().relative_to(Path(old_storage_root).expanduser().resolve())
    except ValueError:
        if Path(old_storage_root).expanduser().resolve() == storage_root:
            return path.resolve()
        raise ValueError("Cannot rebase an upstream absolute directory outside its storage_root; use a BOOTSTRAP directory override.") from None
    return (storage_root / relative).resolve()

def _s5_sector2_directories(source_config, bootstrap, storage_root):
    def declared(value):
        return _s5_rebase_declared_directory(value, source_config["storage_root"], storage_root)
    output = declared(source_config["output_dir"])
    defaults = {
        "results": declared(source_config["results_dir"]) if "results_dir" in source_config else output / "results",
        "manifests": declared(source_config["curated_data_dir"]) / "manifests",
        "configs": output / "configs",
    }
    return {name: Path(bootstrap[f"sector2_{name}_dir"]).expanduser().resolve()
            if bootstrap.get(f"sector2_{name}_dir") else default.resolve()
            for name, default in defaults.items()}

def _s5_completed_state(path):
    state = _s5_json(path)
    if state.get("complete") is not True:
        raise RuntimeError(f"Sector 2 processing is incomplete: {path}")
    return state

def _s5_identity_multiset(table, columns):
    values = table[columns].astype("string")
    for column in columns:
        if column in {"frame_number", "opencv_frame_index", "aligned_frame_index"} or column.startswith("bbox_"):
            values[column] = pd.to_numeric(table[column], errors="coerce").astype("Float64").astype("string")
    return values.fillna("").value_counts(sort=False).sort_index()

def _s5_validate_loaded_frames(labelled, sector3_labelled, videos, split_manifest, confirmed):
    missing = sorted(S5_REQUIRED_LABELLED_COLUMNS - set(labelled))
    if missing:
        raise KeyError(f"Sector 4 labelled-frame artifact lacks Sector 5 input fields: {missing}")
    if labelled.empty:
        raise ValueError("Sector 5 requires at least one accepted labelled annotation.")
    for column in ("annotation_id", "video_key", "finding_class_normalized", "clinical_group", "split"):
        values = labelled[column].astype("string")
        if (values.isna() | values.str.strip().eq("")).any():
            raise ValueError(f"Sector 5 input has missing/empty {column} values.")
    if not labelled["annotation_id"].is_unique:
        raise ValueError("Sector 4 must preserve unique source annotation_id values.")
    for column in ("frame_number", "opencv_frame_index", "frame_index_offset"):
        values = pd.to_numeric(labelled[column], errors="coerce")
        invalid = values.isna() | ~np.isfinite(values.astype("float64")) | values.mod(1).ne(0)
        if column != "frame_index_offset":
            invalid = invalid | values.lt(0)
        if invalid.any() or labelled[column].map(lambda value: isinstance(value, (bool, np.bool_))).any():
            raise ValueError(f"Sector 5 input requires genuine finite integer {column} values.")
        labelled[column] = values.astype("Int64")
    if not labelled["opencv_frame_index"].eq(labelled["frame_number"] + labelled["frame_index_offset"]).all():
        raise ValueError("Accepted decoded indices disagree with source frame numbers plus verified offsets.")
    if labelled.groupby("video_key", observed=True)["frame_index_offset"].nunique().gt(1).any():
        raise ValueError("Accepted annotations have inconsistent verified offsets within a video.")
    if not labelled["annotation_status"].eq("source_expert_annotation_alignment_confirmed").all():
        raise ValueError("Sector 5 can consume only expert annotations with confirmed alignment.")
    if not labelled["alignment_training_eligible"].eq(True).all():
        raise ValueError("Sector 5 input includes annotations rejected by alignment.")
    if videos["video_key"].isna().any() or not videos["video_key"].is_unique:
        raise ValueError("The completed decoded video inventory needs unique non-null video keys.")
    bounds = labelled["video_key"].map(videos.set_index("video_key")["frame_count"])
    if bounds.isna().any() or not labelled["opencv_frame_index"].lt(bounds).all():
        raise ValueError("Accepted decoded frame indices lie outside audited video bounds.")
    if not videos["frame_count_source"].eq("completed_sector2_decode_audit").all():
        raise ValueError("Video bounds must come from completed Sector 2 decoding.")
    if not split_manifest["video_key"].is_unique or split_manifest["video_key"].isna().any():
        raise ValueError("Sector 4 split manifest requires unique non-null video keys.")
    if not split_manifest["split"].isin(["train", "validation", "test"]).all():
        raise ValueError("Sector 4 split manifest contains unsupported split names.")
    split_lookup = split_manifest.set_index("video_key")["split"]
    if set(split_lookup.index) != set(labelled["video_key"]) or not labelled["split"].eq(labelled["video_key"].map(split_lookup)).all():
        raise ValueError("Accepted annotation split assignments disagree with the video-level split manifest.")
    # Preserve every expert bbox row. Compare annotation identities and multiplicities,
    # and never turn duplicate frame/class rows into new labels or discard them here.
    identity_columns = [column for column in (
        "annotation_id", "filename", "image_key", "video_key", "frame_number",
        "opencv_frame_index", "finding_class_normalized", "bbox_xmin", "bbox_ymin", "bbox_xmax", "bbox_ymax",
    ) if column in sector3_labelled]
    if not _s5_identity_multiset(labelled, identity_columns).equals(_s5_identity_multiset(sector3_labelled, identity_columns)):
        raise ValueError("Sector 4 labelled annotations differ from the current Sector 3 evidence.")
    source_columns = [column for column in (
        "filename", "image_key", "video_key", "frame_number", "finding_class_normalized",
        "bbox_xmin", "bbox_ymin", "bbox_xmax", "bbox_ymax",
    ) if column in confirmed and column in sector3_labelled]
    if not _s5_identity_multiset(confirmed, source_columns).equals(_s5_identity_multiset(sector3_labelled, source_columns)):
        raise ValueError("Sector 3 accepted annotations do not preserve current Sector 2 confirmed evidence.")

def _s5_rebase_live_paths(table, storage_root):
    result = table.copy()
    for path_column, relative_column in (("image_path", "image_relpath"), ("video_path", "video_relpath")):
        if path_column not in result or relative_column not in result:
            continue
        paths = []
        for value in result[relative_column]:
            if pd.isna(value):
                paths.append(pd.NA)
                continue
            relative = PurePosixPath(str(value))
            if relative.is_absolute() or "\\" in str(value) or any(part in {"", ".", ".."} for part in str(value).split("/")):
                raise ValueError(f"Unsafe portable path in {relative_column}: {value!r}")
            paths.append(str(storage_root.joinpath(*relative.parts)))
        result[path_column] = pd.Series(paths, index=result.index).astype(result[path_column].dtype)
    return result

def io_load_sector5_context(bootstrap, settings):
    """Read/verify the current 2 -> 3 -> 4 chain; return config, inputs and lineage."""
    if not isinstance(bootstrap, dict) or not isinstance(settings, dict):
        raise TypeError("BOOTSTRAP and SECTOR5_SETTINGS must be dictionaries.")
    allowed_settings = set(SECTOR5_DEFAULTS) | {"local_output_dir", "export_dir", "push", "dry_run", "github_overrides"}
    unknown = sorted(set(settings) - allowed_settings)
    if unknown:
        raise KeyError(f"Unexpected Sector 5 algorithm settings: {unknown}")
    storage_root = Path(bootstrap["storage_root"]).expanduser().resolve()
    root4 = Path(bootstrap["sector4_dir"]).expanduser().resolve() if bootstrap.get("sector4_dir") else storage_root / "outputs/phase2/sector4"
    context4 = io_s5_read_catalog(root4, 4)
    source4 = context4["config"]
    declared3 = source4.get("sector3", {}).get("local_output_dir", "outputs/phase2/sector3")
    root3 = Path(bootstrap["sector3_dir"]).expanduser().resolve() if bootstrap.get("sector3_dir") else _s5_rebase_declared_directory(declared3, source4["storage_root"], storage_root)
    context3 = io_s5_read_catalog(root3, 3)
    inputs3 = _s5_required_tables(context3, S5_SECTOR3_INPUTS)
    inputs4 = _s5_required_tables(context4, S5_SECTOR4_RESULTS)
    source2 = context3["config"].get("source_configs", {}).get("sector2")
    if not isinstance(source2, dict):
        raise KeyError("Current Sector 3 config snapshot lacks its Sector 2 source configuration.")
    dirs2 = _s5_sector2_directories(source2, bootstrap, storage_root)
    state_paths2 = {"alignment": dirs2["results"] / "frame_alignment_state.json",
                    "decode": dirs2["results"] / "decode_audit_state.json",
                    "manifest": dirs2["manifests"] / "video_manifest_state.json"}
    states2 = {name: _s5_completed_state(path) for name, path in state_paths2.items()}
    state_hashes2 = {name: _p3_file_hash(path) for name, path in state_paths2.items()}
    run_id = states2["alignment"].get("run_id")
    source2_path = None
    if run_id:
        matches = [(path, _s5_json(path)) for path in dirs2["configs"].glob("phase2_02_video_frame_validation_*_config.json")]
        matches = [(path, config) for path, config in matches if config.get("run_id") == run_id]
        if not matches:
            raise FileNotFoundError(f"Sector 2 alignment run {run_id!r} requires its saved config snapshot in {dirs2['configs']}.")
        source2_path, source2 = sorted(matches, key=lambda pair: str(pair[0]))[0]
        if any(config != source2 for _, config in matches):
            raise ValueError("Multiple Sector 2 snapshots share the alignment run_id with different configurations.")
        if _s5_sector2_directories(source2, bootstrap, storage_root) != dirs2:
            raise ValueError("Run-matched Sector 2 configuration selects different artifact directories.")
    threshold_keys = {"min_ssim": "frame_alignment_min_ssim", "min_margin": "frame_alignment_min_margin",
                      "candidate_offsets": "frame_index_offset_candidates"}
    if any(states2["alignment"].get("effective_settings", {}).get(recorded) != source2.get(key)
           for recorded, key in threshold_keys.items()):
        raise ValueError("Sector 2 config snapshot disagrees with the completed alignment decision settings.")
    hashes2 = {}
    for family, required in (("alignment", {"frame_alignment_confirmed.parquet", "frame_alignment_excluded.csv"}),
                             ("decode", {"decode_audit.csv"})):
        committed = states2[family].get("output_sha256", {})
        if not required.issubset(committed):
            raise ValueError(f"Sector 2 {family} state lacks required committed output hashes.")
        for filename, digest in committed.items():
            if Path(filename).name != filename or "/" in filename or "\\" in filename:
                raise ValueError("Sector 2 state must list simple committed output filenames.")
            hashes2[filename] = _s5_checked_hash(dirs2["results"] / filename, digest)
    hashes2["video_manifest.csv"] = _s5_checked_hash(dirs2["manifests"] / "video_manifest.csv", states2["manifest"].get("csv_sha256"))
    prior2 = context3["state"]["input_sha256"]
    for filename in ("frame_alignment_confirmed.parquet", "decode_audit.csv", "video_manifest.csv"):
        recorded = {digest for path, digest in prior2.items() if PurePosixPath(str(path)).name == filename}
        if recorded != {hashes2[filename]}:
            raise RuntimeError(f"Current Sector 3 was not built from current Sector 2 {filename}; rerun/save Sector 3 then Sector 4.")
    for name in S5_SECTOR3_INPUTS:
        descriptor = context3["state"]["tables"][name]
        for field in ("relative_path", "schema_path"):
            actual = _p3_file_hash(_p3_safe_path(root3.resolve(), descriptor[field]))
            if context4["state"]["input_sha256"].get(f"{name}/{field}") != actual:
                raise RuntimeError(f"Current Sector 4 was not built from current Sector 3 {name}/{field}; rerun/save Sector 4.")
    raw2 = {"df_alignment_confirmed": pd.read_parquet(dirs2["results"] / "frame_alignment_confirmed.parquet"),
            "frame_alignment_excluded": pd.read_csv(dirs2["results"] / "frame_alignment_excluded.csv", dtype="string"),
            "decode_audit": pd.read_csv(dirs2["results"] / "decode_audit.csv", dtype={"video_key": "string"}),
            "video_manifest": pd.read_csv(dirs2["manifests"] / "video_manifest.csv", dtype={"video_key": "string"})}
    if (len(raw2["df_alignment_confirmed"]) != states2["alignment"].get("confirmed_annotation_rows")
            or len(raw2["frame_alignment_excluded"]) != states2["alignment"].get("excluded_annotation_rows")
            or len(raw2["df_alignment_confirmed"]) + len(raw2["frame_alignment_excluded"]) != states2["alignment"].get("original_annotation_rows")):
        raise ValueError("Sector 2 confirmed/excluded annotation counts disagree with its completed alignment state.")
    if states2["decode"].get("all_videos_decoded") is not True:
        raise ValueError("Every Sector 2 source video must finish decoding before Sector 5.")
    for name, family in (("video_manifest", "manifest"), ("decode_audit", "decode")):
        table = raw2[name]
        if (len(table) != states2[family].get("video_count") or table["video_key"].isna().any()
                or not table["video_key"].is_unique):
            raise ValueError(f"Sector 2 {name} identities/counts disagree with its completed state.")
    if set(raw2["video_manifest"]["video_key"]) != set(raw2["decode_audit"]["video_key"]):
        raise ValueError("Sector 2 video inventory and decode audit contain different video keys.")
    labelled = inputs4["labelled_frame_manifest_with_split"].copy()
    _s5_validate_loaded_frames(labelled, inputs3["labelled_frame_manifest"], inputs3["video_frame_manifest"],
                              inputs4["video_split_manifest"], raw2["df_alignment_confirmed"])
    config = copy.deepcopy(source4)
    decisions = []
    for key, fallback in SECTOR5_DEFAULTS.items():
        override = settings.get(key)
        value = override if override is not None else source4.get(key, fallback)
        minimum = 1 if key == "segment_max_gap_frames" else 2
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
            raise ValueError(f"{key} must be an integer >= {minimum}.")
        config[key] = int(value)
        decisions.append({"config_key": key, "effective_value": int(value),
                          "resolution": "sector5_override" if override is not None else "inherited_sector4" if key in source4 else "declared_sector5_default"})
    sector5_settings = copy.deepcopy(settings)
    sector5_settings.setdefault("local_output_dir", "outputs/phase2/sector5")
    sector5_settings.setdefault("export_dir", "outputs/phase2/sector5")
    if sector5_settings["export_dir"] != "outputs/phase2/sector5":
        raise ValueError("Sector 5 export_dir must be outputs/phase2/sector5.")
    for flag in ("push", "dry_run"):
        if flag in settings and not isinstance(settings[flag], bool):
            raise ValueError(f"SECTOR5_SETTINGS[{flag!r}] must be a boolean.")
    github = copy.deepcopy(source4.get("github", {}))
    github_overrides = copy.deepcopy(settings.get("github_overrides", {}))
    if not isinstance(github_overrides, dict) or set(github_overrides) - {"token_secret_name"}:
        raise ValueError("Sector 5 github_overrides supports only token_secret_name; repository and branch are inherited.")
    if "token_secret_name" in github_overrides:
        name = github_overrides["token_secret_name"]
        if not isinstance(name, str) or not name.strip() or re.search(r"\s", name):
            raise ValueError("token_secret_name must be a nonempty secret reference without whitespace.")
    github.update(github_overrides)
    github.update({"repo_output_dir": sector5_settings["export_dir"],
                   "push": settings.get("push", github.get("push", False)),
                   "dry_run": settings.get("dry_run", github.get("dry_run", False)),
                   "commit_message": "Phase 2 Sector 5: publish finding segments and temporal supervision"})
    github.pop("confirm_push", None)
    config.update({"phase": 2, "sector": 5, "storage_root": str(storage_root),
                   "sector5": sector5_settings, "github": github})
    for key in S4_EXECUTION_METADATA_KEYS:
        config.pop(key, None)
    # Copies used for access are rebased only after validating the saved bytes and identities.
    runtime3 = {name: _s5_rebase_live_paths(table, storage_root) for name, table in inputs3.items()}
    runtime4 = {name: _s5_rebase_live_paths(table, storage_root) for name, table in inputs4.items()}
    runtime4["labelled_frame_manifest_with_split"] = _s5_rebase_live_paths(labelled, storage_root)
    inputs = {"sector2": raw2, "sector3": runtime3, "sector4": runtime4}
    consumed = {f"sector2/{name}": digest for name, digest in hashes2.items()}
    consumed_paths = {f"sector2/{name}": str((dirs2["manifests"] if name == "video_manifest.csv" else dirs2["results"]) / name)
                      for name in hashes2}
    for name, path in state_paths2.items():
        consumed[f"sector2/state/{name}"] = state_hashes2[name]
        consumed_paths[f"sector2/state/{name}"] = str(path)
    if source2_path is not None:
        consumed["sector2/config_snapshot"] = _p3_file_hash(source2_path)
        consumed_paths["sector2/config_snapshot"] = str(source2_path)
    for sector, context, names in ((3, context3, S5_SECTOR3_INPUTS), (4, context4, S5_SECTOR4_RESULTS)):
        for name in names:
            for field in ("relative_path", "schema_path"):
                relative = context["state"]["tables"][name][field]
                consumed[f"sector{sector}/{name}/{field}"] = _p3_file_hash(_p3_safe_path(context["root"], relative))
                consumed_paths[f"sector{sector}/{name}/{field}"] = str(_p3_safe_path(context["root"], relative))
        consumed[f"sector{sector}/state"] = context["state_sha256"]
        consumed_paths[f"sector{sector}/state"] = str(context["state_path"])
        consumed[f"sector{sector}/config_snapshot"] = context["snapshot_sha256"]
        consumed_paths[f"sector{sector}/config_snapshot"] = str(context["snapshot_path"])
    lineage = {"input_sha256": consumed, "input_descriptors": consumed_paths, "sector2_run_id": run_id,
               "sector2_states": states2, "sector2_state_sha256": state_hashes2,
               "sector2_config_snapshot": str(source2_path) if source2_path else None,
               "sector2_config_snapshot_sha256": _p3_file_hash(source2_path) if source2_path else None,
               "upstream": {f"sector{sector}": {"root": str(context["root"]), "content_id": context["state"]["content_id"],
                   "state_sha256": context["state_sha256"], "config_snapshot": str(context["snapshot_path"]),
                   "config_snapshot_sha256": context["snapshot_sha256"]} for sector, context in ((3, context3), (4, context4))},
               "config_inheritance_report": pd.DataFrame.from_records(decisions),
               "config_inheritance_policy": "Current verified Sector 4 effective config; explicit Sector 5 overrides; declared defaults only when absent."}
    for context in (context3, context4):
        _s5_checked_hash(context["state_path"], context["state_sha256"])
    for name, path in state_paths2.items():
        _s5_checked_hash(path, state_hashes2[name])
    for filename, expected in hashes2.items():
        parent = dirs2["manifests"] if filename == "video_manifest.csv" else dirs2["results"]
        _s5_checked_hash(parent / filename, expected)
    return {"config": config, "inputs": inputs, "lineage": lineage}

SECTOR6_DEFAULTS = {
    "temporal_context_enabled": True,
    "temporal_context_extract_frames": True,
    "temporal_context_offsets_seconds": [-2.0, -1.0, 0.0, 1.0, 2.0],
    "temporal_context_image_format": "jpg",
    "temporal_context_jpeg_quality": 95,
    "temporal_context_png_compression": 3,
    "temporal_context_workers": 10,
    "temporal_context_reuse_existing": True,
    "temporal_context_decoder": "nvdec",
    "temporal_context_gpu_id": 0,
    "temporal_context_decode_batch_size": 4,
    "temporal_context_image_dir": "curated/phase2/temporal_frames",
}

S6_SECTOR5_REQUIRED_TABLES = (
    "finding_segment_manifest", "labelled_frame_manifest_with_segments",
)

def _s6_portable_path(value, storage_root, field):
    """Rebase a declared portable path without touching the filesystem."""
    if value is None or value is pd.NA or pd.isna(value):
        return pd.NA
    text = str(value)
    relative = PurePosixPath(text)
    if (not text or relative.is_absolute() or "\\" in text
            or any(part in {"", ".", ".."} for part in text.split("/"))):
        raise ValueError(f"Unsafe storage-relative path in {field}: {value!r}")
    return str(Path(storage_root).joinpath(*relative.parts))

def _s6_rebase_live_paths(table, storage_root):
    """Return a copy with live paths rebuilt only from committed relpaths."""
    result = _s5_rebase_live_paths(table, storage_root)
    for path_column, relative_column in (
        ("target_image_path", "target_image_relpath"),
        ("context_image_path", "context_image_relpath"),
    ):
        if path_column in result and relative_column in result:
            paths = result[relative_column].map(
                lambda value: _s6_portable_path(value, storage_root, relative_column))
            result[path_column] = paths.astype(result[path_column].dtype)
    if "target_annotations_json" in result:
        def rebase_annotations(value):
            if pd.isna(value):
                return pd.NA
            annotations = json.loads(str(value))
            if not isinstance(annotations, list) or any(not isinstance(item, dict) for item in annotations):
                raise ValueError("Sector 5 target_annotations_json must contain an annotation list.")
            for annotation in annotations:
                if "image_path" in annotation:
                    if "image_relpath" not in annotation:
                        raise KeyError("A saved target annotation lacks its portable image_relpath.")
                    path = _s6_portable_path(annotation["image_relpath"], storage_root, "target_annotations_json/image_relpath")
                    annotation["image_path"] = None if pd.isna(path) else path
            return json.dumps(annotations, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        result["target_annotations_json"] = result["target_annotations_json"].map(rebase_annotations).astype("string")
    return result

def _s6_integer_setting(value, name, minimum, maximum=None):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or value < minimum or (maximum is not None and value > maximum)):
        interval = f"{minimum}..{maximum}" if maximum is not None else f">= {minimum}"
        raise ValueError(f"{name} must be an integer {interval}.")
    return int(value)

def _s6_validate_effective_settings(config):
    """Pure validation/normalization of the effective Sector 6 settings."""
    result = copy.deepcopy(config)
    for name in ("temporal_context_enabled", "temporal_context_extract_frames", "temporal_context_reuse_existing"):
        if not isinstance(result[name], bool):
            raise ValueError(f"{name} must be a boolean.")
    offsets = result["temporal_context_offsets_seconds"]
    if not isinstance(offsets, (list, tuple)) or not offsets:
        raise ValueError("temporal_context_offsets_seconds must be a nonempty numeric list containing zero.")
    if any(isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)) for value in offsets):
        raise ValueError("Temporal offsets must contain genuine numbers, not booleans or strings.")
    offsets = [float(value) for value in offsets]
    if not np.isfinite(offsets).all() or len(set(offsets)) != len(offsets) or 0.0 not in offsets:
        raise ValueError("Temporal offsets must be finite, unique and include 0.0 for the verified target.")
    result["temporal_context_offsets_seconds"] = sorted(offsets)
    extension = str(result["temporal_context_image_format"]).strip().lower().lstrip(".")
    if extension not in {"jpg", "jpeg", "png"}:
        raise ValueError("temporal_context_image_format must be jpg, jpeg or png.")
    result["temporal_context_image_format"] = extension
    result["temporal_context_jpeg_quality"] = _s6_integer_setting(result["temporal_context_jpeg_quality"], "temporal_context_jpeg_quality", 1, 100)
    result["temporal_context_png_compression"] = _s6_integer_setting(result["temporal_context_png_compression"], "temporal_context_png_compression", 0, 9)
    result["temporal_context_workers"] = _s6_integer_setting(result["temporal_context_workers"], "temporal_context_workers", 1)
    result["temporal_context_gpu_id"] = _s6_integer_setting(result["temporal_context_gpu_id"], "temporal_context_gpu_id", 0)
    result["temporal_context_decode_batch_size"] = _s6_integer_setting(result["temporal_context_decode_batch_size"], "temporal_context_decode_batch_size", 1)
    if result["temporal_context_decoder"] != "nvdec":
        raise ValueError("Sector 6 supports only temporal_context_decoder='nvdec'; CPU fallback is disabled.")
    image_dir = result["temporal_context_image_dir"]
    if not isinstance(image_dir, str):
        raise ValueError("temporal_context_image_dir must be a storage-relative directory string.")
    _s6_portable_path(image_dir, result["storage_root"], "temporal_context_image_dir")
    return result

def _s6_validate_sector5_tables(inputs5, current4, video_frames):
    """Preserve all source expert annotations and current video-level splits."""
    labelled = inputs5["labelled_frame_manifest_with_segments"]
    segments = inputs5["finding_segment_manifest"]
    required_labelled = set(current4.columns) | {"finding_segment_id"}
    missing = sorted(required_labelled - set(labelled.columns))
    if missing:
        raise KeyError(f"Sector 5 member manifest lacks preserved Sector 4 fields: {missing}")
    if (labelled["annotation_id"].isna().any() or not labelled["annotation_id"].is_unique
            or set(labelled["annotation_id"]) != set(current4["annotation_id"])):
        raise ValueError("Sector 5 annotation IDs do not preserve the current accepted Sector 4 evidence.")
    left = labelled.loc[:, current4.columns].sort_values("annotation_id", kind="stable").reset_index(drop=True)
    right = current4.sort_values("annotation_id", kind="stable").reset_index(drop=True)
    try:
        pd.testing.assert_frame_equal(left, right, check_dtype=False, check_categorical=False)
    except AssertionError as error:
        raise ValueError("Sector 5 changed accepted source annotations, bounding boxes or Sector 4 splits.") from error
    required_segments = {
        "finding_segment_id", "video_key", "finding_class", "finding_class_normalized", "clinical_group", "split",
        "verified_start_frame", "verified_end_frame", "verified_frame_count", "verified_annotation_count",
        "target_frame_number", "target_opencv_frame_index", "target_image_path", "target_image_relpath",
        "target_image_annotation_id", "target_annotation_ids_json", "target_annotations_json",
        "target_annotation_count", "temporal_grounding_candidate", "verified_temporal_boundary_available", "temporal_supervision_level",
    }
    missing = sorted(required_segments - set(segments.columns))
    if missing:
        raise KeyError(f"Sector 5 segment manifest lacks Sector 6 fields: {missing}")
    if segments["finding_segment_id"].isna().any() or not segments["finding_segment_id"].is_unique:
        raise ValueError("Sector 5 segment IDs must be unique and non-null.")
    if labelled["finding_segment_id"].isna().any() or set(labelled["finding_segment_id"]) != set(segments["finding_segment_id"]):
        raise ValueError("Sector 5 segment identities disagree with annotation memberships.")
    if not segments["temporal_grounding_candidate"].isin([True, False]).all():
        raise ValueError("Sector 5 temporal-grounding flags must be non-null booleans.")
    if not segments["verified_temporal_boundary_available"].eq(False).all():
        raise ValueError("Observed finding anchors cannot be treated as verified clinical boundaries.")
    group = labelled.groupby("finding_segment_id", sort=False, observed=True)
    invariants = ["video_key", "finding_class_normalized", "clinical_group", "split"]
    if group[invariants].nunique(dropna=False).gt(1).any(axis=1).any():
        raise ValueError("One finding segment contains conflicting source video, finding, taxonomy or split.")
    expected = group.agg(
        video_key=("video_key", "first"), finding_class_normalized=("finding_class_normalized", "first"),
        clinical_group=("clinical_group", "first"), split=("split", "first"),
        verified_start_frame=("frame_number", "min"), verified_end_frame=("frame_number", "max"),
        verified_frame_count=("frame_number", "nunique"), verified_annotation_count=("annotation_id", "size"),
    ).sort_index()
    actual = segments.set_index("finding_segment_id").loc[expected.index, expected.columns]
    try:
        # Identical validated IDs may use object or StringDtype after Parquet
        # readback/groupby. Ignore index storage dtype only; labels, order,
        # index names and every summary value still have to agree.
        pd.testing.assert_frame_equal(actual, expected, check_dtype=False,
                                      check_index_type=False, check_categorical=False)
    except AssertionError as error:
        raise ValueError("Sector 5 summary counts, observed ranges or split assignments disagree with every retained annotation.") from error
    lookup = labelled.set_index("annotation_id")
    anchor_positions = labelled.groupby(["finding_segment_id", "frame_number"], sort=False, observed=True).indices
    for target in segments.to_dict("records"):
        segment_id = target["finding_segment_id"]
        positions = anchor_positions.get((segment_id, target["target_frame_number"]), np.empty(0, dtype="int64"))
        members = labelled.iloc[positions]
        ids = json.loads(str(target["target_annotation_ids_json"]))
        annotation_records = json.loads(str(target["target_annotations_json"]))
        if (members.empty or not isinstance(ids, list) or len(ids) != len(set(ids))
                or set(ids) != set(members["annotation_id"])
                or int(target["target_annotation_count"]) != len(members)
                or not isinstance(annotation_records, list)
                or any(not isinstance(item, dict) for item in annotation_records)
                or {item.get("annotation_id") for item in annotation_records} != set(ids)
                or len(annotation_records) != len(members)):
            raise ValueError(f"Sector 5 target annotation collection is incomplete: {segment_id}")
        if target["target_image_annotation_id"] not in ids:
            raise ValueError(f"Sector 5 representative annotation does not belong to its target frame: {segment_id}")
        representative = lookup.loc[target["target_image_annotation_id"]]
        for target_column, member_column in (
            ("video_key", "video_key"), ("finding_class_normalized", "finding_class_normalized"),
            ("split", "split"), ("target_frame_number", "frame_number"),
            ("target_opencv_frame_index", "opencv_frame_index"),
            ("target_image_relpath", "image_relpath"), ("target_image_path", "image_path"),
        ):
            saved, source = target[target_column], representative[member_column]
            if pd.isna(saved) or pd.isna(source) or saved != source:
                raise ValueError(f"Sector 5 representative target disagrees with its source annotation: {segment_id}/{target_column}")
        for annotation in annotation_records:
            source = lookup.loc[annotation["annotation_id"]]
            for name, saved in annotation.items():
                if name == "annotation_id":
                    continue
                if name not in source.index:
                    raise ValueError(f"Sector 5 target annotation has an unknown source field: {name}")
                live = source[name]
                saved_missing, live_missing = pd.isna(saved), pd.isna(live)
                if saved_missing != live_missing or (not saved_missing and saved != live):
                    raise ValueError(f"Sector 5 target annotation collection changed expert metadata: {segment_id}/{name}")
    bounds = segments["video_key"].map(video_frames.set_index("video_key")["frame_count"])
    if bounds.isna().any() or not segments["target_opencv_frame_index"].ge(0).all() or not segments["target_opencv_frame_index"].lt(bounds).all():
        raise ValueError("Sector 5 target decoded indices lie outside current completed Sector 2 video bounds.")

def io_load_sector6_context(bootstrap, settings):
    """Read and verify the current 2 -> 3 -> 4 -> 5 artifact chain.

    Every upstream byte hash is checked before input construction and again
    before returning. No artifact is selected by timestamp or notebook globals.
    """
    if not isinstance(bootstrap, dict) or not isinstance(settings, dict):
        raise TypeError("BOOTSTRAP and SECTOR6_SETTINGS must be dictionaries.")
    allowed = set(SECTOR6_DEFAULTS) | {"local_output_dir", "export_dir", "push", "dry_run", "github_overrides"}
    unknown = sorted(set(settings) - allowed)
    if unknown:
        raise KeyError(f"Unexpected Sector 6 settings: {unknown}")
    storage_root = Path(bootstrap["storage_root"]).expanduser().resolve()
    root5 = Path(bootstrap["sector5_dir"]).expanduser().resolve() if bootstrap.get("sector5_dir") else storage_root / "outputs/phase2/sector5"
    context5 = io_s5_read_catalog(root5, 5)
    source5 = context5["config"]
    if source5.get("phase") != 2 or source5.get("sector") != 5:
        raise ValueError("The current Sector 5 catalog must contain a Phase 2 Sector 5 effective configuration.")
    inherited_s5_settings = {key: source5.get(key) for key in SECTOR5_DEFAULTS}
    inherited_s5_settings.update({key: source5.get("sector5", {}).get(key, default) for key, default in (
        ("local_output_dir", "outputs/phase2/sector5"), ("export_dir", "outputs/phase2/sector5"),
    )})
    inherited_s5_settings.update({"push": source5.get("github", {}).get("push", False),
                                  "dry_run": source5.get("github", {}).get("dry_run", False)})
    context234 = io_load_sector5_context(bootstrap, inherited_s5_settings)
    current_hashes = context234["lineage"]["input_sha256"]
    if context5["state"].get("input_sha256") != current_hashes:
        prior = context5["state"].get("input_sha256", {})
        changed = sorted(key for key in set(prior) | set(current_hashes) if prior.get(key) != current_hashes.get(key))
        raise RuntimeError("Current Sector 5 was not built from the current completed Sector 2/3/4 chain. "
                           f"Rerun/save Sector 5 first. Changed lineage entries: {changed}")
    missing = sorted(set(S6_SECTOR5_REQUIRED_TABLES) - set(context5["state"]["tables"]))
    if missing:
        raise KeyError(f"Current Sector 5 catalog lacks required tables: {missing}")
    inputs5 = _s5_required_tables(context5, tuple(context5["state"]["tables"]))
    runtime5 = {name: _s6_rebase_live_paths(table, storage_root) for name, table in inputs5.items()}
    _s6_validate_sector5_tables(runtime5,
        context234["inputs"]["sector4"]["labelled_frame_manifest_with_split"],
        context234["inputs"]["sector3"]["video_frame_manifest"])
    config = copy.deepcopy(source5)
    config["storage_root"] = str(storage_root)
    decisions = []
    for key, fallback in SECTOR6_DEFAULTS.items():
        override = settings.get(key)
        value = override if override is not None else source5.get(key, fallback)
        config[key] = copy.deepcopy(value)
        decisions.append({"config_key": key, "effective_value_json": _s3_semantic_json(value),
                          "resolution": "sector6_override" if override is not None else
                          "inherited_sector5" if key in source5 else "declared_sector6_default"})
    config = _s6_validate_effective_settings(config)
    sector6 = copy.deepcopy(settings)
    sector6.setdefault("local_output_dir", "outputs/phase2/sector6")
    sector6.setdefault("export_dir", "outputs/phase2/sector6")
    if sector6["export_dir"] != "outputs/phase2/sector6":
        raise ValueError("Sector 6 export_dir must be outputs/phase2/sector6.")
    local_dir = sector6["local_output_dir"]
    if not isinstance(local_dir, str) or not local_dir.strip():
        raise ValueError("Sector 6 local_output_dir must be a nonempty directory string.")
    if not Path(local_dir).expanduser().is_absolute():
        _s6_portable_path(local_dir, storage_root, "local_output_dir")
    github = copy.deepcopy(source5.get("github", {}))
    overrides = copy.deepcopy(settings.get("github_overrides", {}))
    if not isinstance(overrides, dict) or set(overrides) - {"token_secret_name"}:
        raise ValueError("Sector 6 github_overrides supports only token_secret_name; repository and branch are inherited.")
    if "token_secret_name" in overrides:
        secret_name = overrides["token_secret_name"]
        if not isinstance(secret_name, str) or not secret_name.strip() or re.search(r"\s", secret_name):
            raise ValueError("token_secret_name must be a nonempty secret reference without whitespace.")
    for flag in ("push", "dry_run"):
        if flag in settings and not isinstance(settings[flag], bool):
            raise ValueError(f"SECTOR6_SETTINGS[{flag!r}] must be a boolean.")
    github.update(overrides)
    github.update({"repo_output_dir": sector6["export_dir"],
                   "push": settings.get("push", github.get("push", False)),
                   "dry_run": settings.get("dry_run", github.get("dry_run", False)),
                   "commit_message": "Phase 2 Sector 6: publish temporal evidence and extraction audit"})
    github.pop("confirm_push", None)
    config.update({"phase": 2, "sector": 6, "sector6": sector6, "github": github})
    for key in S4_EXECUTION_METADATA_KEYS:
        config.pop(key, None)
    lineage = copy.deepcopy(context234["lineage"])
    hashes = lineage["input_sha256"]
    descriptors = lineage["input_descriptors"]
    for name, descriptor in context5["state"]["tables"].items():
        for field in ("relative_path", "schema_path"):
            path = _p3_safe_path(context5["root"], descriptor[field])
            hashes[f"sector5/{name}/{field}"] = _p3_file_hash(path)
            descriptors[f"sector5/{name}/{field}"] = str(path)
    hashes["sector5/state"] = context5["state_sha256"]
    descriptors["sector5/state"] = str(context5["state_path"])
    hashes["sector5/config_snapshot"] = context5["snapshot_sha256"]
    descriptors["sector5/config_snapshot"] = str(context5["snapshot_path"])
    lineage["sector5_state"] = copy.deepcopy(context5["state"])
    lineage["sector5_state_sha256"] = context5["state_sha256"]
    lineage["upstream"]["sector5"] = {
        "root": str(context5["root"]), "content_id": context5["state"]["content_id"],
        "state_sha256": context5["state_sha256"], "config_snapshot": str(context5["snapshot_path"]),
        "config_snapshot_sha256": context5["snapshot_sha256"],
    }
    lineage["config_inheritance_report"] = pd.DataFrame.from_records(decisions)
    lineage["config_inheritance_policy"] = (
        "Current verified Sector 5 effective config; explicit Sector 6 overrides; declared defaults only when absent.")
    for name, expected in hashes.items():
        _s5_checked_hash(descriptors[name], expected)
    inputs = copy.copy(context234["inputs"])
    inputs["sector5"] = runtime5
    return {"config": config, "inputs": inputs, "lineage": lineage}

S6_STATE_PATH = 'configs/sector6_state.json'

S6_RECEIPT_PATH = 'configs/sector6_publication_receipt.json'

S6_LINEAGE_PATH = 'configs/sector6_lineage.json'

def sector6_semantic_config_fingerprint(config):
    # Sector 4's projection already excludes clock metadata recursively.
    return _s3_semantic_digest(sector4_semantic_config(config))

def _p6_read_state(root):
    """Validate the complete flat catalog, current config, schemas and report bytes."""
    root = Path(root).resolve()
    state_path = _p3_safe_path(root, S6_STATE_PATH)
    if not state_path.is_file():
        return None
    state = json.loads(state_path.read_text(encoding='utf-8'))
    if state.get('schema_version') != 2 or state.get('complete') is not True:
        raise RuntimeError('Sector 6 catalog is incomplete or has an unsupported schema.')
    if state.get('content_id') != _p3_json_hash(_p3_catalog_meaning(state)):
        raise RuntimeError('Sector 6 catalog fingerprint is invalid.')
    records = {record['relative_path']: record for record in state['artifacts']}
    if len(records) != len(state['artifacts']) or {S6_STATE_PATH, S6_RECEIPT_PATH} & set(records):
        raise RuntimeError('Sector 6 catalog contains duplicate or operational artifact entries.')
    for relative, record in records.items():
        path = _p3_safe_path(root, relative)
        if not path.is_file() or path.stat().st_size != record['bytes'] or _p3_file_hash(path) != record['sha256']:
            raise RuntimeError(f'Sector 6 saved artifact checksum verification failed: {relative}')
    if state['config_snapshot'] not in records:
        raise RuntimeError('Sector 6 current configuration snapshot is absent from its catalog.')
    envelope = json.loads(_p3_safe_path(root, state['config_snapshot']).read_text(encoding='utf-8'))
    if envelope.get('semantic_config_fingerprint') != state['semantic_config_fingerprint'] or (
        sector6_semantic_config_fingerprint(envelope['effective_config']) != state['semantic_config_fingerprint']
    ):
        raise RuntimeError('Sector 6 saved effective configuration disagrees with its catalog.')
    if S6_LINEAGE_PATH not in records:
        raise RuntimeError('Sector 6 current lineage is absent from its catalog.')
    saved_lineage = json.loads(_p3_safe_path(root, S6_LINEAGE_PATH).read_text(encoding='utf-8'))
    if (saved_lineage.get('input_sha256') != state['input_sha256'] or
            saved_lineage.get('core_source_sha256') != state['core_source_sha256'] or
            not isinstance(saved_lineage.get('input_descriptors'), dict) or
            set(saved_lineage['input_descriptors']) != set(state['input_sha256'])):
        raise RuntimeError('Sector 6 saved current lineage disagrees with its catalog.')
    if _p3_json_hash(saved_lineage) != records[S6_LINEAGE_PATH]['sha256']:
        raise RuntimeError('Sector 6 current lineage is not canonically encoded.')
    for name, descriptor in state['tables'].items():
        if not {descriptor['relative_path'], descriptor['schema_path']}.issubset(records):
            raise RuntimeError(f'Sector 6 table/schema absent from catalog: {name}')
        schema = json.loads(_p3_safe_path(root, descriptor['schema_path']).read_text(encoding='utf-8'))
        if schema.get('artifact') != descriptor['relative_path'] or schema.get('fingerprint') != descriptor['fingerprint']:
            raise RuntimeError(f'Sector 6 saved table schema disagrees with catalog: {name}')
    summary = state['summary']
    if not {summary['summary_relative_path'], summary['report_relative_path']}.issubset(records):
        raise RuntimeError('Sector 6 summary/report is absent from its catalog.')
    saved_summary = json.loads(_p3_safe_path(root, summary['summary_relative_path']).read_text(encoding='utf-8'))
    if _p3_json_hash(saved_summary) != summary['fingerprint'] or (
        records[summary['report_relative_path']]['sha256'] != summary['report_fingerprint']
    ):
        raise RuntimeError('Sector 6 saved summary/report disagrees with its catalog.')
    for relative in state.get('obsolete_paths', []):
        if not _p3_relative(relative).startswith('results/') or relative in records:
            raise RuntimeError('Invalid Sector 6 obsolete-result deletion authorization.')
    return state




def _s7_rebase_extra_image_paths(table, storage_root):
    """Rebase every paired image path and the saved official candidate references."""
    result = table.copy()
    for column in result.columns:
        if column.endswith('image_path'):
            relative_column = column[:-len('path')] + 'relpath'
            if relative_column in result:
                values = result[relative_column].map(
                    lambda value: _s6_portable_path(value, storage_root, relative_column))
                result[column] = values.astype(result[column].dtype)
    if 'context_official_images_json' in result:
        def rebase_candidates(value):
            if pd.isna(value):
                return pd.NA
            candidates = json.loads(str(value))
            if not isinstance(candidates, list) or any(not isinstance(item, dict) for item in candidates):
                raise ValueError('Official candidate images must be a JSON list of descriptors.')
            for candidate in candidates:
                if 'image_path' in candidate:
                    if 'image_relpath' not in candidate:
                        raise KeyError('Official candidate image lacks image_relpath.')
                    path = _s6_portable_path(candidate['image_relpath'], storage_root, 'candidate/image_relpath')
                    candidate['image_path'] = None if pd.isna(path) else path
            return json.dumps(candidates, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(',', ':'))
        result['context_official_images_json'] = result['context_official_images_json'].map(rebase_candidates).astype('string')
    return result


def io_load_sector7_context(bootstrap, settings):
    """Read the verified current Sector 2 -> 3 -> 4 -> 5 -> 6 chain.

    This reader consumes saved metadata only. It neither decodes videos nor
    scans image bytes. Sector 7's source audit binds the images actually used
    for QC to their current content and reports missing or changed images.
    """
    if not isinstance(bootstrap, dict) or not isinstance(settings, dict):
        raise TypeError("BOOTSTRAP and SECTOR7_SETTINGS must be dictionaries.")
    allowed = set(S7_DEFAULTS) | {
        "local_output_dir", "export_dir", "push", "dry_run", "github_overrides"}
    unknown = sorted(set(settings) - allowed)
    if unknown:
        raise KeyError(f"Unexpected Sector 7 settings: {unknown}")
    storage_root = Path(bootstrap["storage_root"]).expanduser().resolve()
    context2345 = io_load_sector6_context(bootstrap, {})
    root6 = (Path(bootstrap["sector6_dir"]).expanduser().resolve()
             if bootstrap.get("sector6_dir") else storage_root / "outputs/phase2/sector6")
    state6 = _p6_read_state(root6)
    if state6 is None:
        raise FileNotFoundError(
            f"Run/save Sector 6 first; current catalog missing: {root6 / S6_STATE_PATH}")
    snapshot_path = _p3_safe_path(root6, state6["config_snapshot"])
    source6 = _s5_json(snapshot_path)["effective_config"]
    if source6.get("phase") != 2 or source6.get("sector") != 6:
        raise ValueError("Current Sector 6 needs a Phase 2 Sector 6 effective configuration.")
    saved_lineage6 = _s5_json(_p3_safe_path(root6, S6_LINEAGE_PATH))
    current_hashes = context2345["lineage"]["input_sha256"]
    prior_hashes = state6["input_sha256"]
    changed = sorted(key for key, digest in current_hashes.items()
                     if prior_hashes.get(key) != digest)
    if changed:
        raise RuntimeError(
            "Current Sector 6 was not built from the current completed Sector 2/3/4/5 chain. "
            f"Rerun/save Sector 6 first. Changed lineage entries: {changed}")
    missing = sorted({"temporal_manifest", "temporal_image_audit"} - set(state6["tables"]))
    if missing:
        raise KeyError(f"Current Sector 6 lacks required QC input tables: {missing}")
    saved6 = {name: _p3_read_table(root6, descriptor)
              for name, descriptor in state6["tables"].items()}
    runtime6 = {name: _s7_rebase_extra_image_paths(_s6_rebase_live_paths(table, storage_root), storage_root)
                for name, table in saved6.items()}
    if "context_image_path" not in runtime6["temporal_manifest"]:
        raise KeyError("Saved Sector 6 temporal_manifest lacks context_image_path.")
    labelled5 = context2345["inputs"]["sector5"]["labelled_frame_manifest_with_segments"]
    if "image_path" not in labelled5:
        raise KeyError("Saved Sector 5 labelled manifest lacks image_path.")

    # S6's saved effective configuration is authoritative, including overrides
    # that were absent from Sector 5. Only the current storage root is rebased.
    config = copy.deepcopy(source6)
    config["storage_root"] = str(storage_root)
    decisions = []
    for key, fallback in S7_DEFAULTS.items():
        override = settings.get(key)
        value = override if override is not None else source6.get(key, fallback)
        config[key] = copy.deepcopy(value)
        decisions.append({"config_key": key, "effective_value_json": _s3_semantic_json(value),
                          "resolution": "sector7_override" if override is not None else
                          "inherited_sector6" if key in source6 else "declared_sector7_default"})
    sector7 = {key: copy.deepcopy(config[key]) for key in S7_DEFAULTS}
    for key, fallback in (("local_output_dir", "outputs/phase2/sector7"),
                          ("export_dir", "outputs/phase2/sector7")):
        sector7[key] = settings.get(key, fallback)
    if sector7["export_dir"] != "outputs/phase2/sector7":
        raise ValueError("Sector 7 export_dir must be outputs/phase2/sector7.")
    local_dir = sector7["local_output_dir"]
    if not isinstance(local_dir, str) or not local_dir.strip():
        raise ValueError("Sector 7 local_output_dir must be a nonempty directory string.")
    if not Path(local_dir).expanduser().is_absolute():
        _s6_portable_path(local_dir, storage_root, "local_output_dir")
    github = copy.deepcopy(source6.get("github", {}))
    overrides = copy.deepcopy(settings.get("github_overrides", {}))
    if not isinstance(overrides, dict) or set(overrides) - {"token_secret_name"}:
        raise ValueError(
            "Sector 7 github_overrides supports only token_secret_name; repository and branch are inherited.")
    if "token_secret_name" in overrides:
        name = overrides["token_secret_name"]
        if not isinstance(name, str) or not name.strip() or re.search(r"\s", name):
            raise ValueError("token_secret_name must be a nonempty secret reference without whitespace.")
    for flag in ("push", "dry_run"):
        value = settings.get(flag, github.get(flag, False))
        if not isinstance(value, bool):
            raise ValueError(f"SECTOR7_SETTINGS[{flag!r}] must be a boolean.")
        sector7[flag] = value
    sector7["github_overrides"] = overrides
    github.update(overrides)
    github.update({"repo_output_dir": sector7["export_dir"],
                   "push": sector7["push"], "dry_run": sector7["dry_run"],
                   "commit_message": "Phase 2 Sector 7: publish image quality metrics and review flags"})
    github.pop("confirm_push", None)
    config.update({"phase": 2, "sector": 7, "sector7": sector7, "github": github})
    for key in S4_EXECUTION_METADATA_KEYS:
        config.pop(key, None)

    lineage = copy.deepcopy(context2345["lineage"])
    hashes, descriptors = lineage["input_sha256"], lineage["input_descriptors"]
    # Catalog artifacts are metadata, including the current lineage JSON.
    # Source-video and official-image hashes remain inherited provenance only;
    # QC binds actual current image inputs in its own source audit.
    for record in state6["artifacts"]:
        relative = record["relative_path"]
        path = _p3_safe_path(root6, relative)
        key = "sector6/artifact/" + relative
        hashes[key] = record["sha256"]
        descriptors[key] = str(path)
    state_path = _p3_safe_path(root6, S6_STATE_PATH)
    hashes["sector6/state"] = _p3_file_hash(state_path)
    descriptors["sector6/state"] = str(state_path)
    hashes["sector6/config_snapshot"] = _p3_file_hash(snapshot_path)
    descriptors["sector6/config_snapshot"] = str(snapshot_path)
    lineage["sector6_saved_input_sha256"] = copy.deepcopy(saved_lineage6["input_sha256"])
    lineage["sector6_saved_input_descriptors"] = copy.deepcopy(saved_lineage6["input_descriptors"])
    lineage["upstream"]["sector6"] = {
        "root": str(root6), "content_id": state6["content_id"],
        "state_sha256": hashes["sector6/state"],
        "config_snapshot": str(snapshot_path),
        "config_snapshot_sha256": hashes["sector6/config_snapshot"],
    }
    lineage["config_inheritance_report"] = pd.DataFrame.from_records(decisions)
    lineage["config_inheritance_policy"] = (
        "Current verified Sector 6 effective config; explicit Sector 7 overrides; declared defaults only when absent.")
    for key, expected in hashes.items():
        _s5_checked_hash(descriptors[key], expected)
    if _p6_read_state(root6) != state6:
        raise RuntimeError("Sector 6 changed during input loading; rerun the input-loading cell.")
    inputs = copy.copy(context2345["inputs"])
    inputs["sector6"] = runtime6
    return {"config": config, "inputs": inputs, "lineage": lineage}


# Pure QC transformations. File/image reads are confined to io_ functions.
IMAGE_QC_COLUMNS = [
    'image_path', 'image_sha256', 'image_readable', 'image_read_error',
    'image_width', 'image_height', 'brightness_mean', 'contrast_std',
    'blur_laplacian', 'underexposed_fraction', 'overexposed_fraction',
    'specular_fraction',
]
IMAGE_QC_FLAG_COLUMNS = [
    'qc_blur_outlier', 'qc_contrast_outlier', 'qc_low_brightness_outlier',
    'qc_high_brightness_outlier', 'qc_thresholds_calibrated',
    'qc_requires_review', 'image_usable_for_modelling',
]
IMAGE_QC_THRESHOLD_COLUMNS = [
    'blur_laplacian_lower_threshold', 'contrast_std_lower_threshold',
    'brightness_mean_lower_threshold', 'brightness_mean_upper_threshold',
    'threshold_calibration_status', 'threshold_fit_split',
    'calibration_candidate_images', 'calibration_readable_images',
    'total_readable_images',
]
IMAGE_QC_INVENTORY_COLUMNS = [
    'image_relpath', 'image_path', 'split',
    'used_by_labelled_manifest', 'used_by_temporal_manifest',
]
IMAGE_QC_MANIFEST_COLUMNS = [
    *IMAGE_QC_INVENTORY_COLUMNS,
    *[name for name in IMAGE_QC_COLUMNS if name != 'image_path'],
    *IMAGE_QC_FLAG_COLUMNS,
]
IMAGE_READ_FAILURE_REPORT_COLUMNS = [
    'image_relpath', 'image_path', 'split', 'image_read_error',
    'used_by_labelled_manifest', 'used_by_temporal_manifest',
]
QC_REFERENCE_COLUMNS = [
    'source_manifest', 'source_row', 'source_reference_id', 'image_path',
    'image_relpath', 'split', 'reference_status',
]
QC_REFERENCE_COVERAGE_COLUMNS = [
    'source_manifest', 'total_references', 'referenced_paths',
    'missing_path_references', 'pending_path_references',
    'qc_covered_references', 'readable_references', 'review_required_references',
]
QC_SUMMARY_COLUMNS = [
    'qc_status', 'total_unique_images', 'readable_images', 'unreadable_images',
    'review_required_images', 'model_usable_images',
    'threshold_calibration_status', 'threshold_fit_split',
    'calibration_candidate_images', 'calibration_readable_images',
]
_QC_BOOL_COLUMNS = frozenset({
    'image_readable', 'used_by_labelled_manifest', 'used_by_temporal_manifest', 'qc_reference_covered',
    *IMAGE_QC_FLAG_COLUMNS,
})
_QC_INT_COLUMNS = frozenset({
    'image_width', 'image_height', 'source_row', 'total_references',
    'referenced_paths', 'missing_path_references', 'pending_path_references',
    'qc_covered_references', 'readable_references', 'review_required_references',
    'total_unique_images', 'readable_images', 'unreadable_images',
    'review_required_images', 'model_usable_images',
    'calibration_candidate_images', 'calibration_readable_images',
    'total_readable_images',
})
_QC_FLOAT_COLUMNS = frozenset({
    'brightness_mean', 'contrast_std', 'blur_laplacian',
    'underexposed_fraction', 'overexposed_fraction', 'specular_fraction',
    *IMAGE_QC_THRESHOLD_COLUMNS[:4],
})


def qc_typed_table(frame):
    """Return stable nullable scalar dtypes, also for empty persisted tables."""
    result = frame.copy()
    for name in result:
        dtype = ('boolean' if name in _QC_BOOL_COLUMNS else
                 'Int64' if name in _QC_INT_COLUMNS else
                 'Float64' if name in _QC_FLOAT_COLUMNS else 'string')
        result[name] = result[name].astype(dtype)
    return result.reset_index(drop=True)


def qc_empty_table(columns):
    return qc_typed_table(pd.DataFrame(columns=columns))


def qc_require_columns(frame, required, name):
    missing = sorted(set(required) - set(frame))
    if missing:
        raise KeyError(f'{name} lacks columns: {missing}')


def validate_qc_quantiles(config):
    """Validate the four quantiles without reading data or mutating config."""
    names = {'blur': 'qc_blur_quantile', 'contrast': 'qc_contrast_quantile',
             'brightness_low': 'qc_brightness_low_quantile',
             'brightness_high': 'qc_brightness_high_quantile'}
    defaults = {'blur': 0.01, 'contrast': 0.01,
                'brightness_low': 0.01, 'brightness_high': 0.99}
    raw = {name: config.get(key, defaults[name]) for name, key in names.items()}
    if any(isinstance(value, (bool, np.bool_)) for value in raw.values()):
        raise ValueError('QC quantiles must be numbers, not booleans.')
    quantiles = pd.to_numeric(pd.Series(raw, dtype='object'), errors='coerce').astype('float64')
    if (quantiles.isna() | ~np.isfinite(quantiles) | quantiles.lt(0) | quantiles.gt(1)).any():
        raise ValueError('QC quantiles must be finite numbers between 0 and 1.')
    if quantiles['brightness_low'] >= quantiles['brightness_high']:
        raise ValueError('The lower brightness quantile must precede the upper quantile.')
    return quantiles


def validate_image_qc_config(config):
    """Fail before image reads when the authored QC settings are invalid."""
    validate_qc_quantiles(config)
    radius = config.get('qc_fov_radius_fraction', 0.95)
    if isinstance(radius, (bool, np.bool_)):
        raise ValueError('qc_fov_radius_fraction must be numeric.')
    try:
        radius = float(radius)
    except (TypeError, ValueError):
        raise ValueError('qc_fov_radius_fraction must be a number in (0, 1].') from None
    if not np.isfinite(radius) or not 0 < radius <= 1:
        raise ValueError('qc_fov_radius_fraction must be a finite number in (0, 1].')
    defaults = {'qc_underexposed_pixel_threshold': 20,
                'qc_overexposed_pixel_threshold': 240,
                'qc_specular_value_threshold': 240,
                'qc_specular_saturation_threshold': 30}
    thresholds = {}
    for name, default in defaults.items():
        value = config.get(name, default)
        if isinstance(value, (bool, np.bool_)):
            raise ValueError(f'{name} must be numeric.')
        try:
            value = float(value)
        except (TypeError, ValueError):
            raise ValueError(f'{name} must be a finite number in [0, 255].') from None
        if not np.isfinite(value) or not 0 <= value <= 255:
            raise ValueError(f'{name} must be a finite number in [0, 255].')
        thresholds[name] = value
    if thresholds['qc_underexposed_pixel_threshold'] >= thresholds['qc_overexposed_pixel_threshold']:
        raise ValueError('The underexposed pixel threshold must be below the overexposed threshold.')
    workers = config.get('qc_workers', 1)
    if isinstance(workers, (bool, np.bool_)) or not isinstance(workers, (int, np.integer)) or workers < 1:
        raise ValueError('qc_workers must be a positive integer.')
    fit_split = config.get('qc_threshold_fit_split', 'train')
    if not isinstance(fit_split, str) or fit_split.strip() != 'train':
        raise ValueError('qc_threshold_fit_split must be train to avoid evaluation leakage.')
    if 'qc_enabled' in config and not isinstance(config['qc_enabled'], (bool, np.bool_)):
        raise ValueError('qc_enabled must be a boolean.')
    if not config.get('qc_enabled', True):
        raise ValueError('Sector 7 QC must be enabled; run it with qc_enabled=True.')
    return {**thresholds, 'qc_fov_radius_fraction': radius,
            'qc_workers': int(workers), 'qc_threshold_fit_split': 'train'}


def build_capsule_fov_mask(image_height, image_width, radius_fraction):
    """Circular field of view; a pure lexical/numeric operation."""
    if (isinstance(radius_fraction, (bool, np.bool_)) or
            not np.isfinite(radius_fraction) or not 0 < radius_fraction <= 1):
        raise ValueError('qc_fov_radius_fraction must be in (0, 1].')
    if (isinstance(image_height, (bool, np.bool_)) or
            isinstance(image_width, (bool, np.bool_)) or
            not isinstance(image_height, (int, np.integer)) or
            not isinstance(image_width, (int, np.integer)) or
            image_height < 1 or image_width < 1):
        raise ValueError('Image dimensions must be positive integers.')
    yy, xx = np.ogrid[:image_height, :image_width]
    radius = min(image_height, image_width) / 2.0 * radius_fraction
    return ((xx - (image_width - 1) / 2.0) ** 2 +
            (yy - (image_height - 1) / 2.0) ** 2 <= radius ** 2)


def build_unreadable_image_qc_record(image_path, error_message):
    return {
        'image_path': pd.NA if pd.isna(image_path) else str(image_path),
        'image_sha256': pd.NA, 'image_readable': False,
        'image_read_error': str(error_message),
        'image_width': pd.NA, 'image_height': pd.NA,
        **{name: np.nan for name in _QC_FLOAT_COLUMNS if name in IMAGE_QC_COLUMNS},
    }


def compute_image_qc(image, config, image_path=None):
    """Compute technical proxies from an already decoded BGR uint8 array.

    This function performs no file reads and never changes the input array.
    The Laplacian uses an eroded FOV so excluded black borders do not inflate
    the sharpness proxy. These values are not clinical image-quality labels.
    """
    parameters = validate_image_qc_config(config)
    if not isinstance(image, np.ndarray) or image.size == 0:
        return build_unreadable_image_qc_record(image_path, 'invalid_or_empty_image_array')
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        return build_unreadable_image_qc_record(image_path, 'expected_bgr_uint8_image')
    height, width = image.shape[:2]
    mask = build_capsule_fov_mask(height, width, parameters['qc_fov_radius_fraction'])
    inner_mask = cv2.erode(mask.astype('uint8'), np.ones((3, 3), dtype='uint8'),
                           borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
    if not mask.any() or not inner_mask.any():
        return build_unreadable_image_qc_record(image_path, 'empty_or_insufficient_fov_mask')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray_pixels, saturation, value = gray[mask], hsv[:, :, 1][mask], hsv[:, :, 2][mask]
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return {
        'image_path': pd.NA if pd.isna(image_path) else str(image_path),
        'image_sha256': pd.NA, 'image_readable': True, 'image_read_error': pd.NA,
        'image_width': int(width), 'image_height': int(height),
        'brightness_mean': float(gray_pixels.mean() / 255.0),
        'contrast_std': float(gray_pixels.std() / 255.0),
        'blur_laplacian': float(laplacian[inner_mask].var()),
        'underexposed_fraction': float(np.mean(gray_pixels < parameters['qc_underexposed_pixel_threshold'])),
        'overexposed_fraction': float(np.mean(gray_pixels > parameters['qc_overexposed_pixel_threshold'])),
        'specular_fraction': float(np.mean(
            (value > parameters['qc_specular_value_threshold']) &
            (saturation < parameters['qc_specular_saturation_threshold']))),
    }


def io_compute_image_qc(image_path, config):
    """Read one immutable byte snapshot, decode it and delegate pure metrics."""
    if pd.isna(image_path) or not str(image_path).strip():
        return build_unreadable_image_qc_record(image_path, 'missing_image_path')
    path = Path(str(image_path))
    if path.is_symlink():
        return build_unreadable_image_qc_record(image_path, 'image_symlink_not_supported')
    try:
        payload = path.read_bytes()
    except FileNotFoundError:
        return build_unreadable_image_qc_record(image_path, 'image_file_not_found')
    except OSError as error:
        return build_unreadable_image_qc_record(image_path, f'{type(error).__name__}: {error}')
    try:
        image = cv2.imdecode(np.frombuffer(payload, dtype='uint8'), cv2.IMREAD_COLOR) if payload else None
        record = compute_image_qc(image, config, image_path)
    except cv2.error as error:
        record = build_unreadable_image_qc_record(image_path, f'opencv_decode_error: {error}')
    # Failed decoding still has a content identity for provenance/reuse checks.
    return {**record, 'image_sha256': hashlib.sha256(payload).hexdigest()}


def _qc_string_or_missing(value):
    return pd.NA if pd.isna(value) or not str(value).strip() else str(value).strip()


def _qc_portable_relpath(value):
    normalized = _qc_string_or_missing(value)
    if pd.isna(normalized):
        return pd.NA
    relative = PurePosixPath(normalized)
    if ('\\' in normalized or relative.is_absolute() or '..' in relative.parts
            or normalized in {'.', ''} or relative.as_posix() != normalized):
        raise ValueError(f'Unsafe or noncanonical image_relpath: {value!r}')
    return normalized


def build_qc_reference_table(labelled_manifest, temporal_manifest, config):
    """Keep every source reference, including missing and pending context paths."""
    root = Path(str(config['storage_root']))
    if not root.is_absolute() or '..' in root.parts:
        raise ValueError('storage_root must be an absolute path without parent traversal.')
    records = []
    specifications = [(labelled_manifest, 'labelled_manifest', 'image_path', 'image_relpath'),
                      (temporal_manifest, 'temporal_manifest', 'context_image_path', 'context_image_relpath')]
    for source, name, path_column, relative_column in specifications:
        qc_require_columns(source, [path_column, relative_column, 'split'], name)
        id_candidates = ['annotation_id', 'image_annotation_id'] if name == 'labelled_manifest' else ['finding_segment_id']
        id_column = next((column for column in id_candidates if column in source), None)
        for position, row in enumerate(source.to_dict('records')):
            path = _qc_string_or_missing(row[path_column])
            relative = _qc_portable_relpath(row[relative_column])
            split = _qc_string_or_missing(row['split'])
            if pd.notna(path):
                supplied_path = Path(path)
                if not supplied_path.is_absolute() or '..' in supplied_path.parts:
                    raise ValueError(f'{name} has an unsafe absolute image path: {path!r}')
                try:
                    inferred_relative = supplied_path.relative_to(root).as_posix()
                except ValueError:
                    raise ValueError(f'{name} image lies outside storage_root: {path}') from None
                if pd.notna(relative) and relative != inferred_relative:
                    raise ValueError(f'{name} image_path disagrees with image_relpath: {path}')
                relative = _qc_portable_relpath(inferred_relative)
                if pd.isna(split):
                    raise ValueError(f'{name} image reference lacks its inherited video split: {relative}')
            status = 'referenced_path' if pd.notna(path) else 'missing_image_path'
            if pd.isna(path) and name == 'temporal_manifest':
                image_source = str(row.get('context_image_source', ''))
                extraction_status = str(row.get('extraction_status', ''))
                if image_source == 'video_extraction_required' or extraction_status in {'not_requested', 'pending', 'not_run'}:
                    status = 'pending_context_image'
            records.append({
                'source_manifest': name, 'source_row': position,
                'source_reference_id': row[id_column] if id_column else pd.NA,
                'image_path': path, 'image_relpath': relative, 'split': split,
                'reference_status': status,
            })
    return qc_typed_table(pd.DataFrame.from_records(records, columns=QC_REFERENCE_COLUMNS))


def build_qc_image_inventory(labelled_manifest, temporal_manifest, config):
    """One physical read per portable image identity; preserve source cardinality."""
    references = build_qc_reference_table(labelled_manifest, temporal_manifest, config)
    available = references.loc[references['image_path'].notna()].copy()
    if available.empty:
        return qc_empty_table(IMAGE_QC_INVENTORY_COLUMNS)
    split_counts = available.groupby('image_relpath', observed=True)['split'].nunique(dropna=False)
    if split_counts.gt(1).any():
        conflicts = split_counts.index[split_counts.gt(1)].tolist()
        raise ValueError(f'An image belongs to different supervised splits: {conflicts[:10]}')
    if available.groupby('image_path', observed=True)['image_relpath'].nunique().gt(1).any():
        raise ValueError('One physical image path maps to multiple portable identities.')
    result = available.assign(
        used_by_labelled_manifest=available['source_manifest'].eq('labelled_manifest'),
        used_by_temporal_manifest=available['source_manifest'].eq('temporal_manifest'),
    ).groupby('image_relpath', as_index=False, observed=True, sort=True).agg(
        image_path=('image_path', 'first'), split=('split', 'first'),
        used_by_labelled_manifest=('used_by_labelled_manifest', 'max'),
        used_by_temporal_manifest=('used_by_temporal_manifest', 'max'))
    return qc_typed_table(result[IMAGE_QC_INVENTORY_COLUMNS])


def io_compute_qc_for_image_inventory(image_inventory, config):
    """Bounded CPU image reads; executor.map retains inventory order."""
    parameters = validate_image_qc_config(config)
    qc_require_columns(image_inventory, IMAGE_QC_INVENTORY_COLUMNS, 'QC inventory')
    if not image_inventory['image_relpath'].is_unique or not image_inventory['image_path'].is_unique:
        raise ValueError('QC inventory identities and paths must be unique.')
    columns = IMAGE_QC_INVENTORY_COLUMNS + [name for name in IMAGE_QC_COLUMNS if name != 'image_path']
    if image_inventory.empty:
        return qc_empty_table(columns)
    paths = image_inventory['image_path'].tolist()
    def read_one(path):
        return io_compute_image_qc(path, config)
    with ThreadPoolExecutor(max_workers=parameters['qc_workers']) as executor:
        records = list(tqdm(executor.map(read_one, paths), total=len(paths), desc='Sector 7 image QC'))
    metrics = qc_typed_table(pd.DataFrame.from_records(records, columns=IMAGE_QC_COLUMNS))
    result = image_inventory.merge(metrics, on='image_path', how='left', sort=False, validate='one_to_one')
    return qc_typed_table(result[columns])


def derive_image_qc_thresholds(image_qc_manifest, config):
    """Train-only calibration; unavailable train images yield explicit status."""
    parameters = validate_image_qc_config(config)
    quantiles = validate_qc_quantiles(config)
    qc_require_columns(image_qc_manifest, ['image_readable', 'split', 'image_relpath',
        'brightness_mean', 'contrast_std', 'blur_laplacian'], 'QC metric manifest')
    if not image_qc_manifest['image_relpath'].is_unique:
        raise ValueError('Threshold fitting requires one row per image identity.')
    readable = image_qc_manifest['image_readable'].fillna(False).astype(bool)
    metrics = image_qc_manifest.loc[readable, ['brightness_mean', 'contrast_std', 'blur_laplacian']].apply(
        pd.to_numeric, errors='coerce')
    if (metrics.isna().any(axis=1) | ~np.isfinite(metrics.astype('float64')).all(axis=1)).any():
        raise RuntimeError('Readable images have missing or non-finite QC metrics.')
    candidates = image_qc_manifest['split'].eq(parameters['qc_threshold_fit_split']).fillna(False)
    fitted = image_qc_manifest.loc[candidates & readable, metrics.columns].astype('float64')
    status = ('calibrated' if not fitted.empty else
              'uncalibrated_no_train_images' if not candidates.any() else
              'uncalibrated_no_readable_train_images')
    record = {name: np.nan for name in IMAGE_QC_THRESHOLD_COLUMNS[:4]}
    if not fitted.empty:
        record.update({
            'blur_laplacian_lower_threshold': fitted['blur_laplacian'].quantile(quantiles['blur']),
            'contrast_std_lower_threshold': fitted['contrast_std'].quantile(quantiles['contrast']),
            'brightness_mean_lower_threshold': fitted['brightness_mean'].quantile(quantiles['brightness_low']),
            'brightness_mean_upper_threshold': fitted['brightness_mean'].quantile(quantiles['brightness_high']),
        })
    record.update({
        'threshold_calibration_status': status, 'threshold_fit_split': 'train',
        'calibration_candidate_images': int(candidates.sum()),
        'calibration_readable_images': len(fitted), 'total_readable_images': int(readable.sum()),
    })
    return qc_typed_table(pd.DataFrame.from_records([record], columns=IMAGE_QC_THRESHOLD_COLUMNS))


def add_image_qc_flags(image_qc_manifest, image_qc_thresholds):
    """Technical flags request review; every readable image remains usable."""
    qc_require_columns(image_qc_thresholds, IMAGE_QC_THRESHOLD_COLUMNS, 'QC thresholds')
    if len(image_qc_thresholds) != 1:
        raise ValueError('Image QC thresholds must contain exactly one status row.')
    threshold = image_qc_thresholds.iloc[0]
    calibrated = threshold['threshold_calibration_status'] == 'calibrated'
    values = pd.to_numeric(threshold[IMAGE_QC_THRESHOLD_COLUMNS[:4]], errors='coerce')
    if calibrated and (values.isna().any() or not np.isfinite(values.astype('float64')).all()):
        raise ValueError('Calibrated thresholds must contain finite metric values.')
    readable = image_qc_manifest['image_readable'].fillna(False).astype(bool)
    result = image_qc_manifest.copy()
    comparisons = {
        'qc_blur_outlier': ('blur_laplacian', 'blur_laplacian_lower_threshold', 'lt'),
        'qc_contrast_outlier': ('contrast_std', 'contrast_std_lower_threshold', 'lt'),
        'qc_low_brightness_outlier': ('brightness_mean', 'brightness_mean_lower_threshold', 'lt'),
        'qc_high_brightness_outlier': ('brightness_mean', 'brightness_mean_upper_threshold', 'gt'),
    }
    for flag, (metric, boundary, operator) in comparisons.items():
        # Strict comparisons avoid labelling all tied minimum/maximum values.
        result[flag] = ((readable & getattr(result[metric], operator)(threshold[boundary]).fillna(False))
                        if calibrated else pd.Series(False, index=result.index, dtype='boolean'))
    result['qc_thresholds_calibrated'] = calibrated
    result['qc_requires_review'] = ~readable | result[list(comparisons)].any(axis=1)
    result['image_usable_for_modelling'] = readable
    return qc_typed_table(result[IMAGE_QC_MANIFEST_COLUMNS].sort_values('image_relpath', kind='stable'))


def build_image_read_failure_report(image_qc_manifest):
    qc_require_columns(image_qc_manifest, IMAGE_READ_FAILURE_REPORT_COLUMNS + ['image_readable'], 'QC manifest')
    return qc_typed_table(image_qc_manifest.loc[
        ~image_qc_manifest['image_readable'].fillna(False), IMAGE_READ_FAILURE_REPORT_COLUMNS
    ].sort_values('image_relpath', kind='stable'))


def build_qc_reference_reports(labelled_manifest, temporal_manifest, image_qc_manifest, config):
    """Report missing reference rows separately from unreadable physical files."""
    references = build_qc_reference_table(labelled_manifest, temporal_manifest, config)
    missing = qc_typed_table(references.loc[references['image_path'].isna(), QC_REFERENCE_COLUMNS])
    metrics = image_qc_manifest[['image_relpath', 'image_readable', 'qc_requires_review']].copy()
    metrics['_qc_found'] = True
    joined = references.merge(metrics, on='image_relpath', how='left', validate='many_to_one')
    records = []
    for source in ('labelled_manifest', 'temporal_manifest'):
        rows = joined.loc[joined['source_manifest'].eq(source)]
        has_path = rows['image_path'].notna()
        covered = has_path & rows['_qc_found'].astype('boolean').fillna(False).astype(bool)
        records.append({
            'source_manifest': source, 'total_references': len(rows),
            'referenced_paths': int(has_path.sum()),
            'missing_path_references': int((~has_path).sum()),
            'pending_path_references': int(rows['reference_status'].eq('pending_context_image').sum()),
            'qc_covered_references': int(covered.sum()),
            'readable_references': int((covered & rows['image_readable'].fillna(False)).sum()),
            'review_required_references': int((~covered | rows['qc_requires_review'].fillna(True)).sum()),
        })
    covered = joined['image_path'].notna() & joined['_qc_found'].astype('boolean').fillna(False).astype(bool)
    joined['qc_reference_covered'] = covered
    joined['image_readable'] = covered & joined['image_readable'].fillna(False)
    joined['qc_requires_review'] = ~covered | joined['qc_requires_review'].fillna(True)
    reference_report = qc_typed_table(joined[QC_REFERENCE_COLUMNS + [
        'qc_reference_covered', 'image_readable', 'qc_requires_review']])
    return {
        'missing_image_reference_report': missing,
        'image_read_failure_report': build_image_read_failure_report(image_qc_manifest),
        'qc_reference_report': reference_report,
        'qc_reference_coverage_report': qc_typed_table(pd.DataFrame.from_records(
            records, columns=QC_REFERENCE_COVERAGE_COLUMNS)),
    }


def attach_image_qc_to_manifests(labelled_manifest, temporal_manifest, image_qc_manifest, config):
    """Many-to-one enrichment retains every annotation, box and context row.

    Metric columns receive a qc_ prefix so Sector 6 image audit fields survive.
    The original image_path/context_image_path and every source column are kept.
    """
    references = build_qc_reference_table(labelled_manifest, temporal_manifest, config)
    metric_columns = [name for name in IMAGE_QC_COLUMNS if name != 'image_path']
    added_columns = {name: 'qc_' + name for name in metric_columns}
    extra_columns = [name for name in image_qc_manifest
                     if name.startswith('qc_') and name not in IMAGE_QC_FLAG_COLUMNS
                     and name not in added_columns.values()]
    attached = image_qc_manifest[['image_relpath'] + metric_columns + IMAGE_QC_FLAG_COLUMNS + extra_columns].rename(columns=added_columns)
    for name in added_columns.values():
        if name in image_qc_manifest and name != 'image_relpath':
            attached[name] = image_qc_manifest[name].array
    if attached['image_relpath'].isna().any() or not attached['image_relpath'].is_unique:
        raise ValueError('QC enrichment requires unique non-null image identities.')
    outputs = {}
    for source, name in [(labelled_manifest, 'labelled_manifest'), (temporal_manifest, 'temporal_manifest')]:
        reserved = set(attached.columns) - {'image_relpath'}
        collisions = sorted(reserved.intersection(source.columns))
        if collisions:
            raise ValueError(f'{name} already contains Sector 7 columns: {collisions}')
        own_references = references.loc[references['source_manifest'].eq(name)]
        result = source.copy()
        result['_qc_source_position'] = np.arange(len(source))
        result['_qc_join_relpath'] = pd.Series(own_references['image_relpath'].array, index=result.index)
        result['_qc_source_has_path'] = pd.Series(own_references['image_path'].notna().to_numpy(), index=result.index)
        lookup = attached.rename(columns={'image_relpath': '_qc_join_relpath'})
        lookup['qc_reference_covered'] = True
        result = result.merge(lookup, on='_qc_join_relpath', how='left', sort=False, validate='many_to_one')
        covered = result['_qc_source_has_path'] & result['qc_reference_covered'].astype('boolean').fillna(False).astype(bool)
        result['qc_reference_covered'] = covered.astype('boolean')
        for flag in IMAGE_QC_FLAG_COLUMNS:
            default = flag == 'qc_requires_review'
            result[flag] = result[flag].astype('boolean').fillna(default)
            if flag in {'qc_requires_review', 'image_usable_for_modelling'}:
                result.loc[~covered, flag] = default
        result['qc_image_readable'] = result['qc_image_readable'].astype('boolean').fillna(False)
        result.loc[~covered, 'qc_image_readable'] = False
        result = result.sort_values('_qc_source_position', kind='stable').drop(
            columns=['_qc_source_position', '_qc_join_relpath', '_qc_source_has_path']).reset_index(drop=True)
        if len(result) != len(source) or list(result.columns[:len(source.columns)]) != list(source.columns):
            raise RuntimeError('QC enrichment changed source manifest cardinality or original schema order.')
        outputs['labelled_frame_manifest_with_qc' if name == 'labelled_manifest' else 'temporal_manifest_with_qc'] = result
    return outputs


def build_image_qc_summary(image_qc_manifest, image_qc_thresholds, reports, config):
    row = image_qc_thresholds.iloc[0]
    readable = image_qc_manifest['image_readable'].fillna(False)
    record = {
        'qc_status': 'completed' if config.get('qc_enabled', True) else 'disabled',
        'total_unique_images': len(image_qc_manifest), 'readable_images': int(readable.sum()),
        'unreadable_images': int((~readable).sum()),
        'review_required_images': int(image_qc_manifest['qc_requires_review'].fillna(False).sum()),
        'model_usable_images': int(image_qc_manifest['image_usable_for_modelling'].fillna(False).sum()),
        **{name: row[name] for name in [
            'threshold_calibration_status', 'threshold_fit_split',
            'calibration_candidate_images', 'calibration_readable_images']},
    }
    return qc_typed_table(pd.DataFrame.from_records([record], columns=QC_SUMMARY_COLUMNS))


def _s5_core_code_value(code):
    """Describe executable code without filenames, source lines or debug positions."""
    fields = (
        "co_argcount", "co_posonlyargcount", "co_kwonlyargcount", "co_nlocals",
        "co_stacksize", "co_flags", "co_code", "co_consts", "co_names", "co_varnames",
        "co_name", "co_qualname", "co_cellvars", "co_freevars", "co_exceptiontable",
    )
    return {field: _s5_core_value(getattr(code, field)) for field in fields if hasattr(code, field)}

def _s5_core_value(value, active=None):
    """Canonicalize scalar/container code values and defaults without changing them."""
    if isinstance(value, CodeType):
        return {"code": _s5_core_code_value(value)}
    if value is Ellipsis:
        return {"ellipsis": True}
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return {"float": value.hex()} if math.isfinite(value) else {"float": repr(value)}
    if isinstance(value, complex):
        return {"complex": [_s5_core_value(value.real), _s5_core_value(value.imag)]}
    if isinstance(value, bytes):
        return {"bytes": value.hex()}
    if not isinstance(value, (dict, list, tuple, set, frozenset)):
        raise TypeError(f"Unsupported active core identity value: {type(value).__name__}")
    active = set() if active is None else active
    identity = id(value)
    if identity in active:
        raise ValueError("Cyclic containers cannot define a deterministic core identity.")
    nested = active | {identity}
    if isinstance(value, dict):
        pairs = [[_s5_core_value(key, nested), _s5_core_value(item, nested)] for key, item in value.items()]
        return {"dict": sorted(pairs, key=lambda pair: _s5_core_identity_json(pair[0]))}
    items = [_s5_core_value(item, nested) for item in value]
    if isinstance(value, (set, frozenset)):
        items = sorted(items, key=_s5_core_identity_json)
    return {type(value).__name__: items}

def _s5_core_identity_json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)

def _s5_core_referenced_names(code):
    """Include globals referenced inside comprehensions and other nested code."""
    children = (item for item in code.co_consts if isinstance(item, CodeType))
    return set(code.co_names).union(*( _s5_core_referenced_names(child) for child in children))

def _s5_core_function_value(function):
    closure = tuple(cell.cell_contents for cell in (function.__closure__ or ()))
    return {
        "code": _s5_core_code_value(function.__code__),
        "defaults": _s5_core_value(function.__defaults__),
        "kwdefaults": _s5_core_value(function.__kwdefaults__),
        "closure": _s5_core_value(closure),
    }

def sector6_runtime_core_sha256(entrypoint, namespace, dependency_versions):
    """Fingerprint active core functions, referenced S3 constants and library versions.

    Only Python functions reachable through executable global references and
    referenced ``S5_*`` constants participate. Unrelated configuration, clocks and
    notebook state are ignored. Recursive function graphs terminate through a
    visited-name set; repeated cell executions keep the same identity.
    """
    if not isinstance(entrypoint, FunctionType) or not isinstance(namespace, dict):
        raise TypeError("Provide a Python entrypoint function and its explicit namespace dictionary.")
    if not isinstance(dependency_versions, dict):
        raise TypeError("Dependency versions must be an explicit dictionary.")
    pending = {"entrypoint": entrypoint}
    functions, constants = {}, {}
    while pending:
        name = min(pending)
        function = pending.pop(name)
        if name in functions:
            continue
        functions[name] = _s5_core_function_value(function)
        for reference in sorted(_s5_core_referenced_names(function.__code__)):
            value = namespace.get(reference)
            if isinstance(value, FunctionType) and reference not in functions:
                pending[reference] = value
            elif reference.isupper() and reference in namespace and isinstance(value, (str, bool, int, float, bytes, dict, list, tuple, set, frozenset, type(None))):
                constants[reference] = _s5_core_value(value)
    payload = {
        "schema_version": 1, "functions": functions, "constants": constants,
        "dependency_versions": _s5_core_value(dependency_versions),
    }
    return hashlib.sha256(_s5_core_identity_json(payload).encode("utf-8")).hexdigest()

sector7_runtime_core_sha256 = sector6_runtime_core_sha256

S7_STATE_PATH = 'configs/sector7_state.json'
S7_RECEIPT_PATH = 'configs/sector7_publication_receipt.json'
S7_LINEAGE_PATH = 'configs/sector7_lineage.json'
S7_RESULT_NAMES = frozenset({'image_qc_inventory', 'image_qc_manifest', 'image_qc_thresholds', 'labelled_frame_manifest_with_qc', 'temporal_manifest_with_qc'})
S7_REPORT_NAMES = frozenset({'image_qc_summary', 'image_read_failure_report', 'qc_reference_report', 'image_qc_source_audit', 'missing_image_reference_report'})
_p7_safe_path = _p3_safe_path
_p7_file_hash = _p3_file_hash

def _p7_write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_p3_json_text(value), encoding='utf-8')
    if _p3_json_text(json.loads(path.read_text(encoding='utf-8'))) != _p3_json_text(value):
        raise RuntimeError(f'Sector 7 JSON readback failed: {path.name}')

def _p7_assert_no_secrets(value, path='configuration'):
    """Save credential references, never actual credential material."""
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key).lower() in {'token', 'access_token', 'github_token', 'password', 'api_key', 'authorization', 'secret', 'private_key', 'client_secret'} and item:
                raise ValueError(f'Credential material cannot be saved at {path}.{key}; use token_secret_name.')
            _p7_assert_no_secrets(item, f'{path}.{key}')
    elif isinstance(value, (list, tuple)):
        for item in value:
            _p7_assert_no_secrets(item, path)
    elif isinstance(value, str) and (re.search('https?://[^/\\s]*:[^/@\\s]+@', value) or re.search('(?:gh[pousr]_[A-Za-z0-9_]{20,}|github_pat_[A-Za-z0-9_]{20,})', value)):
        raise ValueError(f'Credential material cannot be saved at {path}.')

def sector7_semantic_config_fingerprint(config):
    projected = sector4_semantic_config(config)
    projected.pop('github', None)
    sector = projected.get('sector7', {})
    for key in ('push', 'dry_run', 'github_overrides'):
        sector.pop(key, None)
    return _s3_semantic_digest(projected)


def sector7_processing_signature(config, lineage):
    """Pure identity of the effective config, consumed input bytes and active code."""
    hashes, core_hash = (lineage.get('input_sha256'), lineage.get('core_source_sha256'))
    if not isinstance(hashes, dict) or not hashes or (not isinstance(core_hash, str)):
        raise ValueError('Sector 7 lineage needs input_sha256 and core_source_sha256.')
    return _p3_json_hash({'phase': 2, 'sector': 7, 'schema_version': 2, 'semantic_config_fingerprint': sector7_semantic_config_fingerprint(config), 'input_sha256': hashes, 'core_source_sha256': core_hash, 'image_source_audit_sha256': lineage.get('image_source_audit_sha256')})

def _p7_lineage_payload(lineage):
    """Current provenance, excluding clock metadata while retaining input identities.

    Convert diagnostic DataFrames and other supported values before applying
    Sector 4's clock-only projection; hashes, descriptors, dependency versions,
    decoder settings and the upstream artifact chain keep their full meaning.
    """
    return sector4_semantic_config(json.loads(_p3_json_text(lineage)))

def _p7_root(config):
    declared = Path(config.get('sector7', {}).get('local_output_dir', 'outputs/phase2/sector7')).expanduser()
    return (declared if declared.is_absolute() else Path(config['storage_root']) / declared).resolve()

def _p7_validate_lineage(lineage):
    """Recheck every consumed upstream file immediately before publishing local results."""
    hashes, descriptors = (lineage.get('input_sha256'), lineage.get('input_descriptors'))
    if not isinstance(hashes, dict) or not hashes or (not isinstance(descriptors, dict)):
        raise ValueError('Sector 7 lineage needs input_sha256 and input_descriptors from checked input loading.')
    if set(hashes) != set(descriptors):
        raise ValueError('Sector 7 input file descriptors do not match its consumed input hashes.')
    for key, digest in hashes.items():
        if not isinstance(digest, str) or re.fullmatch('[0-9a-f]{64}', digest) is None:
            raise ValueError(f'Invalid consumed input SHA-256 in Sector 7 lineage: {key}')
        path = Path(descriptors[key]).expanduser()
        if not path.is_absolute() or not path.is_file() or path.is_symlink():
            raise RuntimeError(f'Consumed upstream artifact is unavailable or unsafe: {key}')
        if _p3_file_hash(path) != digest:
            raise RuntimeError(f'Consumed upstream artifact changed after loading: {key}; rerun input loading.')
    core_hash = lineage.get('core_source_sha256')
    if not isinstance(core_hash, str) or re.fullmatch('[0-9a-f]{64}', core_hash) is None:
        raise ValueError('Sector 7 lineage lacks the active transformation SHA-256; run the processing-signature cell.')

def _p7_read_state(root):
    """Validate the complete flat catalog, current config, schemas and report bytes."""
    root = Path(root).resolve()
    state_path = _p3_safe_path(root, S7_STATE_PATH)
    if not state_path.is_file():
        return None
    state = json.loads(state_path.read_text(encoding='utf-8'))
    if state.get('schema_version') != 2 or state.get('complete') is not True:
        raise RuntimeError('Sector 7 catalog is incomplete or has an unsupported schema.')
    if state.get('content_id') != _p3_json_hash(_p3_catalog_meaning(state)):
        raise RuntimeError('Sector 7 catalog fingerprint is invalid.')
    records = {record['relative_path']: record for record in state['artifacts']}
    if len(records) != len(state['artifacts']) or {S7_STATE_PATH, S7_RECEIPT_PATH} & set(records):
        raise RuntimeError('Sector 7 catalog contains duplicate or operational artifact entries.')
    for relative, record in records.items():
        path = _p3_safe_path(root, relative)
        if not path.is_file() or path.stat().st_size != record['bytes'] or _p3_file_hash(path) != record['sha256']:
            raise RuntimeError(f'Sector 7 saved artifact checksum verification failed: {relative}')
    if state['config_snapshot'] not in records:
        raise RuntimeError('Sector 7 current configuration snapshot is absent from its catalog.')
    envelope = json.loads(_p3_safe_path(root, state['config_snapshot']).read_text(encoding='utf-8'))
    if state.get('config_snapshot_fingerprint') != _p3_json_hash(sector4_semantic_config(envelope['effective_config'])):
        raise RuntimeError('Sector 7 full effective configuration fingerprint is invalid.')
    if envelope.get('semantic_config_fingerprint') != state['semantic_config_fingerprint'] or sector7_semantic_config_fingerprint(envelope['effective_config']) != state['semantic_config_fingerprint']:
        raise RuntimeError('Sector 7 saved effective configuration disagrees with its catalog.')
    if S7_LINEAGE_PATH not in records:
        raise RuntimeError('Sector 7 current lineage is absent from its catalog.')
    saved_lineage = json.loads(_p3_safe_path(root, S7_LINEAGE_PATH).read_text(encoding='utf-8'))
    if saved_lineage.get('input_sha256') != state['input_sha256'] or saved_lineage.get('core_source_sha256') != state['core_source_sha256'] or (not isinstance(saved_lineage.get('input_descriptors'), dict)) or (set(saved_lineage['input_descriptors']) != set(state['input_sha256'])):
        raise RuntimeError('Sector 7 saved current lineage disagrees with its catalog.')
    if _p3_json_hash(saved_lineage) != records[S7_LINEAGE_PATH]['sha256']:
        raise RuntimeError('Sector 7 current lineage is not canonically encoded.')
    if sector7_processing_signature(envelope['effective_config'], saved_lineage) != state['processing_signature']:
        raise RuntimeError('Sector 7 processing signature disagrees with saved provenance.')
    audit_descriptor = state['tables'].get('image_qc_source_audit')
    if audit_descriptor is None or audit_descriptor['fingerprint'] != saved_lineage.get('image_source_audit_sha256'):
        raise RuntimeError('Sector 7 image audit identity disagrees with saved provenance.')
    for name, descriptor in state['tables'].items():
        if not {descriptor['relative_path'], descriptor['schema_path']}.issubset(records):
            raise RuntimeError(f'Sector 7 table/schema absent from catalog: {name}')
        schema = json.loads(_p3_safe_path(root, descriptor['schema_path']).read_text(encoding='utf-8'))
        if schema.get('artifact') != descriptor['relative_path'] or schema.get('fingerprint') != descriptor['fingerprint']:
            raise RuntimeError(f'Sector 7 saved table schema disagrees with catalog: {name}')
    summary = state['summary']
    if not {summary['summary_relative_path'], summary['report_relative_path']}.issubset(records):
        raise RuntimeError('Sector 7 summary/report is absent from its catalog.')
    saved_summary = json.loads(_p3_safe_path(root, summary['summary_relative_path']).read_text(encoding='utf-8'))
    if _p3_json_hash(saved_summary) != summary['fingerprint'] or records[summary['report_relative_path']]['sha256'] != summary['report_fingerprint']:
        raise RuntimeError('Sector 7 saved summary/report disagrees with its catalog.')
    for relative in state.get('obsolete_paths', []):
        if not _p3_relative(relative).startswith('results/') or relative in records:
            raise RuntimeError('Invalid Sector 7 obsolete-result deletion authorization.')
    return state

def io_load_cached_sector7(config, processing_signature):
    """Optional checked cache; no I/O writes or Git operations."""
    root = _p7_root(config)
    state = _p7_read_state(root)
    if state is None or state['processing_signature'] != processing_signature:
        return None
    return {'tables': {name: _p3_read_table(root, descriptor) for name, descriptor in state['tables'].items()}, 'state': state}

def _p7_summary(tables):
    """Pure summary supplied by the Sector 7 transformation cell."""
    return build_sector7_summary(tables)

def _p7_executive(summary):
    return ('# Phase 2 — Sector 7: Image Quality Control\n\n'
            'Unique labelled and temporal images receive circular-field-of-view technical quality metrics. '
            'Metrics are joined to all original annotation and temporal references without filtering rows.\n\n'
            f"Unique images: {summary['qc']['total_unique_images']}; readable: {summary['qc']['readable_images']}; "
            f"review required: {summary['qc']['review_required_images']}.\n\n"
            'Thresholds are calibrated on the configured training split by default. Unreadable images '
            'are excluded from calibration and marked unusable; technical outliers remain readable and '
            'require review. These proxies are not clinical quality labels.\n\n'
            'All tables are saved as Parquet. Reports also have CSV exports. Config, schemas, lineage and '
            'catalog hashes support checked reuse and later phases. Image binaries remain on Google Drive.\n')


def _p7_write_table(stage, name, frame, fingerprint, stamp):
    is_result = name in S7_RESULT_NAMES
    base = f'results/{name}' if is_result else f'reports/{name}_{stamp}'
    relative = base + '.parquet'
    path = stage / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False, compression='zstd', engine='pyarrow')
    validate_persisted_dataframe(frame, path, name)
    if not is_result:
        frame.to_csv(stage / (base + '.csv'), index=False)
    schema_relative = base + '_schema.json'
    _p7_write_json(stage / schema_relative, {**_p3_schema(frame), 'artifact': relative, 'fingerprint': fingerprint})
    return {'relative_path': relative, 'schema_path': schema_relative, 'fingerprint': fingerprint}


def _p7_install(stage, root, changed_paths, obsolete_paths):
    """Rollback on errors and replace the current state/catalog last."""
    if not root.exists():
        stage.rename(root)
        try:
            installed_state = _p7_read_state(root)
            for descriptor in installed_state['tables'].values():
                _p3_read_table(root, descriptor)
        except BaseException:
            root.rename(stage)
            raise
        return
    backup = Path(tempfile.mkdtemp(prefix='.sector7-backup-', dir=root.parent))
    installed, moved = ([], [])
    sequence = [path for path in changed_paths if path != S7_STATE_PATH] + list(obsolete_paths) + [S7_STATE_PATH]
    try:
        for relative in dict.fromkeys(sequence):
            target = _p3_safe_path(root, relative)
            if target.exists():
                saved = backup / relative
                saved.parent.mkdir(parents=True, exist_ok=True)
                target.replace(saved)
                moved.append(relative)
            source = stage / relative
            if source.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                source.replace(target)
                installed.append(relative)
        installed_state = _p7_read_state(root)
        for descriptor in installed_state['tables'].values():
            _p3_read_table(root, descriptor)
    except BaseException:
        for relative in reversed(installed):
            (root / relative).unlink(missing_ok=True)
        for relative in moved:
            target = root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            (backup / relative).replace(target)
        raise
    finally:
        shutil.rmtree(backup, ignore_errors=True)

def io_persist_sector7(tables, config, lineage):
    """Save/read back Sector 7 outputs; leave all saved files untouched when unchanged."""
    signature = sector7_processing_signature(config, lineage)
    if not isinstance(tables, dict) or not tables or any((not isinstance(frame, pd.DataFrame) for frame in tables.values())):
        raise TypeError('Sector 7 output tables must be a nonempty dictionary of DataFrames.')
    if set(tables) != S7_RESULT_NAMES | S7_REPORT_NAMES:
        raise ValueError('Pass all Sector 7 result and report tables.')
    if sector3_table_fingerprint(tables['image_qc_source_audit']) != lineage.get('image_source_audit_sha256'):
        raise ValueError('Sector 7 source audit changed after signature preparation; rerun cache/computation.')
    _p7_assert_no_secrets(config)
    _p7_assert_no_secrets(lineage)
    _p7_validate_lineage(lineage)
    io_validate_sector7_source_audit(tables['image_qc_source_audit'])
    root = _p7_root(config)
    previous = _p7_read_state(root)
    config_hash = sector7_semantic_config_fingerprint(config)
    snapshot_hash = _p3_json_hash(sector4_semantic_config(config))
    provenance = {'input_sha256': copy.deepcopy(lineage['input_sha256']), 'core_source_sha256': lineage['core_source_sha256']}
    lineage_payload = _p7_lineage_payload(lineage)
    _p7_assert_no_secrets(lineage_payload, 'lineage')
    lineage_hash = _p3_json_hash(lineage_payload)
    fingerprints = {name: sector3_table_fingerprint(table) for name, table in tables.items()}
    summary = _p7_summary(tables)
    summary_hash = _p3_json_hash(summary)
    report_text = _p7_executive(summary)
    report_hash = hashlib.sha256(report_text.encode('utf-8')).hexdigest()
    same_tables = previous is not None and set(fingerprints) == set(previous['tables']) and all((previous['tables'][name]['fingerprint'] == value for name, value in fingerprints.items()))
    if previous and same_tables and all((previous[key] == value for key, value in provenance.items())) and (previous.get('config_snapshot_fingerprint') == snapshot_hash and previous['semantic_config_fingerprint'] == config_hash and previous['summary']['fingerprint'] == summary_hash and (previous['processing_signature'] == signature) and (previous['summary']['report_fingerprint'] == report_hash) and any((record['relative_path'] == S7_LINEAGE_PATH and record['sha256'] == lineage_hash for record in previous['artifacts']))):
        for descriptor in previous['tables'].values():
            _p3_read_table(root, descriptor)
        return {'status': 'unchanged', 'root': str(root), 'local_root': str(root), 'state': previous, 'artifacts': previous['artifacts'], 'changed_paths': []}
    root.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='.sector7-stage-', dir=root.parent))
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    try:
        catalog = {record['relative_path']: record for record in previous['artifacts']} if previous else {}
        descriptors, obsolete = ({}, set(previous.get('obsolete_paths', [])) if previous else set())
        if catalog.get(S7_LINEAGE_PATH, {}).get('sha256') != lineage_hash:
            _p7_write_json(stage / S7_LINEAGE_PATH, lineage_payload)
        config_snapshot = previous['config_snapshot'] if previous else None
        if not previous or previous.get('config_snapshot_fingerprint') != snapshot_hash:
            config_snapshot = f'configs/sector7_config_{stamp}.json'
            _p7_write_json(stage / config_snapshot, {'schema_version': 2, 'changed_at_utc': stamp, 'semantic_config_fingerprint': config_hash, 'effective_config': config, 'current_lineage_path': S7_LINEAGE_PATH, 'lineage': lineage})
        for name, frame in sorted(tables.items()):
            old = previous['tables'].get(name) if previous else None
            if old and old['fingerprint'] == fingerprints[name]:
                _p3_read_table(root, old)
                descriptors[name] = old
            else:
                descriptors[name] = _p7_write_table(stage, name, frame, fingerprints[name], stamp)
                if old and old['relative_path'].startswith('results/') and (old['relative_path'] != descriptors[name]['relative_path']):
                    for relative in (old['relative_path'], old['schema_path']):
                        obsolete.add(relative)
                        catalog.pop(relative, None)
        previous_summary = previous['summary'] if previous else {}
        summary_path = previous_summary.get('summary_relative_path')
        if previous_summary.get('fingerprint') != summary_hash:
            summary_path = f'reports/summary_{stamp}.json'
            _p7_write_json(stage / summary_path, summary)
        report_path = previous_summary.get('report_relative_path')
        if catalog.get(report_path, {}).get('sha256') != report_hash:
            report_path = f'reports/executive_report_{stamp}.md'
            (stage / report_path).write_text(report_text, encoding='utf-8')
        summary_descriptor = {'fingerprint': summary_hash, 'summary_relative_path': summary_path, 'report_fingerprint': report_hash, 'report_relative_path': report_path}
        changed = []
        for path in sorted(stage.rglob('*')):
            if path.is_file():
                relative = path.relative_to(stage).as_posix()
                record = {'relative_path': relative, 'sha256': _p3_file_hash(path), 'bytes': path.stat().st_size}
                if catalog.get(relative) == record:
                    path.unlink()
                else:
                    catalog[relative] = record
                    changed.append(relative)
        obsolete.difference_update(catalog)
        state = {'schema_version': 2, 'complete': True, 'processing_signature': signature, 'semantic_config_fingerprint': config_hash, 'config_snapshot': config_snapshot, 'config_snapshot_fingerprint': snapshot_hash, **provenance, 'tables': descriptors, 'summary': summary_descriptor, 'artifacts': [catalog[path] for path in sorted(catalog)], 'obsolete_paths': sorted(obsolete)}
        state['content_id'] = _p3_json_hash(_p3_catalog_meaning(state))
        state['changed_at_utc'] = stamp
        _p7_write_json(stage / S7_STATE_PATH, state)
        changed.append(S7_STATE_PATH)
        _p7_validate_lineage(lineage)
        io_validate_sector7_source_audit(tables['image_qc_source_audit'])
        _p7_install(stage, root, changed, sorted(obsolete))
        _p7_read_state(root)
        for descriptor in state['tables'].values():
            _p3_read_table(root, descriptor)
        return {'status': 'updated', 'root': str(root), 'local_root': str(root), 'state': state, 'artifacts': state['artifacts'], 'changed_paths': changed}
    finally:
        shutil.rmtree(stage, ignore_errors=True)

def _p7_read_github_token(secret_name):
    if not isinstance(secret_name, str) or not secret_name or re.search('\\s', secret_name):
        raise ValueError('github.token_secret_name must be a nonempty secret reference.')
    token = os.environ.get(secret_name)
    if isinstance(token, str) and token.strip():
        return token.strip()
    if userdata is None:
        raise RuntimeError(f'GitHub secret {secret_name!r} is unavailable; run from Colab or supply an environment variable with this name.')
    try:
        token = userdata.get(secret_name)
    except Exception as error:
        raise RuntimeError(f'Cannot read Colab secret {secret_name!r} ({type(error).__name__}). Enable Notebook access. Saved Sector 7 outputs can be reused.') from None
    if not isinstance(token, str) or not token.strip():
        raise RuntimeError('GitHub secret is empty or invalid.')
    return token.strip()

def _p7_git_options(config):
    """Validate inherited repository settings and confine exports to Sector 7."""
    github, sector = (config.get('github', {}), config.get('sector7', {}))
    if not isinstance(github, dict) or not isinstance(sector, dict):
        raise TypeError('github and sector7 configuration must be dictionaries.')

    def flag(name, default):
        value = sector.get(name, github.get(name, default))
        if not isinstance(value, bool):
            raise TypeError(f'{name} must be a Python/JSON boolean.')
        return value
    push, dry_run = (flag('push', True), flag('dry_run', False))
    if not push and (not dry_run):
        return {'github': github, 'push': False, 'dry_run': False, 'branch': github.get('branch', 'main'), 'export_dir': 'outputs/phase2/sector7', 'export_parts': PurePosixPath('outputs/phase2/sector7')}
    allow_local = github.get('allow_local_test_remote', sector.get('allow_local_test_remote', False))
    if not isinstance(allow_local, bool):
        raise TypeError('allow_local_test_remote must be a Python/JSON boolean.')
    export_dir = sector.get('export_dir', github.get('repo_output_dir', 'outputs/phase2/sector7'))
    if not isinstance(export_dir, str) or export_dir != 'outputs/phase2/sector7':
        raise ValueError('Sector 7 exports must use outputs/phase2/sector7.')
    export_parts = PurePosixPath(export_dir)
    repo_url = github.get('repo_url')
    if not isinstance(repo_url, str) or not repo_url or repo_url != repo_url.strip():
        raise ValueError('github.repo_url must be configured.')
    parsed = urlsplit(repo_url)
    github_remote = parsed.scheme == 'https' and parsed.netloc == 'github.com' and (parsed.username is None) and (parsed.password is None) and (not parsed.query) and (not parsed.fragment) and (re.fullmatch('/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+(?:\\.git)?', parsed.path) is not None)
    if not github_remote:
        if not allow_local:
            raise ValueError('Only credential-free HTTPS github.com repository URLs are allowed.')
        if parsed.scheme == 'file' and (not parsed.netloc) and (not parsed.query) and (not parsed.fragment):
            remote_path = Path(unquote(parsed.path)).resolve()
        elif not parsed.scheme and (not parsed.netloc):
            remote_path = Path(repo_url).resolve()
        else:
            raise ValueError('The explicitly enabled test remote must be a local filesystem repository.')
        if not remote_path.is_dir():
            raise ValueError('The local test repository does not exist.')
        repo_url = str(remote_path)
    branch = github.get('branch', 'main')
    if not isinstance(branch, str) or not branch or branch.startswith('-') or any((c in branch for c in '\r\n\x00')):
        raise ValueError('Invalid Git branch name.')
    author_name = next((github[key] for key in ('git_username', 'git_user_name', 'git_author_name', 'author_name', 'github_username', 'user_name') if github.get(key)), None)
    author_email = next((github[key] for key in ('git_email', 'git_user_email', 'git_author_email', 'author_email', 'user_email') if github.get(key)), None)
    if push and (not dry_run):
        if not isinstance(author_name, str) or not author_name.strip() or (not isinstance(author_email, str)) or (not author_email.strip()):
            raise ValueError('Configure the inherited Git author name and email before publishing.')
        if any((c in author_name + author_email for c in '\r\n\x00')):
            raise ValueError('Git author name and email must be single-line values.')
    return {'github': github, 'repo_url': repo_url, 'github_remote': github_remote, 'branch': branch, 'push': push, 'dry_run': dry_run, 'export_dir': export_dir, 'export_parts': export_parts, 'author_name': author_name, 'author_email': author_email}

def _p7_git_session(options):
    """Return a private, temporary Git session; leave existing checkouts untouched."""

    @contextmanager
    def session():
        secret_name = options['github'].get('token_secret_name', 'GITHUB_TOKEN')
        token = _p7_read_github_token(secret_name) if options['github_remote'] and secret_name else None
        with tempfile.TemporaryDirectory(prefix='sector7-git-') as temporary:
            temporary_root = Path(temporary)
            checkout = temporary_root / 'repository'
            checkout.mkdir()
            askpass = temporary_root / 'askpass.py'
            askpass.write_text("#!/usr/bin/env python3\nimport os, sys\nprompt = sys.argv[1].lower() if len(sys.argv) > 1 else ''\nsys.stdout.write('x-access-token' if 'username' in prompt else os.environ.get('_P7_GITHUB_TOKEN', ''))\n", encoding='utf-8')
            askpass.chmod(448)
            hooks = temporary_root / 'hooks'
            hooks.mkdir()
            child_env = {key: value for key, value in os.environ.items() if not key.startswith(('GIT_TRACE', 'GIT_CONFIG_KEY_', 'GIT_CONFIG_VALUE_')) and key not in {'GIT_CURL_VERBOSE', 'SSH_ASKPASS'}}
            for variable in ('GIT_DIR', 'GIT_WORK_TREE', 'GIT_INDEX_FILE', 'GIT_CONFIG', 'GIT_CONFIG_COUNT', 'GIT_COMMON_DIR', 'GIT_OBJECT_DIRECTORY', 'GIT_ALTERNATE_OBJECT_DIRECTORIES', 'GIT_NAMESPACE', '_P7_GITHUB_TOKEN', 'GIT_AUTHOR_NAME', 'GIT_AUTHOR_EMAIL', 'GIT_COMMITTER_NAME', 'GIT_COMMITTER_EMAIL', 'GIT_AUTHOR_DATE', 'GIT_COMMITTER_DATE'):
                child_env.pop(variable, None)
            child_env.update({'GIT_TERMINAL_PROMPT': '0', 'GIT_ASKPASS': str(askpass), 'GIT_CONFIG_NOSYSTEM': '1', 'GIT_CONFIG_GLOBAL': os.devnull, 'GIT_LITERAL_PATHSPECS': '1'})
            if token:
                child_env['_P7_GITHUB_TOKEN'] = token

            def git(*args, allowed=(0,)):
                result = subprocess.run(['git', '-c', 'credential.helper=', '-c', f'core.askPass={askpass}', '-c', f'core.hooksPath={hooks}', *args], cwd=checkout, env=child_env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
                if result.returncode not in allowed:
                    raise RuntimeError(f'Sector 7 Git {args[0]} failed (exit {result.returncode}).')
                return result
            git('check-ref-format', '--branch', options['branch'])
            git('init', '--quiet')
            git('remote', 'add', 'origin', options['repo_url'])
            yield (checkout, git)
    return session()

def io_fetch_sector7(config):
    """Fetch and report the latest remote tip without modifying the user's checkout."""
    options = _p7_git_options(config)
    if not options['push'] and (not options['dry_run']):
        return {'status': 'not_requested', 'branch': options['branch']}
    with _p7_git_session(options) as (_, git):
        git('fetch', '--quiet', 'origin', f"refs/heads/{options['branch']}")
        tip = git('rev-parse', 'FETCH_HEAD').stdout.strip()
        return {'status': 'fetched', 'branch': options['branch'], 'remote_tip': tip}

def io_push_sector7(persisted, config):
    """Publish a catalog-verified Sector 7 snapshot with an ordinary non-force push.

    Only catalog artifacts, the Sector 7 state, and recorded obsolete result files
    are staged. The commit starts at the latest remote branch; races are retried.
    Existing checkouts, local edits and other pipeline sectors remain untouched.
    """
    options = _p7_git_options(config)
    branch, export_parts = (options['branch'], options['export_parts'])
    outcome = {'status': 'prepared', 'branch': branch, 'export_dir': options['export_dir'], 'pushed': False, 'dry_run': options['dry_run']}
    if not options['push'] and (not options['dry_run']):
        return {**outcome, 'status': 'not_requested'}
    root_value = persisted.get('local_root', persisted.get('root'))
    if root_value is None:
        raise ValueError('The persisted Sector 7 root is missing.')
    root = Path(root_value).resolve()
    state = _p7_read_state(root)
    if state is None:
        raise RuntimeError('Sector 7 has no valid persisted snapshot to publish.')
    if persisted.get('state') != state:
        raise RuntimeError('Sector 7 changed after persistence; save it again before publishing.')
    artifacts = state.get('artifacts')
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError('Sector 7 state must contain a nonempty artifact catalog.')
    relative_paths, records = ([], {})
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            raise ValueError('Invalid Sector 7 artifact record.')
        relative = artifact.get('relative_path')
        source = _p7_safe_path(root, relative)
        if not source.is_file() or source.is_symlink():
            raise RuntimeError(f'Missing or unsafe Sector 7 artifact: {relative}')
        if _p7_file_hash(source) != artifact.get('sha256') or source.stat().st_size != artifact.get('bytes'):
            raise RuntimeError(f'Sector 7 artifact verification failed: {relative}')
        if relative in records:
            raise ValueError('Sector 7 artifact catalog has duplicate paths.')
        records[relative] = artifact
        relative_paths.append(relative)
    state_relative = 'configs/sector7_state.json'
    state_source = _p7_safe_path(root, state_relative)
    if state_source.is_symlink() or json.loads(state_source.read_text(encoding='utf-8')) != state:
        raise RuntimeError('Sector 7 state changed during publisher verification.')
    if state_relative not in relative_paths:
        relative_paths.append(state_relative)
    obsolete_paths = state.get('obsolete_paths', [])
    if not isinstance(obsolete_paths, list) or any((not isinstance(path, str) for path in obsolete_paths)):
        raise ValueError('Sector 7 obsolete_paths must be a list of recorded relative paths.')
    for relative in obsolete_paths:
        _p7_safe_path(root, relative)
        if not relative.startswith('results/') or relative in relative_paths:
            raise ValueError('Only obsolete recorded result files may be deleted from the export.')
    outcome['count'] = len(relative_paths)
    with _p7_git_session(options) as (checkout, git):
        for attempt in range(1, 4):
            git('fetch', '--quiet', 'origin', f'refs/heads/{branch}')
            remote_tip = git('rev-parse', 'FETCH_HEAD').stdout.strip()
            git('checkout', '--quiet', '--detach', remote_tip)

            def export_target(relative):
                relative_path = PurePosixPath(relative)
                if relative_path.is_absolute() or any((p in {'.', '..', '.git'} for p in relative_path.parts)) or '\\' in relative:
                    raise ValueError('Invalid export artifact path.')
                target = checkout.joinpath(*export_parts.parts, *relative_path.parts)
                current = checkout
                for part in (*export_parts.parts, *relative_path.parts):
                    current = current / part
                    if current.is_symlink():
                        raise RuntimeError('The Git export path contains a symlink.')
                return target
            staged_paths = []
            for relative in relative_paths:
                source, target = (_p7_safe_path(root, relative), export_target(relative))
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
                if relative in records:
                    record = records[relative]
                    if _p7_file_hash(target) != record['sha256'] or target.stat().st_size != record['bytes']:
                        raise RuntimeError('A Sector 7 artifact changed while being copied.')
                if relative == state_relative and json.loads(target.read_text(encoding='utf-8')) != state:
                    raise RuntimeError('The exported Sector 7 state changed while being copied.')
                staged_paths.append(target.relative_to(checkout).as_posix())
            for relative in obsolete_paths:
                target = export_target(relative)
                git_relative = target.relative_to(checkout).as_posix()
                tracked = bool(git('ls-files', '--', git_relative).stdout.strip())
                if target.exists():
                    if not target.is_file():
                        raise RuntimeError('An obsolete Sector 7 result path is not a file.')
                    target.unlink()
                if tracked:
                    staged_paths.append(git_relative)
            git('add', '--', *sorted(set(staged_paths)))
            changed = git('diff', '--cached', '--name-only', '-z').stdout
            changed_paths = [path for path in changed.split('\x00') if path]
            if any((not path.startswith(options['export_dir'] + '/') for path in changed_paths)):
                raise RuntimeError('The publication contains a change outside Sector 7.')
            outcome.update({'remote_tip': remote_tip, 'changed_paths': changed_paths, 'attempt': attempt})
            if _p7_read_state(root) != state:
                raise RuntimeError('Sector 7 changed during export; save it again before publishing.')
            if not changed_paths:
                return {**outcome, 'status': 'unchanged'}
            if options['dry_run']:
                return {**outcome, 'status': 'dry_run'}
            git('-c', f"user.name={options['author_name']}", '-c', f"user.email={options['author_email']}", 'commit', '--quiet', '-m', 'Update Phase 2 Sector 7 image quality control')
            commit = git('rev-parse', 'HEAD').stdout.strip()
            git('fetch', '--quiet', 'origin', f'refs/heads/{branch}')
            latest_tip = git('rev-parse', 'FETCH_HEAD').stdout.strip()
            if latest_tip != remote_tip:
                continue
            git('merge-base', '--is-ancestor', remote_tip, 'HEAD')
            pushed = git('push', '--quiet', 'origin', f'HEAD:refs/heads/{branch}', allowed=(0, 1, 128))
            if pushed.returncode == 0:
                return {**outcome, 'status': 'published', 'pushed': True, 'commit': commit}
            git('fetch', '--quiet', 'origin', f'refs/heads/{branch}')
            if git('rev-parse', 'FETCH_HEAD').stdout.strip() == remote_tip:
                raise RuntimeError('Sector 7 Git push failed. Check repository access and token permissions; the saved results remain available for retry.')
        raise RuntimeError('The remote branch kept changing; rerun the Sector 7 publication cell.')


def validate_persisted_dataframe(expected_dataframe, persisted_path, artifact_name):
    """Read back exact Parquet contents, column order and dtypes."""
    path = Path(persisted_path)
    if not path.is_file():
        raise FileNotFoundError(f'Persisted artifact missing: {artifact_name}: {path}')
    restored = pd.read_parquet(path, engine='pyarrow')
    pd.testing.assert_frame_equal(expected_dataframe.reset_index(drop=True), restored,
                                  check_exact=True, check_dtype=True)
    return True


def io_audit_sector7_sources(image_inventory, temporal_image_audit, config):
    """Hash image bytes/status before cache lookup; no decoding or metric calculation."""
    expected = temporal_image_audit[['context_image_relpath', 'image_sha256']].dropna().drop_duplicates()
    if not expected.empty and expected['context_image_relpath'].duplicated().any():
        raise ValueError('Sector 6 recorded conflicting image checksums.')
    lookup = expected.set_index('context_image_relpath')['image_sha256']
    def inspect(row):
        path = Path(row.image_path)
        source_hash, source_bytes, error = pd.NA, pd.NA, pd.NA
        try:
            if path.is_symlink():
                raise OSError('image_symlink_not_supported')
            if not path.is_file():
                raise OSError('image_file_not_found')
            source_bytes = path.stat().st_size
            source_hash = _p3_file_hash(path)
        except OSError as exc:
            error = str(exc)
        previous = lookup.get(row.image_relpath, pd.NA)
        matches = pd.NA if pd.isna(previous) or pd.isna(source_hash) else source_hash == previous
        return {'image_path': row.image_path, 'image_relpath': row.image_relpath,
                'image_sha256': source_hash, 'image_bytes': source_bytes,
                'source_error': error, 'sector6_expected_sha256': previous,
                'source_matches_sector6': matches}
    columns = ['image_path','image_relpath','image_sha256','image_bytes','source_error',
               'sector6_expected_sha256','source_matches_sector6']
    with ThreadPoolExecutor(max_workers=config['qc_workers']) as executor:
        rows = list(tqdm(executor.map(inspect, image_inventory.itertuples(index=False)),
                         total=len(image_inventory), desc='Checking QC source signatures'))
    result = pd.DataFrame.from_records(rows, columns=columns)
    for column in set(columns) - {'image_bytes','source_matches_sector6'}:
        result[column] = result[column].astype('string')
    result['image_bytes'] = result['image_bytes'].astype('Int64')
    result['source_matches_sector6'] = result['source_matches_sector6'].astype('boolean')
    return result


def io_validate_sector7_source_audit(audit):
    """Reject files changed, added or removed between the source audit and persistence."""
    for row in audit.itertuples(index=False):
        path = Path(row.image_path)
        if pd.isna(row.image_sha256):
            if path.is_file() and not path.is_symlink():
                raise RuntimeError('A previously unavailable QC image appeared; rerun Sector 7.')
        elif not path.is_file() or path.is_symlink() or _p3_file_hash(path) != row.image_sha256:
            raise RuntimeError(f'A QC source image changed during this run: {row.image_relpath}')
    return True


def add_qc_source_status(qc_manifest, image_audit):
    """Mark differences from Sector 6 as review conditions without removing images."""
    extra = image_audit[['image_relpath','image_sha256','source_matches_sector6']].rename(
        columns={'image_sha256':'qc_image_sha256','source_matches_sector6':'qc_source_matches_sector6'})
    result = qc_manifest.merge(extra, on='image_relpath', how='left', validate='one_to_one')
    available = result['image_sha256'].notna() & result['qc_image_sha256'].notna()
    if result.loc[available, 'image_sha256'].ne(result.loc[available, 'qc_image_sha256']).any():
        raise RuntimeError('A QC image changed between its source audit and decoding; rerun Sector 7.')
    result['qc_requires_review'] = (result['qc_requires_review'] |
        result['qc_source_matches_sector6'].eq(False).fillna(False)).astype('boolean')
    return result


def build_sector7_summary(tables):
    summary = tables['image_qc_summary'].iloc[0].to_dict()
    return {'schema_version': 1, 'phase':2, 'sector':7,
            'table_row_counts':{name:len(frame) for name,frame in tables.items()},
            'qc':summary, 'threshold_fit_policy':'Configured split only; train by default.',
            'annotation_semantics':'All source annotations and temporal references are preserved.',
            'quality_semantics':'Technical proxies and review flags; no clinical quality labels.'}


def io_compute_sector7_tables(inputs, config, inventory, source_audit):
    """Compose image I/O and pure transformations with explicit inputs."""
    labelled = inputs['sector5']['labelled_frame_manifest_with_segments']
    temporal = inputs['sector6']['temporal_manifest']
    metrics = io_compute_qc_for_image_inventory(inventory, config)
    thresholds = derive_image_qc_thresholds(metrics, config)
    qc = add_qc_source_status(add_image_qc_flags(metrics, thresholds), source_audit)
    enriched = attach_image_qc_to_manifests(labelled, temporal, qc, config)
    reports = build_qc_reference_reports(labelled, temporal, qc, config)
    tables = {'image_qc_inventory':inventory, 'image_qc_manifest':qc,
              'image_qc_thresholds':thresholds, 'image_qc_source_audit':source_audit,
              'labelled_frame_manifest_with_qc':enriched['labelled_frame_manifest_with_qc'],
              'temporal_manifest_with_qc':enriched['temporal_manifest_with_qc'],
              'image_qc_summary':build_image_qc_summary(qc,thresholds,reports,config),
              'image_read_failure_report':reports['image_read_failure_report'],
              'qc_reference_report':reports['qc_reference_report'],
              'missing_image_reference_report':reports['missing_image_reference_report']}
    io_validate_sector7_source_audit(source_audit)
    return tables


def io_sector7_dependency_versions():
    distributions = {'numpy':'numpy','pandas':'pandas','pyarrow':'pyarrow','tqdm':'tqdm'}
    versions = {key:importlib.metadata.version(value) for key,value in distributions.items()}
    versions['opencv'] = cv2.__version__
    versions['python'] = platform.python_version()
    return versions


def io_prepare_sector7(inputs, config, lineage, namespace):
    """Compute current image signatures, then check an existing completed QC run."""
    labelled = inputs['sector5']['labelled_frame_manifest_with_segments']
    temporal = inputs['sector6']['temporal_manifest']
    validate_image_qc_config(config)
    inventory = build_qc_image_inventory(labelled, temporal, config)
    source_audit = io_audit_sector7_sources(inventory, inputs['sector6']['temporal_image_audit'], config)
    versions = io_sector7_dependency_versions()
    current = copy.deepcopy(lineage)
    current['core_source_sha256'] = _s3_semantic_digest({
        'pipeline':sector7_runtime_core_sha256(io_compute_sector7_tables,namespace,versions),
        'source_audit':sector7_runtime_core_sha256(io_audit_sector7_sources,namespace,versions),
        'input_reader':sector7_runtime_core_sha256(io_load_sector7_context,namespace,versions)})
    current['dependency_versions'] = versions
    current['image_source_audit_sha256'] = sector3_table_fingerprint(source_audit)
    signature = sector7_processing_signature(config,current)
    cached = io_load_cached_sector7(config,signature)
    if cached is not None:
        pd.testing.assert_frame_equal(cached['tables']['image_qc_source_audit'],source_audit)
        tables,status = cached['tables'],'reused'
    else:
        tables,status = io_compute_sector7_tables(inputs,config,inventory,source_audit),'computed'
    return {'tables':tables,'lineage':current,'processing_signature':signature,'status':status}


def io_mount_storage(bootstrap):
    """Mount Google Drive when requested; support already mounted local copies."""
    if bootstrap.get('mount_google_drive',False):
        if drive is None:
            if not Path(bootstrap['storage_root']).is_dir():
                raise RuntimeError('Open in Colab or select an existing mounted storage_root.')
        else:
            drive.mount('/content/drive',force_remount=False)
    if not Path(bootstrap['storage_root']).is_dir():
        raise FileNotFoundError('storage_root does not exist; mount/copy the saved Sector 2–6 outputs first.')
