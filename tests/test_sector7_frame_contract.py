"""Regression tests for the S7 reader's authoritative physical-frame splits.

Only pure contract definitions are loaded from notebook ASTs. No notebook I/O,
Colab setup, GPU initialization, data loading or publication cells are run.
"""
import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


_PROJECT_CANDIDATES = (
    Path(__file__).resolve().parents[1],
    Path(__file__).resolve().parents[1] / "pr_work",
    Path.cwd(),
)
ROOT = next(
    root for root in _PROJECT_CANDIDATES
    if (root / "notebooks/phase2/phase2_07_image_quality_control.ipynb").is_file()
)
NOTEBOOKS = ROOT / "notebooks/phase2"


def _load_definitions(path, names):
    namespace = {"pd": pd, "np": np}
    found = set()
    notebook = json.loads(path.read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "\n".join(
            line for line in "".join(cell["source"]).splitlines()
            if not line.startswith(("%", "!"))
        )
        for node in ast.parse(source).body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                defined = {node.name}
            elif isinstance(node, ast.Assign):
                defined = {target.id for target in node.targets if isinstance(target, ast.Name)}
            else:
                continue
            if defined & names:
                exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
                found.update(defined & names)
    assert found == names, f"Missing pure contract definitions: {names - found}"
    return namespace


@pytest.fixture(scope="module")
def contract():
    return _load_definitions(NOTEBOOKS / "phase2_07_image_quality_control.ipynb", {
        "S5_SECTOR4_RESULTS", "S5_REQUIRED_LABELLED_COLUMNS",
        "_s5_identity_multiset", "_s5_validate_frame_splits", "_s5_validate_loaded_frames",
    })


@pytest.fixture
def manifests(contract):
    # Two different boxes annotate the same training frame. All three splits
    # intentionally occur in this single video, as allowed by current S4.
    rows = []
    for annotation_id, frame, split, bbox_xmin in (
        ("train_box_a", 10, "train", 1),
        ("train_box_b", 10, "train", 8),
        ("validation_box", 20, "validation", 2),
        ("test_box", 30, "test", 3),
    ):
        row = {name: None for name in contract["S5_REQUIRED_LABELLED_COLUMNS"]}
        row.update(
            annotation_id=annotation_id,
            annotation_status="source_expert_annotation_alignment_confirmed",
            alignment_training_eligible=True,
            video_key="video_a", finding_class="Polyp", finding_class_normalized="polyp",
            clinical_group="protruding_lesion", frame_number=frame,
            opencv_frame_index=frame, frame_index_offset=0,
            filename=f"video_a_{frame}.jpg", image_path=f"/fixture/video_a_{frame}.jpg",
            image_relpath=f"video_a_{frame}.jpg", has_bbox=True,
            bbox_spatially_usable=True, bbox_validation_status="valid",
            bbox_xmin=bbox_xmin, bbox_ymin=1, bbox_xmax=bbox_xmin + 3, bbox_ymax=5,
            split=split,
        )
        rows.append(row)
    labelled = pd.DataFrame(rows)
    frame_splits = labelled[["video_key", "opencv_frame_index", "split"]].drop_duplicates()
    video_splits = labelled.groupby(["video_key", "split"], as_index=False).agg(
        retained_annotation_count=("annotation_id", "size"),
        retained_unique_frame_count=("opencv_frame_index", "nunique"),
        retained_class_count=("finding_class_normalized", "nunique"),
    )
    videos = pd.DataFrame([{
        "video_key": "video_a", "frame_count": 100,
        "frame_count_source": "completed_sector2_decode_audit",
    }])
    return labelled, frame_splits, video_splits, videos


def test_required_tables_match_current_sector6_contract(contract):
    current6 = _load_definitions(
        NOTEBOOKS / "phase2_06_temporal_context_images.ipynb", {"S5_SECTOR4_RESULTS"},
    )
    assert contract["S5_SECTOR4_RESULTS"] == current6["S5_SECTOR4_RESULTS"] == (
        "labelled_frame_manifest_with_split", "video_split_manifest", "frame_split_manifest",
    )


def test_one_video_can_span_splits_and_preserve_multiple_annotations(contract, manifests):
    labelled, frame_splits, video_splits, videos = manifests
    checked = labelled.copy(deep=True)
    contract["_s5_validate_loaded_frames"](
        checked, labelled.copy(deep=True), videos, video_splits,
        labelled.copy(deep=True), frame_splits,
    )
    assert checked.annotation_id.tolist() == labelled.annotation_id.tolist()
    assert len(checked) == 4
    assert checked.opencv_frame_index.nunique() == 3
    assert checked.loc[checked.opencv_frame_index.eq(10), "bbox_xmin"].tolist() == [1, 8]
    assert checked.split.tolist() == ["train", "train", "validation", "test"]


def test_rejects_changed_authoritative_frame_assignment(contract, manifests):
    labelled, frame_splits, video_splits, _ = manifests
    changed = frame_splits.copy()
    changed.loc[changed.opencv_frame_index.eq(10), "split"] = "test"
    with pytest.raises(ValueError, match="authoritative Sector 4 frame split manifest"):
        contract["_s5_validate_frame_splits"](labelled, video_splits, changed)


def test_rejects_duplicate_physical_frame_identities(contract, manifests):
    labelled, frame_splits, video_splits, _ = manifests
    duplicated = pd.concat([frame_splits, frame_splits.iloc[:1]], ignore_index=True)
    with pytest.raises(ValueError, match="one row per video/decoded-frame key"):
        contract["_s5_validate_frame_splits"](labelled, video_splits, duplicated)


def test_rejects_multiple_split_assignments_for_one_annotated_frame(contract, manifests):
    labelled, frame_splits, video_splits, _ = manifests
    changed = labelled.copy()
    changed.loc[changed.annotation_id.eq("train_box_b"), "split"] = "test"
    with pytest.raises(ValueError, match="one physical frame must share the same split"):
        contract["_s5_validate_frame_splits"](changed, video_splits, frame_splits)


def test_rejects_wrong_saved_membership_counts(contract, manifests):
    labelled, frame_splits, video_splits, _ = manifests
    changed = video_splits.copy()
    changed.loc[changed.split.eq("train"), "retained_annotation_count"] = 1
    with pytest.raises(ValueError, match="inconsistent retained_annotation_count"):
        contract["_s5_validate_frame_splits"](labelled, changed, frame_splits)
