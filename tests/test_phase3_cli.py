"""Integration checks for honest saved status when pixels are unavailable."""

import json
from unittest.mock import patch

import pytest

from mmvqa.phase3.cli import run
from mmvqa.phase3.storage import S8Store
from test_phase3_benchmark import FixtureStore


def test_metadata_only_reports_no_pixel_check(tmp_path):
    store = FixtureStore(tmp_path / "source")
    output = tmp_path / "reports"
    with patch("mmvqa.phase3.cli.S8Store", return_value=store):
        report = run({"variants": ["full", "crop"], "output_dir": str(output)}, metadata_only=True)
    assert report["status"] == "metadata_verified_images_not_checked"
    assert not any(sample["pixels_verified_now"] for sample in report["samples"])
    assert report["training_ready"] is False
    assert json.loads((output / "source_snapshot.json").read_text())["resolved_commit"] == store.resolve()
    assert len(json.loads((output / "provenance.json").read_text())) == 2


def test_missing_mount_replaces_prior_success_with_failure(tmp_path):
    store = FixtureStore(tmp_path / "source")
    output = tmp_path / "reports"
    config = {"variants": ["full"], "output_dir": str(output)}
    with patch("mmvqa.phase3.cli.S8Store", return_value=store):
        run(config, metadata_only=True)
        with pytest.raises(FileNotFoundError):
            run(config)
    report = json.loads((output / "sample_report.json").read_text())
    assert report["status"] == "failed"
    assert report["error_type"] == "FileNotFoundError"
    assert report["training_ready"] is False
    assert json.loads((output / "provenance.json").read_text()) == []
    assert json.loads((output / "evaluator_records.json").read_text()) == []


def test_source_resolution_failure_still_saves_failure_report(tmp_path):
    def unavailable(url, destination):
        raise OSError("GitHub unavailable")
    store = S8Store(cache_dir=tmp_path / "cache", fetch=unavailable)
    assert store.snapshot()["resolution_status"] == "not_resolved"
    output = tmp_path / "reports"
    with patch("mmvqa.phase3.cli.S8Store", return_value=store):
        with pytest.raises(OSError, match="GitHub unavailable"):
            run({"output_dir": str(output)}, metadata_only=True)
    assert json.loads((output / "sample_report.json").read_text())["status"] == "failed"
    assert json.loads((output / "source_snapshot.json").read_text())["resolved_commit"] is None
