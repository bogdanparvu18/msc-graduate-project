"""Offline integration checks for Phase 2 Sector 7 image QC.

These tests use synthetic image files, real Parquet and a local bare Git remote.
They never mount Google Drive, read Colab credentials or contact GitHub.
"""

import copy
import hashlib
import importlib.util
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np
import pandas as pd


MODULE_PATH = Path(__file__).resolve().parents[1] / "notebooks/phase2/phase2_07_image_quality_control.py"
MODULE_SPEC = importlib.util.spec_from_file_location("phase2_07_image_quality_control", MODULE_PATH)
s7 = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(s7)


def changed_compute_image_qc(image, config, image_path=None):
    """An equivalent wrapper with a different executable identity for a test."""
    return compute_image_qc(image, config, image_path)


class Sector7Fixture(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="sector7-tests-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.config = {
            **copy.deepcopy(s7.S7_DEFAULTS),
            "phase": 2,
            "sector": 7,
            "storage_root": str(self.root),
            "sector7": {"local_output_dir": "outputs/phase2/sector7",
                        "export_dir": "outputs/phase2/sector7", "push": False,
                        "dry_run": False},
            "github": {"push": False, "dry_run": False},
        }
        self.config["qc_workers"] = 1
        self.image_dir = self.root / "images"
        self.image_dir.mkdir()
        self.image_a = self.image_dir / "a.png"
        self.image_b = self.image_dir / "b.png"
        self.image_bad = self.image_dir / "bad.png"
        self.image_missing = self.image_dir / "missing.png"
        self.write_image(self.image_a, 90)
        self.write_image(self.image_b, 170)
        self.image_bad.write_bytes(b"not an image")
        self.labelled = pd.DataFrame({
            "annotation_id": pd.Series(["a0", "a1", "b0", "bad0", "missing0", "no-path0"], dtype="string"),
            "image_path": pd.Series([str(self.image_a), str(self.image_a), str(self.image_b),
                                     str(self.image_bad), str(self.image_missing), pd.NA], dtype="string"),
            "image_relpath": pd.Series(["images/a.png", "images/a.png", "images/b.png",
                                        "images/bad.png", "images/missing.png", pd.NA], dtype="string"),
            "split": pd.Series(["train", "train", "test", "train", "train", "train"], dtype="string"),
            "frame_index": pd.Series([1, 1, 2, 3, 4, 5], dtype="Int64"),
            "bounding_box": pd.Series(["box-a", "box-b", "box-c", "box-d", "box-e", "box-f"], dtype="string"),
        })
        self.temporal = pd.DataFrame({
            "finding_segment_id": pd.Series(["seg0", "seg0", "seg1", "seg2", "seg3"], dtype="string"),
            "context_image_path": pd.Series([str(self.image_a), str(self.image_a), str(self.image_b), pd.NA, pd.NA], dtype="string"),
            "context_image_relpath": pd.Series(["images/a.png", "images/a.png", "images/b.png", pd.NA, pd.NA], dtype="string"),
            "split": pd.Series(["train", "train", "test", "test", "train"], dtype="string"),
            "context_image_source": pd.Series(["official", "official", "official", "video_extraction_required", "official"], dtype="string"),
            "extraction_status": pd.Series(["completed", "completed", "completed", "not_requested", "failed"], dtype="string"),
            "image_readable": pd.Series([True, True, True, False, False], dtype="boolean"),
        })
        self.temporal_audit = pd.DataFrame({
            "context_image_relpath": pd.Series(["images/a.png", "images/b.png"], dtype="string"),
            "image_sha256": pd.Series([self.digest(self.image_a), self.digest(self.image_b)], dtype="string"),
        })
        self.inputs = {
            "sector5": {"labelled_frame_manifest_with_segments": self.labelled},
            "sector6": {"temporal_manifest": self.temporal,
                        "temporal_image_audit": self.temporal_audit},
        }
        self.input_file = self.root / "upstream_metadata.txt"
        self.input_file.write_text("verified saved metadata", encoding="utf-8")
        self.lineage = {
            "input_sha256": {"sector6/metadata": self.digest(self.input_file)},
            "input_descriptors": {"sector6/metadata": str(self.input_file)},
            "upstream": {},
        }

    @staticmethod
    def digest(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    @staticmethod
    def write_image(path, brightness):
        image = np.full((48, 48, 3), brightness, dtype=np.uint8)
        image[10:20, 10:20] = np.minimum(255, brightness + 20)
        if not cv2.imwrite(str(path), image):
            raise RuntimeError("Could not write synthetic image")

    def prepare(self, config=None, namespace=None):
        return s7.io_prepare_sector7(self.inputs, config or self.config,
                                    self.lineage, namespace or dict(vars(s7)))

    def persist(self, prepared, config=None):
        return s7.io_persist_sector7(prepared["tables"], config or self.config,
                                    prepared["lineage"])

    def metrics_fixture(self):
        records = []
        for index, (split, brightness, contrast, blur) in enumerate([
            ("train", 0.3, 0.1, 100), ("train", 0.5, 0.2, 200),
            ("train", 0.7, 0.3, 300), ("test", 0.99, 0.9, 1e8),
        ]):
            records.append({
                "image_relpath": f"images/metric{index}.png", "image_path": f"/images/metric{index}.png",
                "split": split, "used_by_labelled_manifest": True,
                "used_by_temporal_manifest": False, "image_sha256": "0" * 64,
                "image_readable": True, "image_read_error": pd.NA,
                "image_width": 48, "image_height": 48,
                "brightness_mean": brightness, "contrast_std": contrast,
                "blur_laplacian": blur, "underexposed_fraction": 0.0,
                "overexposed_fraction": 0.0, "specular_fraction": 0.0,
            })
        return s7.qc_typed_table(pd.DataFrame.from_records(records))


class PureQCTests(Sector7Fixture):
    def test_metrics_on_constant_dark_bright_and_sharp_images(self):
        dark = np.zeros((48, 48, 3), dtype=np.uint8)
        white = np.full((48, 48, 3), 255, dtype=np.uint8)
        sharp = np.repeat((np.indices((48, 48)).sum(axis=0) % 2 * 255)
                          .astype(np.uint8)[..., np.newaxis], 3, axis=2)
        before = sharp.copy()
        dark_qc = s7.compute_image_qc(dark, self.config)
        white_qc = s7.compute_image_qc(white, self.config)
        sharp_qc = s7.compute_image_qc(sharp, self.config)
        self.assertTrue(dark_qc["image_readable"])
        self.assertEqual(dark_qc["brightness_mean"], 0.0)
        self.assertEqual(dark_qc["underexposed_fraction"], 1.0)
        self.assertEqual(white_qc["brightness_mean"], 1.0)
        self.assertEqual(white_qc["overexposed_fraction"], 1.0)
        self.assertEqual(white_qc["specular_fraction"], 1.0)
        self.assertEqual(white_qc["blur_laplacian"], 0.0)
        self.assertGreater(sharp_qc["blur_laplacian"], white_qc["blur_laplacian"])
        np.testing.assert_array_equal(sharp, before)

    def test_fov_excludes_border_and_laplacian_border(self):
        image = np.zeros((48, 48, 3), dtype=np.uint8)
        mask = s7.build_capsule_fov_mask(48, 48, 0.75)
        image[mask] = 128
        config = {**self.config, "qc_fov_radius_fraction": 0.75}
        metrics = s7.compute_image_qc(image, config)
        self.assertAlmostEqual(metrics["brightness_mean"], 128 / 255)
        self.assertEqual(metrics["blur_laplacian"], 0.0)
        self.assertEqual(metrics["contrast_std"], 0.0)

    def test_train_thresholds_ignore_evaluation_extremes_and_do_not_mutate(self):
        metrics = self.metrics_fixture()
        before = metrics.copy(deep=True)
        thresholds = s7.derive_image_qc_thresholds(metrics, self.config)
        train_only = s7.derive_image_qc_thresholds(metrics.loc[metrics["split"].eq("train")], self.config)
        pd.testing.assert_frame_equal(thresholds.iloc[:, :4], train_only.iloc[:, :4])
        self.assertEqual(thresholds.iloc[0]["calibration_readable_images"], 3)
        altered = metrics.copy(deep=True)
        altered.loc[altered["split"].eq("test"), ["brightness_mean", "contrast_std", "blur_laplacian"]] = [0, 0, 0]
        altered_thresholds = s7.derive_image_qc_thresholds(altered, self.config)
        pd.testing.assert_frame_equal(thresholds.iloc[:, :4], altered_thresholds.iloc[:, :4])
        flags = s7.add_image_qc_flags(metrics, thresholds)
        self.assertTrue(flags.iloc[-1]["qc_high_brightness_outlier"])
        self.assertTrue(flags.iloc[-1]["image_usable_for_modelling"])
        pd.testing.assert_frame_equal(metrics, before)

    def test_tied_metrics_are_not_all_flagged(self):
        metrics = self.metrics_fixture()
        metrics.loc[:, "brightness_mean"] = 0.5
        metrics.loc[:, "contrast_std"] = 0.1
        metrics.loc[:, "blur_laplacian"] = 10.0
        flags = s7.add_image_qc_flags(metrics, s7.derive_image_qc_thresholds(metrics, self.config))
        self.assertFalse(flags["qc_requires_review"].any())
        self.assertTrue(flags["image_usable_for_modelling"].all())

    def test_no_train_images_has_explicit_uncalibrated_status(self):
        metrics = self.metrics_fixture()
        metrics["split"] = "test"
        thresholds = s7.derive_image_qc_thresholds(metrics, self.config)
        self.assertEqual(thresholds.iloc[0]["threshold_calibration_status"], "uncalibrated_no_train_images")
        flags = s7.add_image_qc_flags(metrics, thresholds)
        self.assertFalse(flags["qc_thresholds_calibrated"].any())
        self.assertFalse(flags["qc_requires_review"].any())

    def test_split_conflict_is_rejected(self):
        temporal = self.temporal.copy()
        temporal.loc[0, "split"] = "test"
        with self.assertRaisesRegex(ValueError, "different supervised splits"):
            s7.build_qc_image_inventory(self.labelled, temporal, self.config)

    def test_invalid_quantiles_and_unsafe_paths_fail_early(self):
        with self.assertRaises(ValueError):
            s7.validate_image_qc_config({**self.config, "qc_blur_quantile": True})
        with self.assertRaises(ValueError):
            s7.validate_image_qc_config({**self.config, "qc_threshold_fit_split": "test"})
        with self.assertRaises(ValueError):
            s7.build_capsule_fov_mask(0, 48, 0.95)
        labelled = self.labelled.copy()
        labelled.loc[0, "image_relpath"] = "../escape.png"
        with self.assertRaises(ValueError):
            s7.build_qc_image_inventory(labelled, self.temporal, self.config)


class Sector7IntegrationTests(Sector7Fixture):
    def loader_fixture(self, stale=False):
        """Verified catalog fixture: isolate S7 chaining from S6's catalog reader."""
        root6 = self.root / "outputs/phase2/sector6"
        root6.mkdir(parents=True)
        config_path = root6 / "configs/sector6_config_fixture.json"
        config_path.parent.mkdir()
        source6 = copy.deepcopy(self.config)
        source6.update({"phase": 2, "sector": 6, "qc_fov_radius_fraction": 0.72})
        source6["github"]["repo_url"] = "https://github.com/example/verified.git"
        config_path.write_text(json.dumps({"effective_config": source6}), encoding="utf-8")
        lineage_path = root6 / s7.S6_LINEAGE_PATH
        lineage_path.write_text(json.dumps(self.lineage), encoding="utf-8")
        state6 = {
            "config_snapshot": config_path.relative_to(root6).as_posix(),
            "input_sha256": copy.deepcopy(self.lineage["input_sha256"]),
            "content_id": "verified fixture content id",
            "tables": {"temporal_manifest": {"fixture": "temporal_manifest"},
                       "temporal_image_audit": {"fixture": "temporal_image_audit"}},
            "artifacts": [
                {"relative_path": path.relative_to(root6).as_posix(),
                 "sha256": self.digest(path), "bytes": path.stat().st_size}
                for path in (config_path, lineage_path)
            ],
        }
        if stale:
            state6["input_sha256"]["sector6/metadata"] = "0" * 64
        (root6 / s7.S6_STATE_PATH).write_text(json.dumps(state6), encoding="utf-8")
        context = {"inputs": {"sector5": self.inputs["sector5"]},
                   "lineage": copy.deepcopy(self.lineage)}
        return root6, state6, context

    def test_input_loader_inherits_current_sector6_and_preserves_inputs(self):
        root6, state6, context = self.loader_fixture()
        before = copy.deepcopy(context)
        tables6 = self.inputs["sector6"]
        with mock.patch.object(s7, "io_load_sector6_context", return_value=context), \
             mock.patch.object(s7, "_p6_read_state", return_value=state6), \
             mock.patch.object(s7, "_p3_read_table", side_effect=lambda root, descriptor: tables6[descriptor["fixture"]].copy()):
            loaded = s7.io_load_sector7_context({"storage_root": str(self.root), "sector6_dir": str(root6)}, {})
            overridden = s7.io_load_sector7_context({"storage_root": str(self.root), "sector6_dir": str(root6)},
                                                    {"qc_fov_radius_fraction": 0.82})
        self.assertEqual(loaded["config"]["qc_fov_radius_fraction"], 0.72)
        self.assertEqual(overridden["config"]["qc_fov_radius_fraction"], 0.82)
        self.assertEqual(loaded["config"]["github"]["repo_url"], "https://github.com/example/verified.git")
        self.assertEqual(loaded["config"]["sector"], 7)
        self.assertIn("sector6/state", loaded["lineage"]["input_sha256"])
        self.assertIn("sector6", loaded["lineage"]["upstream"])
        self.assertEqual(loaded["lineage"]["sector6_saved_input_sha256"], self.lineage["input_sha256"])
        pd.testing.assert_frame_equal(context["inputs"]["sector5"]["labelled_frame_manifest_with_segments"],
                                      before["inputs"]["sector5"]["labelled_frame_manifest_with_segments"])
        self.assertEqual(context["lineage"], before["lineage"])

    def test_input_loader_rejects_sector6_built_from_stale_prior_chain(self):
        root6, state6, context = self.loader_fixture(stale=True)
        with mock.patch.object(s7, "io_load_sector6_context", return_value=context), \
             mock.patch.object(s7, "_p6_read_state", return_value=state6):
            with self.assertRaisesRegex(RuntimeError, "not built from the current completed"):
                s7.io_load_sector7_context({"storage_root": str(self.root), "sector6_dir": str(root6)}, {})

    def test_reports_and_enrichment_preserve_all_reference_rows(self):
        originals = {"labelled": self.labelled.copy(deep=True),
                     "temporal": self.temporal.copy(deep=True)}
        prepared = self.prepare()
        tables = prepared["tables"]
        self.assertEqual(len(tables["image_qc_inventory"]), 4)
        self.assertEqual(len(tables["image_read_failure_report"]), 2)
        missing = tables["missing_image_reference_report"]
        self.assertEqual(len(missing), 3)
        self.assertEqual(missing["reference_status"].eq("pending_context_image").sum(), 1)
        coverage = tables["qc_reference_report"].groupby("source_manifest", observed=True)
        self.assertEqual(coverage.size()["labelled_manifest"], len(self.labelled))
        self.assertEqual(coverage.size()["temporal_manifest"], len(self.temporal))
        self.assertEqual(coverage["qc_reference_covered"].sum()["labelled_manifest"], 5)
        self.assertEqual(coverage["qc_reference_covered"].sum()["temporal_manifest"], 3)
        for key, original in [("labelled_frame_manifest_with_qc", self.labelled),
                              ("temporal_manifest_with_qc", self.temporal)]:
            enriched = tables[key]
            self.assertEqual(len(enriched), len(original))
            pd.testing.assert_frame_equal(enriched[list(original)], original.reset_index(drop=True))
        self.assertTrue(tables["labelled_frame_manifest_with_qc"].iloc[:2]["qc_reference_covered"].all())
        self.assertFalse(tables["temporal_manifest_with_qc"].iloc[-2:]["qc_reference_covered"].any())
        self.assertTrue(tables["temporal_manifest_with_qc"].iloc[-2:]["qc_requires_review"].all())
        pd.testing.assert_frame_equal(self.labelled, originals["labelled"])
        pd.testing.assert_frame_equal(self.temporal, originals["temporal"])

    def test_exact_parquet_roundtrip_and_unchanged_second_persist(self):
        prepared = self.prepare()
        first = self.persist(prepared)
        self.assertEqual(first["status"], "updated")
        root = Path(first["root"])
        self.assertEqual(set(first["state"]["tables"]), set(prepared["tables"]))
        for name, descriptor in first["state"]["tables"].items():
            self.assertTrue(descriptor["relative_path"].endswith(".parquet"))
            restored = pd.read_parquet(root / descriptor["relative_path"], engine="pyarrow")
            pd.testing.assert_frame_equal(prepared["tables"][name], restored, check_exact=True)
        before = {path.relative_to(root): (path.stat().st_mtime_ns, self.digest(path))
                  for path in root.rglob("*") if path.is_file()}
        second = self.persist(prepared)
        self.assertEqual(second["status"], "unchanged")
        after = {path.relative_to(root): (path.stat().st_mtime_ns, self.digest(path))
                 for path in root.rglob("*") if path.is_file()}
        self.assertEqual(before, after)
        restored = s7.io_load_cached_sector7(self.config, prepared["processing_signature"])
        self.assertIsNotNone(restored)
        for name, expected in prepared["tables"].items():
            pd.testing.assert_frame_equal(restored["tables"][name], expected, check_exact=True)

    def test_cache_reuses_before_any_image_decode(self):
        prepared = self.prepare()
        self.persist(prepared)
        namespace = dict(vars(s7))
        with mock.patch.object(s7, "io_compute_qc_for_image_inventory", side_effect=AssertionError("QC recalculated")), \
             mock.patch.object(s7.cv2, "imdecode", side_effect=AssertionError("image decoded")):
            repeated = self.prepare(namespace=namespace)
        self.assertEqual(repeated["status"], "reused")
        self.assertEqual(prepared["processing_signature"], repeated["processing_signature"])

    def test_cache_invalidated_by_changed_image_bytes(self):
        prepared = self.prepare()
        self.persist(prepared)
        self.write_image(self.image_a, 105)
        changed = self.prepare()
        self.assertEqual(changed["status"], "computed")
        self.assertNotEqual(changed["processing_signature"], prepared["processing_signature"])
        image = changed["tables"]["image_qc_manifest"].set_index("image_relpath").loc["images/a.png"]
        self.assertFalse(image["qc_source_matches_sector6"])
        self.assertTrue(image["qc_requires_review"])

    def test_cache_invalidated_by_config_and_active_runtime_identity(self):
        prepared = self.prepare()
        self.persist(prepared)
        config = copy.deepcopy(self.config)
        config["qc_fov_radius_fraction"] = 0.8
        changed_config = self.prepare(config=config)
        self.assertEqual(changed_config["status"], "computed")
        self.assertNotEqual(changed_config["processing_signature"], prepared["processing_signature"])
        namespace = dict(vars(s7))
        namespace["compute_image_qc"] = changed_compute_image_qc
        changed_runtime = self.prepare(namespace=namespace)
        self.assertEqual(changed_runtime["status"], "computed")
        self.assertNotEqual(changed_runtime["processing_signature"], prepared["processing_signature"])

    def test_publication_toggle_reuses_metrics_and_updates_configuration_snapshot(self):
        prepared = self.prepare()
        saved = self.persist(prepared)
        config = copy.deepcopy(self.config)
        config["github"]["push"] = True
        config["sector7"]["push"] = True
        repeated = self.prepare(config=config)
        self.assertEqual(repeated["status"], "reused")
        self.assertEqual(repeated["processing_signature"], prepared["processing_signature"])
        updated = self.persist(repeated, config=config)
        self.assertEqual(updated["status"], "updated")
        self.assertNotEqual(updated["state"]["config_snapshot"], saved["state"]["config_snapshot"])
        snapshot = json.loads((Path(updated["root"]) / updated["state"]["config_snapshot"]).read_text(encoding="utf-8"))
        self.assertTrue(snapshot["effective_config"]["github"]["push"])
        self.assertTrue(snapshot["effective_config"]["sector7"]["push"])
        self.assertEqual(updated["state"]["tables"], saved["state"]["tables"])

    def test_corrupt_catalog_artifact_is_rejected(self):
        prepared = self.prepare()
        persisted = self.persist(prepared)
        path = Path(persisted["root"]) / persisted["state"]["tables"]["image_qc_manifest"]["relative_path"]
        path.write_bytes(path.read_bytes() + b"corruption")
        with self.assertRaisesRegex(RuntimeError, "checksum verification failed"):
            s7.io_load_cached_sector7(self.config, prepared["processing_signature"])

    def test_changed_upstream_metadata_is_rejected_before_persist(self):
        prepared = self.prepare()
        self.input_file.write_text("metadata changed after loading", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "changed after loading"):
            self.persist(prepared)

    def test_image_changed_during_run_is_rejected(self):
        prepared = self.prepare()
        self.write_image(self.image_a, 110)
        with self.assertRaisesRegex(RuntimeError, "source image changed"):
            self.persist(prepared)

    def test_edited_source_audit_is_rejected_without_touching_saved_snapshot(self):
        prepared = self.prepare()
        persisted = self.persist(prepared)
        root = Path(persisted["root"])
        before = {path.relative_to(root): (path.stat().st_mtime_ns, self.digest(path))
                  for path in root.rglob("*") if path.is_file()}
        edited = copy.deepcopy(prepared)
        edited["tables"]["image_qc_source_audit"].loc[0, "image_bytes"] += 1
        with self.assertRaisesRegex(ValueError, "source audit changed after signature"):
            self.persist(edited)
        after = {path.relative_to(root): (path.stat().st_mtime_ns, self.digest(path))
                 for path in root.rglob("*") if path.is_file()}
        self.assertEqual(before, after)

    def test_failed_installed_readback_rolls_back_previous_snapshot(self):
        prepared = self.prepare()
        persisted = self.persist(prepared)
        root = Path(persisted["root"])
        before = {path.relative_to(root): (path.stat().st_mtime_ns, self.digest(path))
                  for path in root.rglob("*") if path.is_file()}
        self.write_image(self.image_a, 110)
        newer = self.prepare()
        new_fingerprint = s7.sector3_table_fingerprint(newer["tables"]["image_qc_manifest"])
        original_reader = s7._p3_read_table
        injected = []
        def fail_new_readback(read_root, descriptor):
            if Path(read_root).resolve() == root.resolve() and descriptor["fingerprint"] == new_fingerprint:
                injected.append(True)
                raise RuntimeError("injected installed readback failure")
            return original_reader(read_root, descriptor)
        with mock.patch.object(s7, "_p3_read_table", side_effect=fail_new_readback):
            with self.assertRaisesRegex(RuntimeError, "injected installed readback failure"):
                self.persist(newer)
        self.assertTrue(injected)
        after = {path.relative_to(root): (path.stat().st_mtime_ns, self.digest(path))
                 for path in root.rglob("*") if path.is_file()}
        self.assertEqual(before, after)
        self.assertEqual(s7._p7_read_state(root), persisted["state"])

    def test_failed_installed_catalog_validation_rolls_back_previous_snapshot(self):
        prepared = self.prepare()
        persisted = self.persist(prepared)
        root = Path(persisted["root"])
        before = {path.relative_to(root): (path.stat().st_mtime_ns, self.digest(path))
                  for path in root.rglob("*") if path.is_file()}
        self.write_image(self.image_a, 115)
        newer = self.prepare()
        original_reader = s7._p7_read_state
        injected = []
        def fail_new_catalog(read_root):
            path = Path(read_root) / s7.S7_STATE_PATH
            if path.is_file() and json.loads(path.read_text(encoding="utf-8"))["processing_signature"] == newer["processing_signature"]:
                injected.append(True)
                raise RuntimeError("injected installed catalog failure")
            return original_reader(read_root)
        with mock.patch.object(s7, "_p7_read_state", side_effect=fail_new_catalog):
            with self.assertRaisesRegex(RuntimeError, "injected installed catalog failure"):
                self.persist(newer)
        self.assertTrue(injected)
        after = {path.relative_to(root): (path.stat().st_mtime_ns, self.digest(path))
                 for path in root.rglob("*") if path.is_file()}
        self.assertEqual(before, after)
        self.assertEqual(s7._p7_read_state(root), persisted["state"])

    def test_local_git_export_preserves_other_files_and_unchanged_commit(self):
        remote = self.root / "remote.git"
        seed = self.root / "seed"
        subprocess.run(["git", "init", "--bare", "--quiet", str(remote)], check=True)
        subprocess.run(["git", "init", "--quiet", "-b", "main", str(seed)], check=True)
        other = seed / "outputs/phase2/sector6/preserved.txt"
        other.parent.mkdir(parents=True)
        other.write_text("keep sector6", encoding="utf-8")
        (seed / "README.md").write_text("keep repository readme", encoding="utf-8")
        subprocess.run(["git", "-C", str(seed), "add", "."], check=True)
        subprocess.run(["git", "-C", str(seed), "-c", "user.name=QC Test", "-c", "user.email=qc@example.invalid", "commit", "--quiet", "-m", "Seed"], check=True)
        subprocess.run(["git", "-C", str(seed), "remote", "add", "origin", str(remote)], check=True)
        subprocess.run(["git", "-C", str(seed), "push", "--quiet", "origin", "main"], check=True)
        config = copy.deepcopy(self.config)
        config["github"] = {"repo_url": str(remote), "branch": "main", "push": True,
                            "dry_run": False, "allow_local_test_remote": True,
                            "git_username": "QC Test", "git_email": "qc@example.invalid"}
        config["sector7"]["push"] = True
        prepared = self.prepare(config=config)
        persisted = self.persist(prepared, config=config)
        published = s7.io_push_sector7(persisted, config)
        self.assertTrue(published["pushed"])
        self.assertEqual(published["status"], "published")
        self.assertTrue(all(path.startswith("outputs/phase2/sector7/") for path in published["changed_paths"]))
        tip = subprocess.check_output(["git", "--git-dir", str(remote), "rev-parse", "main"], text=True).strip()
        self.assertEqual(subprocess.check_output(["git", "--git-dir", str(remote), "show", "main:outputs/phase2/sector6/preserved.txt"], text=True), "keep sector6")
        self.assertEqual(subprocess.check_output(["git", "--git-dir", str(remote), "show", "main:README.md"], text=True), "keep repository readme")
        unchanged = s7.io_push_sector7(persisted, config)
        self.assertEqual(unchanged["status"], "unchanged")
        self.assertFalse(unchanged["pushed"])
        self.assertEqual(tip, subprocess.check_output(["git", "--git-dir", str(remote), "rev-parse", "main"], text=True).strip())
        self.assertEqual(subprocess.check_output(["git", "--git-dir", str(remote), "rev-list", "--count", "main"], text=True).strip(), "2")


if __name__ == "__main__":
    unittest.main()
