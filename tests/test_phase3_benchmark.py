"""Contract tests with small, genuine Parquet/JSONL publication fixtures."""
import copy
import gzip
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image
import pyarrow as pa
import pyarrow.parquet as pq

from mmvqa.phase3.benchmark import S8Benchmark, _canonical
from mmvqa.phase3.storage import IntegrityError, verify_file


SHA = "a" * 64
PREFIX = f"metadata/{SHA}"


class FixtureStore:
    def __init__(self, root, *, bad_subject=False, missing_links=False, second_cohort=False,
                 bad_derived=False, bad_source=False):
        self.root, self.entries, self.consumed = Path(root), {}, []
        self.tables = {}
        original = {"input_id": "input-full", "artifact_id": "artifact-full", "input_order": 0,
                    "storage": "original", "relative_path": "images/lesion/frame.jpg", "sha256": SHA,
                    "width": 2, "height": 2, "image_role": "target"}
        crop = {**original, "input_id": "input-crop", "artifact_id": "artifact-crop",
                "storage": "derived", "relative_path": "products/roi_crop/crop.jpg", "image_role": "crop"}
        self.rows = [self.recipe("s1", "full", original), self.recipe("s1", "crop", crop)]
        if bad_derived:
            crop["sha256"] = "d" * 64
        if second_cohort:
            self.rows.append(self.recipe("s2", "full", original))
        recipes = gzip.compress("".join(json.dumps(row) + "\n" for row in self.rows).encode())
        self.add(f"{PREFIX}/benchmark/model_inputs.jsonl.gz", recipes)
        truth = [{"ground_truth_ref": "truth-s1", "subject_id": "wrong" if bad_subject else "s1",
                  "annotation_id": "annotation-1", "source_label": "private lesion label",
                  "question": None, "answer": None, "qa_validation_status": "not_authored",
                  "bbox_xyxy_json": "[0, 0, 1, 1]"}]
        links = [{"artifact_id": artifact, "source_index": 0,
                  "image_relpath": original["relative_path"], "source_sha256": SHA,
                  "annotation_ids_json": '["annotation-1"]'}
                 for artifact in ("artifact-full", "artifact-crop")]
        self.table("benchmark_ground_truth_manifest", truth)
        self.table("visual_source_manifest", links[:1] if missing_links else links)
        self.table("source_evidence_manifest", [{"image_relpath": original["relative_path"],
                  "sha256": "e" * 64 if bad_source else SHA, "image_size_bytes": 123,
                  "width": 2, "height": 2, "annotation_ids_json": '["annotation-1"]'}])
        member = {"relative_path": crop["relative_path"], "sha256": SHA, "bytes": 123}
        logical = [*self.entries.values(), member]
        archive = {"relative_path": "archives/roi_crop.tar", "sha256": "b" * 64, "bytes": 10240,
                   "members": [member], "member_count": 1, "kind": "roi_crop"}
        semantic = self.add("configs/sector8_state.json", b'{"semantic": true}\n')
        manifest = {"schema_version": 1, "mode": "tar_per_product_folder", "archive_count": 1,
                    "member_count": 1, "archives": [archive], "source_catalog_sha256": semantic["sha256"],
                    "logical_artifacts_sha256": hashlib.sha256(_canonical(logical)).hexdigest()}
        self.add("configs/sector8_archives.json", json.dumps(manifest).encode())
        self.state = {"schema_version": 1, "phase": 2, "sector": 8, "complete": True,
                      "processing_signature": SHA, "training_ready": False,
                      "logical_artifacts": logical, "tables": self.tables,
                      "archive_delivery": {"mode": "tar_per_product_folder",
                                           "manifest_relative_path": "configs/sector8_archives.json"},
                      "artifacts": [*self.entries.values(), archive]}

    @staticmethod
    def recipe(subject, variant, item):
        return {"example_id": subject + "-" + variant, "subject_id": subject,
                "ground_truth_ref": "truth-" + subject, "shared_source_group_id": "group-1",
                "variant": variant, "ordered_inputs": [item], "image_count": 1,
                "pixels_available": True, "annotation_assisted": variant == "crop",
                "training_ready": False, "final_split": None, "split_assignment_status": "pending"}

    def add(self, path, raw):
        target = self.root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
        record = {"relative_path": path, "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
        self.entries[path] = record
        return record

    def table(self, name, rows):
        table = pa.Table.from_pylist(rows)
        path = f"{PREFIX}/results/{name}.parquet"
        sink = pa.BufferOutputStream()
        pq.write_table(table, sink)
        record = self.add(path, sink.getvalue().to_pybytes())
        schema_path = f"{PREFIX}/results/{name}_schema.json"
        self.add(schema_path, json.dumps({"rows": len(rows), "columns": table.column_names}).encode())
        self.tables[name] = {**record, "schema_path": schema_path, "rows": len(rows)}

    def resolve(self):
        return "c" * 40

    def catalog(self):
        return copy.deepcopy(self.state)

    def load(self, path):
        self.consumed.append(path)
        target = self.root / path
        verify_file(target, self.entries[path]["bytes"], self.entries[path]["sha256"])
        return target

    def read_json(self, path):
        return json.loads(self.load(path).read_bytes())

    def snapshot(self):
        return {"resolved_commit": self.resolve(), "consumed_artifacts": list(self.consumed)}


class Resolver:
    def load_inputs(self, items):
        return ([Image.new("RGB", (2, 2)) for _ in items], [item["image_role"] for item in items],
                [{"verified": True} for _ in items])


class BenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def fixture(self, **kwargs):
        store = FixtureStore(self.tmp.name, **kwargs)
        return store, S8Benchmark(store)

    def test_same_subject_sampling_and_explicit_payload_boundary(self):
        store, benchmark = self.fixture()
        rows = benchmark.sample(("full", "crop"))
        self.assertEqual([r["subject_id"] for r in rows], ["s1", "s1"])
        self.assertEqual(len(benchmark.source_records(rows)), 1)
        result = benchmark.prepare(rows[1], Resolver(), "What is visible?",
                                   allowed_context=("task_instruction",),
                                   context={"task_instruction": "Describe pixels"})
        self.assertEqual(set(result["model"]), {"question", "images", "image_roles", "context"})
        self.assertEqual(result["evaluator"]["source_annotation_evidence"]["source_label"], "private lesion label")
        self.assertEqual(result["evaluator"]["source_annotation_evidence"]["bbox_xyxy"], [0, 0, 1, 1])
        self.assertIsNone(result["evaluator"]["reference_answer"])
        self.assertFalse(result["training_ready"])
        self.assertEqual(result["pixel_verification"], "verified")
        self.assertIn("sources", result["provenance"]["inputs"][0])
        self.assertNotIn("private lesion label", repr(result["model"]))

    def test_inspect_never_claims_pixel_verification(self):
        _, benchmark = self.fixture()
        result = benchmark.inspect(benchmark.sample(("full",))[0])
        self.assertNotIn("model", result)
        self.assertEqual(result["pixel_verification"], "not_run")
        self.assertFalse(result["training_ready"])

    def test_missing_requested_variant_is_error(self):
        _, benchmark = self.fixture()
        with self.assertRaisesRegex(IntegrityError, "Not enough"):
            benchmark.sample(("full", "temporal_multi"))

    def test_requested_count_is_not_silently_reduced(self):
        _, benchmark = self.fixture(second_cohort=True)
        with self.assertRaisesRegex(IntegrityError, "Not enough"):
            benchmark.sample(("full", "crop"), 2)

    def test_recipe_mutation_and_leaking_context_are_rejected(self):
        _, benchmark = self.fixture()
        row = benchmark.sample(("full",))[0]
        changed = copy.deepcopy(row)
        changed["ordered_inputs"][0]["relative_path"] = "another.jpg"
        with self.assertRaises(IntegrityError):
            benchmark.inspect(changed)
        with self.assertRaises(ValueError):
            benchmark.prepare(row, Resolver(), "Question", context={"source_label": "lesion"})
        with self.assertRaises(ValueError):
            benchmark.prepare(row, Resolver(), "Question", allowed_context=("source_label",),
                              context={"source_label": "lesion"})

    def test_ground_truth_cannot_cross_subjects(self):
        _, benchmark = self.fixture(bad_subject=True)
        with self.assertRaisesRegex(IntegrityError, "crosses subjects"):
            benchmark.inspect(benchmark.sample(("full",))[0])

    def test_missing_source_link_is_error(self):
        _, benchmark = self.fixture(missing_links=True)
        with self.assertRaisesRegex(IntegrityError, "coverage/order"):
            benchmark.inspect(benchmark.sample(("crop",))[0])

    def test_catalog_and_index_cannot_select_different_bytes(self):
        store = FixtureStore(self.tmp.name)
        store.entries["configs/sector8_archives.json"] = {
            **store.entries["configs/sector8_archives.json"], "sha256": "f" * 64}
        with self.assertRaisesRegex(IntegrityError, "disagree"):
            S8Benchmark(store)

    def test_nested_private_context_is_rejected(self):
        _, benchmark = self.fixture()
        row = benchmark.sample(("full",))[0]
        with self.assertRaisesRegex(ValueError, "not allowed"):
            benchmark.prepare(row, Resolver(), "Question", allowed_context=("task",),
                              context={"task": [{"nested": {"source_label_json": "lesion"}}]})

    def test_derived_recipe_must_match_archive_member_without_loading_pixels(self):
        _, benchmark = self.fixture(bad_derived=True)
        with self.assertRaisesRegex(IntegrityError, "catalogued archive member"):
            benchmark.inspect(benchmark.sample(("crop",))[0])

    def test_source_link_must_match_source_evidence_without_loading_pixels(self):
        _, benchmark = self.fixture(bad_source=True)
        with self.assertRaisesRegex(IntegrityError, "source-evidence hash"):
            benchmark.inspect(benchmark.sample(("crop",))[0])

    def test_table_schema_rows_are_checked(self):
        store, benchmark = self.fixture()
        benchmark.catalog["tables"]["benchmark_ground_truth_manifest"]["rows"] = 99
        with self.assertRaisesRegex(IntegrityError, "row count"):
            benchmark.inspect(benchmark.sample(("full",))[0])

    def test_metadata_cache_tampering_is_detected_on_later_inspection(self):
        store, benchmark = self.fixture()
        row = benchmark.sample(("full",))[0]
        benchmark.inspect(row)
        target = store.root / benchmark.catalog["tables"]["benchmark_ground_truth_manifest"]["relative_path"]
        target.write_bytes(b"changed after first inspection")
        with self.assertRaisesRegex(IntegrityError, "Integrity check failed"):
            benchmark.inspect(row)

    def test_temporal_order_cannot_be_replaced_by_list_position(self):
        store, _ = self.fixture()
        row = copy.deepcopy(store.rows[0])
        row["variant"] = "temporal_multi"
        first = {**row["ordered_inputs"][0], "frame_index": 10, "is_target": True,
                 "time_offset_seconds": 0.0}
        second = {**first, "input_order": 1, "frame_index": 9, "is_target": False,
                  "time_offset_seconds": -0.5, "image_role": "temporal_context"}
        row.update(ordered_inputs=[first, second], image_count=2)
        with self.assertRaisesRegex(IntegrityError, "increasing frames"):
            S8Benchmark._validate_recipe(row)


if __name__ == "__main__":
    unittest.main()
