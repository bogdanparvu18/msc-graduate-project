"""S8-backed preview recipes with separate model, evaluator and lineage views.

This module unlocks input inspection. It deliberately does not author clinical QA,
assign splits or certify a benchmark for scoring or training.
"""

from __future__ import annotations

from collections import defaultdict
import copy
import gzip
import hashlib
import json
import math
import re

from .storage import IntegrityError, safe_relative_path


DEFAULT_VARIANTS = ("full", "crop", "overlay", "fullcrop", "temporal_multi", "temporal_collage")
_HASH = re.compile(r"[0-9a-f]{64}\Z")
_PRIVATE_CONTEXT = {
    "answer", "reference_answer", "source_label", "clinical_group", "annotation_id",
    "annotation_ids", "bbox_xyxy", "ground_truth_ref", "subject_id", "example_id",
    "relative_path", "image_relpath", "sha256", "artifact_id", "verified_source_labels",
    "source_qc", "final_split", "shared_source_group_id", "source_annotation_evidence",
    "source_sha256", "ground_truth", "evaluator", "provenance", "ordered_inputs",
    "source_refs", "annotation_assisted", "verified_labels", "source_snapshot",
}


def _canonical(value):
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False,
                       allow_nan=False) + "\n").encode("utf-8")


def _identity(record):
    return {key: record.get(key) for key in ("relative_path", "sha256", "bytes")}


def _records(rows, description):
    if not isinstance(rows, list):
        raise IntegrityError(f"{description} must be a list")
    result = {}
    for record in rows:
        if not isinstance(record, dict):
            raise IntegrityError(f"{description} contains a non-object entry")
        path = safe_relative_path(record.get("relative_path"))
        if path in result:
            raise IntegrityError(f"Duplicate {description} path: {path}")
        size = record.get("bytes")
        if type(size) is not int or size < 0 or not _HASH.fullmatch(str(record.get("sha256"))):
            raise IntegrityError(f"Invalid {description} identity: {path}")
        result[path] = record
    return result


def _clean(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _clean(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_clean(item) for item in value]
    return value


def _decoded(row):
    result = _clean(row)
    for key in list(result):
        if key.endswith("_json") and isinstance(result[key], str):
            result[key[:-5]] = json.loads(result.pop(key))
    return result


def _check_context(value):
    """Reject evaluator and lineage fields even inside explicitly allowed containers."""
    if isinstance(value, dict):
        for key, child in value.items():
            if not isinstance(key, str):
                raise ValueError("Context keys must be strings")
            normalized = key.lower()
            if normalized.endswith("_json"):
                normalized = normalized[:-5]
            if normalized in _PRIVATE_CONTEXT:
                raise ValueError(f"Evaluator/provenance field is not allowed in model context: {key}")
            _check_context(child)
    elif isinstance(value, (tuple, list)):
        for child in value:
            _check_context(child)


class S8Benchmark:
    """Discover the active run from its verified catalog and preview its recipes."""

    def __init__(self, store):
        self.store = store
        self.store.resolve()
        self.catalog = self.store.catalog()
        if (self.catalog.get("schema_version") != 1 or self.catalog.get("phase") != 2
                or self.catalog.get("sector") != 8 or self.catalog.get("complete") is not True):
            raise IntegrityError("The selected catalog is not a complete S8 publication")
        self.transport = _records(self.catalog.get("artifacts"), "transport artifact")
        self.logical = _records(self.catalog.get("logical_artifacts"), "logical artifact")
        signature = self.catalog.get("processing_signature")
        if not isinstance(signature, str) or not _HASH.fullmatch(signature):
            raise IntegrityError("S8 catalog has no valid processing signature")
        self.recipes_path = f"metadata/{signature}/benchmark/model_inputs.jsonl.gz"
        delivery = self.catalog.get("archive_delivery", {})
        if delivery.get("mode") != "tar_per_product_folder":
            raise IntegrityError("Unsupported S8 archive delivery mode")
        manifest_path = delivery.get("manifest_relative_path")
        self._bind(manifest_path)
        self.archive_manifest = self.store.read_json(manifest_path)
        manifest = self.archive_manifest
        semantic = self.transport.get("configs/sector8_state.json", {})
        self._bind("configs/sector8_state.json")
        if (manifest.get("schema_version") != 1 or manifest.get("mode") != "tar_per_product_folder"
                or manifest.get("source_catalog_sha256") != semantic.get("sha256")
                or manifest.get("logical_artifacts_sha256") != hashlib.sha256(
                    _canonical(self.catalog["logical_artifacts"])).hexdigest()):
            raise IntegrityError("Archive manifest belongs to a different logical S8 run")
        archives = manifest.get("archives")
        archive_records = _records(archives, "archive")
        if manifest.get("archive_count") != len(archive_records):
            raise IntegrityError("Archive count differs from the manifest")
        total_members = 0
        all_members = set()
        self.members = {}
        for archive in archives:
            if _identity(archive) != _identity(self.transport.get(archive["relative_path"], {})):
                raise IntegrityError("Archive identity differs from the transport catalog")
            members = _records(archive.get("members"), "archive member")
            if archive.get("member_count") != len(members):
                raise IntegrityError("Archive member count differs from the manifest")
            for path, record in members.items():
                if path in all_members or _identity(record) != _identity(self.logical.get(path, {})):
                    raise IntegrityError("Archive member differs from the logical catalog")
                all_members.add(path)
                self.members[path] = record
            total_members += len(members)
        if manifest.get("member_count") != total_members:
            raise IntegrityError("Total archive member count differs from the manifest")
        self._selected = {}
        self._tables = {}

    def _bind(self, path, descriptor=None):
        path = safe_relative_path(path)
        indexed = self.store.entries.get(path)
        declared = self.transport.get(path)
        if indexed is None or declared is None or _identity(indexed) != _identity(declared):
            raise IntegrityError(f"Storage index and catalog disagree: {path}")
        if descriptor is not None and _identity(descriptor) != _identity(declared):
            raise IntegrityError(f"Table descriptor and catalog disagree: {path}")
        if path in self.logical and _identity(self.logical[path]) != _identity(declared):
            raise IntegrityError(f"Logical and transport artifacts disagree: {path}")
        return path

    def _table(self, name, filters=None):
        try:
            import pyarrow.parquet as pq
        except ImportError as error:
            raise RuntimeError("Install pyarrow to load S8 Parquet metadata") from error
        descriptor = self.catalog.get("tables", {}).get(name)
        if not isinstance(descriptor, dict):
            raise IntegrityError(f"Missing S8 table descriptor: {name}")
        path = self._bind(descriptor.get("relative_path"), descriptor)
        schema_path = self._bind(descriptor.get("schema_path"))
        # Revalidate cached bytes whenever they are consumed; a filesystem cache
        # must not become a second, unchecked source of truth.
        local = self.store.load(path)
        if name not in self._tables:
            schema = self.store.read_json(schema_path)
            footer = pq.ParquetFile(local)
            if (footer.metadata.num_rows != descriptor.get("rows")
                    or footer.metadata.num_rows != schema.get("rows")
                    or footer.schema_arrow.names != schema.get("columns")):
                raise IntegrityError(f"Parquet row count or columns differ from S8 schema: {name}")
            self._tables[name] = local
        # Filtering avoids loading whole manifests into Python objects.
        return [_decoded(row) for row in pq.read_table(local, filters=filters).to_pylist()]

    @staticmethod
    def _validate_recipe(row):
        for key in ("example_id", "subject_id", "ground_truth_ref", "shared_source_group_id"):
            if not isinstance(row.get(key), str) or not row[key]:
                raise IntegrityError(f"Recipe has no {key}")
        items = row.get("ordered_inputs")
        if not isinstance(items, list) or not items or row.get("image_count") != len(items):
            raise IntegrityError("Recipe input list/image_count is inconsistent")
        if row.get("pixels_available") is not True:
            raise IntegrityError("Selected recipe has no complete pixel evidence")
        for order, item in enumerate(items):
            if type(item.get("input_order")) is not int or item["input_order"] != order:
                raise IntegrityError("Recipe input order must be contiguous and zero based")
            safe_relative_path(item.get("relative_path"))
            if item.get("storage") not in {"original", "derived"}:
                raise IntegrityError("Recipe has an unsupported image storage kind")
            if not _HASH.fullmatch(str(item.get("sha256"))):
                raise IntegrityError("Recipe image has no valid SHA-256")
            if not isinstance(item.get("artifact_id"), str) or not item["artifact_id"]:
                raise IntegrityError("Recipe image has no artifact identity")
            if not isinstance(item.get("image_role"), str) or not item["image_role"]:
                raise IntegrityError("Recipe image has no role")
            if any(type(item.get(key)) is not int or item[key] <= 0 for key in ("width", "height")):
                raise IntegrityError("Recipe image dimensions must be positive integers")
        if row.get("variant") == "temporal_multi":
            frames = [item.get("frame_index") for item in items]
            if (len(items) < 2 or any(type(frame) is not int or frame < 0 for frame in frames)
                    or any(left >= right for left, right in zip(frames, frames[1:]))
                    or any(type(item.get("is_target")) is not bool for item in items)
                    or sum(item["is_target"] for item in items) != 1):
                raise IntegrityError("Temporal inputs require increasing frames and exactly one target")
            for item in items:
                expected_role = "target" if item["is_target"] else "temporal_context"
                offset = item.get("time_offset_seconds")
                if (item["image_role"] != expected_role or isinstance(offset, bool)
                        or not isinstance(offset, (int, float)) or not math.isfinite(offset)):
                    raise IntegrityError("Temporal role/offset metadata is inconsistent")

    def sample(self, variants=DEFAULT_VARIANTS, per_variant=1, same_subject=True):
        """Select deterministic previews; by default every variant uses the same subjects."""
        variants = tuple(variants)
        if (not variants or len(set(variants)) != len(variants)
                or any(not isinstance(v, str) or not v for v in variants)
                or type(per_variant) is not int or per_variant < 1):
            raise ValueError("Provide distinct variants and a positive per_variant count")
        path = self.store.load(self._bind(self.recipes_path))
        chosen = {variant: [] for variant in variants}
        group = {}
        current_subject = None
        def consider():
            if same_subject and not all(v in group for v in variants):
                return
            for variant in variants:
                if variant in group and len(chosen[variant]) < per_variant:
                    chosen[variant].append(group[variant])
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                subject = row.get("subject_id")
                if not isinstance(subject, str):
                    raise IntegrityError("Recipe has no subject identity")
                if current_subject is not None and subject < current_subject:
                    raise IntegrityError("S8 recipes are not sorted by subject")
                if subject != current_subject:
                    consider()
                    if all(len(chosen[v]) == per_variant for v in variants):
                        break
                    group = {}
                    current_subject = subject
                variant = row.get("variant")
                if variant in chosen:
                    self._validate_recipe(row)
                    if variant in group:
                        raise IntegrityError("Duplicate subject/variant recipe")
                    group[variant] = row
            else:
                consider()
        missing = {v: per_variant - len(rows) for v, rows in chosen.items() if len(rows) < per_variant}
        if missing:
            raise IntegrityError(f"Not enough eligible preview recipes (same_subject={same_subject}): {missing}")
        rows = [row for variant in variants for row in chosen[variant]]
        self._selected.update({row["example_id"]: copy.deepcopy(row) for row in rows})
        return rows

    def source_records(self, examples):
        """Load source records for original image byte-size and identity checks."""
        for example in examples:
            if example != self._selected.get(example.get("example_id")):
                raise IntegrityError("Source lookup requires unchanged recipes returned by sample()")
        paths = sorted({item["relative_path"] for row in examples for item in row["ordered_inputs"]
                        if item["storage"] == "original"})
        if not paths:
            return []
        rows = self._table("source_evidence_manifest", [("image_relpath", "in", paths)])
        if len(rows) != len(paths) or {r["image_relpath"] for r in rows} != set(paths):
            raise IntegrityError("Original image source-evidence coverage is incomplete")
        expected = {item["relative_path"]: item["sha256"] for row in examples
                    for item in row["ordered_inputs"] if item["storage"] == "original"}
        for row in rows:
            if row.get("sha256") != expected[row["image_relpath"]]:
                raise IntegrityError("Original recipe and source-evidence hashes differ")
        return rows

    def inspect(self, example):
        """Verify annotation and source links without reading image bytes."""
        if example != self._selected.get(example.get("example_id")):
            raise IntegrityError("Inspect requires an unchanged recipe returned by sample()")
        self._validate_recipe(example)
        truth = self._table("benchmark_ground_truth_manifest",
                            [("ground_truth_ref", "=", example["ground_truth_ref"])])
        if len(truth) != 1 or truth[0].get("subject_id") != example["subject_id"]:
            raise IntegrityError("Recipe ground-truth linkage is not unique or crosses subjects")
        truth = truth[0]
        artifact_ids = [item["artifact_id"] for item in example["ordered_inputs"]]
        links = self._table("visual_source_manifest", [("artifact_id", "in", artifact_ids)])
        by_artifact = defaultdict(list)
        for link in links:
            safe_relative_path(link.get("image_relpath"))
            if not _HASH.fullmatch(str(link.get("source_sha256"))):
                raise IntegrityError("Provenance link lacks a source hash")
            by_artifact[link["artifact_id"]].append(link)
        source_paths = sorted({link["image_relpath"] for link in links})
        sources = self._table("source_evidence_manifest", [("image_relpath", "in", source_paths)]) if source_paths else []
        by_source = {row["image_relpath"]: row for row in sources}
        if len(sources) != len(source_paths) or set(by_source) != set(source_paths):
            raise IntegrityError("Visual-source links have incomplete source-evidence coverage")
        for link in links:
            source = by_source[link["image_relpath"]]
            if source.get("sha256") != link["source_sha256"]:
                raise IntegrityError("Visual-source link differs from its source-evidence hash")
            for source_key, link_key in (("width", "source_width"), ("height", "source_height"),
                                         ("video_key", "video_key"), ("frame_index", "frame_index")):
                if link_key in link and link[link_key] != source.get(source_key):
                    raise IntegrityError(f"Visual-source {link_key} differs from source evidence")
            if not set(link.get("annotation_ids", [])).issubset(source.get("annotation_ids", [])):
                raise IntegrityError("Visual-source annotations differ from source evidence")
        for item in example["ordered_inputs"]:
            source_links = sorted(by_artifact[item["artifact_id"]], key=lambda r: r["source_index"])
            if not source_links or [r["source_index"] for r in source_links] != list(range(len(source_links))):
                raise IntegrityError("Visual-source provenance coverage/order is incomplete")
            if item["storage"] == "original" and (len(source_links) != 1
                    or source_links[0]["image_relpath"] != item["relative_path"]
                    or source_links[0]["source_sha256"] != item["sha256"]):
                raise IntegrityError("Original input differs from its source provenance")
            if item["storage"] == "original" and "frame_index" in item:
                if item["frame_index"] != source_links[0].get("frame_index"):
                    raise IntegrityError("Temporal frame index differs from source provenance")
            if item["storage"] == "derived":
                member = self.members.get(item["relative_path"])
                if member is None or member["sha256"] != item["sha256"]:
                    raise IntegrityError("Derived recipe differs from its catalogued archive member")
            by_artifact[item["artifact_id"]] = source_links
        input_provenance = []
        for item in example["ordered_inputs"]:
            input_provenance.append({"input": copy.deepcopy(item),
                                     "sources": by_artifact[item["artifact_id"]]})
        return {
            "evaluator": {"reference_answer": None, "qa_validation_status": "not_authored",
                          "source_annotation_evidence": truth,
                          "scope": "Source annotations only; the preview question has no validated answer."},
            "provenance": {"example_id": example["example_id"], "subject_id": example["subject_id"],
                           "ground_truth_ref": example["ground_truth_ref"], "variant": example["variant"],
                           "annotation_assisted": bool(example.get("annotation_assisted")),
                           "shared_source_group_id": example["shared_source_group_id"],
                           "final_split": example.get("final_split"),
                           "split_assignment_status": example.get("split_assignment_status"),
                           "question_status": "preview_only", "inputs": input_provenance,
                           "source_snapshot": self.store.snapshot()},
            "training_ready": False, "preview_only": True, "pixel_verification": "not_run",
        }

    def prepare(self, example, resolver, question, allowed_context=(), context=None):
        """Resolve a selected preview and project three explicitly separate payloads."""
        if not isinstance(question, str) or not question.strip():
            raise ValueError("Provide an explicit preview question; S8 has no authored QA")
        context = {} if context is None else copy.deepcopy(context)
        allowed = set(allowed_context)
        if (not isinstance(context, dict) or not all(isinstance(k, str) for k in allowed)
                or set(context) - allowed or allowed & _PRIVATE_CONTEXT):
            raise ValueError("Context must use an explicit allowlist without evaluator/provenance fields")
        _check_context(context)
        # JSON validation prevents hidden image objects, NaN or metadata wrappers in context.
        json.dumps(context, allow_nan=False)
        result = self.inspect(example)
        images, roles, resolutions = resolver.load_inputs(example["ordered_inputs"])
        if len(images) != example["image_count"] or len(roles) != len(images) or len(resolutions) != len(images):
            raise IntegrityError("Image resolver returned incomplete inputs")
        for item, resolution in zip(result["provenance"]["inputs"], resolutions):
            item["resolution"] = resolution
        result["model"] = {"question": question, "images": images, "image_roles": roles, "context": context}
        result["pixel_verification"] = "verified"
        return result

    def smoke(self, resolver, variants=DEFAULT_VARIANTS, per_variant=1,
              question="Describe only what is visible in the supplied image(s)."):
        """Load all requested variants; a missing image or invalid hash raises immediately."""
        examples = self.sample(variants, per_variant)
        previews = [self.prepare(row, resolver, question) for row in examples]
        return {"complete": True, "preview_only": True, "training_ready": False,
                "samples": [{"example_id": p["provenance"]["example_id"],
                             "variant": p["provenance"]["variant"],
                             "image_count": len(p["model"]["images"]), "status": "verified"}
                            for p in previews],
                "source_snapshot": self.store.snapshot(), "previews": previews}
