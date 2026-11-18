import copy
import hashlib
import json
import math
from pathlib import Path

def canonical_json(value, *, indent=None, trailing_newline=False):
    """Serialize with the project's deterministic JSON conventions, without cleanup.

    The default is compact JSON for terminology, RAG and contract identities.
    Published S8 identities require indent=2 and trailing_newline=True. Keep
    these byte conventions stable; neither mode is an RFC 8785 implementation.
    Non-finite floats and unsupported values must still raise an error.
    """
    separators = (",", ":") if indent is None else (",", ": ")
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=indent,
                      separators=separators, allow_nan=False)
    return text + ("\n" if trailing_newline else "")


def canonical_sha256(value, *, indent=None, trailing_newline=False):
    """Hash the exact UTF-8 serialization selected by the caller."""
    return hashlib.sha256(canonical_json(
        value, indent=indent, trailing_newline=trailing_newline,
    ).encode("utf-8")).hexdigest()


def create_sector1_binding_runtime():
    """Create the same observation validator for S1 and the exported S2 import.

    No QA is generated and no abstract coverage verdict is changed. Concrete
    requests are evaluated only when evaluate/require_eligible is called.
    """
    # This structural interface is shared with the exported contract. It does
    # not replace the source-snapshot and metadata-integrity checks below.
    CORPUS_INTERFACE_SPECIFICATION = {
        "schema_version": "mmvqa-sector1-corpus-interface-v1",
        "members": {
            "_validate_recipe": {"kind": "callable", "required": True},
            "_recipe_hashes": {"kind": "dict", "required": True},
            "visual_by_artifact": {"kind": "dict", "required": True},
            "links_by_artifact": {"kind": "dict", "required": True},
            "sources_by_path": {"kind": "dict", "required": True},
            "truths_by_annotation": {"kind": "dict", "required": True},
            "truths_by_subject": {"kind": "dict", "required": True},
            "source_config": {"kind": "dict", "required": True},
            "source_snapshot": {"kind": "dict", "required": True},
            "validation_report": {"kind": "dict", "required": True},
            "truths_by_frame": {"kind": "dict", "required": False},
        },
        "recipe_validator_contract": "Validate a recipe or raise ValueError; used before comparing its verified S8 hash.",
        "scope": "Structural dependency validation only; source integrity and observation support remain separate checks.",
    }

    def _validate_corpus_interface(corpus):
        missing = object()
        kind_checks = {"dict": lambda value: isinstance(value, dict), "callable": callable}
        errors = []
        for name, spec in CORPUS_INTERFACE_SPECIFICATION["members"].items():
            value = getattr(corpus, name, missing)
            if value is missing:
                if spec["required"]:
                    errors.append(name + ":missing")
            elif not kind_checks[spec["kind"]](value):
                errors.append(name + ":expected_" + spec["kind"])
        if errors:
            raise ValueError("binding_invalid_corpus_interface: " + "; ".join(errors))

    def _binding_label(value):
        return " ".join(value.lower().split()) if isinstance(value, str) else ""


    def _binding_frame(row):
        video, frame = row.get("video_key"), row.get("frame_index")
        if (not isinstance(video, str) or not video or isinstance(frame, bool)
                or not isinstance(frame, (int, float)) or not math.isfinite(frame)
                or frame < 0 or frame != int(frame)):
            raise ValueError("binding_invalid_frame_identity")
        return video, int(frame)


    def _binding_valid_box(box, width, height):
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            return False
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
               for v in [*box, width, height]):
            return False
        return 0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height


    def _binding_source_truths(corpus, source):
        """Validate path-specific source labels before admitting pixel-equivalent aliases."""
        ids, labels = source.get("annotation_ids"), source.get("verified_labels")
        if (not isinstance(ids, list) or len(ids) != len(set(ids))
                or not isinstance(labels, list)
                or any(not isinstance(i, str) or not i for i in ids)
                or any(not _binding_label(label) for label in labels)):
            raise ValueError("binding_invalid_source_annotation_inventory")
        truths = []
        for annotation_id in ids:
            truth = corpus.truths_by_annotation.get(annotation_id)
            if (truth is None or truth.get("annotation_id") != annotation_id
                    or truth.get("image_relpath") != source.get("image_relpath")
                    or _binding_frame(truth) != _binding_frame(source)
                    or not truth.get("ground_truth_ref") or not _binding_label(truth.get("source_label"))):
                raise ValueError("binding_source_annotation_identity_mismatch")
            truths.append(truth)
        if {_binding_label(t["source_label"]) for t in truths} != {_binding_label(x) for x in labels}:
            raise ValueError("binding_source_annotation_label_mismatch")
        return truths


    def _binding_geometry(geometry, operation, panel_index, width, height, display_width, display_height):
        """Check recorded source-to-display transforms, including crop containment."""
        def pair(value, positive=False):
            return (isinstance(value, (list, tuple)) and len(value) == 2
                    and all(not isinstance(v, bool) and isinstance(v, (int, float))
                            and math.isfinite(v) and (not positive or v > 0) for v in value))
        source_geometry = geometry.get("source_geometry", geometry)
        if not isinstance(source_geometry, dict):
            raise ValueError("binding_invalid_source_geometry")
        rect = [0, 0, width, height]
        if operation == "crop":
            rect = source_geometry.get("crop_xyxy")
            translation = source_geometry.get("source_to_crop_translation")
            if (not _binding_valid_box(rect, width, height) or not pair(translation)
                    or any(abs(translation[i] + rect[i]) > 1e-6 for i in (0, 1))
                    or source_geometry.get("resized") is True):
                raise ValueError("binding_crop_geometry_missing_or_invalid")
        if panel_index is None:
            expected = [rect[2] - rect[0], rect[3] - rect[1]]
            if [display_width, display_height] != expected:
                raise ValueError("binding_standalone_dimensions_disagree_with_source_geometry")
            recorded = geometry.get("output_dimensions_wh")
            if recorded is not None and recorded != expected:
                raise ValueError("binding_recorded_output_dimensions_mismatch")
            return
        scale = geometry.get("source_to_panel_scale_xy")
        translation = geometry.get("source_to_panel_translation_xy")
        output_scale = geometry.get("source_to_output_scale_xy")
        output_translation = geometry.get("source_to_output_translation_xy")
        origin, region = geometry.get("panel_origin_xy"), geometry.get("image_region_xyxy")
        if (not pair(scale, positive=True) or not pair(output_scale, positive=True)
                or not pair(translation) or not pair(output_translation) or not pair(origin)
                or any(v < 0 for v in origin)
                or not _binding_valid_box(region, display_width, display_height)):
            raise ValueError("binding_invalid_panel_transform")
        output_region = [region[0] + origin[0], region[1] + origin[1],
                         region[2] + origin[0], region[3] + origin[1]]
        if not _binding_valid_box(output_region, display_width, display_height):
            raise ValueError("binding_panel_outside_displayed_image")
        projected = [rect[i] * scale[i % 2] + translation[i % 2] for i in range(4)]
        if (any(abs(a - b) > 1.0 for a, b in zip(projected, region))
                or any(abs(output_scale[i] - scale[i]) > 1e-9
                       or abs(output_translation[i] - translation[i] - origin[i]) > 1e-6 for i in (0, 1))):
            raise ValueError("binding_panel_transform_disagrees_with_visible_region")


    def _binding_units(corpus, recipe):
        """Resolve positive annotation support in a verified recipe; never decode pixels.

        A full frame does not require boxes. Crops support only annotations with
        valid source boxes entirely inside the recorded crop. Unknown layouts fail
        closed. Equal video/frame identities alone never establish pixel aliases.
        """
        if not isinstance(recipe, dict):
            raise ValueError("binding_recipe_must_be_an_object")
        corpus._validate_recipe(recipe)
        if corpus._recipe_hashes.get(recipe.get("example_id")) != canonical_sha256(recipe, indent=2, trailing_newline=True):
            raise ValueError("binding_recipe_differs_from_verified_S8_manifest")
        units = []
        for image_index, item in enumerate(recipe["ordered_inputs"], 1):
            artifact = corpus.visual_by_artifact[item["artifact_id"]]
            links = corpus.links_by_artifact[item["artifact_id"]]
            metadata, params = artifact.get("metadata") or {}, artifact.get("params") or {}
            if not isinstance(metadata, dict) or not isinstance(params, dict):
                raise ValueError("binding_invalid_visual_metadata")
            panels, specs = metadata.get("panels"), []
            if panels:
                if not isinstance(panels, list) or any(not isinstance(p, dict) for p in panels):
                    raise ValueError("binding_invalid_panel_inventory")
                for panel_index, panel in enumerate(panels):
                    matches = [link for link in links if link.get("image_relpath") == panel.get("image_relpath")]
                    if len(matches) > 1:
                        matches = [link for link in matches if link.get("panel_index") == panel_index]
                    if len(matches) != 1:
                        raise ValueError("binding_ambiguous_panel_source")
                    specs.append((matches[0], panel_index, panel.get("operation"), panel))
            elif artifact.get("kind") in {"full_frame", "roi_crop", "bbox_overlay"} and len(links) == 1:
                operation = {"full_frame": "full", "roi_crop": "crop", "bbox_overlay": "overlay"}[artifact["kind"]]
                specs.append((links[0], None, operation, metadata))
            else:
                raise ValueError("binding_missing_composite_panel_geometry")
            for link, panel_index, operation, geometry in specs:
                if operation not in {"full", "crop", "overlay"}:
                    raise ValueError("binding_unknown_visual_operation")
                source = corpus.sources_by_path[link["image_relpath"]]
                frame = _binding_frame(source)
                source_hash = source.get("sha256")
                if (not isinstance(source_hash, str) or len(source_hash) != 64
                        or any(c not in "0123456789abcdef" for c in source_hash)
                        or link.get("source_sha256") != source_hash or _binding_frame(link) != frame):
                    raise ValueError("binding_source_link_identity_mismatch")
                if panel_index is not None:
                    if (geometry.get("source_sha256") != source_hash
                            or _binding_frame(geometry) != frame
                            or geometry.get("role") != link.get("role")):
                        raise ValueError("binding_panel_source_identity_mismatch")
                own_truths = _binding_source_truths(corpus, source)
                if not set(link.get("annotation_ids") or []).issubset(source["annotation_ids"]):
                    raise ValueError("binding_link_annotations_not_in_source")
                by_id = {t["annotation_id"]: t for t in own_truths}
                for truth in getattr(corpus, "truths_by_frame", {}).get(frame, []):
                    alias = corpus.sources_by_path.get(truth.get("image_relpath"))
                    if (alias is None or alias.get("sha256") != source_hash
                            or _binding_frame(alias) != frame):
                        continue
                    for candidate in _binding_source_truths(corpus, alias):
                        by_id[candidate["annotation_id"]] = candidate
                width, height = source.get("width"), source.get("height")
                source_annotation_ids = sorted(by_id)
                source_geometry = geometry.get("source_geometry", geometry)
                if not isinstance(source_geometry, dict):
                    raise ValueError("binding_invalid_source_geometry")
                _binding_geometry(geometry, operation, panel_index, width, height, item["width"], item["height"])
                if operation == "crop":
                    crop = source_geometry.get("crop_xyxy")
                    if not _binding_valid_box(crop, width, height):
                        raise ValueError("binding_crop_geometry_missing_or_invalid")
                    by_id = {aid: truth for aid, truth in by_id.items()
                             if _binding_valid_box(truth.get("bbox_xyxy"), width, height)
                             and crop[0] <= truth["bbox_xyxy"][0] < truth["bbox_xyxy"][2] <= crop[2]
                             and crop[1] <= truth["bbox_xyxy"][1] < truth["bbox_xyxy"][3] <= crop[3]}
                image_ref = f"image_{image_index}"
                units.append({
                    "unit_id": image_ref if panel_index is None else f"{image_ref}/panel_{panel_index + 1}",
                    "input_id": item["input_id"], "artifact_id": item["artifact_id"],
                    "image_relpath": source["image_relpath"], "video_key": frame[0], "frame_index": frame[1],
                    "source_sha256": source_hash, "operation": operation, "panel_index": panel_index,
                    "link_role": link.get("role"), "annotation_ids": sorted(by_id),
                    "source_annotation_ids": source_annotation_ids,
                    "geometry": geometry, "source_width": width, "source_height": height,
                    "display_width": item["width"], "display_height": item["height"],
                    "is_original": item.get("storage") == "original",
                    "annotation_assisted": artifact.get("annotation_assisted") is True,
                    "reference_annotation_verified": params.get("reference_annotation_verified") is True,
                })
        if not units or len({u["unit_id"] for u in units}) != len(units):
            raise ValueError("binding_missing_or_ambiguous_visual_units")
        return units


    def _binding_index(rows, keys, description):
        result = {}
        for row in rows:
            key = tuple(row.get(k) for k in keys)
            if any(value is None or value == "" for value in key) or key in result:
                raise ValueError("binding_missing_or_duplicate_" + description)
            result[key] = row
        return result


    def _binding_snapshot_matches(left, right):
        keys = ("repository", "resolved_commit", "source_catalog_sha256", "processing_signature")
        return (all(isinstance(left.get(k), str) and left[k] for k in keys[:3])
                and len(left["resolved_commit"]) == 40 and len(left["source_catalog_sha256"]) == 64
                and all(c in "0123456789abcdef" for c in left["resolved_commit"] + left["source_catalog_sha256"])
                and all(left.get(k) == right.get(k) for k in keys))


    # Serializable dispatch policies; names are resolved only by the local
    # validator registry below, never by interpreting a slot/task identifier.
    SLOT_VALIDATORS = {
        "observed_finding": {"validator": "visible_annotations"},
        "observed_finding_set": {"validator": "visible_annotations"},
        "indicated_region_finding": {"validator": "original_region"},
        "normal_reference_finding": {"validator": "normal_reference", "normal_context": True},
        "same_finding_positive_anchors": {"validator": "same_video_anchors"},
        "positive_target_finding": {"validator": "relative_target"},
    }
    DEFAULT_SLOT_VALIDATOR = {"validator": "unresolved"}
    STATE_VALIDATORS = {
        "label_without_box": {"all": ["annotations_without_boxes", "category_available"]},
        "label_with_box": {"all": ["valid_boxes", "category_available"]},
        "multiple_labels": {"all": ["multiple_labels_in_selected_frame"]},
        "missing_category": {"any": ["annotation_category_missing", "category_unavailable"]},
        "unlabelled_neighbor": {"all": ["unlabelled_reference"]},
        "explicit_normal_reference": {"all": ["normal_reference_valid"], "normal_context": True},
        "multiple_verified_anchors": {"all": ["two_anchors"]},
        "annotation_assisted_input": {"all": ["recipe_assisted", "selected_unit_assisted"]},
        "unmapped_or_conflicting_reference": {"all": ["never"]},
    }
    DEFAULT_STATE_VALIDATOR = {"all": ["never"]}
    TASK_GUARDS = {
        "finding_category": [{"all": ["category_available"],
            "reason_code": "unambiguous_original_dataset_category_required"}],
        "evidence_frame_localization": [{"all": ["multiple_displayed_frames"],
            "reason_code": "multiple_displayed_physical_frames_required"}],
        "optional_spatial_localization": [{"all": ["original_full_variant", "unassisted_recipe"],
            "reason_code": "spatial_task_requires_unassisted_original_full_frame"}],
    }

    def _binding_check(spec, facts):
        """Evaluate an explicit all/any condition over verified context facts."""
        operators = [key for key in ("all", "any") if key in spec]
        if len(operators) != 1:
            raise ValueError("binding_invalid_declarative_condition")
        operator = operators[0]
        values = [facts[name] for name in spec[operator]]
        return all(values) if operator == "all" else any(values)

    def _binding_facts(context, labels_per_frame):
        selected, units = context["selected"], context["units"]
        reference, recipe = context["reference"], context["recipe"]
        category_available = context["category_available"]
        return {
            "annotations_without_boxes": all(not t.get("bbox_xyxy") for t in selected),
            "valid_boxes": all(context["valid_boxes"]),
            "category_available": category_available,
            "category_unavailable": not category_available,
            "multiple_labels_in_selected_frame": any(
                len(labels_per_frame.get(frame, set())) > 1 for frame in context["frames"]),
            "annotation_category_missing": any(not t.get("clinical_group") for t in selected),
            "unlabelled_reference": reference is not None and not reference["source_annotation_ids"],
            "normal_reference_valid": context["normal_reason"] is None,
            "two_anchors": context["two_anchors"],
            "recipe_assisted": recipe.get("annotation_assisted") is True,
            "selected_unit_assisted": any(u["annotation_assisted"] for u in context["selected_units"]),
            "multiple_displayed_frames": len({_binding_frame(u) for u in units}) >= 2,
            "original_full_variant": recipe["variant"] == "full",
            "unassisted_recipe": not recipe.get("annotation_assisted"),
            "never": False,
        }

    def _validate_visible_annotations(context):
        # Each selected member already has its own verified, visible annotation.
        return [], None

    def _validate_original_region(context):
        reasons = []
        if not all(context["valid_boxes"]):
            reasons.append("valid_same_class_source_geometry_required")
        # Spatial task coverage currently permits only an original full frame.
        if not all(any(u["is_original"] and u["operation"] == "full" and u["panel_index"] is None
                       and aid in u["annotation_ids"] and u["source_width"] == u["display_width"]
                       and u["source_height"] == u["display_height"] for u in context["selected_units"])
                   for aid in context["annotation_ids"]):
            reasons.append("displayed_original_coordinate_geometry_required")
        return reasons, None

    def _validate_normal_reference(context):
        if context["normal_reason"]:
            return [context["normal_reason"]], None
        if not all(any(aid in u["annotation_ids"] for u in context["references"])
                   for aid in context["annotation_ids"]):
            return ["class_not_in_explicit_verified_normal_reference"], None
        return [], None

    def _validate_same_video_anchors(context):
        reasons = ([] if context["two_anchors"] else
                   ["two_distinct_same_finding_same_video_positive_anchors_required"])
        return reasons, None

    def _validate_relative_target(context):
        reference, direction = context["reference"], context["direction"]
        if reference is None or direction not in {"before", "after"}:
            return ["explicit_reference_unit_and_direction_required"], None
        reference_evidence = {k: reference[k] for k in
            ("unit_id", "input_id", "artifact_id", "image_relpath", "source_sha256", "video_key", "frame_index", "panel_index")}
        reference_evidence["direction"] = direction
        rv, rf = _binding_frame(reference)
        reasons = []
        if any(v != rv or not (f < rf if direction == "before" else f > rf) for v, f in context["frames"]):
            reasons.append("target_reference_temporal_relation_not_supported")
        return reasons, reference_evidence

    def _validate_unresolved_slot(context):
        return ["role_requires_resolved_reference_or_explicit_implementation"], None

    SLOT_VALIDATOR_FUNCTIONS = {
        "visible_annotations": _validate_visible_annotations,
        "original_region": _validate_original_region,
        "normal_reference": _validate_normal_reference,
        "same_video_anchors": _validate_same_video_anchors,
        "relative_target": _validate_relative_target,
        "unresolved": _validate_unresolved_slot,
    }

    class Sector1BindingContract:
        """Check one requested role against an integrity-checked S8 Corpus.

        Mapping approval, abstract applicability and observation support are separate.
        A passing result authorizes this task-input binding only: it does not generate
        QA, certify clinical truth, or authorize arbitrary terminology substitution.
        """

        @staticmethod
        def corpus_interface_specification():
            """Return an independent, JSON-serializable copy of the required interface."""
            return copy.deepcopy(CORPUS_INTERFACE_SPECIFICATION)

        def __init__(self, bindings, mappings, coverage, observations, classes, snapshot, corpus):
            _validate_corpus_interface(corpus)
            if not _binding_snapshot_matches(snapshot, corpus.source_snapshot):
                raise ValueError("binding_Phase2_source_snapshot_mismatch")
            if corpus.validation_report.get("status") != "complete_metadata_verified":
                raise ValueError("binding_requires_verified_Phase2_metadata")
            self.corpus, self.snapshot = corpus, copy.deepcopy(snapshot)
            self.bindings = _binding_index(copy.deepcopy(bindings), ("binding_id",), "binding")
            # Source-required placeholder rows can legitimately have no mapping ID.
            self.mappings = _binding_index([copy.deepcopy(r) for r in mappings if r.get("mapping_id")],
                                          ("mapping_id",), "mapping")
            self.coverage = _binding_index(copy.deepcopy(coverage), ("task", "variant", "evidence_state"), "coverage")
            self.observations = _binding_index(copy.deepcopy(observations), ("annotation_id",), "observation")
            self.classes = _binding_index(copy.deepcopy(classes), ("dataset_class_id",), "class")
            for observation in self.observations.values():
                truth = corpus.truths_by_annotation.get(observation["annotation_id"])
                if (truth is None or observation.get("source_commit") != snapshot["resolved_commit"]
                        or any(observation.get(k) != truth.get(k) for k in
                               ("subject_id", "annotation_id", "ground_truth_ref", "image_relpath", "video_key", "frame_index"))
                        or observation.get("normalized_label") != _binding_label(truth.get("source_label"))
                        or observation.get("dataset_category") != (_binding_label(truth.get("clinical_group")) or None)):
                    raise ValueError("binding_observation_differs_from_Phase2_source")
                cls = self.classes.get((observation.get("dataset_class_id"),))
                if cls is None or cls.get("normalized_label") != observation["normalized_label"]:
                    raise ValueError("binding_observation_class_identity_mismatch")

        def _normal_reference(self, recipe, units):
            subject = self.corpus.truths_by_subject[recipe["subject_id"]]
            reference_path = subject.get("comparison_reference_image_relpath")
            raw_labels = subject.get("comparison_reference_verified_labels")
            if recipe.get("variant") != "comparison" or not isinstance(raw_labels, list):
                return [], "verified_normal_reference_contract_missing"
            labels = {_binding_label(label) for label in raw_labels}
            if not labels or "" in labels:
                return [], "verified_normal_reference_labels_missing"
            configured = self.corpus.source_config.get("normal_reference_labels")
            if "normal_reference_labels" in self.corpus.source_config and (
                    not isinstance(configured, list) or not configured
                    or any(not _binding_label(label) for label in configured)
                    or not labels.issubset({_binding_label(label) for label in configured})):
                return [], "verified_normal_reference_configuration_conflict"
            def unit_labels(unit):
                return {_binding_label(self.corpus.truths_by_annotation[aid]["source_label"])
                        for aid in unit["annotation_ids"]}
            references = [u for u in units if u["link_role"] == "comparison_reference"
                          and u["image_relpath"] == reference_path and u["reference_annotation_verified"]
                          and unit_labels(u) == labels]
            targets = [u for u in units if u["link_role"] == "target"
                       and subject["annotation_id"] in u["annotation_ids"]]
            if not references or not targets:
                return [], "verified_normal_reference_pair_or_panel_missing"
            if any(_binding_frame(r)[0] != _binding_frame(t)[0] or _binding_frame(r) == _binding_frame(t)
                   for r in references for t in targets):
                return [], "normal_reference_requires_distinct_same_video_frames"
            if any(unit_labels(t) & labels for t in targets):
                return [], "normal_reference_target_label_overlap"
            return references, None

        def evaluate(self, recipe, *, binding_id, task, evidence_state, annotation_ids,
                     reference_unit_id=None, direction=None):
            """Return a reproducible verdict; caller supplies observations, never verdicts.

            annotation_ids are the positive operands for this one class/role. The
            reference unit of a relative request may have no positive annotation.
            All selected operands must be supported by the actual displayed context.
            """
            binding = self.bindings.get((binding_id,))
            if binding is None:
                raise ValueError("binding_unknown_binding_id")
            result = {
                "binding_id": binding_id, "example_id": recipe.get("example_id") if isinstance(recipe, dict) else None,
                "task": task, "evidence_state": evidence_state, "clinical_slot": binding["clinical_slot"],
                "mapping_id": binding.get("mapping_id"), "mapping_status": binding.get("mapping_status"),
                "mapping_relation": binding.get("relation"), "mapping_approved": False,
                "task_applicability": "not_evaluated", "evidence_state_verified": False,
                "clinical_instance_binding_status": "not_evaluated_requires_observation_binding",
                "eligible_for_task_input": False, "reason_codes": [], "observation_evidence": [],
                "reference_evidence": None, "context_evidence": [], "source_snapshot": copy.deepcopy(self.snapshot),
                "clinical_truth_established_by_terminology": False, "qa_generated": False,
            }
            reasons = result["reason_codes"]
            mapping = self.mappings.get((binding.get("mapping_id"),))
            identity = ("dataset_class_id", "normalized_label", "terminology", "target_id", "relation", "source_sha256", "mapping_status", "required_attribute_constraints_json")
            result["mapping_approved"] = bool(mapping and mapping.get("mapping_status") == "approved"
                                              and binding.get("source_status") == "imported"
                                              and all(mapping.get(k) == binding.get(k) for k in identity))
            if not result["mapping_approved"]:
                reasons.append("mapping_not_approved_or_identity_mismatch")
            try:
                units = _binding_units(self.corpus, recipe)
            except (ValueError, KeyError, TypeError) as exc:
                reasons.append("invalid_recipe_evidence:" + str(exc))
                result["clinical_instance_binding_status"] = "observation_binding_rejected"
                return result
            policy = self.coverage.get((task, recipe["variant"], evidence_state))
            if policy is None:
                reasons.append("unknown_task_variant_evidence_combination")
            else:
                result["task_applicability"] = policy["expected_status"]
                if policy["expected_status"] != "applicable":
                    reasons.append("task_policy_" + str(policy["expected_status"]))
            # Resolve the requested role through the existing task/evidence binding
            # references, rather than allowing any generic variant/hint binding.
            policy_bindings = json.loads(policy.get("terminology_binding_ids_json", "[]")) if policy else []
            role_allowed = any(
                self.bindings.get((bid,), {}).get("clinical_slot") == binding["clinical_slot"]
                and self.bindings.get((bid,), {}).get("dataset_class_id") == binding["dataset_class_id"]
                and self.bindings.get((bid,), {}).get("mapping_id") == binding.get("mapping_id")
                for bid in policy_bindings)
            if not role_allowed:
                reasons.append("role_not_authorized_by_task_evidence_contract")
            if (not isinstance(annotation_ids, (list, tuple)) or not annotation_ids
                    or any(not isinstance(aid, str) or not aid for aid in annotation_ids)
                    or len(annotation_ids) != len(set(annotation_ids))):
                reasons.append("positive_annotation_ids_required")
                result["clinical_instance_binding_status"] = "observation_binding_rejected"
                return result
            selected, selected_units = [], []
            for aid in annotation_ids:
                observation = self.observations.get((aid,))
                supported = [u for u in units if aid in u["annotation_ids"]]
                if (observation is None or observation["dataset_class_id"] != binding["dataset_class_id"]
                        or observation["normalized_label"] != binding["normalized_label"] or not supported):
                    reasons.append("annotation_not_supported_for_class_in_displayed_context:" + aid)
                    continue
                selected.append(self.corpus.truths_by_annotation[aid])
                selected_units.extend(supported)
                for unit in supported:
                    result["observation_evidence"].append({
                        "annotation_id": aid, "ground_truth_ref": observation["ground_truth_ref"],
                        "annotation_image_relpath": observation["image_relpath"],
                        **{k: unit[k] for k in ("unit_id", "input_id", "artifact_id", "image_relpath", "source_sha256", "video_key", "frame_index", "panel_index")},
                    })
            if len(selected) != len(annotation_ids):
                result["clinical_instance_binding_status"] = "observation_binding_rejected"
                return result
            selected_units = list({u["unit_id"]: u for u in selected_units}.values())
            frames = {_binding_frame(t) for t in selected}
            two_anchors = len(frames) >= 2 and len({v for v, _ in frames}) == 1
            valid_boxes = [
                _binding_valid_box(t.get("bbox_xyxy"), self.corpus.sources_by_path[t["image_relpath"]]["width"],
                                   self.corpus.sources_by_path[t["image_relpath"]]["height"])
                for t in selected]
            cls = self.classes[(binding["dataset_class_id"],)]
            category_available = cls.get("dataset_category_status") == "consistent" and all(t.get("clinical_group") for t in selected)
            references, normal_reason = self._normal_reference(recipe, units)
            slot_policy = SLOT_VALIDATORS.get(binding["clinical_slot"], DEFAULT_SLOT_VALIDATOR)
            state_policy = STATE_VALIDATORS.get(evidence_state, DEFAULT_STATE_VALIDATOR)
            if normal_reason is None and (slot_policy.get("normal_context") or state_policy.get("normal_context")):
                reference_ids = {u["unit_id"] for u in references}
                subject_id = self.corpus.truths_by_subject[recipe["subject_id"]]["annotation_id"]
                for unit in units:
                    if unit["unit_id"] in reference_ids or (unit["link_role"] == "target" and subject_id in unit["annotation_ids"]):
                        result["context_evidence"].append({
                            **{k: unit[k] for k in ("unit_id", "input_id", "artifact_id", "image_relpath", "source_sha256", "video_key", "frame_index", "panel_index", "link_role")},
                            "annotation_ids": list(unit["annotation_ids"]),
                            "ground_truth_refs": [self.corpus.truths_by_annotation[aid]["ground_truth_ref"] for aid in unit["annotation_ids"]],
                        })
            reference = next((u for u in units if u["unit_id"] == reference_unit_id), None)
            context = {
                "recipe": recipe, "selected": selected, "units": units, "selected_units": selected_units,
                "annotation_ids": annotation_ids, "frames": frames, "valid_boxes": valid_boxes,
                "category_available": category_available, "references": references, "normal_reason": normal_reason,
                "reference": reference, "direction": direction, "two_anchors": two_anchors,
            }
            role_reasons, result["reference_evidence"] = SLOT_VALIDATOR_FUNCTIONS[slot_policy["validator"]](context)
            labels_per_frame = {}
            for unit in units:
                labels_per_frame.setdefault(_binding_frame(unit), set()).update(
                    _binding_label(self.corpus.truths_by_annotation[aid]["source_label"]) for aid in unit["annotation_ids"])
            facts = _binding_facts(context, labels_per_frame)
            result["evidence_state_verified"] = _binding_check(state_policy, facts)
            if not result["evidence_state_verified"]:
                reasons.append("requested_evidence_state_not_supported")
            for guard in TASK_GUARDS.get(task, []):
                if not _binding_check(guard, facts):
                    role_reasons.append(guard["reason_code"])
            reasons.extend(role_reasons)
            result["clinical_instance_binding_status"] = (
                "verified_observation_binding" if not role_reasons and result["evidence_state_verified"] and role_allowed
                else "observation_binding_rejected")
            # Reviewed source-label qualifications stay part of the approved mapping.
            # They are not new, independently observed image attributes. Identity
            # checks above require the binding to retain the exact mapping constraints.
            result["reason_codes"] = sorted(set(reasons))
            result["eligible_for_task_input"] = (result["mapping_approved"] and result["task_applicability"] == "applicable"
                                                 and result["clinical_instance_binding_status"] == "verified_observation_binding"
                                                 and not result["reason_codes"])
            return result

        def require_eligible(self, recipe, **request):
            verdict = self.evaluate(recipe, **request)
            if not verdict["eligible_for_task_input"]:
                raise ValueError("sector2_binding_rejected: " + "; ".join(verdict["reason_codes"]))
            return verdict


    def load_sector1_binding_contract(directory, corpus, *, expected_manifest_sha256):
        """Import an exported run against a trusted manifest hash and the same S8 source.

        Obtain expected_manifest_sha256 from the run's trusted publication/storage
        record, not by hashing the untrusted manifest being checked. The caller must
        also verify the validator module's manifest entry before importing its code.
        """
        root = Path(directory).resolve()
        payload = (root / "files_manifest.json").read_bytes()
        if (not isinstance(expected_manifest_sha256, str) or len(expected_manifest_sha256) != 64
                or hashlib.sha256(payload).hexdigest() != expected_manifest_sha256):
            raise ValueError("sector2_manifest_SHA256_mismatch")
        records = json.loads(payload)
        if not isinstance(records, list):
            raise ValueError("sector2_invalid_files_manifest")
        entries = _binding_index(records, ("relative_path",), "manifest_path")
        for (relative,), record in entries.items():
            path = Path(relative)
            resolved = (root / path).resolve()
            if path.is_absolute() or ".." in path.parts or not resolved.is_relative_to(root):
                raise ValueError("sector2_manifest_path_outside_bundle")
        def read_verified(name):
            record = entries.get((name,))
            if record is None:
                raise ValueError("sector2_required_artifact_missing:" + name)
            raw = (root / name).read_bytes()
            if len(raw) != record.get("bytes") or hashlib.sha256(raw).hexdigest() != record.get("sha256"):
                raise ValueError("sector2_artifact_SHA256_mismatch:" + name)
            return json.loads(raw)
        interface = read_verified("sector2_binding_contract.json")
        if interface.get("schema_version") != "mmvqa-sector2-binding-contract-v1":
            raise ValueError("sector2_unsupported_binding_contract")
        return Sector1BindingContract(
            read_verified("contract_terminology_bindings.json"), read_verified("class_crosswalk.json"),
            read_verified("coverage_contract.json"), read_verified("phase2_annotation_provenance.json"),
            read_verified("phase2_classes.json"), read_verified("phase2_source_snapshot.json"), corpus)
    return Sector1BindingContract, load_sector1_binding_contract


Sector1BindingContract, load_sector1_binding_contract = create_sector1_binding_runtime()
