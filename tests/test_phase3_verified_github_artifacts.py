"""Offline regression checks for the actual Phase 3 notebook loaders.

Run with: python -m unittest discover -s tests -p 'test_phase3_verified_github_artifacts.py'
Only loader definitions/constants are compiled; notebook execution, Colab and
its third-party packages are not needed. All downloads use an in-memory server.
"""
import ast
from datetime import date
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import tempfile
import unittest
from unittest.mock import Mock, patch
from urllib.parse import quote


NOTEBOOK = (Path(__file__).resolve().parents[1] / "notebooks" / "phase3"
            / "phase3_01_terminology_mapping.ipynb")


def notebook_loader_namespace():
    namespace = {name: value for name, value in globals().items()
                 if name in {"date", "hashlib", "json", "os", "Path", "PurePosixPath",
                             "re", "tempfile", "quote"}}
    wanted_constants = {
        "RAG_GITHUB_MANIFEST_SCHEMA", "RAG_GITHUB_BUNDLE_SCHEMA",
        "GI_GITHUB_MANIFEST_SCHEMA", "GI_GITHUB_PACKAGE_SCHEMA",
    }
    definitions = []
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        for node in ast.parse("".join(cell["source"])).body:
            if isinstance(node, ast.FunctionDef) and (
                node.name.startswith(("_rag_github_", "_gi_github_"))
                or node.name in {"load_github_rag_evidence", "load_github_gi_supplement",
                                 "load_github_verified_artifact"}
            ):
                definitions.append(node)
            elif isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id in wanted_constants
                for target in node.targets
            ):
                definitions.append(node)
    module = ast.fix_missing_locations(ast.Module(body=definitions, type_ignores=[]))
    exec(compile(module, str(NOTEBOOK), "exec"), namespace)
    return namespace


class ArtifactFixture:
    def __init__(self, kind, root):
        self.kind, self.root = kind, root / kind
        self.repository, self.ref = "example/project", "main"
        self.manifest_path = f"outputs/phase3/sector1/{kind}_manifest.json"
        self.artifact_path = f"outputs/phase3/sector1/{kind}_package.json"
        self.content_hash_key = "bundle_content_sha256" if kind == "rag" else "package_content_sha256"
        self.calls = []
        if kind == "rag":
            self.package = {
                "schema_version": "mmvqa-rag-evidence-v1", "review_date": "2026-11-13",
                "sources": [], "passages": [], "profiles": [], "target_decisions": [],
            }
        else:
            self.package = {
                "schema_version": "mmvqa-gi-supplement-v1",
                **{name: "synthetic test source" for name in (
                    "terminology", "version", "source_url", "doi", "source_locator",
                    "citation", "scope", "patient_attribute_policy")},
                "terms": [{"term_id": term_id, **{name: "fixture" for name in (
                    "preferred_term", "source_label", "term_type", "section", "source_locator")}}
                    for term_id in ("fixture:a", "fixture:b")],
                "synonyms": [{"term_id": "fixture:a", "synonym": "fixture synonym", "language": "en",
                              "relation": "source_attested_equivalent_name", "source_locator": "fixture",
                              "evidence": "synthetic source evidence"}],
                "not_synonyms": [["fixture:a", "fixture:b"]], "excluded_terms": [],
            }
        self.refresh()

    def refresh(self):
        self.payload = (json.dumps(self.package, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
        canonical = json.dumps(self.package, ensure_ascii=False, sort_keys=True,
                               separators=(",", ":"), allow_nan=False).encode("utf-8")
        self.manifest = {
            "schema_version": f"mmvqa-{self.kind}-github-manifest-v1",
            "artifact": {"path": self.artifact_path, "bytes": len(self.payload),
                         "sha256": hashlib.sha256(self.payload).hexdigest(),
                         "schema_version": self.package["schema_version"]},
            self.content_hash_key: hashlib.sha256(canonical).hexdigest(),
        }
        if self.kind == "rag":
            self.manifest.update(bundle_review_date=self.package["review_date"],
                                 phase2_source_commit_at_review="a" * 40)

    def reference_payload_bytes(self):
        self.manifest["artifact"].update(bytes=len(self.payload), sha256=hashlib.sha256(self.payload).hexdigest())

    def fetch(self, url, destination):
        self.calls.append(url)
        if url.endswith(self.manifest_path):
            destination.write_bytes(json.dumps(self.manifest).encode("utf-8"))
        elif url.endswith(self.artifact_path):
            destination.write_bytes(self.payload)
        else:
            raise AssertionError("Unexpected URL: " + url)

    def load(self, namespace, fetch=None, **overrides):
        args = dict(repository=self.repository, ref=self.ref, manifest_path=self.manifest_path,
                    cache_root=self.root, fetch=fetch or self.fetch, attempts=3)
        args.update(overrides)
        wrapper = "load_github_rag_evidence" if self.kind == "rag" else "load_github_gi_supplement"
        return namespace[wrapper](**args)


class VerifiedGitHubArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.namespace = notebook_loader_namespace()

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def fixture(self, kind):
        return ArtifactFixture(kind, self.root)

    def assert_no_partials(self, fixture):
        self.assertFalse(list(fixture.root.rglob("*.partial")))

    def test_both_public_wrappers_use_shared_engine_and_keep_provenance(self):
        engine = self.namespace["load_github_verified_artifact"]
        with patch.dict(self.namespace, {"load_github_verified_artifact": Mock(wraps=engine)}):
            for kind in ("rag", "gi"):
                with self.subTest(kind=kind):
                    fixture = self.fixture(kind)
                    package, source = fixture.load(self.namespace)
                    self.assertEqual(package, fixture.package)
                    self.assertEqual(source["schema_version"], f"mmvqa-{kind}-github-source-v1")
                    self.assertEqual(source[fixture.content_hash_key], fixture.manifest[fixture.content_hash_key])
                    self.assertEqual(source["source_sha256"], source["vm_sha256"])
                    self.assertEqual(Path(source["local_path"]).read_bytes(), fixture.payload)
                    self.assertEqual(json.loads(Path(source["manifest_local_path"]).read_bytes()), fixture.manifest)
                    self.assertEqual((source["manifest_downloads"], source["artifact_downloads"]), (2, 1))
                    self.assertEqual(source["repository_ref"], "main")
                    self.assertEqual(source["attempt_log"][0]["status"], "verified")
                    if kind == "rag":
                        self.assertEqual(source["bundle_review_date"], "2026-11-13")
                        self.assertEqual(source["phase2_source_commit_at_review"], "a" * 40)
                    else:
                        self.assertNotIn("bundle_review_date", source)
                        self.assertIn("gi", Path(source["local_path"]).relative_to(fixture.root).parts)
                    self.assert_no_partials(fixture)
            self.assertEqual(self.namespace["load_github_verified_artifact"].call_count, 2)

    def test_cache_reuse_requires_two_fresh_manifest_reads(self):
        for kind in ("rag", "gi"):
            with self.subTest(kind=kind):
                fixture = self.fixture(kind)
                _, original = fixture.load(self.namespace)
                fixture.calls.clear()
                _, cached = fixture.load(self.namespace)
                self.assertEqual(cached["local_path"], original["local_path"])
                self.assertEqual(cached["cache_status"], "reused_verified")
                self.assertEqual((cached["manifest_downloads"], cached["artifact_downloads"]), (2, 0))
                self.assertEqual(len(fixture.calls), 2)
                self.assertTrue(all(url.endswith(fixture.manifest_path) for url in fixture.calls))

    def test_corrupt_cache_is_replaced_by_verified_download(self):
        for kind in ("rag", "gi"):
            with self.subTest(kind=kind):
                fixture = self.fixture(kind)
                _, original = fixture.load(self.namespace)
                Path(original["local_path"]).write_bytes(b"damaged cache")
                _, repaired = fixture.load(self.namespace)
                self.assertEqual(repaired["artifact_downloads"], 1)
                self.assertEqual(repaired["cache_status"], "downloaded_and_verified")
                self.assertEqual(Path(repaired["local_path"]).read_bytes(), fixture.payload)
                self.assert_no_partials(fixture)

    def test_transfer_failure_retries_without_redefining_manifest(self):
        for kind in ("rag", "gi"):
            with self.subTest(kind=kind):
                fixture = self.fixture(kind)
                downloads = []
                def interrupted_once(url, destination):
                    if url.endswith(fixture.artifact_path):
                        downloads.append(url)
                        if len(downloads) == 1:
                            destination.write_bytes(b"partial transfer")
                            raise OSError("synthetic interruption")
                    fixture.fetch(url, destination)
                _, source = fixture.load(self.namespace, fetch=interrupted_once)
                self.assertEqual((source["attempts_used"], source["manifest_downloads"], source["artifact_downloads"]), (2, 3, 2))
                self.assertEqual([row["status"] for row in source["attempt_log"]],
                                 ["download_or_verification_failed", "verified"])
                self.assertEqual(source["attempt_log"][0]["manifest_sha256"], source["attempt_log"][1]["manifest_sha256"])
                self.assert_no_partials(fixture)

    def test_manifest_update_during_artifact_download_restarts_pair(self):
        for kind in ("rag", "gi"):
            with self.subTest(kind=kind):
                fixture = self.fixture(kind)
                reads = []
                old_payload = fixture.payload
                def update_on_confirmation(url, destination):
                    if url.endswith(fixture.manifest_path):
                        reads.append(url)
                        if len(reads) == 2:
                            fixture.package["publication_revision"] = "second"
                            fixture.refresh()
                    fixture.fetch(url, destination)
                package, source = fixture.load(self.namespace, fetch=update_on_confirmation)
                self.assertEqual(package["publication_revision"], "second")
                self.assertEqual((source["attempts_used"], source["manifest_downloads"], source["artifact_downloads"]), (2, 4, 2))
                self.assertEqual(source["attempt_log"][0]["status"], "manifest_changed_retry")
                self.assertEqual(Path(source["local_path"]).read_bytes(), fixture.payload)
                self.assertNotEqual(fixture.payload, old_payload)
                self.assertEqual(len(list(fixture.root.rglob(Path(fixture.artifact_path).name))), 1)
                self.assert_no_partials(fixture)

    def test_continuously_changing_manifest_exhausts_bounded_retry(self):
        for kind in ("rag", "gi"):
            with self.subTest(kind=kind):
                fixture = self.fixture(kind)
                reads = []
                def unstable(url, destination):
                    if url.endswith(fixture.manifest_path):
                        reads.append(url)
                        fixture.manifest["publication_revision"] = len(reads)
                    fixture.fetch(url, destination)
                with self.assertRaisesRegex(RuntimeError, "after 3 attempts"):
                    fixture.load(self.namespace, fetch=unstable)
                self.assertEqual(len(reads), 6)
                self.assertFalse(list(fixture.root.rglob(Path(fixture.artifact_path).name)))
                self.assert_no_partials(fixture)

    def test_unavailable_manifest_does_not_fall_back_to_verified_cache(self):
        for kind in ("rag", "gi"):
            with self.subTest(kind=kind):
                fixture = self.fixture(kind)
                _, source = fixture.load(self.namespace)
                attempted = []
                def unavailable(url, destination):
                    attempted.append(url)
                    raise OSError("synthetic unavailable server")
                with self.assertRaisesRegex(RuntimeError, "after 3 attempts"):
                    fixture.load(self.namespace, fetch=unavailable)
                self.assertEqual(len(attempted), 3)
                self.assertEqual(Path(source["local_path"]).read_bytes(), fixture.payload)
                self.assert_no_partials(fixture)

    def test_invalid_new_publication_preserves_old_verified_artifact(self):
        for kind in ("rag", "gi"):
            with self.subTest(kind=kind):
                fixture = self.fixture(kind)
                _, source = fixture.load(self.namespace)
                old = fixture.payload
                fixture.package["publication_revision"] = "second"
                fixture.refresh()
                fixture.payload = old  # Server serves bytes incompatible with its new manifest.
                with self.assertRaisesRegex(RuntimeError, "after 3 attempts"):
                    fixture.load(self.namespace)
                self.assertEqual(Path(source["local_path"]).read_bytes(), old)
                self.assertEqual(len(list(fixture.root.rglob(Path(fixture.artifact_path).name))), 1)
                self.assert_no_partials(fixture)

    def test_schema_byte_hash_and_canonical_hash_failures_are_not_cached(self):
        for kind in ("rag", "gi"):
            for failure in ("manifest_schema", "artifact_schema", "bytes", "sha256", "canonical"):
                with self.subTest(kind=kind, failure=failure):
                    fixture = ArtifactFixture(kind, self.root / failure)
                    if failure == "manifest_schema":
                        fixture.manifest["schema_version"] = "unsupported"
                    elif failure == "artifact_schema":
                        fixture.package["schema_version"] = "unsupported"
                        fixture.payload = json.dumps(fixture.package).encode()
                        fixture.reference_payload_bytes()
                    elif failure == "bytes":
                        fixture.manifest["artifact"]["bytes"] += 1
                    elif failure == "sha256":
                        fixture.manifest["artifact"]["sha256"] = "0" * 64
                    else:
                        fixture.manifest[fixture.content_hash_key] = "0" * 64
                    with self.assertRaisesRegex(RuntimeError, "after 2 attempts"):
                        fixture.load(self.namespace, attempts=2)
                    self.assertFalse(list(fixture.root.rglob(Path(fixture.artifact_path).name)))
                    self.assert_no_partials(fixture)

    def test_duplicate_keys_and_nonfinite_json_are_rejected_even_when_byte_hash_matches(self):
        for kind in ("rag", "gi"):
            for malformed in ("duplicate", "nonfinite"):
                with self.subTest(kind=kind, malformed=malformed):
                    fixture = ArtifactFixture(kind, self.root / malformed)
                    prefix = b'{"schema_version":"duplicate",' if malformed == "duplicate" else b'{"probe":NaN,'
                    fixture.payload = prefix + fixture.payload.lstrip()[1:]
                    fixture.reference_payload_bytes()
                    with self.assertRaisesRegex(RuntimeError, "after 1 attempts"):
                        fixture.load(self.namespace, attempts=1)
                    self.assert_no_partials(fixture)

    def test_rag_date_and_gi_source_reference_validators_remain_distinct(self):
        rag = self.fixture("rag")
        rag.manifest["bundle_review_date"] = "2026-11-12"
        with self.assertRaisesRegex(RuntimeError, "after 1 attempts"):
            rag.load(self.namespace, attempts=1)
        gi = self.fixture("gi")
        gi.package["synonyms"][0]["term_id"] = "fixture:missing"
        gi.refresh()
        with self.assertRaisesRegex(RuntimeError, "after 1 attempts"):
            gi.load(self.namespace, attempts=1)
        self.assert_no_partials(rag)
        self.assert_no_partials(gi)

    def test_invalid_configuration_fails_before_network_access(self):
        for kind in ("rag", "gi"):
            for invalid in ({"repository": "../project"}, {"ref": "../main"},
                            {"manifest_path": "../manifest.json"}, {"attempts": True},
                            {"attempts": 0}, {"attempts": 11}):
                with self.subTest(kind=kind, invalid=invalid):
                    fixture = self.fixture(kind)
                    with self.assertRaises(ValueError):
                        fixture.load(self.namespace, **invalid)
                    self.assertEqual(fixture.calls, [])

    def test_manifest_artifact_path_traversal_and_same_path_are_rejected(self):
        for kind in ("rag", "gi"):
            for invalid_path in ("../outside.json", "same_as_manifest"):
                with self.subTest(kind=kind, invalid_path=invalid_path):
                    fixture = self.fixture(kind)
                    fixture.manifest["artifact"]["path"] = (
                        fixture.manifest_path if invalid_path == "same_as_manifest" else invalid_path)
                    with self.assertRaisesRegex(RuntimeError, "after 1 attempts"):
                        fixture.load(self.namespace, attempts=1)
                    self.assertTrue(all(url.endswith(fixture.manifest_path) for url in fixture.calls))
                    self.assert_no_partials(fixture)

    def test_branch_with_slash_is_encoded_without_pinning(self):
        for kind in ("rag", "gi"):
            with self.subTest(kind=kind):
                fixture = self.fixture(kind)
                fixture.ref = "feature/terminology"
                _, source = fixture.load(self.namespace)
                self.assertEqual(source["repository_ref"], fixture.ref)
                self.assertTrue(all("/feature%2Fterminology/" in url for url in fixture.calls))


if __name__ == "__main__":
    unittest.main()
