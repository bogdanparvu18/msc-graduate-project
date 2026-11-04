"""Integrity and snapshot isolation tests using exact-byte fake GitHub responses."""

import copy
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

from mmvqa.phase3.storage import CATALOG_PATH, INDEX_PATH, S8_ROOT, IntegrityError, S8Store


COMMIT = "a" * 40
REPO = "owner/project"


def digest(data):
    return hashlib.sha256(data).hexdigest()


class Publication:
    def __init__(self):
        self.commit = COMMIT
        self.calls = []
        self.payloads = {}
        self.index = {"schema_version": 1, "mode": "exact_byte_parts", "files": []}
        self.add(CATALOG_PATH, b'{"processing_signature":"example"}')
        self.index["source_catalog_sha256"] = self.index["files"][0]["sha256"]

    def add(self, path, data, parts=False):
        entry = {"relative_path": path, "bytes": len(data), "sha256": digest(data)}
        if parts:
            entry.update(storage="parts", parts=[])
            cut = max(1, len(data) // 2)
            offset = 0
            for ordinal, block in enumerate((data[:cut], data[cut:])):
                chunk = f"metadata/_github_chunks/{digest(block)}.bin"
                entry["parts"].append({"relative_path": chunk, "bytes": len(block),
                                       "sha256": digest(block), "offset": offset, "ordinal": ordinal})
                self.payloads[chunk] = block
                offset += len(block)
        else:
            entry.update(storage="direct", github_relative_path=path)
            self.payloads[path] = data
        self.index["files"].append(entry)
        return entry

    def fetch(self, url, target):
        self.calls.append(url)
        if url.startswith("https://api.github.com/"):
            target.write_bytes(json.dumps({"sha": self.commit}).encode())
            return
        prefix = f"https://raw.githubusercontent.com/{REPO}/{COMMIT}/{S8_ROOT}/"
        if not url.startswith(prefix):
            raise AssertionError(f"Unpinned or unexpected request: {url}")
        path = url[len(prefix):]
        data = json.dumps(self.index).encode() if path == INDEX_PATH else self.payloads[path]
        target.write_bytes(data)


class StorageTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.publication = Publication()

    def store(self, **kwargs):
        return S8Store(REPO, cache_dir=self.root / "cache", fetch=self.publication.fetch, **kwargs)

    def test_direct_parts_and_catalog_are_verified_and_recorded(self):
        self.publication.add("metadata/direct.json", b'{"value":1}')
        entry = self.publication.add("metadata/split.bin", b"binary\x00data\xffwith-parts", parts=True)
        # Ordinals define ordering; JSON list order is not relied on.
        entry["parts"].reverse()
        store = self.store()
        self.assertEqual(store.read_json("metadata/direct.json"), {"value": 1})
        self.assertEqual(store.load("metadata/split.bin").read_bytes(), b"binary\x00data\xffwith-parts")
        self.assertEqual(store.catalog()["processing_signature"], "example")
        snapshot = store.snapshot()
        self.assertEqual(snapshot["resolved_commit"], COMMIT)
        self.assertEqual(len(snapshot["consumed_artifacts"]), 3)
        self.assertEqual(snapshot["storage_index"]["sha256"], digest(json.dumps(self.publication.index).encode()))

    def test_branch_is_resolved_only_once_and_never_used_for_raw_requests(self):
        self.publication.add("metadata/data.bin", b"stable")
        store = self.store()
        store.resolve()
        self.publication.commit = "b" * 40
        store.load("metadata/data.bin")
        self.assertEqual(store.snapshot()["resolved_commit"], COMMIT)
        self.assertEqual(sum("api.github.com" in url for url in self.publication.calls), 1)
        self.assertTrue(all("/main/" not in url for url in self.publication.calls))

    def test_explicit_commit_is_validated_against_github(self):
        self.assertEqual(self.store(ref=COMMIT).resolve(), COMMIT)
        self.publication.commit = "b" * 40
        with self.assertRaisesRegex(IntegrityError, "different commit"):
            self.store(ref=COMMIT).resolve()

    def test_corrupt_direct_download_is_not_published(self):
        self.publication.add("metadata/file.bin", b"correct")
        self.publication.payloads["metadata/file.bin"] = b"corrupt"
        store = self.store()
        with self.assertRaises(IntegrityError):
            store.load("metadata/file.bin")
        self.assertFalse(any((self.root / "cache").rglob("file.bin")))
        self.assertEqual(store.consumed, {})
        self.assertFalse(list(self.root.rglob("*.tmp")))

    def test_corrupt_part_and_wrong_final_hash_are_rejected(self):
        entry = self.publication.add("metadata/file.bin", b"correct-parts", parts=True)
        first = entry["parts"][0]["relative_path"]
        original = self.publication.payloads[first]
        self.publication.payloads[first] = b"x" * len(original)
        with self.assertRaises(IntegrityError):
            self.store().load("metadata/file.bin")
        self.publication.payloads[first] = original
        entry["sha256"] = "0" * 64
        with self.assertRaises(IntegrityError):
            self.store().load("metadata/file.bin")

    def test_cache_reused_only_after_hash_check_and_corruption_repaired(self):
        self.publication.add("metadata/file.bin", b"correct")
        store = self.store()
        path = store.load("metadata/file.bin")
        calls = len(self.publication.calls)
        store.load("metadata/file.bin")
        self.assertEqual(len(self.publication.calls), calls)
        path.write_bytes(b"corrupt")
        self.assertEqual(store.load("metadata/file.bin").read_bytes(), b"correct")
        self.assertEqual(len(self.publication.calls), calls + 1)

    def test_split_cache_reconstruction_repairs_corrupt_part(self):
        self.publication.add("metadata/file.bin", b"correct-parts", parts=True)
        store = self.store()
        path = store.load("metadata/file.bin")
        path.unlink()
        cached_part = next(self.root.rglob("parts/*.bin"))
        cached_part.write_bytes(b"broken")
        calls = len(self.publication.calls)
        self.assertEqual(store.load("metadata/file.bin").read_bytes(), b"correct-parts")
        self.assertEqual(len(self.publication.calls), calls + 1)

    def test_invalid_part_layouts_fail_before_payload_download(self):
        entry = self.publication.add("metadata/file.bin", b"correct-parts", parts=True)
        pristine = copy.deepcopy(entry)
        for change in (lambda e: e["parts"][1].update(ordinal=0),
                       lambda e: e["parts"][1].update(offset=0),
                       lambda e: e.update(bytes=1000),
                       lambda e: e["parts"][0].update(bytes=True)):
            with self.subTest(change=change):
                entry.clear()
                entry.update(copy.deepcopy(pristine))
                change(entry)
                self.publication.calls.clear()
                with self.assertRaises(IntegrityError):
                    self.store().resolve()
                self.assertEqual(len(self.publication.calls), 2)

    def test_unsafe_paths_and_duplicate_logical_paths_are_rejected(self):
        pristine = copy.deepcopy(self.publication.index)
        for path in ("../escape", "/absolute", "a/../../escape", "C:/x", "a\\b", "a//b", "a/./b"):
            with self.subTest(path=path):
                self.publication.index = copy.deepcopy(pristine)
                self.publication.add(path, b"data")
                with self.assertRaises(IntegrityError):
                    self.store().resolve()
        self.publication.index = pristine
        self.publication.index["files"].append(copy.deepcopy(pristine["files"][0]))
        with self.assertRaisesRegex(IntegrityError, "Duplicate"):
            self.store().resolve()

    def test_catalog_binding_required(self):
        self.publication.index["source_catalog_sha256"] = "f" * 64
        with self.assertRaisesRegex(IntegrityError, "catalog"):
            self.store().resolve()

    def test_cache_symlink_cannot_escape_publication(self):
        self.publication.add("metadata/file.bin", b"correct")
        store = self.store()
        store.resolve()
        outside = self.root / "outside"
        outside.mkdir()
        store._root.joinpath("logical").symlink_to(outside, target_is_directory=True)
        with self.assertRaisesRegex(IntegrityError, "escapes"):
            store.load("metadata/file.bin")
        self.assertEqual(list(outside.iterdir()), [])

    def test_token_is_absent_from_snapshot(self):
        store = self.store(token="private-test-token")
        self.assertNotIn("private-test-token", json.dumps(store.snapshot()))

    def test_git_checkout_reads_commit_object_not_modified_worktree(self):
        checkout = self.root / "checkout"
        checkout.mkdir()
        subprocess.run(["git", "init", "-q", str(checkout)], check=True)
        self.publication.add("metadata/file.bin", b"committed")
        for path, data in {**self.publication.payloads, INDEX_PATH: json.dumps(self.publication.index).encode()}.items():
            target = checkout / S8_ROOT / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        subprocess.run(["git", "-C", str(checkout), "add", "."], check=True)
        subprocess.run(["git", "-C", str(checkout), "-c", "user.name=Test", "-c", "user.email=test@example.test", "commit", "-qm", "fixture"], check=True)
        self.publication.commit = subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip()
        (checkout / S8_ROOT / "metadata/file.bin").write_bytes(b"mutated worktree")
        store = self.store(checkout=checkout)
        self.assertEqual(store.load("metadata/file.bin").read_bytes(), b"committed")
        self.assertEqual(len(self.publication.calls), 1)


if __name__ == "__main__":
    unittest.main()
