"""Read the S8 publication from one resolved GitHub commit, with byte verification.

The branch is resolved once per store. Every index and artifact request then uses
that immutable commit; a later move of ``main`` cannot mix two publications.
Only explicitly requested logical files are downloaded or reconstructed.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import tempfile
from typing import Callable
from urllib.parse import quote, urlparse
from urllib.request import Request, urlopen


DEFAULT_REPOSITORY = "bogdanparvu18/msc-graduate-project"
S8_ROOT = "outputs/phase2/sector8"
INDEX_PATH = "configs/sector8_github_storage.json"
CATALOG_PATH = "configs/sector8_archive_state.json"
_BLOCK_SIZE = 1024 * 1024
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_COMMIT = re.compile(r"[0-9a-f]{40}\Z")


class IntegrityError(ValueError):
    """The publication, its layout, or an artifact failed verification."""


def safe_relative_path(value: object) -> str:
    """Accept only canonical portable paths beneath a publication root."""
    if not isinstance(value, str) or not value or "\\" in value or ":" in value:
        raise IntegrityError(f"Invalid relative path: {value!r}")
    path = PurePosixPath(value)
    if (path.is_absolute() or any(p in {".", ".."} for p in path.parts)
            or path.as_posix() != value or value == "."
            or any(ord(c) < 32 for c in value)):
        raise IntegrityError(f"Unsafe or noncanonical relative path: {value!r}")
    return value


def file_digest(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(_BLOCK_SIZE), b""):
            size += len(block)
            digest.update(block)
    return size, digest.hexdigest()


def verify_file(path: Path, size: int, sha256: str) -> None:
    actual_size, actual_hash = file_digest(path)
    if actual_size != size or actual_hash != sha256:
        raise IntegrityError(
            f"Integrity check failed for {path.name}: expected {size} bytes / "
            f"{sha256}; received {actual_size} bytes / {actual_hash}"
        )


def _size(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise IntegrityError(f"{field} must be a nonnegative integer")
    return value


def _hash(value: object, field: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise IntegrityError(f"{field} must be a lowercase SHA-256")
    return value


class S8Store:
    """Verified S8 metadata loader.

    ``fetch`` is an optional transport taking ``(url, destination_path)`` and
    writing exact response bytes. ``checkout`` can reuse Git objects locally:
    the commit is still resolved through GitHub, and bytes are read with
    ``git show <resolved-commit>:<path>``, never from a mutable working tree.
    Credentials are kept only in memory and never included in snapshots.
    """

    def __init__(
        self,
        repo: str = DEFAULT_REPOSITORY,
        ref: str = "main",
        cache_dir: str | Path = ".cache/mmvqa-phase3",
        *,
        token: str | None = None,
        fetch: Callable[[str, Path], None] | None = None,
        checkout: str | Path | None = None,
        timeout: float = 120,
    ) -> None:
        if (not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo)
                or any(part in {".", ".."} for part in repo.split("/"))):
            raise ValueError("repo must have the form owner/repository")
        if not isinstance(ref, str) or not ref or any(ord(c) < 32 for c in ref):
            raise ValueError("ref must be a nonempty branch, tag, or commit")
        self.repo, self.ref = repo, ref
        self.cache_dir = Path(cache_dir).expanduser().resolve()
        self.token, self.fetch = token, fetch
        self.checkout = Path(checkout).resolve() if checkout is not None else None
        self.timeout = timeout
        self.commit: str | None = None
        self.index: dict = {}
        self.entries: dict[str, dict] = {}
        self.consumed: dict[str, dict] = {}
        self._index_identity: dict = {}
        self._root: Path | None = None

    def _download(self, url: str, destination: Path) -> None:
        parsed = urlparse(url)
        if self.checkout is not None and parsed.netloc == "raw.githubusercontent.com":
            prefix = f"/{self.repo}/{self.commit}/"
            if not parsed.path.startswith(prefix):
                raise IntegrityError("Local Git read does not match the pinned repository")
            # Paths in the S8 publication are canonical and ASCII; decode URL
            # quoting before passing the whole revision as one argv element.
            from urllib.parse import unquote
            path = safe_relative_path(unquote(parsed.path[len(prefix):]))
            with destination.open("wb") as output:
                process = subprocess.run(
                    ["git", "-C", str(self.checkout), "show", f"{self.commit}:{path}"],
                    stdout=output, stderr=subprocess.PIPE, check=False,
                )
            if process.returncode:
                raise RuntimeError(f"Pinned Git object is unavailable: {path}")
            return
        if self.fetch is not None:
            self.fetch(url, destination)
            return
        headers = {"User-Agent": "MMVQA-Phase3", "Accept": "application/vnd.github+json"}
        if self.token and parsed.netloc in {"api.github.com", "raw.githubusercontent.com"}:
            headers["Authorization"] = f"Bearer {self.token}"
        with urlopen(Request(url, headers=headers), timeout=self.timeout) as response:
            with destination.open("wb") as output:
                shutil.copyfileobj(response, output, _BLOCK_SIZE)

    def _raw_url(self, relative_path: str) -> str:
        path = safe_relative_path(relative_path)
        return f"https://raw.githubusercontent.com/{self.repo}/{self.commit}/{S8_ROOT}/{quote(path, safe='/')}"

    @staticmethod
    def _temp(parent: Path) -> Path:
        parent.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(prefix=".s8-", suffix=".tmp", dir=parent)
        os.close(fd)
        return Path(name)

    def _cache_path(self, namespace: str, relative_path: str) -> Path:
        assert self._root is not None
        relative_path = safe_relative_path(relative_path)
        base = self._root / namespace
        result = base.joinpath(*PurePosixPath(relative_path).parts)
        # Refuse cache symlinks that could redirect a publication outside its root.
        if not result.resolve().is_relative_to(self._root.resolve()):
            raise IntegrityError("Cache path escapes the pinned publication root")
        return result

    def resolve(self) -> str:
        """Resolve the requested GitHub ref and read its immutable storage index."""
        if self.commit is not None and self.entries:
            return self.commit
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        temporary = self._temp(self.cache_dir)
        try:
            self._download(
                f"https://api.github.com/repos/{self.repo}/commits/{quote(self.ref, safe='')}",
                temporary,
            )
            result = json.loads(temporary.read_bytes())
            commit = result.get("sha") if isinstance(result, dict) else None
            if not isinstance(commit, str) or not _COMMIT.fullmatch(commit):
                raise IntegrityError("GitHub did not return an immutable 40-character commit SHA")
            if _COMMIT.fullmatch(self.ref) and commit != self.ref:
                raise IntegrityError("GitHub resolved a different commit than the requested SHA")
            self.commit = commit
            self._root = self.cache_dir / self.repo.replace("/", "__") / commit
            target = self._cache_path("publication", INDEX_PATH)
            target.parent.mkdir(parents=True, exist_ok=True)
            self._download(self._raw_url(INDEX_PATH), temporary)
            index = json.loads(temporary.read_bytes())
            entries = self._validate_index(index)
            byte_count, digest = file_digest(temporary)
            # Copy atomically in the target directory (cache can span mounts).
            staged = self._temp(target.parent)
            try:
                shutil.copyfile(temporary, staged)
                os.replace(staged, target)
            finally:
                staged.unlink(missing_ok=True)
            self.index, self.entries = index, entries
            self._index_identity = {
                "relative_path": INDEX_PATH, "bytes": byte_count, "sha256": digest,
            }
            return commit
        except Exception:
            self.commit = None
            self.entries = {}
            raise
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def _validate_index(index: object) -> dict[str, dict]:
        if (not isinstance(index, dict) or type(index.get("schema_version")) is not int
                or index.get("schema_version") != 1):
            raise IntegrityError("Unsupported S8 storage-index schema")
        if index.get("mode") != "exact_byte_parts":
            raise IntegrityError("Unsupported S8 storage-index mode")
        catalog_hash = _hash(index.get("source_catalog_sha256"), "source_catalog_sha256")
        files = index.get("files")
        if not isinstance(files, list) or not files:
            raise IntegrityError("Storage index has no files")
        entries = {}
        for raw in files:
            if not isinstance(raw, dict):
                raise IntegrityError("Storage entry is not an object")
            entry = dict(raw)
            logical = safe_relative_path(entry.get("relative_path"))
            if logical in entries:
                raise IntegrityError(f"Duplicate logical storage path: {logical}")
            total = _size(entry.get("bytes"), "bytes")
            _hash(entry.get("sha256"), "sha256")
            storage = entry.get("storage")
            if storage == "direct":
                safe_relative_path(entry.get("github_relative_path"))
            elif storage == "parts":
                parts = entry.get("parts")
                if not isinstance(parts, list) or not parts:
                    raise IntegrityError(f"No parts declared for {logical}")
                for part in parts:
                    if not isinstance(part, dict):
                        raise IntegrityError("Part entry is not an object")
                    _size(part.get("ordinal"), "ordinal")
                parts = sorted(parts, key=lambda part: part["ordinal"])
                offset = 0
                for ordinal, part in enumerate(parts):
                    if part["ordinal"] != ordinal:
                        raise IntegrityError(f"Noncontiguous or duplicate part ordinals: {logical}")
                    if _size(part.get("offset"), "offset") != offset:
                        raise IntegrityError(f"Noncontiguous part offsets: {logical}")
                    safe_relative_path(part.get("relative_path"))
                    _hash(part.get("sha256"), "part sha256")
                    part_size = _size(part.get("bytes"), "part bytes")
                    if part_size == 0:
                        raise IntegrityError(f"Empty part in {logical}")
                    offset += part_size
                if offset != total:
                    raise IntegrityError(f"Part sizes do not match logical size: {logical}")
                entry["parts"] = parts
            else:
                raise IntegrityError(f"Unsupported storage kind for {logical}: {storage}")
            entries[logical] = entry
        if CATALOG_PATH not in entries or entries[CATALOG_PATH]["sha256"] != catalog_hash:
            raise IntegrityError("Storage index does not bind its source catalog")
        return entries

    def _ensure(self, target: Path, source: str, size: int, sha256: str) -> None:
        if target.is_file():
            try:
                verify_file(target, size, sha256)
                return
            except IntegrityError:
                pass  # Repair a stale or interrupted cache only with verified bytes.
        temporary = self._temp(target.parent)
        try:
            self._download(self._raw_url(source), temporary)
            verify_file(temporary, size, sha256)
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)

    def load(self, logical_path: str) -> Path:
        """Return a verified logical file, reconstructing ordered parts as needed."""
        self.resolve()
        logical_path = safe_relative_path(logical_path)
        if logical_path not in self.entries:
            raise KeyError(f"Logical path is absent from the pinned S8 index: {logical_path}")
        entry = self.entries[logical_path]
        target = self._cache_path("logical", logical_path)
        valid = False
        if target.is_file():
            try:
                verify_file(target, entry["bytes"], entry["sha256"])
                valid = True
            except IntegrityError:
                pass
        if not valid:
            if entry["storage"] == "direct":
                self._ensure(target, entry["github_relative_path"], entry["bytes"], entry["sha256"])
            else:
                temporary = self._temp(target.parent)
                try:
                    with temporary.open("wb") as output:
                        for part in entry["parts"]:
                            cached = self._cache_path("parts", part["sha256"] + ".bin")
                            self._ensure(cached, part["relative_path"], part["bytes"], part["sha256"])
                            with cached.open("rb") as source:
                                shutil.copyfileobj(source, output, _BLOCK_SIZE)
                    verify_file(temporary, entry["bytes"], entry["sha256"])
                    os.replace(temporary, target)
                finally:
                    temporary.unlink(missing_ok=True)
        self.consumed[logical_path] = {
            "relative_path": logical_path, "bytes": entry["bytes"],
            "sha256": entry["sha256"], "storage": entry["storage"],
            **({"parts": [dict(p) for p in entry["parts"]]} if "parts" in entry else {}),
        }
        return target

    def read_json(self, logical_path: str) -> dict:
        value = json.loads(self.load(logical_path).read_bytes())
        if not isinstance(value, dict):
            raise IntegrityError(f"Expected a JSON object: {logical_path}")
        return value

    def catalog(self) -> dict:
        """Load the catalog selected by the verified index."""
        return self.read_json(CATALOG_PATH)

    def snapshot(self) -> dict:
        """Serializable provenance, including every consumed logical artifact."""
        return {
            "schema_version": 1,
            "repository": self.repo,
            "requested_ref": self.ref,
            "resolved_commit": self.commit,
            "s8_root": S8_ROOT,
            "storage_index": dict(self._index_identity),
            "resolution_status": "verified" if self.commit and self.entries else "not_resolved",
            "source_catalog_sha256": self.index.get("source_catalog_sha256"),
            "consumed_artifacts": [self.consumed[key] for key in sorted(self.consumed)],
        }
