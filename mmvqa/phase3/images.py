"""Resolve S8 originals and archived derivatives without exposing source metadata.

The archive manifest and source records must come from the verified metadata
loader. Image payloads are deliberately read only from caller-configured local
roots (including a mounted Drive); GitHub does not contain S8 image archives.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import math
from pathlib import Path, PurePosixPath
import re
import tarfile
from typing import Any, Iterable, Mapping, Sequence

from PIL import Image


class ImageIntegrityError(ValueError):
    """The selected bytes or their manifest references do not agree."""


def _relative(value: Any) -> str:
    if (not isinstance(value, str) or not value or "\\" in value or ":" in value
            or any(ord(character) < 32 for character in value)):
        raise ImageIntegrityError(f"Unsafe relative path: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != value or value == ".":
        raise ImageIntegrityError(f"Unsafe relative path: {value!r}")
    return value


def _inside(root: Path, relative: str) -> Path:
    root = root.expanduser().resolve()
    candidate = (root / _relative(relative)).resolve()
    if not candidate.is_relative_to(root):
        raise ImageIntegrityError(f"Image path escapes configured root: {relative}")
    return candidate


def _digest(value: Any) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ImageIntegrityError("Expected a lowercase SHA-256 digest")
    return value


def _integer(value: Any, name: str, *, positive: bool = False) -> int:
    # S8 Parquet source byte counts are floats, so exact integral floats are OK.
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ImageIntegrityError(f"Invalid {name}: {value!r}")
    if not math.isfinite(value) or int(value) != value or value < int(positive):
        raise ImageIntegrityError(f"Invalid {name}: {value!r}")
    return int(value)


def _seal(path: Path) -> tuple[int, int, int, int, int]:
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


@dataclass
class ResolvedImage:
    """Pixels for the model and a separate record for provenance reporting."""

    image: Image.Image
    provenance: dict[str, Any]


class ImageResolver:
    """Hash-verify images from original roots or exact S8 TAR members.

    ``original_roots`` contain the S8 ``image_relpath`` hierarchy, normally the
    MMVQA_Clinical directory. ``archive_root`` contains ``archives/<kind>.tar``.
    Optional original overrides map exact portable paths to explicit files.
    No basename search or implicit path fallback can select a different image.
    ``derived_roots`` may point to a previously materialized S8 run directory.
    Full TAR bytes are verified once per unchanged filesystem identity, and
    every consumed member is checked against both the archive manifest and the
    recipe. This class is intended for serial use by a preview/worker instance.
    """

    def __init__(
        self,
        *,
        original_roots: Sequence[str | Path] = (),
        archive_root: str | Path | None = None,
        archive_manifest: Mapping[str, Any] | None = None,
        derived_roots: Sequence[str | Path] = (),
        source_records: Iterable[Mapping[str, Any]] = (),
        original_overrides: Mapping[str, str | Path] | None = None,
        max_image_bytes: int = 128 * 1024 * 1024,
    ) -> None:
        self.original_roots = tuple(Path(p) for p in original_roots)
        self.derived_roots = tuple(Path(p) for p in derived_roots)
        self.archive_root = Path(archive_root) if archive_root is not None else None
        self.max_image_bytes = _integer(max_image_bytes, "max_image_bytes", positive=True)
        self.overrides = {_relative(k): Path(v) for k, v in (original_overrides or {}).items()}
        self.sources: dict[str, dict[str, Any]] = {}
        for source in source_records:
            relative = _relative(source["image_relpath"])
            if relative in self.sources:
                raise ImageIntegrityError(f"Duplicate source path: {relative}")
            self.sources[relative] = dict(source)
        self.members: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
        self._archive_cache: dict[Path, tuple[tuple[int, ...], dict[str, tarfile.TarInfo]]] = {}
        if archive_manifest is not None:
            if archive_manifest.get("schema_version") != 1 or archive_manifest.get("mode") != "tar_per_product_folder":
                raise ImageIntegrityError("Unsupported S8 image archive manifest")
            archives = archive_manifest.get("archives", [])
            if archive_manifest.get("archive_count") != len(archives):
                raise ImageIntegrityError("Archive count differs from manifest")
            archive_paths: set[str] = set()
            for descriptor in archives:
                archive = dict(descriptor)
                path = _relative(archive["relative_path"])
                if path in archive_paths:
                    raise ImageIntegrityError(f"Duplicate archive path: {path}")
                archive_paths.add(path)
                _digest(archive["sha256"])
                _integer(archive["bytes"], "archive byte count", positive=True)
                members = archive.get("members", [])
                if archive.get("member_count") != len(members):
                    raise ImageIntegrityError(f"Member count differs for {path}")
                for row in members:
                    member = dict(row)
                    relative = _relative(member["relative_path"])
                    if relative in self.members:
                        raise ImageIntegrityError(f"Duplicate archived image: {relative}")
                    _digest(member["sha256"])
                    _integer(member["bytes"], "image byte count", positive=True)
                    self.members[relative] = archive, member
            if archive_manifest.get("member_count") != len(self.members):
                raise ImageIntegrityError("Total member count differs from manifest")

    def _read_file(self, path: Path, expected_sha: str, expected_bytes: Any = None) -> bytes:
        before = _seal(path)
        if before[2] > self.max_image_bytes:
            raise ImageIntegrityError(f"Image exceeds configured byte limit: {path.name}")
        if expected_bytes is not None and before[2] != _integer(expected_bytes, "image byte count", positive=True):
            raise ImageIntegrityError(f"Image byte count mismatch: {path.name}")
        with path.open("rb") as stream:
            data = stream.read(self.max_image_bytes + 1)
        if len(data) > self.max_image_bytes or before != _seal(path):
            raise ImageIntegrityError(f"Image changed while reading: {path.name}")
        if len(data) != before[2] or hashlib.sha256(data).hexdigest() != expected_sha:
            raise ImageIntegrityError(f"Image SHA-256 mismatch: {path.name}")
        return data

    def _verify_archive(self, archive: Mapping[str, Any]) -> tuple[Path, tuple[int, ...], dict[str, tarfile.TarInfo]]:
        if self.archive_root is None:
            raise FileNotFoundError("Configure archive_root with the mounted S8 archive directory")
        path = _inside(self.archive_root, archive["relative_path"])
        before = _seal(path)
        cached = self._archive_cache.get(path)
        if cached is not None and cached[0] == before:
            return path, before, cached[1]
        if before[2] != archive["bytes"]:
            raise ImageIntegrityError(f"Archive byte count mismatch: {archive['relative_path']}")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != archive["sha256"] or _seal(path) != before:
            raise ImageIntegrityError(f"Archive SHA-256 mismatch or changed file: {archive['relative_path']}")
        expected = {row["relative_path"]: row for row in archive["members"]}
        index: dict[str, tarfile.TarInfo] = {}
        with tarfile.open(path, mode="r:") as tar:
            for info in tar:
                name = _relative(info.name)
                if not info.isfile() or info.issparse() or name in index:
                    raise ImageIntegrityError(f"Unsafe or duplicate TAR member: {name}")
                if name not in expected or info.size != expected[name]["bytes"]:
                    raise ImageIntegrityError(f"TAR member differs from manifest: {name}")
                index[name] = info
        if set(index) != set(expected) or _seal(path) != before:
            raise ImageIntegrityError("Archive membership mismatch or changed file")
        self._archive_cache[path] = before, index
        return path, before, index

    def _read_derived(self, relative: str, expected_sha: str) -> tuple[bytes, dict[str, Any]]:
        if relative not in self.members:
            raise ImageIntegrityError(f"Derived image absent from archive manifest: {relative}")
        archive, member = self.members[relative]
        if member["sha256"] != expected_sha:
            raise ImageIntegrityError("Recipe and archive member SHA-256 disagree")
        for root in self.derived_roots:
            path = _inside(root, relative)
            if path.is_file():
                return self._read_file(path, expected_sha, member["bytes"]), {
                    "resolution": "materialized_derivative", "local_path": str(path),
                    "archive_relative_path": archive["relative_path"], "archive_sha256": archive["sha256"],
                    "archive_verified": False, "member_sha256": member["sha256"],
                }
        size = member["bytes"]
        if size > self.max_image_bytes:
            raise ImageIntegrityError("Archived image exceeds configured byte limit")
        path, before, index = self._verify_archive(archive)
        with tarfile.open(path, mode="r:") as tar:
            stream = tar.extractfile(index[relative])
            if stream is None:
                raise ImageIntegrityError("TAR image is not a readable regular file")
            with stream:
                data = stream.read(size + 1)
        if _seal(path) != before:
            self._archive_cache.pop(path, None)
            raise ImageIntegrityError("Archive changed while reading the image")
        if len(data) != size or hashlib.sha256(data).hexdigest() != expected_sha:
            raise ImageIntegrityError(f"Archived image integrity mismatch: {relative}")
        return data, {"resolution": "archive_member", "archive_relative_path": archive["relative_path"],
                      "archive_sha256": archive["sha256"], "archive_verified": True,
                      "member_relative_path": relative, "member_sha256": member["sha256"]}

    def resolve(self, item: Mapping[str, Any]) -> ResolvedImage:
        relative = _relative(item["relative_path"])
        digest = _digest(item["sha256"])
        width = _integer(item["width"], "width", positive=True)
        height = _integer(item["height"], "height", positive=True)
        storage = item.get("storage")
        if storage == "original":
            source = self.sources.get(relative)
            size = item.get("bytes")
            if source is not None:
                if source.get("sha256", source.get("source_sha256")) != digest:
                    raise ImageIntegrityError("Recipe and original source SHA-256 disagree")
                size = source.get("image_size_bytes", source.get("bytes", size))
            candidates = ([self.overrides[relative]] if relative in self.overrides else [])
            candidates.extend(_inside(root, relative) for root in self.original_roots)
            path = next((p for p in candidates if p.is_file()), None)
            if path is None:
                raise FileNotFoundError(f"Original image unavailable: {relative}; configure original_roots or original_overrides")
            data = self._read_file(path, digest, size)
            provenance = {"resolution": "original", "local_path": str(path)}
        elif storage == "derived":
            data, provenance = self._read_derived(relative, digest)
        else:
            raise ImageIntegrityError(f"Unknown image storage scope: {storage!r}")
        with Image.open(io.BytesIO(data)) as opened:
            if opened.size != (width, height):
                raise ImageIntegrityError(f"Image dimensions disagree with recipe: {relative}")
            image = opened.convert("RGB").copy()
        image.info.clear()
        if hasattr(image, "filename"):
            image.filename = ""
        provenance.update({"input_id": item.get("input_id"), "artifact_id": item.get("artifact_id"),
                           "input_order": item.get("input_order"), "storage": storage,
                           "relative_path": relative, "sha256": digest, "bytes": len(data),
                           "width": width, "height": height})
        return ResolvedImage(image, provenance)

    def load_inputs(self, inputs: Sequence[Mapping[str, Any]]) -> tuple[list[Image.Image], list[str], list[dict[str, Any]]]:
        """Return ordered pixels, neutral image roles, and separate provenance."""
        if not inputs or [_integer(row.get("input_order"), "input order") for row in inputs] != list(range(len(inputs))):
            raise ImageIntegrityError("Input order must be contiguous and zero based")
        images, roles, provenance = [], [], []
        for item in inputs:
            role = item.get("image_role")
            if not isinstance(role, str) or role not in {
                "target", "context", "full_frame", "region_of_interest", "temporal_context",
                "crop", "overlay", "fullcrop_panel", "temporal_collage", "progression", "qc", "comparison", "combined",
            }:
                raise ImageIntegrityError("Unexpected image role; labels must not enter the model payload")
            loaded = self.resolve(item)
            if item.get("frame_index") is not None:
                role += "; frame=" + str(_integer(item["frame_index"], "frame index"))
            if item.get("time_offset_seconds") is not None:
                offset = item["time_offset_seconds"]
                if isinstance(offset, bool) or not isinstance(offset, (int, float)) or not math.isfinite(offset):
                    raise ImageIntegrityError("Invalid temporal offset")
                role += "; offset_seconds=" + format(offset, "+.6g")
            images.append(loaded.image)
            roles.append(role)
            provenance.append(loaded.provenance)
        return images, roles, provenance
