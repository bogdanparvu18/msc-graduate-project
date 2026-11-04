import hashlib
import io
from pathlib import Path
import tarfile

from PIL import Image
import pytest

from mmvqa.phase3.images import ImageIntegrityError, ImageResolver


def jpeg(color="red"):
    buffer = io.BytesIO()
    Image.new("RGB", (12, 8), color).save(buffer, format="JPEG")
    return buffer.getvalue()


def record(path, data, storage="original", order=0, role="target"):
    return dict(input_id=f"input-{order}", artifact_id=f"artifact-{order}", input_order=order,
                storage=storage, relative_path=path, sha256=hashlib.sha256(data).hexdigest(),
                width=12, height=8, image_role=role)


def archive_fixture(tmp_path, *, duplicate=False, symlink=False, name="products/roi_crop/test.jpg"):
    data = jpeg("green")
    path = tmp_path / "archives/roi_crop.tar"
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "w") as tar:
        info = tarfile.TarInfo(name)
        info.size = len(data)
        if symlink:
            info.type = tarfile.SYMTYPE
            info.linkname = "elsewhere.jpg"
            info.size = 0
        tar.addfile(info, io.BytesIO(data) if not symlink else None)
        if duplicate:
            tar.addfile(info, io.BytesIO(data))
    member = dict(relative_path=name, bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
    entry = dict(relative_path="archives/roi_crop.tar", kind="roi_crop", bytes=path.stat().st_size,
                 sha256=hashlib.sha256(path.read_bytes()).hexdigest(), member_count=1, members=[member])
    manifest = dict(schema_version=1, mode="tar_per_product_folder", archive_count=1, member_count=1, archives=[entry])
    return data, path, manifest


def test_fullcrop_preserves_order_and_separates_provenance(tmp_path):
    original = jpeg()
    (tmp_path / "source.jpg").write_bytes(original)
    crop, _, manifest = archive_fixture(tmp_path)
    inputs = [record("source.jpg", original, role="full_frame"),
              record("products/roi_crop/test.jpg", crop, "derived", 1, "region_of_interest")]
    resolver = ImageResolver(original_roots=[tmp_path], archive_root=tmp_path, archive_manifest=manifest,
                             source_records=[dict(image_relpath="source.jpg", sha256=inputs[0]["sha256"], image_size_bytes=len(original))])
    images, roles, provenance = resolver.load_inputs(inputs)
    assert roles == ["full_frame", "region_of_interest"]
    assert len(images) == 2 and images[0].getpixel((0, 0))[0] > images[0].getpixel((0, 0))[1]
    assert all(not image.info for image in images)
    assert provenance[1]["archive_verified"] is True
    assert not (tmp_path / "products").exists()  # archive is read without extraction


def test_original_byte_hash_and_dimensions_are_checked(tmp_path):
    data = jpeg()
    (tmp_path / "source.jpg").write_bytes(data)
    item = record("source.jpg", data)
    resolver = ImageResolver(original_roots=[tmp_path])
    with pytest.raises(ImageIntegrityError, match="dimensions"):
        resolver.resolve({**item, "width": 13})
    (tmp_path / "source.jpg").write_bytes(jpeg("blue"))
    with pytest.raises(ImageIntegrityError, match="SHA-256"):
        resolver.resolve(item)


@pytest.mark.parametrize("relative", [
    "../escape.jpg", "/absolute.jpg", "a/../escape.jpg", "a//b.jpg", "a\\b.jpg",
    "C:/x.jpg", "C:x.jpg", "x.jpg:stream", "x\ny.jpg", "x\ry.jpg", "x\ty.jpg", "x\x00y.jpg", "x\x1fy.jpg",
])
def test_rejects_unsafe_paths(tmp_path, relative):
    with pytest.raises(ImageIntegrityError, match="relative path"):
        ImageResolver(original_roots=[tmp_path]).resolve(record(relative, jpeg()))


def test_rejects_original_symlink_escape(tmp_path):
    inside = tmp_path / "inside"
    inside.mkdir()
    (tmp_path / "outside.jpg").write_bytes(jpeg())
    (inside / "image.jpg").symlink_to(tmp_path / "outside.jpg")
    with pytest.raises(ImageIntegrityError, match="escapes"):
        ImageResolver(original_roots=[inside]).resolve(record("image.jpg", jpeg()))


@pytest.mark.parametrize("option", ["duplicate", "symlink"])
def test_rejects_unsafe_tar_members(tmp_path, option):
    data, _, manifest = archive_fixture(tmp_path, **{option: True})
    resolver = ImageResolver(archive_root=tmp_path, archive_manifest=manifest)
    with pytest.raises(ImageIntegrityError, match="Unsafe or duplicate"):
        resolver.resolve(record("products/roi_crop/test.jpg", data, "derived"))


def test_archive_corruption_after_cached_read_is_detected(tmp_path):
    data, path, manifest = archive_fixture(tmp_path)
    resolver = ImageResolver(archive_root=tmp_path, archive_manifest=manifest)
    item = record("products/roi_crop/test.jpg", data, "derived")
    resolver.resolve(item)
    payload = bytearray(path.read_bytes())
    payload[-1] ^= 1
    path.write_bytes(payload)
    with pytest.raises(ImageIntegrityError, match="Archive SHA-256"):
        resolver.resolve(item)


def test_recipe_archive_disagreement_is_rejected(tmp_path):
    data, _, manifest = archive_fixture(tmp_path)
    resolver = ImageResolver(archive_root=tmp_path, archive_manifest=manifest)
    item = record("products/roi_crop/test.jpg", data, "derived")
    with pytest.raises(ImageIntegrityError, match="disagree"):
        resolver.resolve({**item, "sha256": "0" * 64})


def test_input_order_and_roles_are_validated(tmp_path):
    resolver = ImageResolver(original_roots=[tmp_path])
    item = record("source.jpg", jpeg(), order=1)
    with pytest.raises(ImageIntegrityError, match="order"):
        resolver.load_inputs([item])
    with pytest.raises(ImageIntegrityError, match="role"):
        resolver.load_inputs([{**item, "input_order": 0, "image_role": "bleeding"}])


def test_source_record_byte_mismatch_is_rejected(tmp_path):
    data = jpeg()
    (tmp_path / "source.jpg").write_bytes(data)
    item = record("source.jpg", data)
    resolver = ImageResolver(original_roots=[tmp_path], source_records=[
        dict(image_relpath="source.jpg", sha256=item["sha256"], image_size_bytes=len(data) + 1)])
    with pytest.raises(ImageIntegrityError, match="byte count"):
        resolver.resolve(item)


def test_temporal_frame_and_offset_roles_preserve_order(tmp_path):
    data = jpeg()
    (tmp_path / "source.jpg").write_bytes(data)
    first = record("source.jpg", data, role="temporal_context")
    second = record("source.jpg", data, order=1, role="target")
    inputs = [{**first, "frame_index": 10, "time_offset_seconds": -0.5},
              {**second, "frame_index": 12, "time_offset_seconds": 0.0}]
    _, roles, _ = ImageResolver(original_roots=[tmp_path]).load_inputs(inputs)
    assert roles == ["temporal_context; frame=10; offset_seconds=-0.5", "target; frame=12; offset_seconds=+0"]


def test_model_image_removes_embedded_exif_and_filename(tmp_path):
    buffer = io.BytesIO()
    exif = Image.Exif()
    exif[270] = "private source annotation"
    exif[315] = "private source author"
    Image.new("RGB", (12, 8), "red").save(buffer, format="JPEG", exif=exif)
    data = buffer.getvalue()
    with Image.open(io.BytesIO(data)) as source:
        assert source.getexif()[270] == "private source annotation"
    (tmp_path / "source.jpg").write_bytes(data)
    image = ImageResolver(original_roots=[tmp_path]).resolve(record("source.jpg", data)).image
    assert not image.info
    assert not image.getexif()
    assert not getattr(image, "filename", "")
