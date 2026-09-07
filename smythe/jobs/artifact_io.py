"""Crash-safe artifact persistence for durable jobs."""

from __future__ import annotations

import hashlib
import io
import os
import stat
import struct
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any


_SUPPORTED_IMAGE_MIME_BY_FORMAT = {
    "PNG": "image/png",
    "JPEG": "image/jpeg",
    "GIF": "image/gif",
    "WEBP": "image/webp",
}

# These limits are deliberately independent from manifest validation. Provider
# responses are untrusted input, so the final persistence boundary must remain
# bounded even when an adapter or SDK returns an unexpected payload.
MAX_ARTIFACT_BYTES = 32 * 1024 * 1024
MAX_IMAGE_PIXELS = 64 * 1024 * 1024
MAX_IMAGE_FRAMES = 256
MAX_TOTAL_DECODED_PIXELS = 256 * 1024 * 1024


class ArtifactInspectionError(ValueError):
    """Raised when declared or detected raster bytes cannot be decoded safely."""


@dataclass(frozen=True, slots=True)
class ArtifactInspection:
    """Objective properties observed from artifact bytes."""

    mime_type: str
    size_bytes: int
    sha256: str
    width: int | None = None
    height: int | None = None


def inspect_artifact(data: bytes, declared_mime_type: str) -> ArtifactInspection:
    """Inspect an artifact without trusting its filename or header alone.

    PNG, JPEG, GIF, and WebP candidates are verified and fully decoded with
    Pillow before their MIME type or dimensions are accepted. Unknown
    non-image binary types retain their declared MIME type and omit dimensions.
    Declared image data fails closed when it cannot be decoded.
    """
    if not isinstance(data, bytes):
        raise ArtifactInspectionError("artifact data must be bytes")
    if len(data) > MAX_ARTIFACT_BYTES:
        raise ArtifactInspectionError(
            f"artifact exceeds the {MAX_ARTIFACT_BYTES}-byte persistence limit"
        )
    header_mime_type, header_width, header_height = _image_header(
        data, declared_mime_type
    )
    looks_like_supported_image = header_mime_type in _SUPPORTED_IMAGE_MIME_BY_FORMAT.values()
    declared_as_image = declared_mime_type.strip().lower().startswith("image/")
    if looks_like_supported_image or declared_as_image:
        if (
            header_width is not None
            and header_height is not None
            and header_width > 0
            and header_height > 0
        ):
            _validate_image_shape(header_width, header_height, frames=1)
        mime_type, width, height = _decode_image(data)
    else:
        mime_type, width, height = declared_mime_type, None, None
    return ArtifactInspection(
        mime_type=mime_type,
        size_bytes=len(data),
        sha256=hashlib.sha256(data).hexdigest(),
        width=width,
        height=height,
    )


def _decode_image(data: bytes) -> tuple[str, int, int]:
    try:
        from PIL import Image, UnidentifiedImageError
    except ImportError as exc:  # pragma: no cover - exercised without the extra
        raise ImportError(
            "Image artifact validation requires Pillow; install smythe[jobs]"
        ) from exc

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as candidate:
                _validate_image_shape(
                    *candidate.size,
                    frames=getattr(candidate, "n_frames", 1),
                )
                candidate.verify()
            with Image.open(io.BytesIO(data)) as decoded:
                format_name = (decoded.format or "").upper()
                size = decoded.size
                frame_count = getattr(decoded, "n_frames", 1)
                _validate_image_shape(*size, frames=frame_count)
                for frame_index in range(frame_count):
                    decoded.seek(frame_index)
                    decoded.load()
    except (
        UnidentifiedImageError,
        OSError,
        SyntaxError,
        ValueError,
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
    ) as exc:
        if isinstance(exc, ArtifactInspectionError):
            raise
        raise ArtifactInspectionError(
            f"image artifact could not be fully decoded: {type(exc).__name__}"
        ) from exc

    mime_type = _SUPPORTED_IMAGE_MIME_BY_FORMAT.get(format_name)
    if mime_type is None:
        raise ArtifactInspectionError(
            f"unsupported decoded image format: {format_name or 'unknown'}"
        )
    width, height = size
    if width < 1 or height < 1:
        raise ArtifactInspectionError(f"decoded image has invalid dimensions: {size!r}")
    return mime_type, width, height


def _validate_image_shape(width: int, height: int, *, frames: int) -> None:
    if width < 1 or height < 1:
        raise ArtifactInspectionError(
            f"decoded image has invalid dimensions: {(width, height)!r}"
        )
    if width * height > MAX_IMAGE_PIXELS:
        raise ArtifactInspectionError(
            f"decoded image exceeds the {MAX_IMAGE_PIXELS}-pixel limit"
        )
    if frames < 1 or frames > MAX_IMAGE_FRAMES:
        raise ArtifactInspectionError(
            f"decoded image frame count {frames} exceeds the {MAX_IMAGE_FRAMES}-frame limit"
        )
    if width * height * frames > MAX_TOTAL_DECODED_PIXELS:
        raise ArtifactInspectionError(
            "decoded image frames exceed the aggregate pixel limit"
        )


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Atomically replace *path* after flushing a same-directory temp file."""
    if not isinstance(data, bytes):
        raise TypeError("artifact data must be bytes")
    if len(data) > MAX_ARTIFACT_BYTES:
        raise ValueError(
            f"artifact exceeds the {MAX_ARTIFACT_BYTES}-byte persistence limit"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
        _fsync_directory(path.parent)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def atomic_publish_bytes(path: Path, data: bytes) -> None:
    """Durably publish new bytes, atomically refusing an existing destination.

    The fully flushed temporary file is linked into place without replacement.
    Filesystems without hard-link support fail closed; Jobs probes this before
    admitting provider calls. Existing destinations, including identical bytes,
    always remain untouched.
    """
    if not isinstance(data, bytes):
        raise TypeError("artifact data must be bytes")
    if len(data) > MAX_ARTIFACT_BYTES:
        raise ValueError(f"artifact exceeds the {MAX_ARTIFACT_BYTES}-byte persistence limit")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", dir=path.parent, prefix=f".{path.name}.",
                                         suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _regular_path(root: Path, relative: str) -> Path:
    windows, posix = PureWindowsPath(relative), PurePosixPath(relative.replace("\\", "/"))
    if (not relative or windows.drive or windows.root or posix.is_absolute() or ".." in posix.parts
            or not posix.parts or any(":" in part for part in posix.parts)):
        raise ValueError("Unsafe recorded artifact path")
    path = root
    for part in posix.parts:
        path = path / part
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise ValueError("Artifact ownership does not follow symbolic links or junctions")
    if not stat.S_ISREG(info.st_mode):
        raise ValueError("Artifact ownership requires regular files")
    return path


def claim_artifact_root(root: Path, owner_bytes: bytes, *, legacy: bool,
                        legacy_artifacts: list[dict[str, Any]]) -> None:
    """Bind a directory to one persistent run identity before dispatch.

    Legacy receipt paths remain in place. An existing conflicting or incomplete
    marker is never replaced. No-clobber publication also protects artifacts
    belonging to an unknown legacy ledger that has not claimed this root yet.
    """
    root.mkdir(parents=True, exist_ok=True)
    marker = root / ".smythe-run-owner.json"
    try:
        marker.lstat()
    except FileNotFoundError:
        if not legacy and any(root.iterdir()):
            raise ValueError("New artifact namespace already contains unowned files")
        for artifact in legacy_artifacts:
            if not artifact["accepted"]:
                continue
            path = _regular_path(root, artifact["relative_path"])
            size = artifact["size_bytes"]
            if type(size) is not int or not 0 <= size <= MAX_ARTIFACT_BYTES:
                raise ValueError("Invalid legacy artifact size")
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                remaining = size
                while remaining:
                    chunk = stream.read(min(1024 * 1024, remaining))
                    if not chunk:
                        raise ValueError("Legacy accepted artifact is truncated")
                    digest.update(chunk)
                    remaining -= len(chunk)
                if stream.read(1) or digest.hexdigest() != artifact["sha256"]:
                    raise ValueError("Legacy accepted artifact differs from its receipt")
        try:
            atomic_publish_bytes(marker, owner_bytes)
        except FileExistsError:
            pass  # Another claimant won; exact ownership is checked below.
    marker = _regular_path(root, marker.name)
    with marker.open("rb") as stream:
        if stream.read(len(owner_bytes) + 1) != owner_bytes:
            raise ValueError("Artifact directory belongs to another run or has an incomplete owner marker")
    # Check the actual mounted filesystem even when a marker predates this
    # process. All probe paths are owned by this fresh temporary directory.
    with tempfile.TemporaryDirectory(prefix=".smythe-publication-", dir=root) as directory:
        probe = Path(directory) / "probe"
        atomic_publish_bytes(probe, b"publication probe")
        try:
            atomic_publish_bytes(probe, b"must not replace")
        except FileExistsError:
            if probe.read_bytes() != b"publication probe":
                raise ValueError("Filesystem did not preserve an existing artifact")
        else:
            raise ValueError("Filesystem does not support exclusive artifact publication")


def _fsync_directory(directory: Path) -> None:
    """Persist the directory entry after replace where the OS supports it."""

    if os.name == "nt":
        # Windows does not support opening directories with ``os.open``. The
        # file itself is flushed before replacement or exclusive publication;
        # the standard library cannot separately flush the directory entry.
        return
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(directory, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _image_header(
    data: bytes,
    declared_mime_type: str,
) -> tuple[str, int | None, int | None]:
    if len(data) >= 24 and data.startswith(b"\x89PNG\r\n\x1a\n"):
        width, height = struct.unpack(">II", data[16:24])
        return "image/png", width, height

    if len(data) >= 10 and data[:6] in (b"GIF87a", b"GIF89a"):
        width, height = struct.unpack("<HH", data[6:10])
        return "image/gif", width, height

    if len(data) >= 12 and data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        dimensions = _webp_dimensions(data)
        return "image/webp", *(dimensions or (None, None))

    if len(data) >= 4 and data.startswith(b"\xff\xd8"):
        dimensions = _jpeg_dimensions(data)
        return "image/jpeg", *(dimensions or (None, None))

    return declared_mime_type, None, None


def _jpeg_dimensions(data: bytes) -> tuple[int, int] | None:
    offset = 2
    start_of_frame = {
        0xC0,
        0xC1,
        0xC2,
        0xC3,
        0xC5,
        0xC6,
        0xC7,
        0xC9,
        0xCA,
        0xCB,
        0xCD,
        0xCE,
        0xCF,
    }
    while offset + 4 <= len(data):
        if data[offset] != 0xFF:
            offset += 1
            continue
        marker = data[offset + 1]
        offset += 2
        if marker in (0xD8, 0xD9) or 0xD0 <= marker <= 0xD7:
            continue
        if offset + 2 > len(data):
            break
        segment_length = struct.unpack(">H", data[offset : offset + 2])[0]
        if segment_length < 2 or offset + segment_length > len(data):
            break
        if marker in start_of_frame and segment_length >= 7:
            height, width = struct.unpack(">HH", data[offset + 3 : offset + 7])
            return width, height
        offset += segment_length
    return None


def _webp_dimensions(data: bytes) -> tuple[int, int] | None:
    if len(data) < 30:
        return None
    chunk = data[12:16]
    if chunk == b"VP8X" and len(data) >= 30:
        width = 1 + int.from_bytes(data[24:27], "little")
        height = 1 + int.from_bytes(data[27:30], "little")
        return width, height
    if chunk == b"VP8L" and len(data) >= 25 and data[20] == 0x2F:
        bits = int.from_bytes(data[21:25], "little")
        return (bits & 0x3FFF) + 1, ((bits >> 14) & 0x3FFF) + 1
    return None
