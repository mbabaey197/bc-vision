from pathlib import Path

import cv2
import numpy as np
import pytest

import app.media_storage as media_storage


@pytest.fixture(autouse=True)
def _isolated_media_reservation(monkeypatch):
    class Reservation:
        def close(self, **_kwargs):
            return None

    monkeypatch.setattr(
        media_storage,
        "begin_media_write",
        lambda *_args, **_kwargs: Reservation(),
    )


def test_retry_reuses_valid_deterministic_targets_without_rewrite(
    tmp_path,
    monkeypatch,
):
    frame = np.full((120, 220, 3), 140, dtype=np.uint8)
    result = {
        "bbox": (60, 70, 160, 100),
        "crop": frame[70:100, 60:160].copy(),
    }
    plate_target = tmp_path / "plates" / "plate-live-token-77.jpg"
    vehicle_target = (
        tmp_path / "snapshots" / "vehicle-live-token-77.jpg"
    )

    first = media_storage.save_event_images(
        result,
        frame,
        plate_target=plate_target,
        vehicle_target=vehicle_target,
    )

    assert first.media_status == "complete"
    assert first.media_error == ""
    assert first.plate_path == str(plate_target)
    assert first.image_path == str(vehicle_target)
    assert cv2.imread(str(plate_target)) is not None
    assert cv2.imread(str(vehicle_target)) is not None
    original_plate = plate_target.read_bytes()
    original_vehicle = vehicle_target.read_bytes()

    def unexpected_rewrite(*_args, **_kwargs):
        raise AssertionError("verified deterministic media must be reused")

    monkeypatch.setattr(
        media_storage,
        "write_jpeg_atomic",
        unexpected_rewrite,
    )
    recovered = media_storage.save_event_images(
        {},
        None,
        plate_target=plate_target,
        vehicle_target=vehicle_target,
        reuse_existing_targets=True,
    )

    assert recovered.media_status == "complete"
    assert recovered.media_error == ""
    assert recovered.plate_path == str(plate_target)
    assert recovered.image_path == str(vehicle_target)
    assert plate_target.read_bytes() == original_plate
    assert vehicle_target.read_bytes() == original_vehicle


def test_atomic_jpeg_fsyncs_parent_after_publish(tmp_path, monkeypatch):
    target = tmp_path / "nested" / "evidence.jpg"
    image = np.full((32, 64, 3), 90, dtype=np.uint8)
    fsynced = []

    def record_directory_fsync(directory: Path):
        fsynced.append((Path(directory), target.is_file()))

    monkeypatch.setattr(
        media_storage,
        "_fsync_directory",
        record_directory_fsync,
    )

    published = media_storage.write_jpeg_atomic(target, image)

    assert published == target
    # First persist creation of the nested directory in its parent; then
    # persist the final JPEG entry after the atomic publish.
    assert fsynced == [(tmp_path, False), (target.parent, True)]
    assert cv2.imread(str(target)) is not None
    assert list(target.parent.glob(".*.tmp")) == []


def test_post_publish_failure_removes_new_file_before_quota_rollback(
    tmp_path,
    monkeypatch,
):
    target = tmp_path / "nested" / "evidence.jpg"
    image = np.full((32, 64, 3), 90, dtype=np.uint8)
    closes = []
    fsync_calls = 0

    class Reservation:
        def close(self, **kwargs):
            closes.append(kwargs)

    def fail_first_directory_fsync(_directory):
        nonlocal fsync_calls
        fsync_calls += 1
        if target.is_file():
            raise OSError("directory fsync failed")

    monkeypatch.setattr(
        media_storage,
        "begin_media_write",
        lambda *_args, **_kwargs: Reservation(),
    )
    monkeypatch.setattr(
        media_storage,
        "_fsync_directory",
        fail_first_directory_fsync,
    )

    with pytest.raises(media_storage.MediaWriteError):
        media_storage.write_jpeg_atomic(target, image)

    assert not target.exists()
    assert closes == [{"success": False}]
    assert fsync_calls == 3
    assert list(target.parent.glob(".*.tmp")) == []


def test_existing_evidence_is_never_overwritten(tmp_path, monkeypatch):
    target = tmp_path / "evidence.jpg"
    target.write_bytes(b"original-evidence")
    image = np.full((32, 64, 3), 90, dtype=np.uint8)
    closes = []

    class Reservation:
        def close(self, **kwargs):
            closes.append(kwargs)

    monkeypatch.setattr(
        media_storage,
        "begin_media_write",
        lambda *_args, **_kwargs: Reservation(),
    )

    with pytest.raises(
        media_storage.MediaWriteError,
        match="refusing to overwrite",
    ):
        media_storage.write_jpeg_atomic(target, image)

    assert target.read_bytes() == b"original-evidence"
    assert closes == [{"success": False}]


def test_encoded_jpeg_validation_reads_dimensions_without_decompression():
    image = np.full((37, 91, 3), 120, dtype=np.uint8)
    payload = media_storage.encode_jpeg_bytes(image)

    assert media_storage.validate_encoded_jpeg_bytes(payload) == (91, 37)


def test_encoded_jpeg_validation_rejects_unsafe_header_dimensions():
    image = np.full((37, 91, 3), 120, dtype=np.uint8)
    payload = bytearray(media_storage.encode_jpeg_bytes(image))
    marker = next(
        index
        for index in range(2, len(payload) - 9)
        if payload[index] == 0xFF
        and payload[index + 1]
        in media_storage._JPEG_START_OF_FRAME_MARKERS
    )
    # SOF layout: marker, segment length, precision, height, width.
    payload[marker + 5 : marker + 9] = b"\xff\xff\xff\xff"

    with pytest.raises(
        media_storage.MediaWriteError,
        match="dimensions are unsafe",
    ):
        media_storage.validate_encoded_jpeg_bytes(bytes(payload))


def test_atomic_encoded_jpeg_publish_uses_normal_quota_path(tmp_path):
    image = np.full((37, 91, 3), 120, dtype=np.uint8)
    payload = media_storage.encode_jpeg_bytes(image)
    target = tmp_path / "encoded" / "evidence.jpg"

    published = media_storage.write_jpeg_bytes_atomic(target, payload)

    assert published == target
    assert target.read_bytes() == payload
