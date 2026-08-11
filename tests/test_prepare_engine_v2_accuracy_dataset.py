from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from app.engine_v2.benchmark import REQUIRED_ACCURACY_CATEGORIES
from tools.prepare_engine_v2_accuracy_dataset import prepare_accuracy_dataset


def _draft(tmp_path: Path, *, all_categories: bool = True) -> Path:
    categories = (
        REQUIRED_ACCURACY_CATEGORIES if all_categories else ("multiple_vehicles",)
    )
    samples = []
    for index, category in enumerate(categories):
        media = tmp_path / f"source-{index}.mp4"
        media.write_bytes(f"verified-media-{category}".encode("utf-8"))
        sample = {
            "id": f"sample-{index}",
            "category": category,
            "input": {"path": media.name, "media_type": "video"},
            "label_status": "verified",
            "label_source": "operator-test",
        }
        if category == "multiple_vehicles":
            sample["expected_events"] = [
                {"plate": "12ب34567", "start_ms": 0, "end_ms": 500},
                {"plate": "34د76543", "start_ms": 501, "end_ms": 1000},
            ]
        else:
            sample["expected_plate"] = "12ب34567"
        samples.append(sample)
    negative = tmp_path / "negative.mp4"
    negative.write_bytes(b"verified-empty-lane")
    samples.append(
        {
            "id": "negative-0",
            "category": "clear_plate",
            "input": {"path": negative.name, "media_type": "video"},
            "expected_plate": None,
            "label_status": "verified",
            "label_source": "operator-test",
        }
    )
    path = tmp_path / "draft.json"
    path.write_text(
        json.dumps(
            {
                "schema": "bcvision.anpr.accuracy-manifest/v1",
                "dataset_id": "unit-test-dataset",
                "training_allowed": False,
                "label_source": "operator-test",
                "samples": samples,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return path


def test_preparer_copies_content_addressed_media_and_binds_every_byte(tmp_path: Path) -> None:
    draft = _draft(tmp_path)
    output = tmp_path / "immutable-dataset"

    result = prepare_accuracy_dataset(draft, output)

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    integrity = json.loads((output / "integrity.json").read_text(encoding="utf-8"))
    assert result["coverage_complete"] is True
    assert manifest["training_allowed"] is False
    assert len(manifest["samples"]) == len(REQUIRED_ACCURACY_CATEGORIES) + 1
    assert integrity["dataset_fingerprint_sha256"] == manifest[
        "dataset_fingerprint_sha256"
    ]
    for sample in manifest["samples"]:
        media = output / sample["input"]["path"]
        assert media.is_file()
        assert hashlib.sha256(media.read_bytes()).hexdigest() == sample["input"][
            "sha256"
        ]
        assert media.stat().st_size == sample["input"]["size_bytes"]
        assert media.name.startswith(sample["input"]["sha256"])


def test_preparer_output_is_deterministic_for_same_reviewed_inputs(tmp_path: Path) -> None:
    draft = _draft(tmp_path)
    first = tmp_path / "first"
    second = tmp_path / "second"

    one = prepare_accuracy_dataset(draft, first)
    two = prepare_accuracy_dataset(draft, second)

    assert one["dataset_fingerprint_sha256"] == two["dataset_fingerprint_sha256"]
    assert (first / "manifest.json").read_bytes() == (second / "manifest.json").read_bytes()
    assert (first / "integrity.json").read_bytes() == (second / "integrity.json").read_bytes()


def test_preparer_refuses_unverified_or_symlinked_media(tmp_path: Path) -> None:
    draft = _draft(tmp_path, all_categories=False)
    payload = json.loads(draft.read_text(encoding="utf-8"))
    payload["samples"][0]["label_status"] = "pending"
    draft.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="label_status"):
        prepare_accuracy_dataset(
            draft,
            tmp_path / "pending",
            require_all_categories=False,
            require_negative_sample=False,
        )

    payload["samples"][0]["label_status"] = "verified"
    source = tmp_path / payload["samples"][0]["input"]["path"]
    link = tmp_path / "linked.mp4"
    try:
        link.symlink_to(source)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable on this platform")
    payload["samples"][0]["input"]["path"] = link.name
    draft.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="symlink"):
        prepare_accuracy_dataset(
            draft,
            tmp_path / "linked",
            require_all_categories=False,
            require_negative_sample=False,
        )


def test_preparer_honors_reviewed_source_hash_and_size(tmp_path: Path) -> None:
    draft = _draft(tmp_path, all_categories=False)
    payload = json.loads(draft.read_text(encoding="utf-8"))
    sample = payload["samples"][0]
    source = tmp_path / sample["input"]["path"]
    sample["input"]["sha256"] = "0" * 64
    sample["input"]["size_bytes"] = source.stat().st_size
    draft.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="reviewed SHA-256"):
        prepare_accuracy_dataset(
            draft,
            tmp_path / "hash-mismatch",
            require_all_categories=False,
            require_negative_sample=False,
        )

    sample["input"]["sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    sample["input"]["size_bytes"] += 1
    draft.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="reviewed byte size"):
        prepare_accuracy_dataset(
            draft,
            tmp_path / "size-mismatch",
            require_all_categories=False,
            require_negative_sample=False,
        )


def test_partial_dataset_is_materialized_but_never_coverage_complete(tmp_path: Path) -> None:
    draft = _draft(tmp_path, all_categories=False)
    payload = json.loads(draft.read_text(encoding="utf-8"))
    payload["samples"] = payload["samples"][:1]
    draft.write_text(json.dumps(payload), encoding="utf-8")

    result = prepare_accuracy_dataset(
        draft,
        tmp_path / "partial",
        require_all_categories=False,
        require_negative_sample=False,
    )

    assert result["coverage_complete"] is False
    assert "night" in result["coverage"]["missing_categories"]
    assert result["coverage"]["negative_samples"] == 0


def test_preparer_never_overwrites_an_existing_dataset(tmp_path: Path) -> None:
    draft = _draft(tmp_path)
    output = tmp_path / "existing"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("preserve", encoding="utf-8")

    with pytest.raises(FileExistsError):
        prepare_accuracy_dataset(draft, output)

    assert marker.read_text(encoding="utf-8") == "preserve"


def test_coverage_complete_requires_readable_labels_in_every_category(tmp_path: Path) -> None:
    draft = _draft(tmp_path)
    payload = json.loads(draft.read_text(encoding="utf-8"))
    for sample in payload["samples"]:
        if "expected_plate" in sample:
            sample["expected_plate"] = None
    draft.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    result = prepare_accuracy_dataset(
        draft,
        tmp_path / "unreadable-categories",
        require_all_categories=False,
    )

    assert result["coverage_complete"] is False
    assert "night" in result["coverage"]["unreadable_categories"]
    assert result["coverage"]["readable_event_counts"]["night"] == 0


def test_preparer_rejects_any_per_sample_adapter_input(tmp_path: Path) -> None:
    draft = _draft(tmp_path, all_categories=False)
    payload = json.loads(draft.read_text(encoding="utf-8"))
    payload["samples"][0]["adapter_input"] = {
        "runtime": [{"expected_events": [{"plate": "12ب34567"}]}]
    }
    draft.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(ValueError, match="adapter_input is not permitted"):
        prepare_accuracy_dataset(
            draft,
            tmp_path / "leaking-adapter-input",
            require_all_categories=False,
            require_negative_sample=False,
        )


def test_preparer_rejects_symlinked_parent_directory(tmp_path: Path) -> None:
    draft = _draft(tmp_path, all_categories=False)
    payload = json.loads(draft.read_text(encoding="utf-8"))
    real_directory = tmp_path / "real-media"
    real_directory.mkdir()
    source = tmp_path / payload["samples"][0]["input"]["path"]
    moved = real_directory / source.name
    source.replace(moved)
    linked_directory = tmp_path / "linked-media"
    try:
        linked_directory.symlink_to(real_directory, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("directory symlinks unavailable on this platform")
    payload["samples"][0]["input"]["path"] = str(
        Path(linked_directory.name) / moved.name
    )
    draft.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(ValueError, match="symlink or junction"):
        prepare_accuracy_dataset(
            draft,
            tmp_path / "linked-parent",
            require_all_categories=False,
            require_negative_sample=False,
        )


@pytest.mark.parametrize("missing_field", ["training_allowed", "label_source"])
def test_preparer_requires_explicit_dataset_provenance(
    tmp_path: Path,
    missing_field: str,
) -> None:
    draft = _draft(tmp_path, all_categories=False)
    payload = json.loads(draft.read_text(encoding="utf-8"))
    payload.pop(missing_field)
    draft.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(ValueError, match=missing_field):
        prepare_accuracy_dataset(
            draft,
            tmp_path / f"missing-{missing_field}",
            require_all_categories=False,
            require_negative_sample=False,
        )


def test_preparer_rejects_boolean_timestamps_and_invalid_track_ids(tmp_path: Path) -> None:
    draft = _draft(tmp_path, all_categories=False)
    payload = json.loads(draft.read_text(encoding="utf-8"))
    payload["samples"][0]["input"]["start_ms"] = True
    draft.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError, match="input.start_ms"):
        prepare_accuracy_dataset(
            draft,
            tmp_path / "boolean-time",
            require_all_categories=False,
            require_negative_sample=False,
        )

    payload["samples"][0]["input"].pop("start_ms")
    payload["samples"][0]["expected_events"][0]["track_id"] = ["invalid"]
    draft.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError, match="track_id"):
        prepare_accuracy_dataset(
            draft,
            tmp_path / "invalid-track",
            require_all_categories=False,
            require_negative_sample=False,
        )


def test_schema_requires_prepared_media_identity_and_provenance() -> None:
    schema = json.loads(
        Path("docs/ANPR_ENGINE_V2_BENCHMARK_MANIFEST.schema.json").read_text(
            encoding="utf-8"
        )
    )
    assert {"label_source", "training_allowed"} <= set(schema["required"])
    required_input = set(schema["$defs"]["sample"]["properties"]["input"]["required"])
    assert {"path", "media_type", "sha256", "size_bytes"} <= required_input


def test_preparer_preserves_known_positive_scope_as_non_promotional(tmp_path: Path) -> None:
    draft = _draft(tmp_path, all_categories=False)
    payload = json.loads(draft.read_text(encoding="utf-8"))
    payload["samples"] = payload["samples"][:1]
    payload["samples"][0]["label_scope"] = "known_positives"
    draft.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    result = prepare_accuracy_dataset(
        draft,
        tmp_path / "known-positive",
        require_all_categories=False,
        require_negative_sample=False,
    )
    manifest = json.loads(
        (tmp_path / "known-positive" / "manifest.json").read_text(encoding="utf-8")
    )

    assert result["coverage_complete"] is False
    assert result["coverage"]["known_positive_samples"] == 1
    assert manifest["samples"][0]["label_scope"] == "known_positives"
