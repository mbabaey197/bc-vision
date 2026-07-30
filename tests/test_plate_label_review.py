import json
from pathlib import Path

from PIL import Image

from tools.build_plate_label_review import (
    _json_for_script,
    build_review_page,
    collect_samples,
)


def _write_image(path: Path, color, size=(128, 32)):
    Image.new("RGB", size, color).save(path, quality=92)


def test_review_page_keeps_shadow_suggestions_unconfirmed(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    _write_image(images / "a.jpg", (230, 190, 60))
    _write_image(images / "b.jpg", (40, 40, 40), size=(72, 18))
    archive = tmp_path / "source.zip"
    archive.write_bytes(b"private-source")
    suggestions = tmp_path / "suggestions.json"
    suggestions.write_text(
        json.dumps([{
            "file_name": "a.jpg",
            "proposed_plate": "12ع34567",
            "confidence": 0.91,
            "min_position_confidence": 0.62,
            "min_position_margin": 0.12,
            "layout_conflict": False,
        }]),
        encoding="utf-8",
    )
    output = tmp_path / "review.html"

    metadata = build_review_page(
        images,
        output,
        source_archive=archive,
        suggestions_path=suggestions,
        good_count=1,
        review_count=0,
        ownership_evidence=(
            "user-attestation-chat-2026-07-30-company-owned-01"
        ),
    )
    page = output.read_text(encoding="utf-8")

    assert metadata["image_count"] == 2
    assert metadata["ownership_attested"] is True
    assert metadata["model_suggestions_are_labels"] is False
    assert metadata["quality_buckets"] == {
        "good": 1,
        "review": 0,
        "hard": 1,
    }
    assert '"status":"untrusted-shadow-suggestion"' in page
    assert "پیشنهاد آزمایشی مدل" in page
    assert "operator-confirmed-only" in page
    assert "data:image/jpeg;base64," in page
    assert "localStorage" in page
    assert "bcvision-operator-plate-review" in page


def test_review_page_escapes_source_filename_from_script_context():
    script_json = _json_for_script({
        "file_name": "unsafe<script>.jpg",
    })

    assert "unsafe<script>" not in script_json
    assert "unsafe\\u003Cscript\\u003E.jpg" in script_json


def test_collect_samples_rejects_invalid_quality_bucket_counts(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    _write_image(images / "a.jpg", (255, 255, 255))

    try:
        collect_samples(images, good_count=2)
    except ValueError as exc:
        assert "exceed" in str(exc)
    else:
        raise AssertionError("invalid quality bucket counts were accepted")
