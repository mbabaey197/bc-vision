import csv
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from app.ai.plate_rules import ALLOWED_PLATE_LETTERS, plausible_plate
from app.ai.training import operator_dataset_fingerprint
from tools.generate_cct_synthetic_dataset import (
    CONDITION_PROFILES,
    FOCUS_LETTERS,
    _base_plate,
    _degrade,
    _display_letter,
    _style_for_letter,
    _unique_plates,
    _validate_font_rendering,
    _validate_font_stack,
    generate,
    main as generate_main,
)
from tools.iran_plate_renderer import (
    BASE_PLATE_SIZE,
    LAYOUT_PROFILE,
    REFERENCE_GEOMETRY,
    SPECIAL_LAYOUT_PROFILE,
)
from tools.prepare_cct_dataset import _load_rows, prepare
from tools.train_fastplate_cct import (
    _copy_pretrained_backbone,
    _dataset_contract,
    _deployment_policy,
    _empty_metric_bucket,
    _finalize_metric_bucket,
    _training_plate_config,
    _update_metric_bucket,
)


def _font() -> Path:
    path = Path(
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    )
    if not path.is_file():
        pytest.skip("DejaVu Sans is not installed")
    return path


def _plate_texts(path: Path) -> set[str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {
            row["plate_text"]
            for row in csv.DictReader(handle)
        }


def _operator_manifest(samples, *, golden=False) -> dict:
    return {
        "schema": 2,
        "training_source": "operator-confirmed-only",
        "source_license": "operator-confirmed-rights-unverified",
        "ownership_attested": False,
        "distribution_allowed": False,
        "license_evidence": "",
        "golden_benchmark_data": golden,
        "dataset_fingerprint": operator_dataset_fingerprint(samples),
        "samples": samples,
    }


def test_synthetic_cct_dataset_has_disjoint_plate_identities(tmp_path):
    output = tmp_path / "synthetic"

    manifest = generate(
        output=output,
        train_plates=20,
        validation_plates=5,
        views_per_plate=2,
        font=_font(),
        font_license="DejaVu-font-license",
        seed=20260728,
        jpeg_quality=82,
    )

    train = _plate_texts(output / "train" / "annotations.csv")
    validation = _plate_texts(output / "val" / "annotations.csv")
    assert manifest["identity_overlap"] == 0
    assert train.isdisjoint(validation)
    assert manifest["train_images"] == 40
    assert manifest["validation_images"] == 10
    contract = _dataset_contract(output)
    assert contract["manifest"]["golden_benchmark_data"] is False
    assert contract["synthetic_only"] is True
    assert _deployment_policy(contract) == {
        "usage_scope": "production-candidate",
        "distribution_allowed": True,
        "activation_allowed": False,
        "activation_gate": "independent-real-camera-pass",
    }


def test_synthetic_cct_test_split_is_held_out(tmp_path):
    output = tmp_path / "synthetic"

    manifest = generate(
        output=output,
        train_plates=20,
        validation_plates=5,
        test_plates=7,
        views_per_plate=2,
        font=_font(),
        font_license="DejaVu-font-license",
        seed=20260730,
        jpeg_quality=82,
    )

    train = _plate_texts(output / "train" / "annotations.csv")
    validation = _plate_texts(output / "val" / "annotations.csv")
    test = _plate_texts(output / "test" / "annotations.csv")
    assert manifest["identity_overlap"] == 0
    assert manifest["identity_overlaps"] == {
        "train_validation": 0,
        "train_test": 0,
        "validation_test": 0,
    }
    assert train.isdisjoint(validation)
    assert train.isdisjoint(test)
    assert validation.isdisjoint(test)
    assert manifest["test_unique_plates"] == 7
    assert manifest["test_images"] == 14
    assert _dataset_contract(output)["test"] == (
        output / "test" / "annotations.csv"
    )


def test_synthetic_contract_rejects_renderer_hash_tampering(tmp_path):
    output = tmp_path / "synthetic"
    generate(
        output=output,
        train_plates=10,
        validation_plates=5,
        views_per_plate=1,
        font=_font(),
        font_license="DejaVu-font-license",
        seed=20260730,
        jpeg_quality=82,
    )
    manifest_path = output / "dataset-license.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["renderer_sha256"] = "0" * 64
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="provenance"):
        _dataset_contract(output)


def test_synthetic_generator_rejects_unapproved_font_license(tmp_path):
    with pytest.raises(ValueError, match="Font license is not approved"):
        generate(
            output=tmp_path / "synthetic",
            train_plates=20,
            validation_plates=5,
            views_per_plate=1,
            font=_font(),
            font_license="gpl-3.0",
            seed=20260728,
            jpeg_quality=82,
        )


def test_synthetic_generator_cli_requires_license_for_custom_font(tmp_path):
    with pytest.raises(SystemExit):
        generate_main([
            "--output",
            str(tmp_path / "synthetic"),
            "--font",
            str(_font()),
        ])


def test_synthetic_generator_cli_rejects_license_override_for_default_font(
    tmp_path,
):
    with pytest.raises(SystemExit):
        generate_main([
            "--output",
            str(tmp_path / "synthetic"),
            "--font-license",
            "bcvision-company-owned",
        ])


def test_synthetic_generator_rejects_font_without_persian_glyphs():
    font = Path(
        "/usr/share/fonts/opentype/urw-base35/NimbusSans-Regular.otf"
    )
    if not font.is_file():
        pytest.skip("A Latin-only font fixture is not installed")

    with pytest.raises(ValueError, match="missing required"):
        _validate_font_rendering(font)


def test_synthetic_plate_generation_balances_every_supported_letter():
    import random

    plates = _unique_plates(
        len(ALLOWED_PLATE_LETTERS),
        random.Random(20260728),
    )

    assert {plate[2] for plate in plates} == set(
        ALLOWED_PLATE_LETTERS
    )


def test_synthetic_generator_covers_conditions_and_cct_shape(tmp_path):
    output = tmp_path / "synthetic"
    manifest = generate(
        output=output,
        train_plates=len(CONDITION_PROFILES),
        validation_plates=5,
        views_per_plate=1,
        font=_font(),
        font_license="DejaVu-font-license",
        seed=20260728,
        jpeg_quality=82,
    )

    assert manifest["schema"] == 3
    assert manifest["layout_profile"] == LAYOUT_PROFILE
    assert manifest["renderer"] == "iran_plate_renderer.py"
    assert len(manifest["renderer_sha256"]) == 64
    assert manifest["layout_profiles"] == {
        "private": LAYOUT_PROFILE,
        "special": SPECIAL_LAYOUT_PROFILE,
    }
    assert manifest["base_plate_width"] == BASE_PLATE_SIZE[0]
    assert manifest["base_plate_height"] == BASE_PLATE_SIZE[1]
    assert manifest["output_width"] == 128
    assert manifest["output_height"] == 64
    assert set(manifest["train_conditions"]) == set(CONDITION_PROFILES)
    assert manifest["condition_profile_versions"][
        "overexposed_defocus"
    ] == "rear-plate-overexposed-defocus-v1"
    assert all(
        count == 1
        for count in manifest["train_conditions"].values()
    )
    with (output / "train" / "annotations.csv").open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == len(CONDITION_PROFILES)
    assert {row["condition_profile"] for row in rows} == set(
        CONDITION_PROFILES
    )
    for row in rows:
        assert plausible_plate(row["plate_text"])
        image = cv2.imread(
            str(output / "train" / row["image_path"]),
            cv2.IMREAD_COLOR,
        )
        assert image is not None
        assert image.shape == (64, 128, 3)
        assert 0.0 <= float(row["quality_score"]) <= 1.0


def test_overexposed_defocus_is_bounded_and_reproducible():
    import random

    plate = _base_plate(
        "28ف46195",
        _font(),
        random.Random(20260730),
        calibration=True,
    )
    first, first_metadata = _degrade(
        plate,
        random.Random(7081),
        profile="overexposed_defocus",
        return_metadata=True,
    )
    second, second_metadata = _degrade(
        plate,
        random.Random(7081),
        profile="overexposed_defocus",
        return_metadata=True,
    )

    assert np.array_equal(first, second)
    assert first_metadata == second_metadata
    assert first.shape == (64, 128, 3)
    assert first_metadata["degradation_profile_version"] == (
        "rear-plate-overexposed-defocus-v1"
    )
    assert first_metadata["difficulty"] == "hard"
    assert 0.40 <= first_metadata["defocus_sigma_output_px"] <= 1.45
    assert 0.10 <= first_metadata["exposure_ev"] <= 0.75
    assert 0.05 <= first_metadata["bloom_strength"] <= 0.16
    assert first_metadata["edge_retention"] >= 0.20
    assert "motion_blur_length" not in first_metadata


def test_synthetic_generator_is_byte_reproducible(tmp_path):
    outputs = [tmp_path / "first", tmp_path / "second"]
    for output in outputs:
        generate(
            output=output,
            train_plates=20,
            validation_plates=5,
            views_per_plate=2,
            font=_font(),
            font_license="DejaVu-font-license",
            seed=77,
            jpeg_quality=82,
        )

    for split in ("train", "val"):
        first_annotations = (
            outputs[0] / split / "annotations.csv"
        ).read_bytes()
        second_annotations = (
            outputs[1] / split / "annotations.csv"
        ).read_bytes()
        assert first_annotations == second_annotations
        first_images = sorted((outputs[0] / split / "images").glob("*.jpg"))
        second_images = sorted((outputs[1] / split / "images").glob("*.jpg"))
        assert [path.name for path in first_images] == [
            path.name for path in second_images
        ]
        assert [
            hashlib.sha256(path.read_bytes()).hexdigest()
            for path in first_images
        ] == [
            hashlib.sha256(path.read_bytes()).hexdigest()
            for path in second_images
        ]


def test_synthetic_generator_oversamples_known_weak_letters():
    import random

    plates = _unique_plates(
        250,
        random.Random(20260728),
    )
    counts = {
        letter: sum(plate[2] == letter for plate in plates)
        for letter in ALLOWED_PLATE_LETTERS
    }
    focus_average = sum(
        counts[letter] for letter in FOCUS_LETTERS
    ) / len(FOCUS_LETTERS)
    other = [
        letter
        for letter in ALLOWED_PLATE_LETTERS
        if letter not in FOCUS_LETTERS
    ]
    other_average = sum(counts[letter] for letter in other) / len(other)

    assert focus_average > other_average * 1.7


def test_synthetic_generator_rejects_unknown_condition(tmp_path):
    with pytest.raises(ValueError, match="Unknown condition"):
        generate(
            output=tmp_path / "synthetic",
            train_plates=20,
            validation_plates=5,
            views_per_plate=1,
            font=_font(),
            font_license="DejaVu-font-license",
            seed=20260728,
            jpeg_quality=82,
            profiles=("clean", "snowstorm"),
        )


def test_synthetic_plate_styles_cover_special_classes():
    assert _style_for_letter("ب") == "private"
    assert _style_for_letter("ع") == "public"
    assert _style_for_letter("ا") == "government"
    assert _style_for_letter("ث") == "military"
    assert _style_for_letter("D") == "diplomatic"
    assert _style_for_letter("S") == "service"
    assert _display_letter("ا") == "الف"
    assert _display_letter("ژ") == "♿"


def test_private_plate_renderer_uses_measured_fixed_slot_geometry():
    import random

    image = _base_plate(
        "28ف46195",
        _font(),
        random.Random(20260730),
        calibration=True,
    )

    assert image.size == BASE_PLATE_SIZE
    width = REFERENCE_GEOMETRY["canvas"][2]
    blue = REFERENCE_GEOMETRY["blue_band"]
    assert 0.09 <= (blue[2] - blue[0]) / width <= 0.105
    assert (
        0.795
        <= REFERENCE_GEOMETRY["region_separator_x"] / width
        <= 0.81
    )
    prefix_centers = [
        (left + right) / 2 / width
        for left, _, right, _ in REFERENCE_GEOMETRY["prefix_cells"]
    ]
    letter = REFERENCE_GEOMETRY["letter_cell"]
    letter_center = (letter[0] + letter[2]) / 2 / width
    serial_centers = [
        (left + right) / 2 / width
        for left, _, right, _ in REFERENCE_GEOMETRY["serial_cells"]
    ]
    region_centers = [
        (left + right) / 2 / width
        for left, _, right, _ in REFERENCE_GEOMETRY["region_cells"]
    ]
    assert prefix_centers == pytest.approx([0.1583, 0.2512], abs=0.003)
    assert 0.40 <= letter_center <= 0.43
    assert serial_centers == pytest.approx(
        [0.5464, 0.6488, 0.75],
        abs=0.004,
    )
    assert region_centers == pytest.approx(
        [0.8601, 0.9494],
        abs=0.004,
    )

    array = np.asarray(image)
    assert array[35:80, 8:35, 2].mean() > array[35:80, 8:35, 0].mean()
    separator_x = round(
        REFERENCE_GEOMETRY["region_separator_x"]
        / REFERENCE_GEOMETRY["canvas"][2]
        * BASE_PLATE_SIZE[0]
    )
    assert array[10:-10, separator_x - 1:separator_x + 2].mean() < 120


def test_font_stack_allows_an_approved_fallback_for_missing_glyphs():
    primary = Path(
        "/usr/share/fonts/opentype/urw-base35/NimbusSans-Regular.otf"
    )
    if not primary.is_file():
        pytest.skip("A Latin-only primary font fixture is not installed")

    _validate_font_stack(primary, _font())


def test_synthetic_generator_cli_requires_paired_fallback_license(tmp_path):
    with pytest.raises(SystemExit):
        generate_main([
            "--output",
            str(tmp_path / "synthetic"),
            "--fallback-font",
            str(_font()),
        ])

    with pytest.raises(SystemExit):
        generate_main([
            "--output",
            str(tmp_path / "synthetic"),
            "--fallback-font-license",
            "DejaVu-font-license",
        ])


def test_company_crop_import_rejects_golden_benchmark_rows(tmp_path):
    image = tmp_path / "plate.png"
    assert cv2.imwrite(
        str(image),
        np.zeros((32, 128, 3), dtype=np.uint8),
    )
    source = tmp_path / "source.csv"
    source.write_text(
        "image_path,plate_text,group_id,source_license,usage\n"
        "plate.png,31ط55674,track-1,bcvision-company-owned,golden\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Golden/benchmark"):
        _load_rows(source)


def test_company_crop_import_rejects_unapproved_license(tmp_path):
    image = tmp_path / "plate.png"
    assert cv2.imwrite(
        str(image),
        np.zeros((32, 128, 3), dtype=np.uint8),
    )
    source = tmp_path / "source.csv"
    source.write_text(
        "image_path,plate_text,group_id,source_license,usage\n"
        "plate.png,31ط55674,track-1,gpl-3.0,train\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Unapproved source license"):
        _load_rows(source)


def test_company_crop_import_rejects_conflicting_track_labels(tmp_path):
    for index in range(2):
        image = tmp_path / f"plate-{index}.png"
        assert cv2.imwrite(
            str(image),
            np.full((32, 128, 3), index, dtype=np.uint8),
        )
    source = tmp_path / "source.csv"
    source.write_text(
        "image_path,plate_text,group_id,source_license,usage\n"
        "plate-0.png,31ط55674,track-a,bcvision-company-owned,train\n"
        "plate-1.png,55ط63974,track-a,bcvision-company-owned,train\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="multiple plate identities"):
        _load_rows(source)


def test_operator_feedback_json_is_accepted_as_cct_source(tmp_path):
    image = tmp_path / "confirmed.png"
    assert cv2.imwrite(
        str(image),
        np.full((32, 128, 3), 120, dtype=np.uint8),
    )
    digest = hashlib.sha256(image.read_bytes()).hexdigest().upper()
    source = tmp_path / "operator-feedback.json"
    samples = [{
        "feedback_id": 1,
        "image_path": str(image),
        "plate": "31ط55674",
        "group_id": "31ط55674",
        "sha256": digest,
        "split": "train",
    }]
    source.write_text(
        json.dumps(_operator_manifest(samples), ensure_ascii=False),
        encoding="utf-8",
    )

    rows = _load_rows(source)

    assert len(rows) == 1
    assert rows[0]["plate_text"] == "31ط55674"
    assert rows[0]["source_license"] == (
        "operator-confirmed-rights-unverified"
    )
    assert rows[0]["sha256"] == digest


def test_operator_feedback_json_rejects_golden_or_changed_crop(tmp_path):
    image = tmp_path / "confirmed.png"
    assert cv2.imwrite(
        str(image),
        np.full((32, 128, 3), 120, dtype=np.uint8),
    )
    source = tmp_path / "operator-feedback.json"
    samples = [{
        "feedback_id": 1,
        "image_path": str(image),
        "plate": "31ط55674",
        "group_id": "31ط55674",
        "sha256": "0" * 64,
        "split": "train",
    }]
    source.write_text(
        json.dumps(
            _operator_manifest(samples, golden=True),
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="non-Golden"):
        _load_rows(source)

    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["golden_benchmark_data"] = False
    source.write_text(
        json.dumps(payload, ensure_ascii=False),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="hash mismatch"):
        _load_rows(source)


def test_cct_prepare_preserves_declared_operator_splits(tmp_path):
    images = []
    for name, value in (("train.png", 90), ("validation.png", 180)):
        image = tmp_path / name
        assert cv2.imwrite(
            str(image),
            np.full((32, 128, 3), value, dtype=np.uint8),
        )
        images.append(image)
    source = tmp_path / "operator-feedback.json"
    samples = [
        {
            "feedback_id": index,
            "image_path": image.name,
            "plate": plate,
            "group_id": plate,
            "sha256": hashlib.sha256(
                image.read_bytes()
            ).hexdigest().upper(),
            "split": split,
        }
        for index, (image, plate, split) in enumerate(
            (
                (images[0], "31ط55674", "train"),
                (images[1], "55ط63974", "validation"),
            ),
            1,
        )
    ]
    source.write_text(
        json.dumps(_operator_manifest(samples), ensure_ascii=False),
        encoding="utf-8",
    )

    manifest = prepare(
        source_manifest=source,
        output=tmp_path / "prepared",
        validation_ratio=0.49,
        seed=999,
    )

    assert manifest["train_samples"] == 1
    assert manifest["validation_samples"] == 1
    assert (
        manifest["source_license"]
        == "operator-confirmed-rights-unverified"
    )
    assert manifest["ownership_attested"] is False
    assert manifest["distribution_allowed"] is False
    assert (
        manifest["activation_policy"]
        == "shadow-only-until-rights-attested"
    )
    assert _plate_texts(
        tmp_path / "prepared" / "train" / "annotations.csv"
    ) == {"31ط55674"}
    assert _plate_texts(
        tmp_path / "prepared" / "val" / "annotations.csv"
    ) == {"55ط63974"}


def test_operator_feedback_json_rejects_changed_label_metadata(tmp_path):
    image = tmp_path / "confirmed.png"
    assert cv2.imwrite(
        str(image),
        np.full((32, 128, 3), 120, dtype=np.uint8),
    )
    samples = [{
        "feedback_id": 1,
        "image_path": image.name,
        "plate": "31ط55674",
        "group_id": "31ط55674",
        "sha256": hashlib.sha256(
            image.read_bytes()
        ).hexdigest().upper(),
        "split": "train",
    }]
    payload = _operator_manifest(samples)
    payload["samples"][0]["plate"] = "31ط56674"
    source = tmp_path / "operator-feedback.json"
    source.write_text(
        json.dumps(payload, ensure_ascii=False),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="fingerprint mismatch"):
        _load_rows(source)


def test_operator_company_ownership_requires_explicit_evidence(tmp_path):
    image = tmp_path / "confirmed.png"
    assert cv2.imwrite(
        str(image),
        np.full((32, 128, 3), 120, dtype=np.uint8),
    )
    samples = [{
        "feedback_id": 1,
        "image_path": image.name,
        "plate": "31ط55674",
        "group_id": "31ط55674",
        "sha256": hashlib.sha256(
            image.read_bytes()
        ).hexdigest().upper(),
        "split": "train",
    }]
    payload = _operator_manifest(samples)
    payload.update({
        "source_license": "operator-confirmed-company-owned",
        "ownership_attested": False,
        "distribution_allowed": True,
    })
    source = tmp_path / "operator-feedback.json"
    source.write_text(
        json.dumps(payload, ensure_ascii=False),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="ownership attestation"):
        _load_rows(source)

    payload["ownership_attested"] = True
    payload["license_evidence"] = "camera-fleet-rights-attestation-1"
    source.write_text(
        json.dumps(payload, ensure_ascii=False),
        encoding="utf-8",
    )
    rows = _load_rows(source)

    assert rows[0]["source_license"] == (
        "operator-confirmed-company-owned"
    )
    assert rows[0]["distribution_allowed"] is True


def test_operator_feedback_json_rejects_crop_outside_snapshot(tmp_path):
    snapshot = tmp_path / "anpr-training"
    manifests = snapshot / "manifests"
    manifests.mkdir(parents=True)
    golden = tmp_path / "anpr-golden"
    golden.mkdir()
    image = golden / "forbidden.png"
    assert cv2.imwrite(
        str(image),
        np.full((32, 128, 3), 120, dtype=np.uint8),
    )
    samples = [{
        "feedback_id": 1,
        "image_path": "../../anpr-golden/forbidden.png",
        "plate": "31ط55674",
        "group_id": "31ط55674",
        "sha256": hashlib.sha256(
            image.read_bytes()
        ).hexdigest().upper(),
        "split": "train",
    }]
    source = manifests / "run-1.json"
    source.write_text(
        json.dumps(_operator_manifest(samples), ensure_ascii=False),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="escapes snapshot root"):
        _load_rows(source)


def test_training_contract_rejects_golden_manifest(tmp_path):
    dataset = tmp_path / "dataset"
    (dataset / "train").mkdir(parents=True)
    (dataset / "val").mkdir()
    for split in ("train", "val"):
        (dataset / split / "annotations.csv").write_text(
            "image_path,plate_text\n",
            encoding="utf-8",
        )
    (dataset / "dataset-license.json").write_text(
        json.dumps({
            "source_license": "bcvision-company-owned",
            "golden_benchmark_data": True,
        }),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Golden benchmark"):
        _dataset_contract(dataset)


@pytest.mark.parametrize("value", [None, 0, "", "false"])
def test_training_contract_requires_explicit_non_golden_false(
    tmp_path,
    value,
):
    dataset = tmp_path / "dataset"
    (dataset / "train").mkdir(parents=True)
    (dataset / "val").mkdir()
    for split in ("train", "val"):
        (dataset / split / "annotations.csv").write_text(
            "image_path,plate_text\n",
            encoding="utf-8",
        )
    payload = {"source_license": "bcvision-company-owned"}
    if value is not None:
        payload["golden_benchmark_data"] = value
    (dataset / "dataset-license.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="explicitly forbid"):
        _dataset_contract(dataset)


def test_training_contract_rejects_synthetic_font_license(tmp_path):
    dataset = tmp_path / "dataset"
    (dataset / "train").mkdir(parents=True)
    (dataset / "val").mkdir()
    for split in ("train", "val"):
        (dataset / split / "annotations.csv").write_text(
            "image_path,plate_text\n",
            encoding="utf-8",
        )
    (dataset / "dataset-license.json").write_text(
        json.dumps({
            "source_license": "synthetic-bcvision-company-owned",
            "third_party_plate_dataset": False,
            "golden_benchmark_data": False,
            "font_license": "gpl-3.0",
        }),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="font license"):
        _dataset_contract(dataset)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema", 1),
        ("third_party_plate_dataset", 0),
        ("procedural_only", 1),
        ("real_plate_pixels_used", 0),
    ],
)
def test_training_contract_requires_exact_synthetic_provenance(
    tmp_path,
    field,
    value,
):
    dataset = tmp_path / "dataset"
    (dataset / "train").mkdir(parents=True)
    (dataset / "val").mkdir()
    for split in ("train", "val"):
        (dataset / split / "annotations.csv").write_text(
            "image_path,plate_text\n",
            encoding="utf-8",
        )
    payload = {
        "schema": 2,
        "source_license": "synthetic-bcvision-company-owned",
        "golden_benchmark_data": False,
        "third_party_plate_dataset": False,
        "procedural_only": True,
        "real_plate_pixels_used": False,
        "font_license": "dejavu-font-license",
    }
    payload[field] = value
    (dataset / "dataset-license.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Synthetic data provenance"):
        _dataset_contract(dataset)


def test_company_dataset_keeps_plate_identity_in_one_split(tmp_path):
    for index in range(4):
        image = tmp_path / f"plate-{index}.png"
        assert cv2.imwrite(
            str(image),
            np.full((32, 128, 3), index, dtype=np.uint8),
        )
    source = tmp_path / "source.csv"
    source.write_text(
        "image_path,plate_text,group_id,source_license,usage\n"
        "plate-0.png,31ط55674,track-a,bcvision-company-owned,train\n"
        "plate-1.png,31ط55674,track-b,bcvision-company-owned,train\n"
        "plate-2.png,55ط63974,track-c,bcvision-company-owned,train\n"
        "plate-3.png,84ب57133,track-d,bcvision-company-owned,train\n",
        encoding="utf-8",
    )

    manifest = prepare(
        source,
        tmp_path / "prepared",
        validation_ratio=0.5,
        seed=20260728,
    )

    assert manifest["plate_identity_overlap"] == 0


class _FakeLayer:
    def __init__(self, name, weights):
        self.name = name
        self._weights = [np.asarray(weight) for weight in weights]

    def get_weights(self):
        return [weight.copy() for weight in self._weights]

    def set_weights(self, weights):
        self._weights = [np.asarray(weight).copy() for weight in weights]


class _FakeModel:
    def __init__(self, layers):
        self.layers = layers


def test_pretrained_transfer_copies_backbone_but_never_ocr_head():
    source = _FakeModel([
        _FakeLayer("backbone", [np.full((2, 2), 7)]),
        _FakeLayer("plate", [np.full((2, 3), 9)]),
    ])
    target = _FakeModel([
        _FakeLayer("backbone", [np.zeros((2, 2))]),
        _FakeLayer("plate", [np.zeros((2, 3))]),
    ])

    transferred = _copy_pretrained_backbone(source, target)

    assert transferred == ["backbone"]
    assert np.all(target.layers[0].get_weights()[0] == 7)
    assert np.all(target.layers[1].get_weights()[0] == 0)


def test_pretrained_transfer_rejects_different_backbone_shape():
    source = _FakeModel([
        _FakeLayer("backbone", [np.zeros((3, 2))]),
    ])
    target = _FakeModel([
        _FakeLayer("backbone", [np.zeros((2, 2))]),
    ])

    with pytest.raises(ValueError, match="does not match"):
        _copy_pretrained_backbone(source, target)


def test_cct_metric_bucket_reports_acceptance_and_character_accuracy():
    bucket = _empty_metric_bucket()
    _update_metric_bucket(
        bucket,
        expected="12ب34567",
        raw="12ب34567",
        result={"accepted": True, "plate_norm": "12ب34567"},
    )
    _update_metric_bucket(
        bucket,
        expected="98ک76543",
        raw="98گ76543",
        result={"accepted": False, "plate_norm": ""},
    )

    metrics = _finalize_metric_bucket(bucket)

    assert metrics["samples"] == 2
    assert metrics["raw_exact_accuracy"] == 0.5
    assert metrics["raw_character_accuracy"] == 0.9375
    assert metrics["accepted_precision"] == 1.0
    assert metrics["rejection_rate"] == 0.5


def test_dual_view_training_uses_aspect_preserving_plate_config():
    root = Path(__file__).resolve().parents[1]

    legacy = _training_plate_config(root, "stretch-v1")
    dual = _training_plate_config(
        root,
        "stretch-letterbox-geomean-v1",
    )

    assert legacy.name == "iran_plate_config.yaml"
    assert "keep_aspect_ratio: false" in legacy.read_text(
        encoding="utf-8"
    )
    assert dual.name == "iran_plate_letterbox_config.yaml"
    assert "keep_aspect_ratio: true" in dual.read_text(
        encoding="utf-8"
    )


def test_training_rejects_pretrained_and_resume_together(tmp_path):
    from tools.train_fastplate_cct import train_and_export

    dataset = tmp_path / "dataset"
    output = tmp_path / "output"
    pretrained = tmp_path / "pretrained.keras"
    resume = tmp_path / "resume.keras"
    pretrained.write_bytes(b"pretrained")
    resume.write_bytes(b"resume")

    with pytest.raises(ValueError, match="either pretrained backbone"):
        train_and_export(
            dataset=dataset,
            output=output,
            variant="xs",
            pretrained_backbone=pretrained,
            resume_checkpoint=resume,
            checkpoint_metric="char",
            epochs=4,
            batch_size=4,
            seed=1,
        )


def test_company_cct_candidate_stays_locked_until_independent_gates():
    policy = _deployment_policy({
        "research_only": False,
        "synthetic_only": False,
    })

    assert policy == {
        "usage_scope": "production-candidate",
        "distribution_allowed": True,
        "activation_allowed": False,
        "activation_gate": "independent-golden-and-real-camera-pass",
    }


def test_unverified_operator_candidate_is_non_distributable():
    policy = _deployment_policy({
        "research_only": False,
        "rights_unverified": True,
        "synthetic_only": False,
    })

    assert policy == {
        "usage_scope": "research-shadow-only",
        "distribution_allowed": False,
        "activation_allowed": False,
        "activation_gate": "rights-attestation-and-real-camera-pass",
    }
