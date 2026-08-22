import hashlib
import json
from pathlib import Path
import shutil
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import pytest

from app.ai import model_manager, training
from app.ai.training_manifest import operator_dataset_fingerprint
from app.ai.training_worker import _load_manifest
from tools.prepare_cct_dataset import _load_rows


@pytest.fixture(autouse=True)
def _reset_pending_feedback_pins():
    training.close_pending_feedback_source_pins()
    yield
    training.close_pending_feedback_source_pins()


def test_training_digest_validation_requires_hexadecimal_sha256():
    assert training._valid_sha256("A" * 64) is True
    assert training._valid_sha256("G" * 64) is False
    assert training._valid_sha256("A" * 63) is False


def _isolated_database(tmp_path, monkeypatch):
    import app.config
    import app.database

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    db_path = data_dir / "bcvision.db"
    monkeypatch.setattr(app.config, "DATA_DIR", data_dir)
    monkeypatch.setattr(app.database, "DB_PATH", db_path)
    app.database.init_db()
    # Feedback crops in these unit tests live directly below tmp_path. Model
    # the same configured/current media-root invariant used in production so
    # capture_feedback_sample exercises the real read-pin policy.
    storage_root = tmp_path.parent
    with app.database.connect() as connection:
        connection.executemany(
            "INSERT INTO settings(key,value) VALUES(?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (
                ("storage_root", str(storage_root)),
                ("plate_path", str(tmp_path)),
                ("snapshot_path", str(storage_root / f"{tmp_path.name}-snapshots")),
                ("video_path", str(storage_root / f"{tmp_path.name}-videos")),
                ("backup_path", str(storage_root / f"{tmp_path.name}-backups")),
                ("media_roots_history", "[]"),
                ("max_storage_gb", "0"),
            ),
        )
    return app.database, data_dir


def _write_operator_manifest(
    path,
    plate="12ب34567",
    *,
    rights_verified=True,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    train_image = path.parent / "train-sample.png"
    validation_image = path.parent / "validation-sample.png"
    train_image.write_bytes(b"train-sample")
    validation_image.write_bytes(b"validation-sample")
    validation_plate = (
        "34پ56789" if plate != "34پ56789" else "56ت78901"
    )
    samples = [
        {
            "feedback_id": 1,
            "image_path": train_image.name,
            "sha256": hashlib.sha256(
                train_image.read_bytes()
            ).hexdigest().upper(),
            "plate": plate,
            "group_id": plate,
            "split": "train",
        },
        {
            "feedback_id": 2,
            "image_path": validation_image.name,
            "sha256": hashlib.sha256(
                validation_image.read_bytes()
            ).hexdigest().upper(),
            "plate": validation_plate,
            "group_id": validation_plate,
            "split": "validation",
        },
    ]
    path.write_text(
        json.dumps({
            "schema": 2,
            "training_source": "operator-confirmed-only",
            "source_license": (
                "operator-confirmed-company-owned"
                if rights_verified
                else "operator-confirmed-rights-unverified"
            ),
            "ownership_attested": bool(rights_verified),
            "distribution_allowed": bool(rights_verified),
            "license_evidence": (
                "test-company-camera-rights-attestation"
                if rights_verified
                else ""
            ),
            "golden_benchmark_data": False,
            "dataset_fingerprint": operator_dataset_fingerprint(samples),
            "samples": samples,
        }),
        encoding="utf-8",
    )
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _passing_golden_decision(digest="B" * 64):
    return {
        "promote": True,
        "reasons": [],
        "baseline_exact_accuracy": 0.90,
        "candidate_exact_accuracy": 0.95,
        "baseline_false_accept_rate": 0.0,
        "candidate_false_accept_rate": 0.0,
        "baseline_mean_character_error": 0.10,
        "candidate_mean_character_error": 0.05,
        "evaluation_kind": "verified-ocr-crop-golden",
        "golden_manifest_sha256": digest,
        "samples": 40,
    }


def _ready_golden(digest="B" * 64, plate="31ط55674"):
    return {
        "ready": True,
        "manifest_sha256": digest,
        "errors": [],
        "samples": 40,
        "unique_plates": 20,
        "slice_counts": {},
        "rows": [{
            "expected_plate": plate,
            "media_kind": "ocr-crop",
        }],
    }


def _candidate_ready_run(
    tmp_path,
    monkeypatch,
    *,
    manifest_rights_verified=True,
):
    database, data_dir = _isolated_database(tmp_path, monkeypatch)
    baseline = data_dir / "models" / "crnn" / "ocr_crnn.onnx"
    baseline.parent.mkdir(parents=True)
    baseline.write_bytes(b"verified-baseline")
    baseline_digest = hashlib.sha256(
        baseline.read_bytes()
    ).hexdigest().upper()
    monkeypatch.setattr(model_manager, "CRNN_SHA256", baseline_digest)
    monkeypatch.setattr(
        model_manager,
        "CRNN_SIZE",
        baseline.stat().st_size,
    )
    candidate = tmp_path / "candidate.onnx"
    candidate.write_bytes(b"candidate-crnn")
    candidate_digest = hashlib.sha256(
        candidate.read_bytes()
    ).hexdigest().upper()
    manifest = (
        data_dir
        / "anpr-training"
        / "manifests"
        / "run-1.json"
    )
    manifest_digest = _write_operator_manifest(
        manifest,
        rights_verified=manifest_rights_verified,
    )
    decision = _passing_golden_decision("C" * 64)
    report = {
        "schema": 1,
        "promote": True,
        "reasons": [],
        "baseline_sha256": baseline_digest,
        "validation_samples": 12,
        "baseline_accuracy": 0.80,
        "candidate_accuracy": 0.85,
        "baseline_mean_character_error": 0.40,
        "candidate_mean_character_error": 0.30,
        "validation_regressions": 0,
        "initialization_mode": "active-model-distillation",
        "training_rights_verified": True,
        "golden": {
            "ready": True,
            "manifest_sha256": "C" * 64,
            "samples": 40,
        },
        "golden_decision": decision,
    }
    from app.ai import golden

    current_golden = _ready_golden("C" * 64)
    monkeypatch.setattr(
        golden,
        "golden_status",
        lambda: current_golden,
    )
    with database.connect() as con:
        run_id = con.execute(
            "INSERT INTO anpr_training_runs("
            "status,candidate_path,candidate_sha256,"
            "promotion_report,dataset_manifest_path,"
            "dataset_manifest_sha256"
            ") VALUES('candidate-ready',?,?,?,?,?)",
            (
                str(candidate),
                candidate_digest,
                json.dumps(report),
                str(manifest),
                manifest_digest,
            ),
        ).lastrowid
    return {
        "database": database,
        "data_dir": data_dir,
        "run_id": run_id,
        "candidate_digest": candidate_digest,
        "decision": decision,
    }


def _pending_candidate_run(
    tmp_path,
    monkeypatch,
    *,
    training_plate="12ب34567",
    manifest_rights_verified=True,
):
    database, _data_dir = _isolated_database(tmp_path, monkeypatch)
    baseline = tmp_path / "baseline.onnx"
    candidate = tmp_path / "candidate.onnx"
    baseline.write_bytes(b"baseline")
    candidate.write_bytes(b"candidate")
    baseline_sha256 = hashlib.sha256(
        baseline.read_bytes()
    ).hexdigest().upper()
    candidate_sha256 = hashlib.sha256(
        candidate.read_bytes()
    ).hexdigest().upper()
    manifest = tmp_path / "dataset.json"
    manifest_sha256 = _write_operator_manifest(
        manifest,
        training_plate,
        rights_verified=manifest_rights_verified,
    )
    with database.connect() as con:
        run_id = con.execute(
            "INSERT INTO anpr_training_runs("
            "status,validation_samples,baseline_accuracy,"
            "candidate_accuracy,baseline_mean_character_error,"
            "candidate_mean_character_error,baseline_sha256,"
            "candidate_path,candidate_sha256,promotion_report,"
            "dataset_manifest_path,dataset_manifest_sha256"
            ") VALUES('awaiting-golden',?,?,?,?,?,?,?,?,?,?,?)",
            (
                12,
                0.80,
                0.85,
                0.40,
                0.30,
                baseline_sha256,
                str(candidate),
                candidate_sha256,
                json.dumps({
                    "validation_regressions": 0,
                    "initialization_mode":
                        "active-model-distillation",
                }),
                str(manifest),
                manifest_sha256,
            ),
        ).lastrowid
    monkeypatch.setattr(
        model_manager,
        "active_crnn_model",
        lambda: (
            baseline,
            baseline_sha256,
            baseline.stat().st_size,
        ),
    )
    return {
        "database": database,
        "baseline": baseline,
        "manifest": manifest,
        "run_id": run_id,
    }


def test_confirmed_feedback_becomes_verified_training_sample(
    tmp_path,
    monkeypatch,
):
    database, data_dir = _isolated_database(tmp_path, monkeypatch)
    image_path = tmp_path / "plate.jpg"
    assert cv2.imwrite(
        str(image_path),
        np.full((48, 180, 3), 170, dtype=np.uint8),
    )
    with database.connect() as con:
        event_id = con.execute(
            "INSERT INTO plate_events(plate_text,plate_norm) VALUES(?,?)",
            ("12-ب-345-67", "12ب34567"),
        ).lastrowid
        feedback_id = con.execute(
            "INSERT INTO anpr_feedback("
            "event_id,observed_text,observed_norm,corrected_text,"
            "corrected_norm,plate_image_path,status"
            ") VALUES(?,?,?,?,?,?,?)",
            (
                event_id,
                "12-ب-345-76",
                "12ب34576",
                "12-ب-345-67",
                "12ب34567",
                str(image_path),
                "confirmed",
            ),
        ).lastrowid

    result = training.capture_feedback_sample(feedback_id)

    assert result["ready"] is True
    sample = Path(result["path"])
    assert sample.is_file()
    assert sample.is_relative_to(data_dir / "anpr-training")
    assert hashlib.sha256(sample.read_bytes()).hexdigest().upper() == (
        result["sha256"]
    )
    with database.connect() as con:
        row = con.execute(
            "SELECT training_status,sample_path,sample_sha256 "
            "FROM anpr_feedback WHERE id=?",
            (feedback_id,),
        ).fetchone()
    assert row["training_status"] == "ready"
    assert row["sample_path"] == str(sample)
    assert row["sample_sha256"] == result["sha256"]


def test_transient_capture_failure_keeps_source_pinned_for_retry(
    tmp_path,
    monkeypatch,
):
    database, _data_dir = _isolated_database(tmp_path, monkeypatch)
    image_path = tmp_path / "retry-source.jpg"
    assert cv2.imwrite(
        str(image_path),
        np.full((48, 180, 3), 170, dtype=np.uint8),
    )
    with database.connect() as con:
        event_id = con.execute(
            "INSERT INTO plate_events(plate_text,plate_norm) VALUES(?,?)",
            ("12-ب-345-67", "12ب34567"),
        ).lastrowid
        feedback_id = con.execute(
            "INSERT INTO anpr_feedback("
            "event_id,observed_text,observed_norm,corrected_text,"
            "corrected_norm,plate_image_path,status"
            ") VALUES(?,?,?,?,?,?,'confirmed')",
            (
                event_id,
                "12-ب-345-67",
                "12ب34567",
                "12-ب-345-67",
                "12ب34567",
                str(image_path),
            ),
        ).lastrowid

    original_capture = training._capture_feedback_sample

    def fail_capture(_feedback_id):
        raise RuntimeError("transient encoder failure")

    monkeypatch.setattr(training, "_capture_feedback_sample", fail_capture)
    with pytest.raises(RuntimeError, match="transient encoder failure"):
        training.capture_feedback_sample(feedback_id)

    assert feedback_id in training._PENDING_SOURCE_PINS
    assert image_path.is_file()

    monkeypatch.setattr(training, "_capture_feedback_sample", original_capture)
    assert training.capture_feedback_sample(feedback_id)["ready"] is True
    assert feedback_id not in training._PENDING_SOURCE_PINS


def test_refresh_pending_pins_rejects_directory_row_individually(
    tmp_path,
    monkeypatch,
):
    database, _data_dir = _isolated_database(tmp_path, monkeypatch)
    image_path = tmp_path / "valid-source.jpg"
    assert cv2.imwrite(
        str(image_path),
        np.full((48, 180, 3), 170, dtype=np.uint8),
    )
    with database.connect() as con:
        event_ids = [
            con.execute(
                "INSERT INTO plate_events(plate_text,plate_norm) VALUES(?,?)",
                ("12-ب-345-67", "12ب34567"),
            ).lastrowid
            for _ in range(2)
        ]
        feedback_ids = [
            con.execute(
                "INSERT INTO anpr_feedback("
                "event_id,observed_text,observed_norm,corrected_text,"
                "corrected_norm,plate_image_path,status"
                ") VALUES(?,?,?,?,?,?,'confirmed')",
                (
                    event_id,
                    "12-ب-345-67",
                    "12ب34567",
                    "12-ب-345-67",
                    "12ب34567",
                    str(source),
                ),
            ).lastrowid
            for event_id, source in zip(
                event_ids,
                (tmp_path, image_path),
                strict=True,
            )
        ]

    status = training.refresh_pending_feedback_source_pins()

    assert status["pending"] == 2
    assert status["pinned"] == 1
    assert status["failed_ids"] == [feedback_ids[0]]
    assert set(training._PENDING_SOURCE_PINS) == {feedback_ids[1]}


def test_reconcile_corrupt_training_sample_recaptures_source(
    tmp_path,
    monkeypatch,
):
    database, _data_dir = _isolated_database(tmp_path, monkeypatch)
    image_path = tmp_path / "reconcile-source.jpg"
    assert cv2.imwrite(
        str(image_path),
        np.full((48, 180, 3), 170, dtype=np.uint8),
    )
    with database.connect() as con:
        event_id = con.execute(
            "INSERT INTO plate_events(plate_text,plate_norm) VALUES(?,?)",
            ("12-ب-345-67", "12ب34567"),
        ).lastrowid
        feedback_id = con.execute(
            "INSERT INTO anpr_feedback("
            "event_id,observed_text,observed_norm,corrected_text,"
            "corrected_norm,plate_image_path,status"
            ") VALUES(?,?,?,?,?,?,'confirmed')",
            (
                event_id,
                "12-ب-345-67",
                "12ب34567",
                "12-ب-345-67",
                "12ب34567",
                str(image_path),
            ),
        ).lastrowid

    captured = training.capture_feedback_sample(feedback_id)
    sample_path = Path(captured["path"])
    sample_path.write_bytes(b"corrupt")
    with database.connect() as con:
        con.execute(
            "UPDATE anpr_feedback SET training_status='trained',"
            "trained_run_id=99 WHERE id=?",
            (feedback_id,),
        )

    reconciled = training.reconcile_feedback_sample_files()
    pinned = training.refresh_pending_feedback_source_pins()
    recovered = training.recover_pending_feedback_samples(1)

    assert reconciled["reset_ids"] == [feedback_id]
    assert pinned["pinned"] == 1
    assert recovered["recovered"] == 1
    with database.connect() as con:
        row = con.execute(
            "SELECT training_status,trained_run_id,sample_path "
            "FROM anpr_feedback WHERE id=?",
            (feedback_id,),
        ).fetchone()
    assert row["training_status"] == "ready"
    assert row["trained_run_id"] is None
    assert Path(row["sample_path"]).is_file()
    assert feedback_id not in training._PENDING_SOURCE_PINS


def test_duplicate_confirmed_crop_is_not_counted_twice(
    tmp_path,
    monkeypatch,
):
    database, _data_dir = _isolated_database(tmp_path, monkeypatch)
    image_path = tmp_path / "plate.jpg"
    assert cv2.imwrite(
        str(image_path),
        np.full((48, 180, 3), 170, dtype=np.uint8),
    )
    with database.connect() as con:
        event_ids = [
            con.execute(
                "INSERT INTO plate_events(plate_text,plate_norm) VALUES(?,?)",
                ("12-ب-345-67", "12ب34567"),
            ).lastrowid
            for _ in range(2)
        ]
        feedback_ids = [
            con.execute(
                "INSERT INTO anpr_feedback("
                "event_id,observed_text,observed_norm,corrected_text,"
                "corrected_norm,plate_image_path,status"
                ") VALUES(?,?,?,?,?,?,?)",
                (
                    event_id,
                    "12-ب-345-67",
                    "12ب34567",
                    "12-ب-345-67",
                    "12ب34567",
                    str(image_path),
                    "confirmed",
                ),
            ).lastrowid
            for event_id in event_ids
        ]

    first = training.capture_feedback_sample(feedback_ids[0])
    with database.connect() as con:
        con.execute(
            "UPDATE anpr_feedback SET training_status='trained' WHERE id=?",
            (feedback_ids[0],),
        )
    duplicate = training.capture_feedback_sample(feedback_ids[1])

    assert first["ready"] is True
    assert duplicate == {
        "ready": False,
        "reason": "duplicate-image",
        "duplicate_of": feedback_ids[0],
    }
    assert training.dataset_status()["samples"] == 1
    with database.connect() as con:
        status = con.execute(
            "SELECT training_status FROM anpr_feedback WHERE id=?",
            (feedback_ids[1],),
        ).fetchone()[0]
    assert status == "duplicate"


def test_conflicting_duplicate_quarantines_trained_and_duplicate_rows(
    tmp_path,
    monkeypatch,
):
    database, _data_dir = _isolated_database(tmp_path, monkeypatch)
    image_path = tmp_path / "plate.jpg"
    assert cv2.imwrite(
        str(image_path),
        np.full((48, 180, 3), 170, dtype=np.uint8),
    )
    labels = ("12ب34567", "12ب34567", "12پ34567")
    with database.connect() as con:
        feedback_ids = []
        for label in labels:
            event_id = con.execute(
                "INSERT INTO plate_events(plate_text,plate_norm) VALUES(?,?)",
                (label, label),
            ).lastrowid
            feedback_ids.append(
                con.execute(
                    "INSERT INTO anpr_feedback("
                    "event_id,observed_text,observed_norm,corrected_text,"
                    "corrected_norm,plate_image_path,status"
                    ") VALUES(?,?,?,?,?,?,?)",
                    (
                        event_id,
                        label,
                        label,
                        label,
                        label,
                        str(image_path),
                        "confirmed",
                    ),
                ).lastrowid
            )

    first = training.capture_feedback_sample(feedback_ids[0])
    assert first["ready"] is True
    with database.connect() as con:
        con.execute(
            "UPDATE anpr_feedback SET training_status='trained' WHERE id=?",
            (feedback_ids[0],),
        )
    duplicate = training.capture_feedback_sample(feedback_ids[1])
    assert duplicate["reason"] == "duplicate-image"

    conflict = training.capture_feedback_sample(feedback_ids[2])

    assert conflict == {
        "ready": False,
        "reason": "duplicate-label-conflict",
    }
    with database.connect() as con:
        statuses = [
            row[0]
            for row in con.execute(
                "SELECT training_status FROM anpr_feedback ORDER BY id"
            ).fetchall()
        ]
    assert statuses == ["label-conflict"] * 3
    assert training._verified_samples() == []


@pytest.mark.parametrize("letter", ["ف", "ک", "گ", "D", "S"])
def test_capture_rejects_plate_letters_missing_from_crnn_alphabet(
    tmp_path,
    monkeypatch,
    letter,
):
    database, data_dir = _isolated_database(tmp_path, monkeypatch)
    label = f"12{letter}34567"
    image_path = tmp_path / f"plate-{ord(letter):x}.jpg"
    assert cv2.imwrite(
        str(image_path),
        np.full((48, 180, 3), 170, dtype=np.uint8),
    )
    with database.connect() as con:
        event_id = con.execute(
            "INSERT INTO plate_events(plate_text,plate_norm) VALUES(?,?)",
            (label, label),
        ).lastrowid
        feedback_id = con.execute(
            "INSERT INTO anpr_feedback("
            "event_id,observed_text,observed_norm,corrected_text,"
            "corrected_norm,plate_image_path,status"
            ") VALUES(?,?,?,?,?,?,?)",
            (
                event_id,
                label,
                label,
                label,
                label,
                str(image_path),
                "confirmed",
            ),
        ).lastrowid

    assert training.capture_feedback_sample(feedback_id) == {
        "ready": False,
        "reason": "unsupported-alphabet",
    }
    with database.connect() as con:
        status = con.execute(
            "SELECT training_status FROM anpr_feedback WHERE id=?",
            (feedback_id,),
        ).fetchone()[0]
    assert status == "unsupported-alphabet"
    assert not (data_dir / "anpr-training" / "samples").exists()


def test_training_snapshot_rejects_legacy_unsupported_alphabet(
    tmp_path,
    monkeypatch,
):
    database, _data_dir = _isolated_database(tmp_path, monkeypatch)
    label = "12ف34567"
    with database.connect() as con:
        event_id = con.execute(
            "INSERT INTO plate_events(plate_text,plate_norm) VALUES(?,?)",
            (label, label),
        ).lastrowid
        feedback_id = con.execute(
            "INSERT INTO anpr_feedback("
            "event_id,observed_text,observed_norm,corrected_text,"
            "corrected_norm,status,training_status"
            ") VALUES(?,?,?,?,?,'confirmed','ready')",
            (event_id, label, label, label, label),
        ).lastrowid
    manifest = tmp_path / "dataset.json"
    manifest_sha256 = _write_operator_manifest(manifest, plate=label)

    with pytest.raises(ValueError, match="unsupported alphabet"):
        training._reject_unsupported_manifest_labels(
            manifest,
            expected_sha256=manifest_sha256,
        )

    with database.connect() as con:
        status = con.execute(
            "SELECT training_status FROM anpr_feedback WHERE id=?",
            (feedback_id,),
        ).fetchone()[0]
    assert status == "unsupported-alphabet"


def test_pending_feedback_recovery_is_bounded_and_idempotent(
    tmp_path,
    monkeypatch,
):
    database, _data_dir = _isolated_database(tmp_path, monkeypatch)
    good_image = tmp_path / "good.jpg"
    assert cv2.imwrite(
        str(good_image),
        np.full((48, 180, 3), 170, dtype=np.uint8),
    )
    with database.connect() as con:
        feedback_ids = []
        for index, image_path in enumerate(
            (tmp_path / "missing.jpg", good_image),
            start=1,
        ):
            label = f"1{index}ب34{index}67"
            event_id = con.execute(
                "INSERT INTO plate_events(plate_text,plate_norm) VALUES(?,?)",
                (label, label),
            ).lastrowid
            feedback_ids.append(
                con.execute(
                    "INSERT INTO anpr_feedback("
                    "event_id,observed_text,observed_norm,corrected_text,"
                    "corrected_norm,plate_image_path,status,training_status"
                    ") VALUES(?,?,?,?,?,?,?,'pending')",
                    (
                        event_id,
                        label,
                        label,
                        label,
                        label,
                        str(image_path),
                        "confirmed",
                    ),
                ).lastrowid
            )

    first = training.recover_pending_feedback_samples(limit=2)
    second = training.recover_pending_feedback_samples(limit=2)

    assert first["selected"] == 2
    assert first["recovered"] == 1
    assert first["failed"] == 1
    assert [row["feedback_id"] for row in first["results"]] == feedback_ids
    assert first["results"][0]["reason"] == "missing-image"
    assert first["results"][1]["ready"] is True
    assert second == {
        "selected": 0,
        "recovered": 0,
        "failed": 0,
        "results": [],
    }
    with database.connect() as con:
        statuses = [
            row[0]
            for row in con.execute(
                "SELECT training_status FROM anpr_feedback ORDER BY id"
            ).fetchall()
        ]
    assert statuses == ["missing-image", "ready"]


def test_pending_feedback_recovery_continues_after_unexpected_error(
    tmp_path,
    monkeypatch,
):
    database, _data_dir = _isolated_database(tmp_path, monkeypatch)
    with database.connect() as con:
        feedback_ids = []
        for index in range(3):
            label = f"1{index}ب34{index}67"
            event_id = con.execute(
                "INSERT INTO plate_events(plate_text,plate_norm) VALUES(?,?)",
                (label, label),
            ).lastrowid
            feedback_ids.append(
                con.execute(
                    "INSERT INTO anpr_feedback("
                    "event_id,observed_text,observed_norm,corrected_text,"
                    "corrected_norm,status,training_status"
                    ") VALUES(?,?,?,?,?,'confirmed','pending')",
                    (event_id, label, label, label, label),
                ).lastrowid
            )
    calls = []

    def capture(feedback_id):
        calls.append(feedback_id)
        if feedback_id == feedback_ids[0]:
            raise RuntimeError("one broken sample")
        return {"ready": True}

    monkeypatch.setattr(training, "capture_feedback_sample", capture)

    result = training.recover_pending_feedback_samples(limit=2)

    assert calls == feedback_ids[:2]
    assert result["selected"] == 2
    assert result["recovered"] == 1
    assert result["failed"] == 1
    assert result["results"][0]["reason"] == "capture-error:RuntimeError"


def test_conflicting_duplicate_label_is_rejected_if_old_copy_is_corrupt(
    tmp_path,
    monkeypatch,
):
    database, _data_dir = _isolated_database(tmp_path, monkeypatch)
    image_path = tmp_path / "plate.jpg"
    assert cv2.imwrite(
        str(image_path),
        np.full((48, 180, 3), 170, dtype=np.uint8),
    )
    with database.connect() as con:
        event_ids = [
            con.execute(
                "INSERT INTO plate_events(plate_text,plate_norm) VALUES(?,?)",
                ("12-ب-345-67", "12ب34567"),
            ).lastrowid
            for _ in range(2)
        ]
        feedback_ids = []
        for event_id, label in zip(
            event_ids,
            ("12ب34567", "12پ34567"),
            strict=True,
        ):
            feedback_ids.append(
                con.execute(
                    "INSERT INTO anpr_feedback("
                    "event_id,observed_text,observed_norm,corrected_text,"
                    "corrected_norm,plate_image_path,status"
                    ") VALUES(?,?,?,?,?,?,?)",
                    (
                        event_id,
                        label,
                        label,
                        label,
                        label,
                        str(image_path),
                        "confirmed",
                    ),
                ).lastrowid
            )

    first = training.capture_feedback_sample(feedback_ids[0])
    Path(first["path"]).write_bytes(b"corrupt")
    conflict = training.capture_feedback_sample(feedback_ids[1])

    assert conflict == {
        "ready": False,
        "reason": "duplicate-label-conflict",
    }
    with database.connect() as con:
        status = con.execute(
            "SELECT training_status FROM anpr_feedback WHERE id=?",
            (feedback_ids[1],),
        ).fetchone()[0]
    assert status == "label-conflict"


def test_concurrent_duplicate_confirmations_create_one_sample(
    tmp_path,
    monkeypatch,
):
    database, _data_dir = _isolated_database(tmp_path, monkeypatch)
    image_path = tmp_path / "plate.jpg"
    assert cv2.imwrite(
        str(image_path),
        np.full((48, 180, 3), 170, dtype=np.uint8),
    )
    with database.connect() as con:
        feedback_ids = []
        for _index in range(2):
            event_id = con.execute(
                "INSERT INTO plate_events(plate_text,plate_norm) VALUES(?,?)",
                ("12-ب-345-67", "12ب34567"),
            ).lastrowid
            feedback_ids.append(
                con.execute(
                    "INSERT INTO anpr_feedback("
                    "event_id,observed_text,observed_norm,corrected_text,"
                    "corrected_norm,plate_image_path,status"
                    ") VALUES(?,?,?,?,?,?,?)",
                    (
                        event_id,
                        "12ب34567",
                        "12ب34567",
                        "12ب34567",
                        "12ب34567",
                        str(image_path),
                        "confirmed",
                    ),
                ).lastrowid
            )

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(
            training.capture_feedback_sample,
            feedback_ids,
        ))

    assert sorted(result["reason"] for result in results if not result["ready"]) == [
        "duplicate-image"
    ]
    assert sum(result["ready"] for result in results) == 1
    assert training.dataset_status()["samples"] == 1


def test_verified_samples_deduplicate_legacy_rows_and_reject_conflicts(
    tmp_path,
    monkeypatch,
):
    database, _data_dir = _isolated_database(tmp_path, monkeypatch)
    image_path = tmp_path / "plate.jpg"
    assert cv2.imwrite(
        str(image_path),
        np.full((48, 180, 3), 170, dtype=np.uint8),
    )
    event_id = None
    with database.connect() as con:
        event_id = con.execute(
            "INSERT INTO plate_events(plate_text,plate_norm) VALUES(?,?)",
            ("12-ب-345-67", "12ب34567"),
        ).lastrowid
        feedback_id = con.execute(
            "INSERT INTO anpr_feedback("
            "event_id,observed_text,observed_norm,corrected_text,"
            "corrected_norm,plate_image_path,status"
            ") VALUES(?,?,?,?,?,?,?)",
            (
                event_id,
                "12-ب-345-67",
                "12ب34567",
                "12-ب-345-67",
                "12ب34567",
                str(image_path),
                "confirmed",
            ),
        ).lastrowid
    captured = training.capture_feedback_sample(feedback_id)
    assert captured["ready"] is True

    with database.connect() as con:
        for label in ("12ب34567", "12ب34567"):
            duplicate_event_id = con.execute(
                "INSERT INTO plate_events(plate_text,plate_norm) VALUES(?,?)",
                (label, label),
            ).lastrowid
            con.execute(
                "INSERT INTO anpr_feedback("
                "event_id,observed_text,observed_norm,"
                "corrected_text,corrected_norm,status,"
                "sample_path,sample_sha256,dataset_split,training_status"
                ") VALUES(?,?,?,?,?,?,?,?,?,'ready')",
                (
                    duplicate_event_id,
                    label,
                    label,
                    label,
                    label,
                    "confirmed",
                    captured["path"],
                    captured["sha256"],
                    captured["split"],
                ),
            )

    assert training.dataset_status()["samples"] == 1

    with database.connect() as con:
        con.execute(
            "UPDATE anpr_feedback SET corrected_text=?,corrected_norm=? "
            "WHERE id=(SELECT MAX(id) FROM anpr_feedback)",
            ("12پ34567", "12پ34567"),
        )

    assert training.dataset_status()["samples"] == 0


def test_exported_operator_manifest_is_portable_after_relocation(
    tmp_path,
    monkeypatch,
):
    database, data_dir = _isolated_database(tmp_path, monkeypatch)
    image_path = tmp_path / "plate.jpg"
    assert cv2.imwrite(
        str(image_path),
        np.full((48, 180, 3), 170, dtype=np.uint8),
    )
    with database.connect() as con:
        event_id = con.execute(
            "INSERT INTO plate_events(plate_text,plate_norm) VALUES(?,?)",
            ("12-ب-345-67", "12ب34567"),
        ).lastrowid
        feedback_id = con.execute(
            "INSERT INTO anpr_feedback("
            "event_id,observed_text,observed_norm,"
            "corrected_text,corrected_norm,plate_image_path,status"
            ") VALUES(?,?,?,?,?,?,?)",
            (
                event_id,
                "12-ب-345-67",
                "12ب34567",
                "12-ب-345-67",
                "12ب34567",
                str(image_path),
                "confirmed",
            ),
        ).lastrowid
    assert training.capture_feedback_sample(feedback_id)["ready"] is True
    manifest = training.export_manifest(run_id=7)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    exported_path = payload["samples"][0]["image_path"]
    assert (
        payload["source_license"]
        == "operator-confirmed-rights-unverified"
    )
    assert payload["ownership_attested"] is False
    assert payload["distribution_allowed"] is False
    assert not Path(exported_path).is_absolute()
    assert exported_path.startswith("../samples/")

    attested_manifest = training.export_manifest(
        run_id=8,
        rights_attested=True,
        attested_by="admin",
    )
    attested_payload = json.loads(
        attested_manifest.read_text(encoding="utf-8")
    )
    attested_digest = hashlib.sha256(
        attested_manifest.read_bytes()
    ).hexdigest().upper()
    assert attested_payload["source_license"] == (
        "operator-confirmed-company-owned"
    )
    assert attested_payload["ownership_attested"] is True
    assert attested_payload["distribution_allowed"] is True
    assert attested_payload["license_evidence"].startswith(
        "bcvision-admin-attestation:admin:"
    )
    assert training._training_manifest_rights_verified(
        attested_manifest,
        expected_sha256=attested_digest,
    ) is True

    relocated = tmp_path / "relocated-training"
    shutil.copytree(data_dir / "anpr-training", relocated)
    rows = _load_rows(relocated / "manifests" / "run-7.json")

    assert len(rows) == 1
    assert rows[0]["sha256"] == payload["samples"][0]["sha256"]
    assert rows[0]["split"] == payload["samples"][0]["split"]


def test_crnn_worker_loads_portable_operator_snapshot(tmp_path):
    snapshot = tmp_path / "snapshot"
    samples_dir = snapshot / "samples"
    manifests_dir = snapshot / "manifests"
    samples_dir.mkdir(parents=True)
    manifests_dir.mkdir()
    samples = []
    for index, (plate, split, value) in enumerate((
        ("12ب34567", "train", 90),
        ("31ط55674", "validation", 180),
    ), 1):
        image = samples_dir / f"{index}.png"
        assert cv2.imwrite(
            str(image),
            np.full((32, 128, 3), value, dtype=np.uint8),
        )
        samples.append({
            "feedback_id": index,
            "image_path": f"../samples/{image.name}",
            "plate": plate,
            "group_id": plate,
            "sha256": hashlib.sha256(
                image.read_bytes()
            ).hexdigest().upper(),
            "split": split,
        })
    manifest = manifests_dir / "run-1.json"
    manifest.write_text(
        json.dumps({
            "schema": 2,
            "training_source": "operator-confirmed-only",
            "golden_benchmark_data": False,
            "dataset_fingerprint": operator_dataset_fingerprint(samples),
            "samples": samples,
        }, ensure_ascii=False),
        encoding="utf-8",
    )

    train, validation = _load_manifest(manifest)

    assert [row["label"] for row in train] == ["12ب34567"]
    assert [row["label"] for row in validation] == ["31ط55674"]
    assert train[0]["image"] == (samples_dir / "1.png").resolve()


def test_crnn_worker_rejects_golden_operator_snapshot(tmp_path):
    manifest = tmp_path / "dataset.json"
    samples = []
    manifest.write_text(
        json.dumps({
            "schema": 2,
            "training_source": "operator-confirmed-only",
            "golden_benchmark_data": True,
            "dataset_fingerprint": operator_dataset_fingerprint(samples),
            "samples": samples,
        }),
        encoding="utf-8",
    )

    try:
        _load_manifest(manifest)
    except ValueError as exc:
        assert "non-Golden" in str(exc)
    else:
        raise AssertionError("Golden snapshot entered the CRNN worker")


@pytest.mark.parametrize(
    ("mode", "message"),
    [
        ("escape", "escapes snapshot root"),
        ("corrupt", "Invalid or changed training sample"),
        ("cross-split", "crosses train and validation"),
    ],
)
def test_crnn_worker_rejects_invalid_snapshot_rows(
    tmp_path,
    mode,
    message,
):
    snapshot = tmp_path / "snapshot"
    samples_dir = snapshot / "samples"
    manifests_dir = snapshot / "manifests"
    samples_dir.mkdir(parents=True)
    manifests_dir.mkdir()
    outside = tmp_path / "outside.png"
    paths = [samples_dir / "1.png", samples_dir / "2.png", outside]
    for index, image in enumerate(paths):
        assert cv2.imwrite(
            str(image),
            np.full((32, 128, 3), 80 + index * 40, dtype=np.uint8),
        )
    second_plate = "12ب34567" if mode == "cross-split" else "31ط55674"
    second_path = (
        "../../outside.png"
        if mode == "escape"
        else "../samples/2.png"
    )
    samples = [
        {
            "feedback_id": 1,
            "image_path": "../samples/1.png",
            "plate": "12ب34567",
            "group_id": "12ب34567",
            "sha256": (
                "0" * 64
                if mode == "corrupt"
                else hashlib.sha256(paths[0].read_bytes()).hexdigest().upper()
            ),
            "split": "train",
        },
        {
            "feedback_id": 2,
            "image_path": second_path,
            "plate": second_plate,
            "group_id": second_plate,
            "sha256": hashlib.sha256(
                (outside if mode == "escape" else paths[1]).read_bytes()
            ).hexdigest().upper(),
            "split": "validation",
        },
    ]
    manifest = manifests_dir / "run-1.json"
    manifest.write_text(
        json.dumps({
            "schema": 2,
            "training_source": "operator-confirmed-only",
            "golden_benchmark_data": False,
            "dataset_fingerprint": operator_dataset_fingerprint(samples),
            "samples": samples,
        }, ensure_ascii=False),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=message):
        _load_manifest(manifest)


def test_training_start_requires_twelve_validation_samples(monkeypatch):
    rows = [
        {
            "feedback_id": index,
            "image_path": f"/sample/{index}.png",
            "sha256": f"{index:064X}",
            "plate": f"{10 + index % 20:02d}ب{100 + index % 20:03d}"
            f"{20 + index % 20:02d}",
            "group_id": f"group-{index}",
            "split": "train" if index < 24 else "validation",
        }
        for index in range(30)
    ]
    monkeypatch.setattr(training, "_verified_samples", lambda: rows)

    status = training.dataset_status()

    assert status["validation_samples"] == 6
    assert status["minimum_validation"] == 12
    assert status["ready"] is False


def test_pending_candidate_can_reach_ready_after_golden_evaluation(
    tmp_path,
    monkeypatch,
):
    database, _data_dir = _isolated_database(tmp_path, monkeypatch)
    baseline = tmp_path / "baseline.onnx"
    candidate = tmp_path / "candidate.onnx"
    baseline.write_bytes(b"baseline")
    candidate.write_bytes(b"candidate")
    baseline_sha256 = hashlib.sha256(
        baseline.read_bytes()
    ).hexdigest().upper()
    candidate_sha256 = hashlib.sha256(
        candidate.read_bytes()
    ).hexdigest().upper()
    initial_report = {
        "validation_regressions": 0,
        "initialization_mode": "active-model-distillation",
    }
    manifest = tmp_path / "dataset.json"
    manifest_sha256 = _write_operator_manifest(manifest)
    with database.connect() as con:
        run_id = con.execute(
            "INSERT INTO anpr_training_runs("
            "status,validation_samples,baseline_accuracy,"
            "candidate_accuracy,baseline_mean_character_error,"
            "candidate_mean_character_error,baseline_sha256,"
            "candidate_path,candidate_sha256,promotion_report,"
            "dataset_manifest_path,dataset_manifest_sha256"
            ") VALUES('awaiting-golden',?,?,?,?,?,?,?,?,?,?,?)",
            (
                12,
                0.80,
                0.85,
                0.40,
                0.30,
                baseline_sha256,
                str(candidate),
                candidate_sha256,
                json.dumps(initial_report),
                str(manifest),
                manifest_sha256,
            ),
        ).lastrowid
    monkeypatch.setattr(
        model_manager,
        "active_crnn_model",
        lambda: (
            baseline,
            baseline_sha256,
            baseline.stat().st_size,
        ),
    )
    from app.ai import benchmark, golden

    monkeypatch.setattr(
        golden,
        "golden_status",
        _ready_golden,
    )
    monkeypatch.setattr(
        benchmark,
        "compare_crnn_candidate_on_golden",
        lambda *_args, **_kwargs: _passing_golden_decision(),
    )

    result = training.evaluate_candidate_on_golden(run_id)

    assert result["status"] == "candidate-ready"
    assert result["report"]["promote"] is True
    with database.connect() as con:
        row = con.execute(
            "SELECT status,promotion_report FROM anpr_training_runs "
            "WHERE id=?",
            (run_id,),
        ).fetchone()
    assert row["status"] == "candidate-ready"
    assert json.loads(row["promotion_report"])["promote"] is True


def test_pending_candidate_is_rejected_on_training_golden_overlap(
    tmp_path,
    monkeypatch,
):
    state = _pending_candidate_run(
        tmp_path,
        monkeypatch,
        training_plate="31ط55674",
    )
    from app.ai import benchmark, golden

    monkeypatch.setattr(golden, "golden_status", _ready_golden)
    monkeypatch.setattr(
        benchmark,
        "compare_crnn_candidate_on_golden",
        lambda *_args, **_kwargs: pytest.fail(
            "overlap must block before inference"
        ),
    )

    result = training.evaluate_candidate_on_golden(state["run_id"])

    assert result["status"] == "rejected"
    assert (
        "golden:training-identity-overlap"
        in result["report"]["reasons"]
    )


def test_pending_candidate_with_unverified_rights_is_rejected(
    tmp_path,
    monkeypatch,
):
    state = _pending_candidate_run(
        tmp_path,
        monkeypatch,
        manifest_rights_verified=False,
    )
    from app.ai import benchmark, golden

    monkeypatch.setattr(golden, "golden_status", _ready_golden)
    monkeypatch.setattr(
        benchmark,
        "compare_crnn_candidate_on_golden",
        lambda *_args, **_kwargs: _passing_golden_decision(),
    )

    result = training.evaluate_candidate_on_golden(state["run_id"])

    assert result["status"] == "rejected"
    assert "training-rights-unverified" in result["report"]["reasons"]


def test_training_snapshot_requires_stored_sha256(tmp_path):
    manifest = tmp_path / "dataset.json"
    _write_operator_manifest(manifest)

    with pytest.raises(ValueError, match="integrity"):
        training._training_golden_identity_overlap(
            manifest,
            _ready_golden(),
            expected_sha256="",
        )


def test_pending_golden_rejects_tampered_baseline(
    tmp_path,
    monkeypatch,
):
    state = _pending_candidate_run(tmp_path, monkeypatch)
    from app.ai import golden

    state["baseline"].write_bytes(b"tampered-baseline")
    monkeypatch.setattr(golden, "golden_status", _ready_golden)

    with pytest.raises(ValueError, match="مدل پایه"):
        training.evaluate_candidate_on_golden(state["run_id"])


def test_transient_golden_evaluation_error_remains_retryable():
    golden = _ready_golden()
    result = {
        "validation_samples": 12,
        "baseline_accuracy": 0.80,
        "candidate_accuracy": 0.85,
        "baseline_mean_character_error": 0.40,
        "candidate_mean_character_error": 0.30,
        "validation_regressions": 0,
        "baseline_sha256": "A" * 64,
        "initialization_mode": "active-model-distillation",
        "training_rights_verified": True,
        "golden_decision": {
            "promote": False,
            "reasons": ["evaluation-error:RuntimeError"],
            "golden_manifest_sha256": "B" * 64,
        },
    }
    from app.ai.benchmark import assess_training_candidate

    report = assess_training_candidate(result, golden)

    assert report["promote"] is False
    assert training._promotion_status(report) == "awaiting-golden"


def test_only_verified_candidate_can_be_promoted(tmp_path, monkeypatch):
    state = _candidate_ready_run(tmp_path, monkeypatch)
    from app.ai import benchmark

    monkeypatch.setattr(
        benchmark,
        "compare_crnn_candidate_on_golden",
        lambda *_args, **_kwargs: state["decision"],
    )

    promoted = training.apply_candidate(state["run_id"], "admin")
    active_path, active_digest, active_size = (
        model_manager.active_crnn_model()
    )

    assert promoted["sha256"] == state["candidate_digest"]
    assert active_path.is_file()
    assert active_path.is_relative_to(
        state["data_dir"] / "models" / "crnn" / "custom"
    )
    assert active_digest == state["candidate_digest"]
    assert active_size == len(b"candidate-crnn")
    with state["database"].connect() as con:
        row = con.execute(
            "SELECT status,applied_by FROM anpr_training_runs WHERE id=?",
            (state["run_id"],),
        ).fetchone()
    assert row["status"] == "applied"
    assert row["applied_by"] == "admin"


def test_live_golden_rejection_blocks_apply(tmp_path, monkeypatch):
    state = _candidate_ready_run(tmp_path, monkeypatch)
    from app.ai import benchmark

    monkeypatch.setattr(
        benchmark,
        "compare_crnn_candidate_on_golden",
        lambda *_args, **_kwargs: {
            "promote": False,
            "reasons": ["candidate-accuracy-floor"],
            "golden_manifest_sha256": "C" * 64,
        },
    )

    with pytest.raises(ValueError, match="بازآزمایی Golden"):
        training.apply_candidate(state["run_id"], "admin")
    with state["database"].connect() as con:
        status = con.execute(
            "SELECT status FROM anpr_training_runs WHERE id=?",
            (state["run_id"],),
        ).fetchone()[0]
    assert status == "candidate-ready"


def test_apply_rechecks_training_rights_against_snapshot(
    tmp_path,
    monkeypatch,
):
    state = _candidate_ready_run(
        tmp_path,
        monkeypatch,
        manifest_rights_verified=False,
    )

    with pytest.raises(ValueError, match="مجوز و گواه مالکیت"):
        training.apply_candidate(state["run_id"], "admin")


def test_concurrent_candidate_apply_rejects_stale_second_baseline(
    tmp_path,
    monkeypatch,
):
    state = _candidate_ready_run(tmp_path, monkeypatch)
    from app.ai import benchmark

    monkeypatch.setattr(
        benchmark,
        "compare_crnn_candidate_on_golden",
        lambda *_args, **_kwargs: state["decision"],
    )
    second_candidate = tmp_path / "candidate-second.onnx"
    second_candidate.write_bytes(b"candidate-crnn-second")
    second_digest = hashlib.sha256(
        second_candidate.read_bytes()
    ).hexdigest().upper()
    with state["database"].connect() as con:
        first = con.execute(
            "SELECT promotion_report,dataset_manifest_path,"
            "dataset_manifest_sha256 FROM anpr_training_runs WHERE id=?",
            (state["run_id"],),
        ).fetchone()
        second_run_id = con.execute(
            "INSERT INTO anpr_training_runs("
            "status,candidate_path,candidate_sha256,promotion_report,"
            "dataset_manifest_path,dataset_manifest_sha256"
            ") VALUES('candidate-ready',?,?,?,?,?)",
            (
                str(second_candidate),
                second_digest,
                first["promotion_report"],
                first["dataset_manifest_path"],
                first["dataset_manifest_sha256"],
            ),
        ).lastrowid

    def attempt(run_id):
        try:
            training.apply_candidate(run_id, "admin")
            return "applied"
        except ValueError:
            return "rejected"

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(
            attempt,
            (state["run_id"], second_run_id),
        ))

    assert sorted(outcomes) == ["applied", "rejected"]
    with state["database"].connect() as con:
        statuses = [
            row[0]
            for row in con.execute(
                "SELECT status FROM anpr_training_runs "
                "WHERE id IN (?,?) ORDER BY id",
                (state["run_id"], second_run_id),
            ).fetchall()
        ]
    assert statuses.count("applied") == 1
    assert statuses.count("candidate-ready") == 1


def test_minimal_fabricated_promotion_report_is_rejected(
    tmp_path,
    monkeypatch,
):
    database, _data_dir = _isolated_database(tmp_path, monkeypatch)
    candidate = tmp_path / "candidate.onnx"
    candidate.write_bytes(b"candidate-crnn")
    digest = hashlib.sha256(candidate.read_bytes()).hexdigest().upper()
    with database.connect() as con:
        run_id = con.execute(
            "INSERT INTO anpr_training_runs("
            "status,candidate_path,candidate_sha256,promotion_report"
            ") VALUES('candidate-ready',?,?,?)",
            (
                str(candidate),
                digest,
                json.dumps({
                    "schema": 1,
                    "promote": True,
                    "baseline_sha256": "A" * 64,
                }),
            ),
        ).lastrowid

    with pytest.raises(ValueError, match="کامل نیست"):
        training.apply_candidate(run_id, "admin")


def test_candidate_without_promotion_evidence_is_rejected(
    tmp_path,
    monkeypatch,
):
    database, _data_dir = _isolated_database(tmp_path, monkeypatch)
    candidate = tmp_path / "candidate.onnx"
    candidate.write_bytes(b"candidate-crnn")
    digest = hashlib.sha256(candidate.read_bytes()).hexdigest().upper()
    with database.connect() as con:
        run_id = con.execute(
            "INSERT INTO anpr_training_runs("
            "status,candidate_path,candidate_sha256"
            ") VALUES('candidate-ready',?,?)",
            (str(candidate), digest),
        ).lastrowid

    try:
        training.apply_candidate(run_id, "admin")
    except ValueError as exc:
        assert "گزارش ارتقای مدل" in str(exc)
    else:
        raise AssertionError("promotion evidence was bypassed")
