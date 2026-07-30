import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from app.ai import model_manager, training


def _isolated_database(tmp_path, monkeypatch):
    import app.config
    import app.database

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    db_path = data_dir / "bcvision.db"
    monkeypatch.setattr(app.config, "DATA_DIR", data_dir)
    monkeypatch.setattr(app.database, "DB_PATH", db_path)
    app.database.init_db()
    return app.database, data_dir


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


def test_only_verified_candidate_can_be_promoted(tmp_path, monkeypatch):
    database, data_dir = _isolated_database(tmp_path, monkeypatch)
    candidate = tmp_path / "candidate.onnx"
    candidate.write_bytes(b"candidate-crnn")
    digest = hashlib.sha256(candidate.read_bytes()).hexdigest().upper()
    _active_path, active_digest, _active_size = (
        model_manager.active_crnn_model()
    )
    manifest = (
        data_dir
        / "anpr-training"
        / "manifests"
        / "run-1.json"
    )
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        json.dumps({"schema": 2, "samples": []}),
        encoding="utf-8",
    )
    manifest_digest = hashlib.sha256(
        manifest.read_bytes()
    ).hexdigest().upper()
    promotion_report = json.dumps({
        "schema": 1,
        "promote": True,
        "baseline_sha256": active_digest,
    })
    with database.connect() as con:
        run_id = con.execute(
            "INSERT INTO anpr_training_runs("
            "status,candidate_path,candidate_sha256,"
            "promotion_report,dataset_manifest_path,"
            "dataset_manifest_sha256"
            ") VALUES('candidate-ready',?,?,?,?,?)",
            (
                str(candidate),
                digest,
                promotion_report,
                str(manifest),
                manifest_digest,
            ),
        ).lastrowid

    promoted = training.apply_candidate(run_id, "admin")
    active_path, active_digest, active_size = (
        model_manager.active_crnn_model()
    )

    assert promoted["sha256"] == digest
    assert active_path.is_file()
    assert active_path.is_relative_to(
        data_dir / "models" / "crnn" / "custom"
    )
    assert active_digest == digest
    assert active_size == len(b"candidate-crnn")
    with database.connect() as con:
        row = con.execute(
            "SELECT status,applied_by FROM anpr_training_runs WHERE id=?",
            (run_id,),
        ).fetchone()
    assert row["status"] == "applied"
    assert row["applied_by"] == "admin"


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
