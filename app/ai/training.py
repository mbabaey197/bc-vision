"""Persistent, operator-confirmed ANPR dataset and controlled CRNN training."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import threading

import cv2

from .dataset_split import stable_split_for_group
from .plate_rules import normalize_plate, plausible_plate


MIN_TRAIN_SAMPLES = 24
MIN_VALIDATION_SAMPLES = 6
MIN_UNIQUE_PLATES = 8
_TRAINING_LOCK = threading.RLock()
_TRAINING_THREAD: threading.Thread | None = None


def _training_root() -> Path:
    from app.config import DATA_DIR

    root = Path(DATA_DIR) / "anpr-training"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest().upper()


def capture_feedback_sample(feedback_id: int) -> dict:
    """Copy one confirmed plate crop into the immutable training dataset."""

    from app.database import connect

    with connect() as con:
        row = con.execute(
            "SELECT * FROM anpr_feedback WHERE id=? AND status='confirmed'",
            (int(feedback_id),),
        ).fetchone()
    if not row:
        return {"ready": False, "reason": "feedback-not-found"}

    label = normalize_plate(row["corrected_norm"])
    if not plausible_plate(label):
        return {"ready": False, "reason": "invalid-label"}
    source = Path(row["plate_image_path"] or "")
    if not source.is_file():
        with connect() as con:
            con.execute(
                "UPDATE anpr_feedback SET training_status='missing-image' "
                "WHERE id=?",
                (int(feedback_id),),
            )
        return {"ready": False, "reason": "missing-image"}

    image = cv2.imread(str(source), cv2.IMREAD_COLOR)
    if (
        image is None
        or getattr(image, "size", 0) == 0
        or image.shape[0] < 8
        or image.shape[1] < 24
    ):
        with connect() as con:
            con.execute(
                "UPDATE anpr_feedback SET training_status='invalid-image' "
                "WHERE id=?",
                (int(feedback_id),),
            )
        return {"ready": False, "reason": "invalid-image"}

    samples = _training_root() / "samples"
    samples.mkdir(parents=True, exist_ok=True)
    encoded_ok, encoded = cv2.imencode(
        ".png",
        image,
        [cv2.IMWRITE_PNG_COMPRESSION, 3],
    )
    if not encoded_ok:
        return {"ready": False, "reason": "encode-failed"}
    payload = encoded.tobytes()
    digest = hashlib.sha256(payload).hexdigest().upper()
    target = samples / f"{int(feedback_id):08d}-{digest[:16]}.png"
    temporary = target.with_suffix(".tmp")
    temporary.write_bytes(payload)
    if _sha256(temporary) != digest:
        temporary.unlink(missing_ok=True)
        return {"ready": False, "reason": "hash-mismatch"}
    os.replace(temporary, target)
    # All frames carrying the same confirmed plate stay in one split.  The
    # previous image-hash split could put neighbouring frames of one vehicle
    # in both train and validation and inflate validation accuracy.
    split = stable_split_for_group(label)
    with connect() as con:
        con.execute(
            "UPDATE anpr_feedback SET sample_path=?,sample_sha256=?,"
            "dataset_split=?,training_status='ready' WHERE id=?",
            (str(target), digest, split, int(feedback_id)),
        )
    return {
        "ready": True,
        "path": str(target),
        "sha256": digest,
        "split": split,
        "label": label,
    }


def _verified_samples() -> list[dict]:
    from app.database import connect

    with connect() as con:
        rows = con.execute(
            "SELECT id,corrected_norm,sample_path,sample_sha256,"
            "dataset_split FROM anpr_feedback "
            "WHERE status='confirmed' AND training_status='ready' "
            "ORDER BY id"
        ).fetchall()
    samples = []
    for row in rows:
        path = Path(row["sample_path"] or "")
        label = normalize_plate(row["corrected_norm"])
        if (
            not plausible_plate(label)
            or not path.is_file()
            or _sha256(path) != str(row["sample_sha256"]).upper()
        ):
            continue
        samples.append({
            "feedback_id": int(row["id"]),
            "image_path": str(path),
            "sha256": str(row["sample_sha256"]).upper(),
            "plate": label,
            "group_id": label,
            "split": stable_split_for_group(label),
        })
    return samples


def dataset_status() -> dict:
    samples = _verified_samples()
    train = sum(row["split"] == "train" for row in samples)
    validation = sum(
        row["split"] == "validation" for row in samples
    )
    unique = len({row["plate"] for row in samples})
    ready = (
        train >= MIN_TRAIN_SAMPLES
        and validation >= MIN_VALIDATION_SAMPLES
        and unique >= MIN_UNIQUE_PLATES
    )
    return {
        "samples": len(samples),
        "train_samples": train,
        "validation_samples": validation,
        "unique_plates": unique,
        "minimum_train": MIN_TRAIN_SAMPLES,
        "minimum_validation": MIN_VALIDATION_SAMPLES,
        "minimum_unique_plates": MIN_UNIQUE_PLATES,
        "identity_overlap": 0,
        "ready": ready,
    }


def export_manifest(run_id: int | None = None) -> Path:
    samples = _verified_samples()
    root = _training_root()
    if run_id is None:
        manifest = root / "dataset.json"
    else:
        manifest = root / "manifests" / f"run-{int(run_id)}.json"
        manifest.parent.mkdir(parents=True, exist_ok=True)
    temporary = manifest.with_suffix(".tmp")
    sample_fingerprint = hashlib.sha256(
        json.dumps(
            [
                {
                    "feedback_id": row["feedback_id"],
                    "sha256": row["sha256"],
                    "plate": row["plate"],
                    "group_id": row["group_id"],
                    "split": row["split"],
                }
                for row in samples
            ],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest().upper()
    payload = {
        "schema": 2,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "training_source": "operator-confirmed-only",
        "golden_benchmark_data": False,
        "group_key": "group_id",
        "dataset_fingerprint": sample_fingerprint,
        "samples": samples,
    }
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary, manifest)
    return manifest


def latest_training_status() -> dict:
    from app.database import connect

    data = dataset_status()
    with connect() as con:
        row = con.execute(
            "SELECT * FROM anpr_training_runs ORDER BY id DESC LIMIT 1"
        ).fetchone()
    data["run"] = dict(row) if row else None
    return data


def _run_training(
    run_id: int,
    manifest: Path,
    device: str,
    epochs: int,
) -> None:
    from app.database import connect
    from .training_worker import train_candidate

    try:
        with connect() as con:
            con.execute(
                "UPDATE anpr_training_runs SET status='running',"
                "started_at=CURRENT_TIMESTAMP WHERE id=?",
                (run_id,),
            )
        result = train_candidate(
            manifest=manifest,
            output_dir=_training_root() / "candidates" / str(run_id),
            device=device,
            epochs=epochs,
        )
        from .benchmark import assess_training_candidate
        from .golden import golden_status

        golden = golden_status()
        report = assess_training_candidate(result, golden)
        golden_only = all(
            reason == "golden-not-ready"
            or reason == "golden-comparison-missing"
            or reason.startswith("golden:")
            for reason in report["reasons"]
        )
        status = (
            "candidate-ready"
            if report["promote"]
            else "awaiting-golden"
            if golden_only
            else "rejected"
        )
        message = (
            "مدل نامزد همه دروازه‌های Validation و Golden را گذراند."
            if status == "candidate-ready"
            else (
                "مدل آموزش دید، اما تا تکمیل و قبولی Golden فعال نمی‌شود."
                if status == "awaiting-golden"
                else "مدل نامزد یک یا چند دروازه ارتقا را رد کرد."
            )
        )
        with connect() as con:
            con.execute(
                "UPDATE anpr_training_runs SET status=?,"
                "baseline_accuracy=?,candidate_accuracy=?,"
                "baseline_mean_character_error=?,"
                "candidate_mean_character_error=?,"
                "baseline_sha256=?,promotion_report=?,"
                "candidate_checkpoint_path=?,"
                "candidate_checkpoint_sha256=?,"
                "candidate_path=?,candidate_sha256=?,message=?,"
                "finished_at=CURRENT_TIMESTAMP WHERE id=?",
                (
                    status,
                    float(result["baseline_accuracy"]),
                    float(result["candidate_accuracy"]),
                    float(
                        result["baseline_mean_character_error"]
                    ),
                    float(
                        result["candidate_mean_character_error"]
                    ),
                    str(result["baseline_sha256"]).upper(),
                    json.dumps(
                        report,
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                    result["candidate_checkpoint_path"],
                    result["candidate_checkpoint_sha256"],
                    result["candidate_path"],
                    result["candidate_sha256"],
                    message,
                    run_id,
                ),
            )
    except Exception as exc:
        with connect() as con:
            con.execute(
                "UPDATE anpr_training_runs SET status='error',message=?,"
                "finished_at=CURRENT_TIMESTAMP WHERE id=?",
                (f"{type(exc).__name__}: {exc}", run_id),
            )


def start_training(device="auto", epochs=12) -> dict:
    global _TRAINING_THREAD
    from app.database import connect

    state = dataset_status()
    if not state["ready"]:
        raise ValueError(
            "نمونه‌های تأییدشده برای آموزش کنترل‌شده هنوز کافی نیست."
        )
    device = device if device in {"auto", "cpu", "gpu"} else "auto"
    epochs = max(4, min(40, int(epochs)))
    with _TRAINING_LOCK:
        if _TRAINING_THREAD is not None and _TRAINING_THREAD.is_alive():
            raise ValueError("یک آموزش دیگر در حال اجرا است.")
        with connect() as con:
            # A daemon worker cannot survive an application shutdown. Recover
            # stale rows before accepting the next explicit administrator run.
            con.execute(
                "UPDATE anpr_training_runs SET status='interrupted',"
                "message='برنامه پیش از پایان آموزش بسته شد.',"
                "finished_at=CURRENT_TIMESTAMP "
                "WHERE status IN ('queued','running')"
            )
            active = con.execute(
                "SELECT id FROM anpr_training_runs "
                "WHERE status IN ('queued','running') LIMIT 1"
            ).fetchone()
            if active:
                raise ValueError("یک آموزش دیگر در حال اجرا است.")
            cursor = con.execute(
                "INSERT INTO anpr_training_runs("
                "status,device,epochs,train_samples,validation_samples"
                ") VALUES('queued',?,?,?,?)",
                (
                    device,
                    epochs,
                    state["train_samples"],
                    state["validation_samples"],
                ),
            )
            run_id = int(cursor.lastrowid)
        manifest = export_manifest(run_id)
        manifest_sha256 = _sha256(manifest)
        with connect() as con:
            con.execute(
                "UPDATE anpr_training_runs SET "
                "dataset_manifest_path=?,dataset_manifest_sha256=? "
                "WHERE id=?",
                (str(manifest), manifest_sha256, run_id),
            )
        _TRAINING_THREAD = threading.Thread(
            target=_run_training,
            args=(run_id, manifest, device, epochs),
            daemon=True,
            name=f"bc-anpr-train-{run_id}",
        )
        _TRAINING_THREAD.start()
    return {"run_id": run_id, "status": "queued"}


def apply_candidate(run_id: int, username: str) -> dict:
    from app.database import connect
    from .model_manager import active_crnn_model, promote_crnn_candidate

    with connect() as con:
        row = con.execute(
            "SELECT * FROM anpr_training_runs WHERE id=?",
            (int(run_id),),
        ).fetchone()
    if not row or row["status"] != "candidate-ready":
        raise ValueError("مدل نامزد آماده و تأییدشده‌ای وجود ندارد.")
    try:
        report = json.loads(row["promotion_report"] or "")
    except Exception as exc:
        raise ValueError("گزارش ارتقای مدل معتبر نیست.") from exc
    if not report.get("promote"):
        raise ValueError("دروازه ارتقای مدل نامزد تأیید نشده است.")
    _active_path, active_sha256, _active_size = active_crnn_model()
    if str(active_sha256).upper() != str(
        report.get("baseline_sha256", "")
    ).upper():
        raise ValueError(
            "مدل پایه پس از آموزش تغییر کرده است؛ ارزیابی باید تکرار شود."
        )
    candidate = Path(row["candidate_path"] or "")
    expected = str(row["candidate_sha256"] or "").upper()
    if not candidate.is_file() or _sha256(candidate) != expected:
        raise ValueError("فایل مدل نامزد یا SHA-256 آن معتبر نیست.")
    manifest = Path(row["dataset_manifest_path"] or "")
    manifest_digest = str(
        row["dataset_manifest_sha256"] or ""
    ).upper()
    if (
        not manifest.is_file()
        or _sha256(manifest) != manifest_digest
    ):
        raise ValueError("Snapshot دیتاست آموزش معتبر نیست.")
    manifest_payload = json.loads(
        manifest.read_text(encoding="utf-8")
    )
    trained_feedback_ids = sorted({
        int(sample["feedback_id"])
        for sample in manifest_payload.get("samples", [])
        if int(sample.get("feedback_id", 0)) > 0
    })
    promoted = promote_crnn_candidate(
        candidate,
        expected,
        source_run_id=int(run_id),
        training_checkpoint=(
            Path(row["candidate_checkpoint_path"])
            if row["candidate_checkpoint_path"]
            else None
        ),
        training_checkpoint_sha256=str(
            row["candidate_checkpoint_sha256"] or ""
        ),
    )
    with connect() as con:
        con.execute(
            "UPDATE anpr_training_runs SET status='applied',"
            "applied_at=CURRENT_TIMESTAMP,applied_by=? WHERE id=?",
            (username, int(run_id)),
        )
        if trained_feedback_ids:
            placeholders = ",".join(
                "?" for _ in trained_feedback_ids
            )
            con.execute(
                "UPDATE anpr_feedback SET training_status='trained',"
                f"trained_run_id=? WHERE id IN ({placeholders}) "
                "AND training_status='ready'",
                (int(run_id), *trained_feedback_ids),
            )
    return promoted
