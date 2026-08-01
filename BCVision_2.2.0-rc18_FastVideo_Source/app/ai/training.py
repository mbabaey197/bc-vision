"""Persistent, operator-confirmed ANPR dataset and controlled CRNN training."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import threading

import cv2

from .dataset_split import stable_split_for_group
from .plate_rules import normalize_plate, plausible_plate
from .training_manifest import operator_dataset_fingerprint


MIN_TRAIN_SAMPLES = 24
MIN_VALIDATION_SAMPLES = 12
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


def _store_unique_feedback_sample(
    *,
    feedback_id: int,
    label: str,
    payload: bytes,
    digest: str,
    samples: Path,
) -> dict:
    from app.database import connect

    # Serialize the digest check and write within this process. Without this
    # lock, two simultaneous confirmations of the same crop can both pass the
    # read check and inflate the verified dataset.
    with _TRAINING_LOCK:
        with connect() as con:
            duplicate = con.execute(
                "SELECT id,corrected_norm,sample_path,sample_sha256 "
                "FROM anpr_feedback WHERE id<>? "
                "AND training_status IN ('ready','trained') "
                "AND sample_sha256=? ORDER BY id LIMIT 1",
                (int(feedback_id), digest),
            ).fetchone()
        if duplicate:
            duplicate_path = Path(duplicate["sample_path"] or "")
            duplicate_label = normalize_plate(duplicate["corrected_norm"])
            duplicate_valid = (
                duplicate_path.is_file()
                and _sha256(duplicate_path) == digest
            )
            if duplicate_label != label:
                with connect() as con:
                    con.execute(
                        "UPDATE anpr_feedback SET "
                        "training_status='label-conflict' WHERE id=?",
                        (int(feedback_id),),
                    )
                return {
                    "ready": False,
                    "reason": "duplicate-label-conflict",
                }
            if duplicate_valid:
                split = stable_split_for_group(label)
                with connect() as con:
                    con.execute(
                        "UPDATE anpr_feedback SET sample_path=?,"
                        "sample_sha256=?,dataset_split=?,"
                        "training_status='duplicate' WHERE id=?",
                        (
                            str(duplicate_path),
                            digest,
                            split,
                            int(feedback_id),
                        ),
                    )
                return {
                    "ready": False,
                    "reason": "duplicate-image",
                    "duplicate_of": int(duplicate["id"]),
                }

        target = samples / f"{int(feedback_id):08d}-{digest[:16]}.png"
        temporary = target.with_suffix(".tmp")
        temporary.write_bytes(payload)
        if _sha256(temporary) != digest:
            temporary.unlink(missing_ok=True)
            return {"ready": False, "reason": "hash-mismatch"}
        os.replace(temporary, target)
        # All frames carrying the same confirmed plate stay in one split. The
        # previous image-hash split could put neighbouring frames of one
        # vehicle in both train and validation and inflate validation accuracy.
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
    return _store_unique_feedback_sample(
        feedback_id=int(feedback_id),
        label=label,
        payload=payload,
        digest=digest,
        samples=samples,
    )


def _verified_samples() -> list[dict]:
    from app.database import connect

    with connect() as con:
        rows = con.execute(
            "SELECT id,corrected_norm,sample_path,sample_sha256,"
            "dataset_split FROM anpr_feedback "
            "WHERE status='confirmed' "
            "AND training_status IN ('ready','trained') "
            "ORDER BY id"
        ).fetchall()
    labels_by_digest = {}
    for row in rows:
        digest = str(row["sample_sha256"] or "").upper()
        label = normalize_plate(row["corrected_norm"])
        if len(digest) == 64 and plausible_plate(label):
            labels_by_digest.setdefault(digest, set()).add(label)
    conflicting = {
        digest
        for digest, labels in labels_by_digest.items()
        if len(labels) != 1
    }
    candidates = []
    for row in rows:
        path = Path(row["sample_path"] or "")
        label = normalize_plate(row["corrected_norm"])
        digest = str(row["sample_sha256"] or "").upper()
        if (
            not plausible_plate(label)
            or not path.is_file()
            or digest in conflicting
            or _sha256(path) != digest
        ):
            continue
        candidates.append({
            "feedback_id": int(row["id"]),
            "image_path": str(path),
            "sha256": digest,
            "plate": label,
            "group_id": label,
            "split": stable_split_for_group(label),
        })
    unique = {}
    for sample in candidates:
        digest = sample["sha256"]
        if digest in conflicting:
            continue
        unique.setdefault(digest, sample)
    return list(unique.values())


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
    portable_samples = []
    for row in samples:
        portable = dict(row)
        portable["image_path"] = os.path.relpath(
            row["image_path"],
            start=manifest.parent,
        ).replace("\\", "/")
        portable_samples.append(portable)
    temporary = manifest.with_suffix(".tmp")
    sample_fingerprint = operator_dataset_fingerprint(samples)
    payload = {
        "schema": 2,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "training_source": "operator-confirmed-only",
        "source_license": "operator-confirmed-rights-unverified",
        "ownership_attested": False,
        "distribution_allowed": False,
        "license_evidence": "",
        "golden_benchmark_data": False,
        "group_key": "group_id",
        "dataset_fingerprint": sample_fingerprint,
        "samples": portable_samples,
    }
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary, manifest)
    return manifest


def _training_manifest_payload(
    manifest: Path,
    *,
    expected_sha256: str = "",
) -> dict:
    manifest = Path(manifest)
    expected = str(expected_sha256).strip().upper()
    try:
        manifest_bytes = manifest.read_bytes()
    except OSError as exc:
        raise ValueError(
            "Training snapshot integrity verification failed"
        ) from exc
    if (
        len(expected) != 64
        or any(
            character not in "0123456789ABCDEF"
            for character in expected
        )
        or hashlib.sha256(manifest_bytes).hexdigest().upper()
        != expected
    ):
        raise ValueError("Training snapshot integrity verification failed")
    payload = json.loads(manifest_bytes.decode("utf-8"))
    samples = payload.get("samples", [])
    fingerprint = str(
        payload.get("dataset_fingerprint", "")
    ).strip().upper()
    if (
        int(payload.get("schema", 0)) != 2
        or payload.get("training_source") != "operator-confirmed-only"
        or payload.get("golden_benchmark_data") is not False
        or not isinstance(samples, list)
        or len(fingerprint) != 64
        or operator_dataset_fingerprint(samples) != fingerprint
    ):
        raise ValueError("Training snapshot contract is invalid")
    return payload


def _training_manifest_rights_verified(
    manifest: Path,
    *,
    expected_sha256: str,
) -> bool:
    payload = _training_manifest_payload(
        manifest,
        expected_sha256=expected_sha256,
    )
    license_name = str(
        payload.get("source_license", "")
    ).strip().lower()
    evidence = str(payload.get("license_evidence", "")).strip()
    if (
        payload.get("ownership_attested") is not True
        or payload.get("distribution_allowed") is not True
        or not evidence
    ):
        return False
    return license_name in {
        "bcvision-company-owned",
        "operator-confirmed-company-owned",
        "cc0-1.0",
    }


def _training_golden_identity_overlap(
    manifest: Path,
    golden: dict,
    *,
    expected_sha256: str = "",
) -> int:
    """Return only the count of identities shared by Train and Golden."""

    payload = _training_manifest_payload(
        manifest,
        expected_sha256=expected_sha256,
    )
    samples = payload["samples"]
    training_identities = set()
    for sample in samples:
        identity = normalize_plate(sample.get("plate", ""))
        if not plausible_plate(identity):
            raise ValueError("Training snapshot contains an invalid label")
        training_identities.add(identity)
    golden_rows = golden.get("rows", [])
    if not isinstance(golden_rows, list):
        raise ValueError("Golden rows are invalid")
    golden_identities = {
        identity
        for row in golden_rows
        if plausible_plate(
            identity := normalize_plate(
                row.get("expected_plate", "")
            )
        )
    }
    return len(training_identities & golden_identities)


def latest_training_status() -> dict:
    from app.database import connect

    data = dataset_status()
    with connect() as con:
        row = con.execute(
            "SELECT * FROM anpr_training_runs ORDER BY id DESC LIMIT 1"
        ).fetchone()
    data["run"] = dict(row) if row else None
    return data


def _promotion_status(report: dict) -> str:
    if report.get("promote"):
        return "candidate-ready"
    reasons = set(report.get("reasons") or [])
    pending_reasons = {
        "golden-not-ready",
        "golden-comparison-missing",
        "golden:ocr-crop-media-required",
    }
    if reasons and reasons <= pending_reasons:
        return "awaiting-golden"
    if reasons and all(
        reason in pending_reasons
        or reason.startswith("golden:evaluation-error:")
        for reason in reasons
    ):
        return "awaiting-golden"
    return "rejected"


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
            snapshot = con.execute(
                "SELECT dataset_manifest_sha256 "
                "FROM anpr_training_runs WHERE id=?",
                (int(run_id),),
            ).fetchone()
        manifest_sha256 = str(
            snapshot["dataset_manifest_sha256"]
            if snapshot
            else ""
        ).upper()
        result = train_candidate(
            manifest=manifest,
            output_dir=_training_root() / "candidates" / str(run_id),
            device=device,
            epochs=epochs,
            manifest_sha256=manifest_sha256,
        )
        result["training_rights_verified"] = (
            _training_manifest_rights_verified(
                manifest,
                expected_sha256=manifest_sha256,
            )
        )
        from .benchmark import (
            assess_training_candidate,
            compare_crnn_candidate_on_golden,
        )
        from .golden import golden_status

        golden = golden_status()
        if golden.get("ready"):
            try:
                from .model_manager import (
                    active_crnn_model,
                    verify_file,
                )

                baseline_path, baseline_sha256, baseline_size = (
                    active_crnn_model()
                )
                candidate_path = Path(result["candidate_path"])
                golden_sha256 = str(
                    golden.get("manifest_sha256", "")
                ).upper()
                if not verify_file(
                    baseline_path,
                    baseline_sha256,
                    baseline_size,
                ):
                    result["golden_decision"] = {
                        "promote": False,
                        "reasons": ["baseline-integrity-failed"],
                        "golden_manifest_sha256": golden_sha256,
                    }
                elif (
                    str(baseline_sha256).upper()
                    != str(result["baseline_sha256"]).upper()
                ):
                    result["golden_decision"] = {
                        "promote": False,
                        "reasons": ["baseline-identity-changed"],
                        "golden_manifest_sha256": golden_sha256,
                    }
                elif (
                    not candidate_path.is_file()
                    or _sha256(candidate_path)
                    != str(result["candidate_sha256"]).upper()
                ):
                    result["golden_decision"] = {
                        "promote": False,
                        "reasons": ["candidate-integrity-failed"],
                        "golden_manifest_sha256": golden_sha256,
                    }
                elif _training_golden_identity_overlap(
                    manifest,
                    golden,
                    expected_sha256=manifest_sha256,
                ):
                    result["golden_decision"] = {
                        "promote": False,
                        "reasons": ["training-identity-overlap"],
                        "golden_manifest_sha256": golden_sha256,
                    }
                else:
                    result["golden_decision"] = (
                        compare_crnn_candidate_on_golden(
                            baseline_path,
                            candidate_path,
                            golden,
                        )
                    )
            except Exception as exc:
                result["golden_decision"] = {
                    "promote": False,
                    "reasons": [
                        "evaluation-error:"
                        + type(exc).__name__
                    ],
                    "golden_manifest_sha256": str(
                        golden.get("manifest_sha256", "")
                    ).upper(),
                }
        report = assess_training_candidate(result, golden)
        status = _promotion_status(report)
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


def evaluate_candidate_on_golden(run_id: int) -> dict:
    """Run a pending candidate against the verified OCR-crop Golden set."""

    from app.database import connect
    from .benchmark import (
        assess_training_candidate,
        compare_crnn_candidate_on_golden,
    )
    from .golden import golden_status
    from .model_manager import active_crnn_model, verify_file

    with connect() as con:
        row = con.execute(
            "SELECT * FROM anpr_training_runs WHERE id=?",
            (int(run_id),),
        ).fetchone()
    if not row or row["status"] != "awaiting-golden":
        raise ValueError("مدل نامزد در انتظار ارزیابی Golden نیست.")
    golden = golden_status()
    if not golden.get("ready"):
        raise ValueError("Golden Dataset کامل و معتبر نیست.")
    try:
        previous = json.loads(row["promotion_report"] or "")
    except Exception as exc:
        raise ValueError("گزارش اولیه آموزش معتبر نیست.") from exc
    candidate = Path(row["candidate_path"] or "")
    candidate_sha256 = str(row["candidate_sha256"] or "").upper()
    if (
        not candidate.is_file()
        or _sha256(candidate) != candidate_sha256
    ):
        raise ValueError("فایل مدل نامزد یا SHA-256 آن معتبر نیست.")
    baseline, baseline_sha256, baseline_size = active_crnn_model()
    if not verify_file(
        baseline,
        baseline_sha256,
        baseline_size,
    ):
        raise ValueError("فایل مدل پایه یا SHA-256 آن معتبر نیست.")
    if str(baseline_sha256).upper() != str(
        row["baseline_sha256"] or ""
    ).upper():
        raise ValueError(
            "مدل پایه تغییر کرده است؛ آموزش و ارزیابی باید تکرار شود."
        )
    manifest = Path(row["dataset_manifest_path"] or "")
    manifest_sha256 = str(
        row["dataset_manifest_sha256"] or ""
    ).upper()
    overlap = _training_golden_identity_overlap(
        manifest,
        golden,
        expected_sha256=manifest_sha256,
    )
    training_rights_verified = _training_manifest_rights_verified(
        manifest,
        expected_sha256=manifest_sha256,
    )
    from .training_worker import _load_manifest

    _load_manifest(
        manifest,
        expected_sha256=manifest_sha256,
    )
    golden_decision = (
        {
            "promote": False,
            "reasons": ["training-identity-overlap"],
            "golden_manifest_sha256": str(
                golden.get("manifest_sha256", "")
            ).upper(),
        }
        if overlap
        else compare_crnn_candidate_on_golden(
            baseline,
            candidate,
            golden,
        )
    )
    result = {
        "validation_samples": int(row["validation_samples"] or 0),
        "baseline_accuracy": float(row["baseline_accuracy"] or 0.0),
        "candidate_accuracy": float(row["candidate_accuracy"] or 0.0),
        "baseline_mean_character_error": float(
            row["baseline_mean_character_error"] or 0.0
        ),
        "candidate_mean_character_error": float(
            row["candidate_mean_character_error"] or 0.0
        ),
        "validation_regressions": int(
            previous.get("validation_regressions", 0)
        ),
        "baseline_sha256": str(row["baseline_sha256"] or "").upper(),
        "initialization_mode": str(
            previous.get("initialization_mode", "")
        ),
        "training_rights_verified": training_rights_verified,
        "golden_decision": golden_decision,
    }
    report = assess_training_candidate(result, golden)
    status = _promotion_status(report)
    message = (
        "مدل نامزد همه دروازه‌های Validation و Golden را گذراند."
        if status == "candidate-ready"
        else (
            "Golden باید از Cropهای OCR معتبر تکمیل و دوباره اجرا شود."
            if status == "awaiting-golden"
            else "مدل نامزد در ارزیابی Golden رد شد."
        )
    )
    with connect() as con:
        con.execute(
            "UPDATE anpr_training_runs SET status=?,promotion_report=?,"
            "message=?,finished_at=CURRENT_TIMESTAMP WHERE id=?",
            (
                status,
                json.dumps(
                    report,
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                message,
                int(run_id),
            ),
        )
    return {"run_id": int(run_id), "status": status, "report": report}


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
    """Apply one candidate atomically against the active baseline."""

    with _TRAINING_LOCK:
        return _apply_candidate_locked(run_id, username)


def _apply_candidate_locked(run_id: int, username: str) -> dict:
    from app.database import connect
    from .benchmark import (
        compare_crnn_candidate_on_golden,
        validate_golden_decision_evidence,
    )
    from .model_manager import (
        active_crnn_model,
        promote_crnn_candidate,
        verify_file,
    )
    from .golden import golden_status

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
    current_golden = golden_status()
    try:
        golden_summary = report["golden"]
        stored_validation_samples = int(report["validation_samples"])
        stored_baseline_accuracy = float(report["baseline_accuracy"])
        stored_candidate_accuracy = float(report["candidate_accuracy"])
        stored_baseline_error = float(
            report["baseline_mean_character_error"]
        )
        stored_candidate_error = float(
            report["candidate_mean_character_error"]
        )
        stored_regressions = int(report["validation_regressions"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("گزارش ارتقای مدل کامل نیست.") from exc
    if (
        int(report.get("schema", 0)) != 1
        or report.get("promote") is not True
        or report.get("reasons") != []
        or stored_validation_samples < MIN_VALIDATION_SAMPLES
        or not all(
            math.isfinite(value)
            for value in (
                stored_baseline_accuracy,
                stored_candidate_accuracy,
                stored_baseline_error,
                stored_candidate_error,
            )
        )
        or not 0.0 <= stored_baseline_accuracy <= 1.0
        or not 0.0 <= stored_candidate_accuracy <= 1.0
        or stored_baseline_error < 0.0
        or stored_candidate_error < 0.0
        or stored_candidate_accuracy
        < max(0.70, stored_baseline_accuracy)
        or stored_candidate_error > stored_baseline_error + 1e-9
        or stored_regressions != 0
        or report.get("initialization_mode") not in {
            "active-checkpoint",
            "active-model-distillation",
        }
        or report.get("training_rights_verified") is not True
        or not isinstance(golden_summary, dict)
        or golden_summary.get("ready") is not True
        or str(
            golden_summary.get("manifest_sha256", "")
        ).upper()
        != str(
            current_golden.get("manifest_sha256", "")
        ).upper()
        or int(golden_summary.get("samples", 0))
        != int(current_golden.get("samples", 0))
    ):
        raise ValueError("دروازه ارتقای مدل نامزد تأیید نشده است.")
    evaluated_golden = report.get("golden_decision")
    if (
        not current_golden.get("ready")
        or validate_golden_decision_evidence(
            evaluated_golden,
            current_golden,
        )
    ):
        raise ValueError(
            "شواهد Golden ناقص، تغییرکرده یا نامعتبر است."
        )
    active_path, active_sha256, active_size = active_crnn_model()
    if not verify_file(active_path, active_sha256, active_size):
        raise ValueError("فایل مدل پایه یا SHA-256 آن معتبر نیست.")
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
    try:
        overlap = _training_golden_identity_overlap(
            manifest,
            current_golden,
            expected_sha256=manifest_digest,
        )
    except Exception as exc:
        raise ValueError("Snapshot دیتاست آموزش معتبر نیست.") from exc
    if overlap:
        raise ValueError(
            "هویت پلاک میان دیتاست آموزش و Golden هم‌پوشانی دارد."
        )
    if not _training_manifest_rights_verified(
        manifest,
        expected_sha256=manifest_digest,
    ):
        raise ValueError(
            "مجوز و گواه مالکیت دیتاست آموزش برای اعمال مدل کافی نیست."
        )
    from .training_worker import _load_manifest

    try:
        _load_manifest(
            manifest,
            expected_sha256=manifest_digest,
        )
    except Exception as exc:
        raise ValueError("Snapshot دیتاست آموزش معتبر نیست.") from exc
    live_golden_decision = compare_crnn_candidate_on_golden(
        active_path,
        candidate,
        current_golden,
    )
    if (
        live_golden_decision.get("promote") is not True
        or live_golden_decision.get("reasons") != []
        or validate_golden_decision_evidence(
            live_golden_decision,
            current_golden,
        )
    ):
        raise ValueError("مدل نامزد در بازآزمایی Golden رد شد.")
    final_golden = golden_status()
    if (
        not final_golden.get("ready")
        or str(final_golden.get("manifest_sha256", "")).upper()
        != str(current_golden.get("manifest_sha256", "")).upper()
    ):
        raise ValueError(
            "Golden Dataset هنگام بازآزمایی تغییر کرده است."
        )
    final_active_path, final_active_sha256, final_active_size = (
        active_crnn_model()
    )
    if (
        str(final_active_sha256).upper()
        != str(active_sha256).upper()
        or not verify_file(
            final_active_path,
            final_active_sha256,
            final_active_size,
        )
    ):
        raise ValueError(
            "مدل پایه هنگام بازآزمایی تغییر کرده است."
        )
    try:
        manifest_bytes = manifest.read_bytes()
        if (
            hashlib.sha256(manifest_bytes).hexdigest().upper()
            != manifest_digest
        ):
            raise ValueError
        manifest_payload = json.loads(
            manifest_bytes.decode("utf-8")
        )
    except Exception as exc:
        raise ValueError("Snapshot دیتاست آموزش معتبر نیست.") from exc
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
