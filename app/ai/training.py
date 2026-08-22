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
MAX_PENDING_FEEDBACK_RECOVERY = 256
_TRAINING_LOCK = threading.RLock()
_TRAINING_THREAD: threading.Thread | None = None
_PENDING_SOURCE_PINS: dict[int, object] = {}


def _training_root() -> Path:
    from app.config import DATA_DIR

    root = Path(DATA_DIR) / "anpr-training"
    _mkdir_durable(root)
    return root


def _mkdir_durable(directory: Path) -> None:
    from app.storage_policy import fsync_parent_directory

    missing = []
    current = Path(directory)
    while not current.exists() and current != current.parent:
        missing.append(current)
        current = current.parent
    directory.mkdir(parents=True, exist_ok=True)
    for created in reversed(missing):
        fsync_parent_directory(created)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _valid_sha256(value: str) -> bool:
    return len(value) == 64 and all(
        character in "0123456789ABCDEF" for character in value
    )


def _release_pending_source_pin(feedback_id: int) -> None:
    with _TRAINING_LOCK:
        lease = _PENDING_SOURCE_PINS.pop(int(feedback_id), None)
    if lease is not None:
        lease.close()


def close_pending_feedback_source_pins() -> None:
    """Release every process-local lease during orderly shutdown."""

    with _TRAINING_LOCK:
        leases = list(_PENDING_SOURCE_PINS.values())
        _PENDING_SOURCE_PINS.clear()
    for lease in leases:
        lease.close()


def _pin_pending_feedback_source(
    feedback_id: int,
    source_value: str | None = None,
) -> bool:
    """Retain one exact pending crop without allowing subtree/root pins."""

    from app.database import connect
    from app.storage_policy import StoragePolicyError, pin_media_paths

    feedback_id = int(feedback_id)
    with _TRAINING_LOCK:
        if feedback_id in _PENDING_SOURCE_PINS:
            return True
    if source_value is None:
        with connect() as con:
            row = con.execute(
                "SELECT plate_image_path FROM anpr_feedback "
                "WHERE id=? AND status='confirmed' "
                "AND training_status='pending'",
                (feedback_id,),
            ).fetchone()
        if not row:
            return False
        source_value = row["plate_image_path"]
    source_text = str(source_value or "").strip()
    if not source_text:
        return False
    source = Path(source_text)
    try:
        lease = pin_media_paths((source,))
    except (OSError, RuntimeError, StoragePolicyError, ValueError):
        return False
    # Acquire the policy lease first, then require an exact regular-file leaf.
    # A hostile DB row pointing at the media root must never pin a subtree.
    if not source.is_file():
        lease.close()
        return False
    with _TRAINING_LOCK:
        existing = _PENDING_SOURCE_PINS.get(feedback_id)
        if existing is None:
            _PENDING_SOURCE_PINS[feedback_id] = lease
            return True
    lease.close()
    return True


def refresh_pending_feedback_source_pins() -> dict:
    """Individually protect all retryable feedback before retention starts."""

    from app.database import connect

    with connect() as con:
        rows = con.execute(
            "SELECT id,plate_image_path FROM anpr_feedback "
            "WHERE status='confirmed' AND training_status='pending' "
            "ORDER BY id"
        ).fetchall()
    pending_ids = {int(row["id"]) for row in rows}
    with _TRAINING_LOCK:
        stale_ids = set(_PENDING_SOURCE_PINS) - pending_ids
    for feedback_id in stale_ids:
        _release_pending_source_pin(feedback_id)
    pinned = 0
    failed = []
    for row in rows:
        feedback_id = int(row["id"])
        if _pin_pending_feedback_source(
            feedback_id,
            row["plate_image_path"],
        ):
            pinned += 1
        else:
            failed.append(feedback_id)
    return {
        "pending": len(rows),
        "pinned": pinned,
        "failed_ids": failed,
    }


def _training_label_supported(label: str) -> bool:
    """Return whether every canonical label character is trainable by CRNN."""

    from .onnx_crnn import CRNN_LABELS

    normalized = normalize_plate(label)
    alphabet = set(CRNN_LABELS)
    return plausible_plate(normalized) and all(
        character in alphabet for character in normalized
    )


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
            duplicates = con.execute(
                "SELECT id,corrected_norm,sample_path,sample_sha256,"
                "training_status FROM anpr_feedback WHERE id<>? "
                "AND sample_sha256=? ORDER BY id",
                (int(feedback_id), digest),
            ).fetchall()
        if any(
            normalize_plate(row["corrected_norm"]) != label
            for row in duplicates
        ):
            # A digest represents one immutable crop and may never carry two
            # truths. Quarantine every historical copy, including rows that
            # were already trained or recorded as duplicates, so the oldest
            # label cannot silently remain in the verified dataset.
            with connect() as con:
                con.execute(
                    "UPDATE anpr_feedback SET "
                    "training_status='label-conflict' "
                    "WHERE sample_sha256=?",
                    (digest,),
                )
                con.execute(
                    "UPDATE anpr_feedback SET sample_sha256=?,"
                    "training_status='label-conflict' WHERE id=?",
                    (digest, int(feedback_id)),
                )
            return {
                "ready": False,
                "reason": "duplicate-label-conflict",
            }
        for duplicate in duplicates:
            duplicate_path = Path(duplicate["sample_path"] or "")
            duplicate_valid = (
                duplicate_path.is_file()
                and _sha256(duplicate_path) == digest
            )
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
        with temporary.open("wb") as sample_file:
            sample_file.write(payload)
            sample_file.flush()
            os.fsync(sample_file.fileno())
        if _sha256(temporary) != digest:
            temporary.unlink(missing_ok=True)
            return {"ready": False, "reason": "hash-mismatch"}
        os.replace(temporary, target)
        from app.storage_policy import fsync_parent_directory

        fsync_parent_directory(target)
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


def _capture_feedback_sample(feedback_id: int) -> dict:
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
        with connect() as con:
            con.execute(
                "UPDATE anpr_feedback SET training_status='invalid-label' "
                "WHERE id=?",
                (int(feedback_id),),
            )
        return {"ready": False, "reason": "invalid-label"}
    if not _training_label_supported(label):
        with connect() as con:
            con.execute(
                "UPDATE anpr_feedback SET "
                "training_status='unsupported-alphabet' WHERE id=?",
                (int(feedback_id),),
            )
        return {"ready": False, "reason": "unsupported-alphabet"}
    source = Path(row["plate_image_path"] or "")
    from app.storage_policy import StoragePolicyError, pin_media_paths

    source_exists = False
    image = None
    try:
        with pin_media_paths((source,)):
            source_exists = source.is_file()
            if source_exists:
                image = cv2.imread(str(source), cv2.IMREAD_COLOR)
    except StoragePolicyError:
        # A DB value outside the configured current/history media roots is not
        # trusted as training evidence, even if an attacker created that file.
        source_exists = False
    if not source_exists:
        with connect() as con:
            con.execute(
                "UPDATE anpr_feedback SET training_status='missing-image' "
                "WHERE id=?",
                (int(feedback_id),),
            )
        return {"ready": False, "reason": "missing-image"}

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
    _mkdir_durable(samples)
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


def capture_feedback_sample(feedback_id: int) -> dict:
    """Capture one sample while retaining retryable source evidence."""

    from app.database import connect

    feedback_id = int(feedback_id)
    _pin_pending_feedback_source(feedback_id)
    try:
        return _capture_feedback_sample(feedback_id)
    finally:
        # Expected outcomes all move the row out of pending. Unexpected
        # failures deliberately retain the process-local lease for retry.
        lookup_failed = False
        try:
            with connect() as con:
                row = con.execute(
                    "SELECT status,training_status FROM anpr_feedback "
                    "WHERE id=?",
                    (feedback_id,),
                ).fetchone()
        except Exception:
            lookup_failed = True
            row = None
        if not lookup_failed and (
            row is None
            or row["status"] != "confirmed"
            or row["training_status"] != "pending"
        ):
            _release_pending_source_pin(feedback_id)


def reconcile_feedback_sample_files() -> dict:
    """Move invalid ready/trained rows back to the recoverable state."""

    from app.database import connect

    samples_root = (_training_root() / "samples").resolve()
    with _TRAINING_LOCK:
        with connect() as con:
            rows = con.execute(
                "SELECT id,sample_path,sample_sha256 FROM anpr_feedback "
                "WHERE status='confirmed' "
                "AND training_status IN ('ready','trained') "
                "ORDER BY id"
            ).fetchall()
        invalid_ids = []
        for row in rows:
            digest = str(row["sample_sha256"] or "").upper()
            try:
                path = Path(row["sample_path"] or "").resolve()
                valid = (
                    _valid_sha256(digest)
                    and path.is_relative_to(samples_root)
                    and path.is_file()
                    and _sha256(path) == digest
                )
            except (OSError, RuntimeError, ValueError):
                valid = False
            if not valid:
                invalid_ids.append(int(row["id"]))
        if invalid_ids:
            with connect() as con:
                con.executemany(
                    "UPDATE anpr_feedback SET training_status='pending',"
                    "sample_path='',sample_sha256='',dataset_split='',"
                    "trained_run_id=NULL WHERE id=? "
                    "AND status='confirmed' "
                    "AND training_status IN ('ready','trained')",
                    ((feedback_id,) for feedback_id in invalid_ids),
                )
    return {
        "checked": len(rows),
        "reset": len(invalid_ids),
        "reset_ids": invalid_ids,
    }


def recover_pending_feedback_samples(limit=64) -> dict:
    """Retry a bounded batch of confirmed feedback still marked pending.

    The recovery is safe to call repeatedly: terminal rows leave ``pending``
    during capture and are not selected again. An unexpected failure remains
    retryable, while processing continues for every other row in this batch.
    """

    from app.database import connect

    try:
        bounded_limit = int(limit)
    except (OverflowError, TypeError, ValueError):
        bounded_limit = 0
    bounded_limit = max(
        0,
        min(MAX_PENDING_FEEDBACK_RECOVERY, bounded_limit),
    )
    if bounded_limit == 0:
        return {
            "selected": 0,
            "recovered": 0,
            "failed": 0,
            "results": [],
        }

    with _TRAINING_LOCK:
        with connect() as con:
            feedback_ids = [
                int(row["id"])
                for row in con.execute(
                    "SELECT id FROM anpr_feedback "
                    "WHERE status='confirmed' "
                    "AND training_status='pending' "
                    "ORDER BY id LIMIT ?",
                    (bounded_limit,),
                ).fetchall()
            ]

        results = []
        recovered = 0
        for feedback_id in feedback_ids:
            try:
                outcome = dict(capture_feedback_sample(feedback_id))
            except Exception as exc:
                outcome = {
                    "ready": False,
                    "reason": "capture-error:"
                    + type(exc).__name__,
                }
            recovered += int(bool(outcome.get("ready")))
            results.append({"feedback_id": feedback_id, **outcome})

    return {
        "selected": len(feedback_ids),
        "recovered": recovered,
        "failed": len(feedback_ids) - recovered,
        "results": results,
    }


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
    unsupported_ids = [
        int(row["id"])
        for row in rows
        if plausible_plate(normalize_plate(row["corrected_norm"]))
        and not _training_label_supported(row["corrected_norm"])
    ]
    if unsupported_ids:
        placeholders = ",".join("?" for _ in unsupported_ids)
        with connect() as con:
            con.execute(
                "UPDATE anpr_feedback SET "
                "training_status='unsupported-alphabet' "
                f"WHERE id IN ({placeholders})",
                unsupported_ids,
            )
        unsupported = set(unsupported_ids)
        rows = [
            row for row in rows if int(row["id"]) not in unsupported
        ]
    labels_by_digest = {}
    for row in rows:
        digest = str(row["sample_sha256"] or "").upper()
        label = normalize_plate(row["corrected_norm"])
        if _valid_sha256(digest) and plausible_plate(label):
            labels_by_digest.setdefault(digest, set()).add(label)
    conflicting = {
        digest
        for digest, labels in labels_by_digest.items()
        if len(labels) != 1
    }
    samples_root = (_training_root() / "samples").resolve()
    candidates = []
    for row in rows:
        try:
            path = Path(row["sample_path"] or "").resolve()
            path_is_trusted = path.is_relative_to(samples_root)
        except (OSError, RuntimeError, ValueError):
            path = Path()
            path_is_trusted = False
        label = normalize_plate(row["corrected_norm"])
        digest = str(row["sample_sha256"] or "").upper()
        try:
            digest_matches = path.is_file() and _sha256(path) == digest
        except OSError:
            digest_matches = False
        if (
            not plausible_plate(label)
            or not _valid_sha256(digest)
            or not path_is_trusted
            or not digest_matches
            or digest in conflicting
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


def export_manifest(
    run_id: int | None = None,
    *,
    rights_attested: bool = False,
    attested_by: str = "",
) -> Path:
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
    generated_at = datetime.now(timezone.utc).isoformat()
    attester = str(attested_by or "").strip()[:120]
    rights_verified = bool(rights_attested is True and attester)
    payload = {
        "schema": 2,
        "generated_at": generated_at,
        "training_source": "operator-confirmed-only",
        "source_license": (
            "operator-confirmed-company-owned"
            if rights_verified
            else "operator-confirmed-rights-unverified"
        ),
        "ownership_attested": rights_verified,
        "distribution_allowed": rights_verified,
        "license_evidence": (
            f"bcvision-admin-attestation:{attester}:{generated_at}"
            if rights_verified
            else ""
        ),
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


def _reject_unsupported_manifest_labels(
    manifest: Path,
    *,
    expected_sha256: str,
) -> None:
    """Fail before CRNN target encoding and quarantine referenced feedback."""

    from app.database import connect

    payload = _training_manifest_payload(
        manifest,
        expected_sha256=expected_sha256,
    )
    unsupported_ids = sorted({
        int(sample.get("feedback_id", 0))
        for sample in payload["samples"]
        if not _training_label_supported(sample.get("plate", ""))
        and int(sample.get("feedback_id", 0)) > 0
    })
    if not unsupported_ids:
        return
    placeholders = ",".join("?" for _ in unsupported_ids)
    with connect() as con:
        con.execute(
            "UPDATE anpr_feedback SET "
            "training_status='unsupported-alphabet' "
            f"WHERE id IN ({placeholders})",
            unsupported_ids,
        )
    raise ValueError("Training snapshot contains an unsupported alphabet")


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
        _reject_unsupported_manifest_labels(
            manifest,
            expected_sha256=manifest_sha256,
        )
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


def start_training(
    device="auto",
    epochs=12,
    *,
    rights_attested=False,
    attested_by="",
) -> dict:
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
        manifest = export_manifest(
            run_id,
            rights_attested=bool(rights_attested),
            attested_by=str(attested_by or ""),
        )
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
