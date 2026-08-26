from dataclasses import replace
import json
import sqlite3

import numpy as np
import pytest

from app.ai import persistence_outbox
from app.ai.persistence_outbox import (
    OutboxEntry,
    OutboxError,
    OutboxPayloadError,
    PersistenceOutbox,
)


OBSERVED_AT = "2026-08-21T09:30:00.123456Z"


def entry(
    retry_id,
    *,
    frame_value=80,
    plate="31ط55674",
    track_id=1,
    event_id=None,
    extra_result=None,
):
    frame = np.full((72, 160, 3), frame_value, dtype=np.uint8)
    crop = frame[20:52, 25:135].copy()
    result = {
        "plate": plate,
        "plate_norm": plate,
        "valid": True,
        "confidence": 0.94,
        "ocr_confidence": np.float32(0.91),
        "track_id": track_id,
        "bbox": (25, 20, 135, 52),
        "crop": crop,
        "vehicle_crop": frame.copy(),
        "private_runtime_object": object(),
    }
    if extra_result:
        result.update(extra_result)
    return OutboxEntry.from_images(
        retry_id=retry_id,
        state_scope="runtime-a:camera-1",
        camera_id=1,
        camera_name="دروازه شمالی",
        result=result,
        frame=frame,
        crop=crop,
        observed_at_utc=OBSERVED_AT,
        processing_ms=18.25,
        duplicate_seconds=30.0,
        detector_generation=2,
        detector_revision="yolo11n:test",
        track_id=track_id,
        identity=plate,
        emission_kind="confirmed",
        event_id=event_id,
        ledger_key=plate,
    )


def create_legacy_outbox(path, rows):
    """Create the v1 layout used before durable lineage/media policy."""

    with sqlite3.connect(path) as con:
        con.executescript("""
        CREATE TABLE retry_outbox(
            seq INTEGER PRIMARY KEY AUTOINCREMENT,
            retry_id TEXT NOT NULL UNIQUE,
            state_scope TEXT NOT NULL,
            schema_version INTEGER NOT NULL,
            camera_id INTEGER NOT NULL,
            camera_name TEXT NOT NULL,
            detector_generation INTEGER NOT NULL,
            detector_revision TEXT NOT NULL DEFAULT '',
            track_id INTEGER NOT NULL,
            identity TEXT NOT NULL DEFAULT '',
            emission_kind TEXT NOT NULL,
            result_json TEXT NOT NULL,
            frame_jpeg BLOB NOT NULL,
            crop_jpeg BLOB,
            event_id INTEGER,
            ledger_key TEXT NOT NULL DEFAULT '',
            observed_at_utc TEXT NOT NULL,
            processing_ms REAL NOT NULL,
            duplicate_seconds REAL NOT NULL,
            attempts INTEGER NOT NULL DEFAULT 0,
            first_failed_at_utc TEXT NOT NULL DEFAULT '',
            next_attempt_at_epoch REAL NOT NULL DEFAULT 0,
            last_error TEXT NOT NULL DEFAULT '',
            payload_sha256 TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            quarantine_reason TEXT NOT NULL DEFAULT '',
            quarantined_at_utc TEXT NOT NULL DEFAULT '',
            created_at_utc TEXT NOT NULL,
            updated_at_utc TEXT NOT NULL
        );
        PRAGMA user_version=1;
        """)
        digests = {}
        for item in rows:
            result_json = persistence_outbox._result_json(item.result)
            digest = persistence_outbox._payload_digest(
                item,
                result_json,
                item.frame_jpeg,
                item.crop_jpeg,
                schema_version=1,
                include_media_policy=False,
            )
            digests[item.retry_id] = digest
            con.execute(
                "INSERT INTO retry_outbox("
                "retry_id,state_scope,schema_version,camera_id,camera_name,"
                "detector_generation,detector_revision,track_id,identity,"
                "emission_kind,result_json,frame_jpeg,crop_jpeg,event_id,"
                "ledger_key,observed_at_utc,processing_ms,duplicate_seconds,"
                "attempts,first_failed_at_utc,next_attempt_at_epoch,last_error,"
                "payload_sha256,status,quarantine_reason,quarantined_at_utc,"
                "created_at_utc,updated_at_utc"
                ") VALUES("
                ":retry_id,:state_scope,1,:camera_id,:camera_name,"
                ":detector_generation,:detector_revision,:track_id,:identity,"
                ":emission_kind,:result_json,:frame_jpeg,:crop_jpeg,:event_id,"
                ":ledger_key,:observed_at_utc,:processing_ms,"
                ":duplicate_seconds,:attempts,:first_failed_at_utc,"
                ":next_attempt_at_epoch,:last_error,:payload_sha256,'pending',"
                "'','','2026-08-21T09:30:01.000000Z',"
                "'2026-08-21T09:30:01.000000Z')",
                {
                    "retry_id": item.retry_id,
                    "state_scope": item.state_scope,
                    "camera_id": item.camera_id,
                    "camera_name": item.camera_name,
                    "detector_generation": item.detector_generation,
                    "detector_revision": item.detector_revision,
                    "track_id": item.track_id,
                    "identity": item.identity,
                    "emission_kind": item.emission_kind,
                    "result_json": result_json,
                    "frame_jpeg": sqlite3.Binary(item.frame_jpeg),
                    "crop_jpeg": (
                        sqlite3.Binary(item.crop_jpeg)
                        if item.crop_jpeg is not None else None
                    ),
                    "event_id": item.event_id,
                    "ledger_key": item.ledger_key,
                    "observed_at_utc": item.observed_at_utc,
                    "processing_ms": item.processing_ms,
                    "duplicate_seconds": item.duplicate_seconds,
                    "attempts": item.attempts,
                    "first_failed_at_utc": item.first_failed_at_utc,
                    "next_attempt_at_epoch": item.next_attempt_at_epoch,
                    "last_error": item.last_error,
                    "payload_sha256": digest,
                },
            )
    return digests


def test_outbox_uses_wal_full_sync_and_blob_schema(tmp_path):
    path = tmp_path / "retry.db"
    PersistenceOutbox(path)

    with sqlite3.connect(path) as con:
        assert con.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert con.execute("PRAGMA synchronous").fetchone()[0] == 2
        assert con.execute("PRAGMA user_version").fetchone()[0] == 2
        columns = {
            row[1]: row[2]
            for row in con.execute("PRAGMA table_info(retry_outbox)")
        }

    assert columns["result_json"] == "TEXT"
    assert columns["frame_jpeg"] == "BLOB"
    assert columns["crop_jpeg"] == "BLOB"
    assert columns["payload_sha256"] == "TEXT"
    assert columns["quarantine_reason"] == "TEXT"
    assert columns["predecessor_id"] == "TEXT"
    assert columns["plate_root"] == "TEXT"
    assert columns["snapshot_root"] == "TEXT"
    assert columns["save_plate"] == "INTEGER"
    assert columns["save_vehicle"] == "INTEGER"
    with sqlite3.connect(path) as con:
        indexes = {
            row[1] for row in con.execute("PRAGMA index_list(retry_outbox)")
        }
    assert "idx_retry_outbox_camera_status" in indexes


def test_v1_upgrade_validates_legacy_checksum_before_rewriting(tmp_path):
    path = tmp_path / "legacy-retry.db"
    original_digests = create_legacy_outbox(
        path,
        [
            entry("legacy-valid", event_id=41),
            entry("legacy-corrupt", track_id=2),
        ],
    )
    with sqlite3.connect(path) as con:
        # Keep the payload canonical but make it disagree with its v1 digest.
        con.execute(
            "UPDATE retry_outbox SET result_json='{}' "
            "WHERE retry_id='legacy-corrupt'"
        )

    upgraded = PersistenceOutbox(path)
    recovered = upgraded.recover()

    assert [item.retry_id for item in recovered.entries] == ["legacy-valid"]
    valid = recovered.entries[0]
    assert valid.event_id == 41
    assert valid.predecessor_id == ""
    assert valid.plate_root == ""
    assert valid.snapshot_root == ""
    assert valid.save_plate is True
    assert valid.save_vehicle is True
    assert [item.retry_id for item in recovered.quarantined] == [
        "legacy-corrupt"
    ]
    assert "checksum mismatch" in recovered.quarantined[0].quarantine_reason

    with sqlite3.connect(path) as con:
        assert con.execute("PRAGMA user_version").fetchone()[0] == 2
        columns = {
            row[1] for row in con.execute("PRAGMA table_info(retry_outbox)")
        }
        stored = {
            row[0]: row[1:]
            for row in con.execute(
                "SELECT retry_id,schema_version,status,payload_sha256 "
                "FROM retry_outbox ORDER BY seq"
            )
        }
    assert {
        "predecessor_id",
        "plate_root",
        "snapshot_root",
        "save_plate",
        "save_vehicle",
    } <= columns
    assert stored["legacy-valid"][0:2] == (2, "pending")
    assert (
        stored["legacy-valid"][2]
        != original_digests["legacy-valid"]
    )
    assert stored["legacy-corrupt"] == (
        1,
        "quarantined",
        original_digests["legacy-corrupt"],
    )

    # The migration is safe to run again and does not legitimize quarantine.
    reopened = PersistenceOutbox(path).recover()
    assert [item.retry_id for item in reopened.entries] == ["legacy-valid"]
    assert [item.retry_id for item in reopened.quarantined] == [
        "legacy-corrupt"
    ]


def test_newer_outbox_schema_is_never_downgraded(tmp_path):
    path = tmp_path / "future-retry.db"
    with sqlite3.connect(path) as con:
        con.execute("PRAGMA user_version=99")

    with pytest.raises(OutboxError, match="newer BC Vision"):
        PersistenceOutbox(path)

    with sqlite3.connect(path) as con:
        assert con.execute("PRAGMA user_version").fetchone()[0] == 99
        assert con.execute(
            "SELECT COUNT(*) FROM sqlite_master "
            "WHERE type='table' AND name='retry_outbox'"
        ).fetchone()[0] == 0


def test_round_trip_is_ordered_whitelisted_json_with_jpeg_evidence(tmp_path):
    path = tmp_path / "retry.db"
    outbox = PersistenceOutbox(path)
    second_seq = outbox.upsert(entry("retry-2", frame_value=120, track_id=2))
    first_seq = outbox.upsert(entry("retry-1", frame_value=60, track_id=1))

    recovered = PersistenceOutbox(path).load()

    assert [item.retry_id for item in recovered] == ["retry-2", "retry-1"]
    assert [item.seq for item in recovered] == [second_seq, first_seq]
    assert recovered[0].result["plate_norm"] == "31ط55674"
    assert recovered[0].result["bbox"] == [25, 20, 135, 52]
    assert recovered[0].result["ocr_confidence"] == pytest.approx(0.91)
    assert "crop" not in recovered[0].result
    assert "vehicle_crop" not in recovered[0].result
    assert "private_runtime_object" not in recovered[0].result
    assert recovered[0].decode_frame().shape == (72, 160, 3)
    assert recovered[0].decode_crop().shape == (32, 110, 3)
    assert len(recovered[0].payload_sha256) == 64

    with sqlite3.connect(path) as con:
        row = con.execute(
            "SELECT result_json,typeof(frame_jpeg) FROM retry_outbox "
            "WHERE retry_id='retry-2'"
        ).fetchone()
    assert row[1] == "blob"
    stored_result = json.loads(row[0])
    assert set(stored_result) <= set(recovered[0].result)


def test_upsert_keeps_seq_event_id_and_failure_history(tmp_path):
    outbox = PersistenceOutbox(tmp_path / "retry.db")
    original_seq = outbox.upsert(entry("same", event_id=77))
    assert outbox.update_failure(
        "same",
        "database busy",
        next_attempt_at_epoch=123.5,
        failed_at_utc="2026-08-21T09:31:00Z",
    ) == 1

    replacement = entry(
        "same",
        frame_value=190,
        event_id=88,
        extra_result={"confidence": 0.98},
    )
    assert outbox.upsert(replacement) == original_seq

    loaded = outbox.load()[0]
    assert loaded.seq == original_seq
    assert loaded.event_id == 77
    assert loaded.attempts == 1
    assert loaded.first_failed_at_utc == "2026-08-21T09:31:00.000000Z"
    assert loaded.next_attempt_at_epoch == 123.5
    assert loaded.last_error == "database busy"
    assert loaded.result["confidence"] == 0.98
    assert int(loaded.decode_frame().mean()) == pytest.approx(190, abs=2)


def test_lineage_and_media_policy_fields_round_trip_on_insert_and_upsert(
    tmp_path,
):
    outbox = PersistenceOutbox(tmp_path / "retry.db")
    original = replace(
        entry("lineage-media"),
        predecessor_id="prior-emission-41",
        plate_root=str(tmp_path / "old" / "plates"),
        snapshot_root=str(tmp_path / "old" / "snapshots"),
        save_plate=False,
        save_vehicle=True,
    )

    seq = outbox.upsert(original)
    loaded = outbox.load()[0]

    assert loaded.seq == seq
    assert loaded.predecessor_id == "prior-emission-41"
    assert loaded.plate_root == str(tmp_path / "old" / "plates")
    assert loaded.snapshot_root == str(tmp_path / "old" / "snapshots")
    assert loaded.save_plate is False
    assert loaded.save_vehicle is True

    replacement = replace(
        original,
        predecessor_id="prior-emission-42",
        plate_root=str(tmp_path / "new" / "plates"),
        snapshot_root=str(tmp_path / "new" / "snapshots"),
        save_plate=True,
        save_vehicle=False,
    )
    assert outbox.upsert(replacement) == seq

    refreshed = outbox.load()[0]
    assert refreshed.predecessor_id == "prior-emission-42"
    assert refreshed.plate_root == str(tmp_path / "new" / "plates")
    assert refreshed.snapshot_root == str(tmp_path / "new" / "snapshots")
    assert refreshed.save_plate is True
    assert refreshed.save_vehicle is False

    with sqlite3.connect(outbox.path) as con:
        raw = con.execute(
            "SELECT predecessor_id,plate_root,snapshot_root,"
            "save_plate,save_vehicle FROM retry_outbox WHERE retry_id=?",
            ("lineage-media",),
        ).fetchone()
    assert raw == (
        "prior-emission-42",
        str(tmp_path / "new" / "plates"),
        str(tmp_path / "new" / "snapshots"),
        1,
        0,
    )


def test_due_filter_and_delete_acknowledged_row(tmp_path):
    outbox = PersistenceOutbox(tmp_path / "retry.db")
    outbox.upsert(entry("now", track_id=1))
    outbox.upsert(entry("later", track_id=2))
    outbox.update_failure(
        "later",
        "offline",
        next_attempt_at_epoch=500.0,
    )

    assert [item.retry_id for item in outbox.load(due_at_epoch=499.0)] == [
        "now"
    ]
    assert [item.retry_id for item in outbox.load(due_at_epoch=500.0)] == [
        "now",
        "later",
    ]
    assert outbox.delete("now") is True
    assert outbox.delete("now") is False
    assert outbox.pending_count() == 1


def test_after_seq_pages_pending_rows_without_overlap(tmp_path):
    outbox = PersistenceOutbox(tmp_path / "retry.db")
    inserted = [
        outbox.upsert(entry(f"page-{index}", track_id=index))
        for index in range(1, 6)
    ]

    first = outbox.load(limit=2)
    second = outbox.load(limit=2, after_seq=first[-1].seq)
    third = outbox.load(limit=2, after_seq=second[-1].seq)

    recovered = first + second + third
    assert [item.seq for item in recovered] == inserted
    assert [item.retry_id for item in recovered] == [
        "page-1",
        "page-2",
        "page-3",
        "page-4",
        "page-5",
    ]
    assert outbox.load(after_seq=inserted[-1]) == []


def test_pending_stats_are_camera_scoped_and_ignore_quarantine(tmp_path):
    outbox = PersistenceOutbox(tmp_path / "retry.db")
    camera_one_a = entry("camera-1-a", frame_value=50, track_id=1)
    camera_one_b = entry("camera-1-b", frame_value=80, track_id=2)
    camera_two = replace(
        entry("camera-2", frame_value=110, track_id=3),
        camera_id=2,
        state_scope="runtime-a:camera-2",
    )
    for item in (camera_one_a, camera_one_b, camera_two):
        outbox.upsert(item)

    one_bytes = sum(
        len(item.frame_jpeg) + len(item.crop_jpeg or b"")
        for item in (camera_one_a, camera_one_b)
    )
    two_bytes = len(camera_two.frame_jpeg) + len(camera_two.crop_jpeg or b"")
    assert outbox.pending_stats(1) == (2, one_bytes)
    assert outbox.pending_stats(2) == (1, two_bytes)
    assert outbox.pending_stats() == (3, one_bytes + two_bytes)

    assert outbox.quarantine("camera-1-a", "test quarantine") is True
    remaining_one_bytes = (
        len(camera_one_b.frame_jpeg) + len(camera_one_b.crop_jpeg or b"")
    )
    assert outbox.pending_stats(1) == (1, remaining_one_bytes)
    assert outbox.pending_stats() == (2, remaining_one_bytes + two_bytes)


def test_failure_metadata_survives_a_new_store_instance(tmp_path):
    path = tmp_path / "retry.db"
    first = PersistenceOutbox(path)
    first.upsert(entry("survives", event_id=91))
    first.update_failure(
        "survives",
        "OSError: storage temporarily unavailable",
        next_attempt_at_epoch=777.25,
        failed_at_utc="2026-08-21T10:00:00+00:00",
    )

    report = PersistenceOutbox(path).recover()

    assert len(report.entries) == 1
    restored = report.entries[0]
    assert restored.retry_id == "survives"
    assert restored.event_id == 91
    assert restored.attempts == 1
    assert restored.next_attempt_at_epoch == 777.25
    assert "temporarily unavailable" in restored.last_error
    assert report.quarantined == ()


def test_corrupt_payload_is_quarantined_and_never_deleted(tmp_path):
    path = tmp_path / "retry.db"
    outbox = PersistenceOutbox(path)
    outbox.upsert(entry("corrupt"))
    outbox.upsert(entry("healthy", track_id=2))
    with sqlite3.connect(path) as con:
        con.execute(
            "UPDATE retry_outbox SET result_json='not-json' "
            "WHERE retry_id='corrupt'"
        )

    report = PersistenceOutbox(path).recover()

    assert [item.retry_id for item in report.entries] == ["healthy"]
    assert len(report.quarantined) == 1
    quarantine = report.quarantined[0]
    assert quarantine.retry_id == "corrupt"
    assert "payload" in quarantine.quarantine_reason.lower()
    assert outbox.pending_count() == 1
    assert outbox.quarantined_count() == 1
    with sqlite3.connect(path) as con:
        retained = con.execute(
            "SELECT status,result_json,quarantine_reason "
            "FROM retry_outbox WHERE retry_id='corrupt'"
        ).fetchone()
    assert retained[0] == "quarantined"
    assert retained[1] == "not-json"
    assert retained[2]


def test_corrupt_jpeg_checksum_is_quarantined(tmp_path):
    path = tmp_path / "retry.db"
    outbox = PersistenceOutbox(path)
    outbox.upsert(entry("bad-image"))
    with sqlite3.connect(path) as con:
        con.execute(
            "UPDATE retry_outbox SET frame_jpeg=X'00010203' "
            "WHERE retry_id='bad-image'"
        )

    report = outbox.recover()

    assert report.entries == ()
    assert len(report.quarantined) == 1
    assert report.quarantined[0].retry_id == "bad-image"
    assert "checksum" in report.quarantined[0].quarantine_reason.lower()


def test_invalid_whitelisted_value_is_rejected_without_pickle(tmp_path):
    outbox = PersistenceOutbox(tmp_path / "retry.db")
    invalid = entry(
        "invalid",
        extra_result={"bbox": [1, 2, object(), 4]},
    )

    with pytest.raises(OutboxPayloadError, match="unsupported type"):
        outbox.upsert(invalid)

    assert outbox.pending_count() == 0


def test_text_only_entry_can_survive_when_frame_encoding_failed(tmp_path):
    outbox = PersistenceOutbox(tmp_path / "retry.db")
    text_only = OutboxEntry(
        retry_id="text-only",
        state_scope="runtime-a:camera-1",
        camera_id=1,
        camera_name="Gate",
        result={
            "plate": "ناخوانا",
            "valid": False,
            "confidence": 0.20,
            "track_id": 1,
            "bbox": [20, 20, 140, 60],
            "unreadable_final": True,
        },
        frame_jpeg=b"",
        observed_at_utc=OBSERVED_AT,
        processing_ms=12.0,
        duplicate_seconds=30.0,
        detector_generation=1,
        detector_revision="detector-a",
        track_id=1,
        identity="",
        emission_kind="unreadable",
    )

    outbox.upsert(text_only)
    recovered = outbox.load()[0]

    assert recovered.retry_id == "text-only"
    assert recovered.frame_jpeg == b""
    assert recovered.decode_frame() is None


def test_backup_includes_pending_rows_from_wal(tmp_path):
    source_path = tmp_path / "retry.db"
    backup_path = tmp_path / "moved" / "bcvision-retry.db"
    source = PersistenceOutbox(source_path)
    pending = replace(
        entry("pending-wal", event_id=73),
        predecessor_id="predecessor-72",
        plate_root=str(tmp_path / "plates"),
        snapshot_root=str(tmp_path / "snapshots"),
        save_plate=False,
        save_vehicle=True,
    )
    source.upsert(pending)
    source.update_failure(
        "pending-wal",
        "database offline",
        next_attempt_at_epoch=1234.5,
    )

    assert source.backup(backup_path) == backup_path.resolve()

    recovered = PersistenceOutbox(backup_path).recover()
    assert [item.retry_id for item in recovered.entries] == ["pending-wal"]
    assert recovered.entries[0].event_id == 73
    assert recovered.entries[0].attempts == 1
    assert recovered.entries[0].next_attempt_at_epoch == 1234.5
    assert recovered.entries[0].predecessor_id == "predecessor-72"
    assert recovered.entries[0].plate_root == str(tmp_path / "plates")
    assert recovered.entries[0].snapshot_root == str(tmp_path / "snapshots")
    assert recovered.entries[0].save_plate is False
    assert recovered.entries[0].save_vehicle is True
    with sqlite3.connect(backup_path) as con:
        assert con.execute("PRAGMA quick_check").fetchone()[0] == "ok"
