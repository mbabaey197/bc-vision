import csv
import sqlite3
from io import StringIO

import app.csv_export as csv_export
from app.csv_export import csv_safe_cell, iter_event_csv


def _connect(path):
    def open_connection():
        connection = sqlite3.connect(path)
        connection.row_factory = sqlite3.Row
        return connection

    return open_connection


def _create_events_table(connection, *, id_type="INTEGER PRIMARY KEY"):
    connection.execute(
        "CREATE TABLE plate_events("
        f"id {id_type},plate_text TEXT,confidence REAL,"
        "camera_name TEXT,city TEXT,plate_region TEXT,"
        "vehicle_type TEXT,vehicle_color TEXT,"
        "vehicle_confidence REAL,created_at TEXT)"
    )


def _read_rows(chunks):
    payload = "".join(chunks)
    return list(csv.reader(StringIO(payload.lstrip("\ufeff"))))


def test_event_export_is_keyset_batched_complete_and_formula_safe(tmp_path):
    database = tmp_path / "events.db"
    connect = _connect(database)
    with connect() as connection:
        _create_events_table(connection)
        connection.executemany(
            "INSERT INTO plate_events VALUES(?,?,?,?,?,?,?,?,?,?)",
            [
                (
                    index,
                    "=HYPERLINK('bad')" if index == 1 else f"12ب{index:05d}",
                    0.9,
                    "+camera" if index == 1 else "Gate",
                    "تهران",
                    "67",
                    "خودرو",
                    "سفید",
                    0.8,
                    f"2026-08-21 10:00:{index:02d}",
                )
                for index in range(1, 8)
            ],
        )

    chunks = list(iter_event_csv(
        connect,
        lambda value: f"FA:{value}",
        batch_size=3,
    ))

    assert len(chunks) == 3
    assert chunks[0].startswith("\ufeff")
    assert all("\ufeff" not in chunk for chunk in chunks[1:])
    rows = _read_rows(chunks)
    assert len(rows) == 8
    assert [int(row[0]) for row in rows[1:]] == list(range(7, 0, -1))
    assert rows[-1][1].startswith("'=HYPERLINK")
    assert rows[-1][3] == "'+camera"
    assert rows[1][-1].startswith("FA:")


def test_empty_export_still_has_one_header_chunk(tmp_path):
    database = tmp_path / "empty.db"
    connect = _connect(database)
    with connect() as connection:
        _create_events_table(connection)

    chunks = list(iter_event_csv(connect, str, batch_size=2))

    assert len(chunks) == 1
    assert chunks[0].startswith("\ufeffردیف,")


def test_csv_safe_cell_blocks_formula_and_whitespace_control_prefixes():
    unsafe = (
        "=x",
        "+x",
        "-x",
        "@x",
        "\tx",
        "\rx",
        "\nx",
        " =x",
        "\v@x",
        "\x00+x",
        "\u200b-x",
        "\u00a0@x",
    )

    assert [csv_safe_cell(value) for value in unsafe] == [
        "'" + value for value in unsafe
    ]
    assert csv_safe_cell("camera-1") == "camera-1"
    assert csv_safe_cell(0.95) == "0.95"


def test_every_export_column_is_formula_safe_including_text_in_real(tmp_path):
    database = tmp_path / "hostile.db"
    connect = _connect(database)
    values = (
        " =id",
        "\t=plate",
        "\x00+confidence",
        "\u200b-camera",
        "\n@city",
        "\r=region",
        "\v+type",
        "\f-color",
        "\u00a0@vehicle-confidence",
        "ignored by formatter",
    )
    with connect() as connection:
        _create_events_table(connection, id_type="TEXT PRIMARY KEY")
        connection.execute(
            "INSERT INTO plate_events VALUES(?,?,?,?,?,?,?,?,?,?)",
            values,
        )

    rows = _read_rows(iter_event_csv(
        connect,
        lambda _value: "\x1f=formatted-date",
        batch_size=1,
    ))

    assert len(rows) == 2
    assert all(cell.startswith("'") for cell in rows[1])
    assert rows[1][2] == "'\x00+confidence"
    assert rows[1][8] == "'\u00a0@vehicle-confidence"


def test_slow_consumer_does_not_hold_reader_or_block_wal_checkpoint(tmp_path):
    database = tmp_path / "wal.db"
    connect = _connect(database)
    with connect() as connection:
        assert connection.execute("PRAGMA journal_mode=WAL").fetchone()[0] == "wal"
        _create_events_table(connection)
        connection.executemany(
            "INSERT INTO plate_events VALUES(?,?,?,?,?,?,?,?,?,?)",
            [
                (
                    index,
                    f"plate-{index}",
                    0.9,
                    "Gate",
                    "",
                    "",
                    "",
                    "",
                    0.8,
                    "now",
                )
                for index in range(1, 4)
            ],
        )

    export = iter_event_csv(connect, str, batch_size=1)
    first_chunk = next(export)
    assert "plate-3" in first_chunk

    # Leave the streaming iterator suspended to model a slow downloader. A
    # writer and TRUNCATE checkpoint must still complete immediately.
    with connect() as writer:
        writer.execute(
            "INSERT INTO plate_events VALUES(?,?,?,?,?,?,?,?,?,?)",
            (4, "plate-4", 0.9, "Gate", "", "", "", "", 0.8, "now"),
        )
    with connect() as checkpoint:
        checkpoint.execute("PRAGMA busy_timeout=0")
        busy, _log_frames, _checkpointed = checkpoint.execute(
            "PRAGMA wal_checkpoint(TRUNCATE)"
        ).fetchone()

    export.close()
    assert busy == 0


def test_export_has_row_byte_and_chunk_limits_with_explicit_notice(tmp_path):
    database = tmp_path / "bounded.db"
    connect = _connect(database)
    with connect() as connection:
        _create_events_table(connection)
        connection.executemany(
            "INSERT INTO plate_events VALUES(?,?,?,?,?,?,?,?,?,?)",
            [
                (
                    index,
                    "P" * 300,
                    0.9,
                    "C" * 300,
                    "city",
                    "region",
                    "type",
                    "color",
                    0.8,
                    "now",
                )
                for index in range(1, 6)
            ],
        )

    max_bytes = 900
    chunks = list(iter_event_csv(
        connect,
        str,
        batch_size=5,
        max_rows=2,
        max_bytes=max_bytes,
        max_chunk_bytes=64,
    ))
    rows = _read_rows(chunks)

    assert chunks
    assert all(len(chunk.encode("utf-8")) <= 64 for chunk in chunks)
    assert sum(len(chunk.encode("utf-8")) for chunk in chunks) <= max_bytes
    assert rows[-1][0] == "#LIMIT"
    assert "خروجی محدود شد" in rows[-1][1]
    assert len([row for row in rows[1:] if row[0] != "#LIMIT"]) <= 2


def test_multi_megabyte_database_cell_is_bounded_inside_sql(tmp_path):
    database = tmp_path / "oversized-cell.db"
    connect = _connect(database)
    hostile = "=X" + "z" * (5 * 1024 * 1024)
    with connect() as connection:
        _create_events_table(connection)
        connection.execute(
            "INSERT INTO plate_events VALUES(?,?,?,?,?,?,?,?,?,?)",
            (1, hostile, 0.9, "Gate", "", "", "", "", 0.8, "now"),
        )

    chunks = list(iter_event_csv(connect, str, batch_size=1))
    rows = _read_rows(chunks)

    assert len(rows) == 2
    assert rows[1][1].startswith("'=X")
    assert rows[1][1].endswith("…[کوتاه‌شده]")
    assert len(rows[1][1]) <= 4096
    assert len("".join(chunks).encode("utf-8")) < 64 * 1024


def test_requested_large_batch_is_capped_for_slow_consumer_memory(
    tmp_path,
    monkeypatch,
):
    database = tmp_path / "batch-cap.db"
    connect = _connect(database)
    with connect() as connection:
        _create_events_table(connection)
        connection.executemany(
            "INSERT INTO plate_events VALUES(?,?,?,?,?,?,?,?,?,?)",
            [
                (
                    index,
                    "P" * 5000,
                    0.9,
                    "C" * 5000,
                    "city",
                    "region",
                    "type",
                    "color",
                    0.8,
                    "now",
                )
                for index in range(1, 21)
            ],
        )
    limits = []
    real_fetch = csv_export._fetch_page

    def tracked_fetch(open_connection, *, after_id, limit):
        limits.append(limit)
        return real_fetch(
            open_connection,
            after_id=after_id,
            limit=limit,
        )

    monkeypatch.setattr(csv_export, "_fetch_page", tracked_fetch)
    list(iter_event_csv(connect, str, batch_size=500, max_rows=20))

    assert limits
    assert max(limits) <= csv_export.MAX_EVENT_EXPORT_BATCH_SIZE + 1
