import sqlite3

from app import database
from app.ai.plate_rules import PLATE_NORMALIZATION_VERSION


def _connection():
    con = sqlite3.connect(":memory:")
    con.row_factory = sqlite3.Row
    con.executescript(
        """
        CREATE TABLE settings(key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE plate_events(
            id INTEGER PRIMARY KEY,
            plate_text TEXT,
            plate_norm TEXT,
            raw_guess_text TEXT,
            raw_guess_norm TEXT
        );
        CREATE TABLE anpr_feedback(
            id INTEGER PRIMARY KEY,
            observed_text TEXT,
            observed_norm TEXT,
            corrected_text TEXT,
            corrected_norm TEXT
        );
        CREATE TABLE plate_watchlist(
            id INTEGER PRIMARY KEY,
            plate_text TEXT,
            plate_norm TEXT UNIQUE
        );
        """
    )
    return con


def test_versioned_migration_recanonicalizes_existing_nonempty_keys():
    con = _connection()
    con.execute(
        "INSERT INTO plate_events VALUES(1,?,?,?,?)",
        (
            "۵۵،ط،۶۳۹،ايران،۷۴",
            "55،ط،639،ايران،74",
            "۳۱ طِ ۵۵۶ ایران ۷۴",
            "31طِ55674",
        ),
    )
    con.execute(
        "INSERT INTO anpr_feedback VALUES(1,?,?,?,?)",
        (
            "۵۵،ط،۶۳۹،ايران،۷۴",
            "55،ط،639،ايران،74",
            "۵۵ ط ۶۳۹ ایران ۷۴",
            "55ط63974",
        ),
    )
    con.execute(
        "INSERT INTO plate_watchlist VALUES(1,?,?)",
        ("۵۵،ط،۶۳۹،ايران،۷۴", "legacy-key"),
    )

    database._backfill_plate_norm(con)

    event = con.execute("SELECT * FROM plate_events").fetchone()
    assert event["plate_norm"] == "55ط63974"
    assert event["raw_guess_norm"] == "31ط55674"
    feedback = con.execute("SELECT * FROM anpr_feedback").fetchone()
    assert feedback["observed_norm"] == "55ط63974"
    assert feedback["corrected_norm"] == "55ط63974"
    assert con.execute(
        "SELECT plate_norm FROM plate_watchlist"
    ).fetchone()[0] == "55ط63974"
    assert int(con.execute(
        "SELECT value FROM settings WHERE key='plate_normalization_version'"
    ).fetchone()[0]) == PLATE_NORMALIZATION_VERSION


def test_current_version_still_fills_new_empty_keys():
    con = _connection()
    con.execute(
        "INSERT INTO settings VALUES('plate_normalization_version',?)",
        (str(PLATE_NORMALIZATION_VERSION),),
    )
    con.execute(
        "INSERT INTO plate_events VALUES(1,?,?,?,?)",
        ("۳۱ ط ۵۵۶ ایران ۷۴", "", "", ""),
    )

    database._backfill_plate_norm(con)

    assert con.execute(
        "SELECT plate_norm FROM plate_events"
    ).fetchone()[0] == "31ط55674"
