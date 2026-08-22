import sqlite3

from app.ai.evaluation import (
    character_distance,
    feedback_quality_summary,
)


def test_character_distance_measures_plate_errors():
    assert character_distance("31-ط-566-74", "31-ط-556-74") == 1
    assert character_distance("84-ب-579-32", "84-ب-571-33") == 2
    assert character_distance("55-ط-639-74", "55-ط-639-74") == 0


def test_feedback_summary_compares_model_revisions():
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    connection.execute(
        "CREATE TABLE anpr_feedback("
        "id INTEGER PRIMARY KEY, observed_norm TEXT, corrected_norm TEXT,"
        "observed_engine TEXT, observed_model_revision TEXT,"
        "character_distance INTEGER, exact_match INTEGER,"
        "status TEXT)"
    )
    connection.executemany(
        "INSERT INTO anpr_feedback VALUES(?,?,?,?,?,?,?,?)",
        [
            (1, "31ط56674", "31ط55674", "cct", "before", 1, 0, "confirmed"),
            (2, "55ط63974", "55ط63974", "cct", "after", 0, 1, "confirmed"),
            (3, "84ب57133", "84ب57133", "cct", "after", 0, 1, "confirmed"),
        ],
    )

    result = feedback_quality_summary(connection)

    assert result["guessed"] == 3
    assert result["exact"] == 2
    assert result["mean_character_error"] == 0.3333
    assert result["by_model"][0]["model_revision"] == "after"
    assert result["by_model"][0]["exact_accuracy"] == 1.0


def test_feedback_accuracy_counts_missing_guesses_end_to_end():
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    connection.execute(
        "CREATE TABLE anpr_feedback("
        "id INTEGER PRIMARY KEY, observed_norm TEXT, corrected_norm TEXT,"
        "observed_engine TEXT, observed_model_revision TEXT,"
        "character_distance INTEGER, exact_match INTEGER,"
        "status TEXT)"
    )
    connection.executemany(
        "INSERT INTO anpr_feedback VALUES(?,?,?,?,?,?,?,?)",
        [
            (
                index,
                "31ط55674" if index == 1 else "",
                "31ط55674",
                "hezar",
                "revision-a",
                0,
                int(index == 1),
                "confirmed",
            )
            for index in range(1, 101)
        ],
    )

    result = feedback_quality_summary(connection)

    assert result["reviewed"] == 100
    assert result["guessed"] == 1
    assert result["exact"] == 1
    assert result["exact_accuracy"] == 0.01
    assert result["coverage"] == 0.01
    assert result["accepted_precision"] == 1.0
    assert result["miss_count"] == 99
    assert result["miss_rate"] == 0.99
    assert result["mean_character_error"] == 0.0
    assert result["mean_character_error_end_to_end"] == 7.92
