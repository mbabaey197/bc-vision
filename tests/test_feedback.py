import sqlite3

import pytest

from app.ai.feedback import apply_learned_correction, validate_correction
from app.ai.plate_rules import iran_plate_parts, persian_digits


def test_persian_plate_display_parts():
    assert persian_digits("1405/05/06") == "۱۴۰۵/۰۵/۰۶"
    assert iran_plate_parts("12-ب-345-67") == {
        "prefix": "۱۲",
        "letter": "ب",
        "serial": "۳۴۵",
        "region": "۶۷",
    }


def test_correction_requires_complete_iranian_plate():
    assert validate_correction("۱۲ ب ۳۴۵ ایران ۶۷") == (
        "12-ب-345-67",
        "12ب34567",
    )
    with pytest.raises(ValueError):
        validate_correction("۱۲۳")


def test_exact_confirmed_feedback_is_reused(tmp_path, monkeypatch):
    db_path = tmp_path / "feedback.db"
    with sqlite3.connect(db_path) as con:
        con.executescript("""
        CREATE TABLE anpr_feedback(
            id INTEGER PRIMARY KEY,
            observed_norm TEXT,
            corrected_text TEXT,
            corrected_norm TEXT,
            status TEXT
        );
        INSERT INTO anpr_feedback(
            observed_norm,corrected_text,corrected_norm,status
        ) VALUES('12ب34576','12-ب-345-67','12ب34567','confirmed');
        """)

    import app.database
    monkeypatch.setattr(app.database, "DB_PATH", db_path)
    result = apply_learned_correction({
        "plate": "12-ب-345-76",
        "plate_norm": "12ب34576",
        "valid": True,
    })

    assert result["plate"] == "12-ب-345-67"
    assert result["plate_norm"] == "12ب34567"
    assert result["operator_learned"] is True
