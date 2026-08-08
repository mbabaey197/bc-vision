from app.main import jalali_date, jalali_datetime


def test_all_visible_date_and_time_digits_are_persian():
    date_text = jalali_date("2026-07-28")
    datetime_text = jalali_datetime("2026-07-28 14:05:09")

    assert date_text
    assert datetime_text
    assert not any(character.isascii() and character.isdigit() for character in date_text)
    assert not any(
        character.isascii() and character.isdigit()
        for character in datetime_text
    )
    # SQLite timestamps are UTC; the Persian UI displays Tehran local time.
    assert "۱۷:۳۵:۰۹" in datetime_text
