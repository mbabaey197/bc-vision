from app.ai.plate_rules import (
    PLATE_NORMALIZATION_VERSION,
    normalize_plate,
    plausible_plate,
)


def test_plate_normalization_v2_is_unicode_safe_and_idempotent():
    canonical = "55ط63974"
    variants = (
        "۵۵ ط ۶۳۹ ایران ۷۴",
        "٥٥،ط،٦٣٩،ايران،٧٤",
        "۵۵ ط ۶۳۹ اِیْران ۷۴",
        "۵۵ ط ۶۳۹ اﻳران ۷۴",
        "۵۵ ط ۶۳۹ اىران ۷۴",
    )

    assert PLATE_NORMALIZATION_VERSION >= 2
    for value in variants:
        normalized = normalize_plate(value)
        assert normalized == canonical
        assert normalize_plate(normalized) == canonical
        assert plausible_plate(normalized)


def test_plate_normalization_drops_formatting_not_letters():
    assert normalize_plate("۳۱،طِ،۵۵۶،ایران،۷۴") == "31ط55674"
    assert normalize_plate("31-D-556-74 IRAN") == "31D55674"

