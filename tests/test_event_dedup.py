from app.ai.event_dedup import (
    PlateVisitLedger,
    candidate_plate_key,
    strict_plate_key,
)


def plate_result(plate, track_id):
    return {
        "plate": plate,
        "plate_norm": plate,
        "valid": True,
        "track_id": track_id,
    }


def test_plate_key_canonicalizes_persian_digits_and_separators():
    result = plate_result("۳۱-ط-۵۵۶-۷۴", 1)

    assert strict_plate_key(result) == "31ط55674"


def test_review_candidate_has_exact_key_without_becoming_confirmed():
    result = {
        "plate": "31-ط-556-74",
        "plate_norm": "",
        "raw_guess_norm": "۳۱ط۵۵۶۷۴",
        "valid": False,
        "needs_review": True,
    }

    assert strict_plate_key(result) == ""
    assert candidate_plate_key(result) == "31ط55674"


def test_continuous_visit_survives_fragmented_tracks_and_expired_timer():
    ledger = PlateVisitLedger()
    first = plate_result("31-ط-556-74", 1)
    ledger.register(first, 41, 0.0)

    for timestamp, track_id in ((31.0, 2), (62.0, 3)):
        current = plate_result("31ط55674", track_id)
        ledger.observe([current], {track_id}, timestamp, 30.0)
        key, event_ref = ledger.event_ref(current, timestamp, 30.0)

        assert key == "31ط55674"
        assert event_ref == 41

    assert ledger.seen["31ط55674"] == 62.0


def test_confirmed_absence_allows_same_plate_to_start_new_visit():
    ledger = PlateVisitLedger()
    first = plate_result("31ط55674", 1)
    ledger.register(first, 41, 0.0)

    for timestamp in (31.0, 32.0, 33.0):
        ledger.observe([], set(), timestamp, 30.0)

    returned = plate_result("31ط55674", 2)
    ledger.observe([returned], {2}, 34.0, 30.0)
    key, event_ref = ledger.event_ref(returned, 34.0, 30.0)

    assert key == "31ط55674"
    assert event_ref is None
    ledger.register(returned, 42, 34.0)
    assert ledger.event_refs["31ط55674"] == 42


def test_exact_keys_do_not_merge_one_character_difference():
    ledger = PlateVisitLedger()
    first = plate_result("12ب34567", 1)
    different = plate_result("12ب34568", 2)
    ledger.register(first, 11, 0.0)
    ledger.observe([different], {1, 2}, 0.2, 30.0)

    first_key, first_ref = ledger.event_ref(first, 0.2, 30.0)
    second_key, second_ref = ledger.event_ref(different, 0.2, 30.0)

    assert (first_key, first_ref) == ("12ب34567", 11)
    assert (second_key, second_ref) == ("12ب34568", None)


def test_distinct_review_candidate_does_not_inherit_bound_event():
    ledger = PlateVisitLedger()
    first = plate_result("31ط55674", 1)
    ledger.register(first, 11, 0.0)
    candidate = {
        "plate": "ناخوانا",
        "plate_norm": "",
        "raw_guess_text": "12-ب-345-67",
        "raw_guess_norm": "12ب34567",
        "valid": False,
        "needs_review": True,
        "track_id": 1,
    }

    retired = ledger.observe([candidate], {1}, 0.2, 30.0)
    key, event_ref = ledger.event_ref(
        candidate,
        0.2,
        30.0,
        allow_candidate=True,
    )

    assert retired == {1}
    assert (key, event_ref) == ("12ب34567", None)
    assert 1 not in ledger.track_event_refs()


def test_same_plate_is_independent_between_camera_ledgers():
    camera_one = PlateVisitLedger()
    camera_two = PlateVisitLedger()
    result = plate_result("31ط55674", 1)

    camera_one.register(result, 101, 0.0)
    camera_two.register(result, 202, 0.0)

    assert camera_one.event_ref(result, 1.0, 30.0)[1] == 101
    assert camera_two.event_ref(result, 1.0, 30.0)[1] == 202


def test_confirmed_absence_retires_still_active_one_shot_track():
    ledger = PlateVisitLedger()
    result = plate_result("31ط55674", 7)
    ledger.register(result, 41, 0.0)

    retired = set()
    for timestamp in (0.8, 1.2, 1.6):
        retired.update(ledger.observe([], {7}, timestamp, 0.0))

    assert retired == {7}
    assert ledger.event_refs == {}


def test_zero_event_index_is_a_valid_visit_reference():
    ledger = PlateVisitLedger()
    result = plate_result("31ط55674", 1)
    ledger.register(result, 0, 0.0)

    assert ledger.event_ref(result, 1.0, 30.0)[1] == 0
