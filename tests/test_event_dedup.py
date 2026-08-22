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


def test_unstable_temporal_guess_cannot_end_confirmed_visit():
    ledger = PlateVisitLedger()
    confirmed = plate_result("31ط55674", 1)
    ledger.register(confirmed, 41, 0.0)
    unstable = {
        "plate": "31-ط-558-74",
        "plate_norm": "",
        "raw_guess_text": "31-ط-558-74",
        "raw_guess_norm": "31ط55874",
        "valid": False,
        "needs_review": True,
        "visit_identity_stable": False,
        "track_id": 1,
    }

    retired = ledger.observe([unstable], {1}, 0.2, 30.0)

    assert retired == set()
    assert ledger.track_keys == {1: "31ط55674"}
    assert ledger.active == {"31ط55674"}
    assert ledger.seen["31ط55674"] == 0.2


def test_unstable_fragment_reuses_the_only_bound_confirmed_event():
    ledger = PlateVisitLedger()
    confirmed = {
        **plate_result("31ط55674", 1),
        "bbox": (20, 30, 170, 70),
    }
    ledger.register(confirmed, 77, 0.0)
    unstable = {
        "plate": "31-ط-558-74",
        "plate_norm": "",
        "raw_guess_text": "31-ط-558-74",
        "raw_guess_norm": "31ط55874",
        "valid": False,
        "needs_review": True,
        "visit_identity_stable": False,
        "track_id": 2,
        "bbox": (24, 30, 174, 70),
    }

    retired = ledger.observe([unstable], {2}, 0.2, 30.0)

    assert retired == set()
    assert ledger.track_event_refs() == {2: 77}
    assert ledger.can_reuse_track_event(2, unstable) is True


def test_delayed_registration_never_moves_visit_time_backwards():
    ledger = PlateVisitLedger()
    result = plate_result("31ط55674", 1)
    ledger.register(result, 77, 0.0)
    ledger.observe([result], {1}, 10.0, 30.0)

    ledger.register(result, 77, 1.0)

    assert ledger.seen["31ط55674"] == 10.0


def test_delayed_identity_migration_preserves_newest_visit_time():
    ledger = PlateVisitLedger()
    review = {
        "plate": "31-ط-558-74",
        "plate_norm": "",
        "raw_guess_norm": "31ط55874",
        "valid": False,
        "needs_review": True,
        "track_id": 1,
    }
    ledger.register(review, 77, 0.0, allow_candidate=True)
    ledger.observe([review], {1}, 10.0, 30.0)
    corrected = {
        **plate_result("31ط55674", 1),
        "raw_guess_norm": "31ط55674",
    }

    ledger.register(corrected, 77, 1.0, allow_candidate=True)

    assert ledger.event_refs == {"31ط55674": 77}
    assert ledger.seen["31ط55674"] == 10.0


def test_review_identity_migrates_without_leaving_event_aliases():
    ledger = PlateVisitLedger()
    first = {
        "plate": "31-ط-558-74",
        "plate_norm": "",
        "raw_guess_norm": "31ط55874",
        "valid": False,
        "needs_review": True,
        "track_id": 1,
    }
    second = {
        **first,
        "plate": "31-ط-556-74",
        "raw_guess_norm": "31ط55674",
    }
    ledger.register(first, 41, 0.0, allow_candidate=True)
    ledger.register(second, 41, 0.2, allow_candidate=True)

    assert ledger.event_refs == {"31ط55674": 41}
    assert ledger.track_keys == {1: "31ط55674"}
    assert ledger.active == {"31ط55674"}
    assert "31ط55874" not in ledger.seen

    confirmed = plate_result("31ط55674", 1)
    ledger.register(confirmed, 41, 0.3, allow_candidate=True)
    conflicting_review = {
        **first,
        "plate": "12-ب-345-67",
        "raw_guess_norm": "12ب34567",
    }
    canonical = ledger.register(
        conflicting_review,
        41,
        0.4,
        allow_candidate=True,
    )

    assert canonical == "31ط55674"
    assert ledger.event_refs == {"31ط55674": 41}
    assert ledger.confirmed_keys == {"31ط55674"}


def test_fragmented_track_migrates_only_with_spatial_one_slot_correction():
    ledger = PlateVisitLedger()
    review = {
        "plate": "31-ط-558-74",
        "plate_norm": "",
        "raw_guess_norm": "31ط55874",
        "valid": False,
        "needs_review": True,
        "visit_identity_stable": False,
        "track_id": 1,
        "bbox": (20, 30, 170, 70),
    }
    ledger.register(review, 41, 0.0, allow_candidate=True)
    corrected = {
        "plate": "31-ط-556-74",
        "plate_norm": "31ط55674",
        "valid": True,
        "needs_review": False,
        "track_id": 2,
        "bbox": (25, 30, 175, 70),
    }

    retired = ledger.observe([corrected], {2}, 0.2, 30.0)

    assert retired == set()
    assert ledger.track_event_refs() == {2: 41}
    ledger.register(corrected, 41, 0.2, allow_candidate=True)
    assert ledger.event_refs == {"31ط55674": 41}
    assert ledger.track_keys == {2: "31ط55674"}


def test_same_track_strict_read_replaces_multislot_provisional_identity():
    ledger = PlateVisitLedger()
    review = {
        "plate": "31-ط-558-74",
        "plate_norm": "",
        "raw_guess_norm": "31ط55874",
        "valid": False,
        "needs_review": True,
        "visit_identity_stable": False,
        "track_id": 1,
        "bbox": (20, 30, 170, 70),
    }
    ledger.register(review, 41, 0.0, allow_candidate=True)
    corrected = {
        "plate": "31-ط-526-74",
        "plate_norm": "31ط52674",
        "raw_guess_norm": "31ط52674",
        "valid": True,
        "needs_review": False,
        "visit_identity_stable": True,
        "track_id": 1,
        "bbox": (24, 30, 174, 70),
    }

    retired = ledger.observe([corrected], {1}, 0.2, 30.0)
    key, event_ref = ledger.event_ref(
        corrected,
        0.2,
        30.0,
        allow_candidate=True,
    )

    assert retired == set()
    assert key == "31ط52674"
    # _process retains the event id already bound to this same physical track.
    assert event_ref is None
    assert ledger.can_reuse_track_event(1, corrected) is True
    ledger.register(corrected, 41, 0.2, allow_candidate=True)
    assert ledger.event_refs == {"31ط52674": 41}
    assert ledger.confirmed_keys == {"31ط52674"}
    assert ledger.track_event_refs() == {1: 41}


def test_multislot_provisional_upgrade_does_not_cross_tracks():
    ledger = PlateVisitLedger()
    review = {
        "plate": "31-ط-558-74",
        "plate_norm": "",
        "raw_guess_norm": "31ط55874",
        "valid": False,
        "needs_review": True,
        "visit_identity_stable": False,
        "track_id": 1,
        "bbox": (20, 30, 170, 70),
    }
    ledger.register(review, 41, 0.0, allow_candidate=True)
    corrected = {
        "plate": "31-ط-526-74",
        "plate_norm": "31ط52674",
        "valid": True,
        "needs_review": False,
        "track_id": 2,
        "bbox": (24, 30, 174, 70),
    }

    retired = ledger.observe([corrected], {2}, 0.2, 30.0)
    key, event_ref = ledger.event_ref(
        corrected,
        0.2,
        30.0,
        allow_candidate=True,
    )

    assert retired == set()
    assert (key, event_ref) == ("31ط52674", None)
    assert ledger.track_event_refs() == {}
    assert ledger.event_refs == {"31ط55874": 41}


def test_unknown_fragment_does_not_gain_multislot_migration_authority():
    ledger = PlateVisitLedger()
    review = {
        "plate": "31-ط-558-74",
        "plate_norm": "",
        "raw_guess_norm": "31ط55874",
        "valid": False,
        "needs_review": True,
        "visit_identity_stable": False,
        "track_id": 1,
        "bbox": (20, 30, 170, 70),
    }
    ledger.register(review, 41, 0.0, allow_candidate=True)
    unknown_fragment = {
        "plate": "ناخوانا",
        "plate_norm": "",
        "valid": False,
        "visit_identity_stable": False,
        "track_id": 2,
        "bbox": (22, 30, 172, 70),
    }
    ledger.observe([unknown_fragment], {2}, 0.1, 30.0)
    assert ledger.track_event_refs() == {2: 41}
    corrected = {
        "plate": "31-ط-526-74",
        "plate_norm": "31ط52674",
        "valid": True,
        "needs_review": False,
        "track_id": 2,
        "bbox": (24, 30, 174, 70),
    }

    retired = ledger.observe([corrected], {2}, 0.2, 30.0)

    assert retired == {2}
    assert ledger.can_reuse_track_event(2, corrected) is True
    assert ledger.event_ref(
        corrected,
        0.2,
        30.0,
        allow_candidate=True,
    ) == ("31ط52674", None)
    assert ledger.event_refs == {"31ط55874": 41}


def test_unrelated_strict_identity_retires_provisional_review_track():
    ledger = PlateVisitLedger()
    review = {
        "plate": "31-ط-558-74",
        "plate_norm": "",
        "raw_guess_norm": "31ط55874",
        "valid": False,
        "needs_review": True,
        "track_id": 1,
        "bbox": (20, 30, 170, 70),
    }
    ledger.register(review, 41, 0.0, allow_candidate=True)
    unrelated = {
        "plate": "12-ب-345-67",
        "plate_norm": "12ب34567",
        "valid": True,
        "needs_review": False,
        "track_id": 1,
        "bbox": (22, 30, 172, 70),
    }

    retired = ledger.observe([unrelated], {1}, 0.2, 30.0)
    key, event_ref = ledger.event_ref(
        unrelated,
        0.2,
        30.0,
        allow_candidate=True,
    )

    assert retired == {1}
    assert (key, event_ref) == ("12ب34567", None)
    assert ledger.track_event_refs() == {}
    assert ledger.can_reuse_track_event(1, unrelated) is True


def test_unrelated_fragmented_track_cannot_claim_provisional_event():
    ledger = PlateVisitLedger()
    review = {
        "plate": "31-ط-558-74",
        "plate_norm": "",
        "raw_guess_norm": "31ط55874",
        "valid": False,
        "needs_review": True,
        "track_id": 1,
        "bbox": (20, 30, 170, 70),
    }
    ledger.register(review, 41, 0.0, allow_candidate=True)
    unrelated = {
        "plate": "12-ب-345-67",
        "plate_norm": "12ب34567",
        "valid": True,
        "needs_review": False,
        "track_id": 2,
        "bbox": (24, 30, 174, 70),
    }

    retired = ledger.observe([unrelated], {2}, 0.2, 30.0)

    assert retired == set()
    assert ledger.track_event_refs() == {}
    key, event_ref = ledger.event_ref(
        unrelated,
        0.2,
        30.0,
        allow_candidate=True,
    )
    assert (key, event_ref) == ("12ب34567", None)


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
