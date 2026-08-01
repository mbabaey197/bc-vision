from app.ai.dataset_split import grouped_train_validation_split


def test_related_track_frames_never_cross_dataset_split():
    samples = [
        {"frame": index, "track_id": f"track-{index // 3}"}
        for index in range(18)
    ]

    train, validation = grouped_train_validation_split(
        samples,
        group_key="track_id",
        validation_ratio=0.33,
    )

    train_groups = {row["track_id"] for row in train}
    validation_groups = {row["track_id"] for row in validation}
    assert train_groups
    assert validation_groups
    assert train_groups.isdisjoint(validation_groups)
