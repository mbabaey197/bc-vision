from __future__ import annotations

import numpy as np
import pytest

from app.engine_v2.model_adapters import (
    CTCPlateOCR,
    CTCPlateOCRConfig,
    _ctc_prefix_beam_search,
)


class _Backend:
    input_names = ("input",)
    output_names = ("output",)


def test_constrained_ctc_beam_returns_top_k_iranian_plate_candidates() -> None:
    charset = tuple("0123456789") + ("ب",)
    blank_index = len(charset)
    plate_indices = [1, 2, 10, 3, 4, 5, 6, 7]
    timesteps = len(plate_indices) * 2 - 1
    probabilities = np.full((timesteps, len(charset) + 1), 1e-5, dtype=np.float64)
    for position, class_index in enumerate(plate_indices):
        timestep = position * 2
        probabilities[timestep, class_index] = 0.98
        if position == len(plate_indices) - 1:
            probabilities[timestep, class_index] = 0.55
            probabilities[timestep, 8] = 0.44
        if timestep + 1 < timesteps:
            probabilities[timestep + 1, blank_index] = 0.99
    probabilities /= probabilities.sum(axis=1, keepdims=True)

    candidates = _ctc_prefix_beam_search(
        probabilities,
        charset=charset,
        blank_index=blank_index,
        beam_width=8,
        top_k=2,
        constrain_iranian_layout=True,
    )

    assert [candidate[0] for candidate in candidates] == ["12ب34567", "12ب34568"]
    assert sum(candidate[1] for candidate in candidates) == pytest.approx(1.0)
    assert all(len(candidate[2]) == 8 for candidate in candidates)


def test_ctc_beam_configuration_is_bounded() -> None:
    with pytest.raises(ValueError, match="top_k"):
        CTCPlateOCRConfig(beam_width=2, top_k=3)


def test_ctc_adapter_publishes_beam_candidates_for_temporal_fusion() -> None:
    charset = tuple("0123456789") + ("ب",)
    blank_index = len(charset)
    plate_indices = [1, 2, 10, 3, 4, 5, 6, 7]
    probabilities = np.full((15, len(charset) + 1), 1e-5, dtype=np.float64)
    for position, class_index in enumerate(plate_indices):
        timestep = position * 2
        probabilities[timestep, class_index] = 0.98
        if position == 7:
            probabilities[timestep, class_index] = 0.55
            probabilities[timestep, 8] = 0.44
        if timestep + 1 < 15:
            probabilities[timestep + 1, blank_index] = 0.99
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    adapter = CTCPlateOCR(
        _Backend(),
        CTCPlateOCRConfig(
            charset=charset,
            blank_index=blank_index,
            output_layout="tc",
            activation="probabilities",
            beam_width=8,
            top_k=2,
            constrain_iranian_layout=True,
        ),
    )

    result = adapter.decode(probabilities)

    assert result.text == "12ب34567"
    assert result.metadata["decoder"] == "prefix_beam"
    assert [item["text"] for item in result.metadata["candidates"]] == [
        "12ب34567",
        "12ب34568",
    ]
