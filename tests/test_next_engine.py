import numpy as np

from app.ai import next_engine


def test_shadow_failure_never_changes_baseline_output(monkeypatch):
    expected = [{"plate": "31-ط-556-74"}]

    def fail(*_args, **_kwargs):
        raise RuntimeError("candidate failed")

    monkeypatch.setattr(next_engine, "process_frame_next", fail)
    router = next_engine.EngineRouter()

    result = router.process(
        np.zeros((32, 64, 3), dtype=np.uint8),
        baseline=lambda: expected,
        mode="shadow",
    )

    assert result.primary is expected
    assert result.shadow == []
    assert result.mode == "shadow"
    assert "candidate failed" in result.error


def test_next_runtime_failure_rolls_back_and_uses_baseline(monkeypatch):
    rollbacks = []

    def fail(*_args, **_kwargs):
        raise RuntimeError("invalid model output")

    monkeypatch.setattr(next_engine, "process_frame_next", fail)
    monkeypatch.setattr(
        next_engine,
        "rollback_to_baseline",
        lambda reason: rollbacks.append(reason),
    )
    router = next_engine.EngineRouter()

    result = router.process(
        np.zeros((32, 64, 3), dtype=np.uint8),
        baseline=lambda: [{"plate": "baseline"}],
        mode="next",
    )

    assert result.primary == [{"plate": "baseline"}]
    assert result.mode == "baseline"
    assert result.degraded is True
    assert rollbacks and "invalid model output" in rollbacks[0]
