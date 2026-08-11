from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np
import pytest

from app.engine_v2.types import PlateEvent
from tools import anpr_accuracy_adapters as adapter_module
from tools.anpr_accuracy_adapters import (
    EngineV2OfflineAccuracyAdapter,
    LegacyVideoAccuracyAdapter,
    LegacyVideoAccuracyConfig,
    V2OfflineAccuracyConfig,
)
from tools import benchmark_engine_v2 as benchmark_cli


def _sample(path: Path, *, media_type: str = "video", sample_id: str = "sample-01") -> dict[str, Any]:
    return {
        "id": sample_id,
        "category": "clear_plate",
        "input": {
            "path": path.name,
            "resolved_path": str(path),
            "media_type": media_type,
        },
    }


def test_legacy_adapter_is_lazy_uses_explicit_temp_settings_and_timestamps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    media = tmp_path / "same-input.mp4"
    media.write_bytes(b"verified-video-placeholder")
    detector_model = tmp_path / "legacy-detector.onnx"
    fallback_model = tmp_path / "legacy-detector-fallback.onnx"
    crnn_model = tmp_path / "legacy-crnn.onnx"
    cnn_model = tmp_path / "legacy-cnn.onnx"
    hezar_model = tmp_path / "legacy-hezar.onnx"
    for index, model in enumerate(
        (detector_model, fallback_model, crnn_model, cnn_model, hezar_model),
        start=1,
    ):
        model.write_bytes(f"legacy-model-{index}".encode("ascii"))
    import_calls: list[str] = []
    temporary_roots: list[Path] = []
    observed: dict[str, Any] = {}

    def fake_process_video(**kwargs: Any):
        observed.update(kwargs)
        plate_dir = Path(kwargs["plate_dir"])
        snapshot_dir = Path(kwargs["snapshot_dir"])
        assert plate_dir.parent == snapshot_dir.parent
        assert plate_dir.parent.is_dir()
        temporary_roots.append(plate_dir.parent)
        plate_dir.mkdir()
        snapshot_dir.mkdir()
        return (
            {"fps": 20.0, "frames": 60, "detector_variant": kwargs["detector_variant"]},
            [
                {
                    "plate": "12ب34567",
                    "confidence": 0.91,
                    "video_second": 1.25,
                    "frame": 25,
                    "plate_path": str(plate_dir / "deleted.jpg"),
                },
                {
                    "plate_norm": "34د76543",
                    "confidence": 0.82,
                    "frame": 40,
                },
                {
                    "plate_norm": "درحالبررسی",
                    "confidence": 0.10,
                    "capture_only": True,
                    "valid": False,
                    "frame": 45,
                },
                {
                    "plate_norm": "11ب11111",
                    "confidence": 0.80,
                    "valid": False,
                    "frame": 50,
                },
            ],
        )

    def fake_import(name: str):
        import_calls.append(name)
        if name == "app.ai.video_test":
            return SimpleNamespace(process_video=fake_process_video)
        assert name == "app.ai.model_manager"
        return SimpleNamespace(
            detector_variant_spec=lambda _variant: {
                "path": detector_model,
                "sha256": hashlib.sha256(detector_model.read_bytes()).hexdigest(),
                "size": detector_model.stat().st_size,
            },
            detector_fallback_path=lambda: fallback_model,
            DETECTOR_FALLBACK_SHA256=hashlib.sha256(fallback_model.read_bytes()).hexdigest(),
            DETECTOR_FALLBACK_SIZE=fallback_model.stat().st_size,
            active_crnn_model=lambda: (
                crnn_model,
                hashlib.sha256(crnn_model.read_bytes()).hexdigest(),
                crnn_model.stat().st_size,
            ),
            cnn_path=lambda: cnn_model,
            CNN_SHA256=hashlib.sha256(cnn_model.read_bytes()).hexdigest(),
            CNN_SIZE=cnn_model.stat().st_size,
            hezar_path=lambda: hezar_model,
            HEZAR_ONNX_SHA256=hashlib.sha256(hezar_model.read_bytes()).hexdigest(),
            HEZAR_ONNX_SIZE=hezar_model.stat().st_size,
        )

    monkeypatch.setattr(adapter_module.importlib, "import_module", fake_import)
    adapter = LegacyVideoAccuracyAdapter(
        LegacyVideoAccuracyConfig(
            frame_step=3,
            max_events=17,
            min_confidence=0.42,
            duplicate_seconds=1.75,
            detector_variant="yolo11n",
            roi=(1.0, 2.0, 80.0, 70.0),
        )
    )
    assert import_calls == []

    prediction = adapter.predict(_sample(media))

    assert import_calls == ["app.ai.video_test"]
    assert observed["video_path"] == str(media)
    assert observed["frame_step"] == 3
    assert observed["max_events"] == 17
    assert observed["min_confidence"] == pytest.approx(0.42)
    assert observed["duplicate_seconds"] == pytest.approx(1.75)
    assert observed["roi"] == (1.0, 2.0, 80.0, 70.0)
    assert observed["include_candidate_shadow"] is False
    assert observed["detector_variant"] == "yolo11n"
    assert [event["timestamp_ms"] for event in prediction["events"]] == [1250.0, 2000.0]
    assert [event["timestamp_source"] for event in prediction["events"]] == [
        "legacy-frame/fps",
        "legacy-frame/fps",
    ]
    assert all("plate_path" not in event for event in prediction["events"])
    assert prediction["run_metadata"]["input_sha256"] == hashlib.sha256(
        media.read_bytes()
    ).hexdigest()
    model_identity = adapter.reproducibility_metadata()["model_identity"]
    assert import_calls == ["app.ai.video_test", "app.ai.model_manager"]
    assert [row["role"] for row in model_identity["files"]] == [
        "detector-selected",
        "detector-fallback",
        "ocr-crnn-active",
        "ocr-cnn-fallback",
        "ocr-hezar-primary",
    ]
    assert all(row["sha256"] and row["matches_expected_sha256"] for row in model_identity["files"])
    assert model_identity["execution_provider_contract"] == "CPUExecutionProvider"
    assert temporary_roots and all(not root.exists() for root in temporary_roots)


class _FakeEngine:
    def __init__(self, callback) -> None:
        self.callback = callback
        self.reset_calls = 0
        self.packets = []
        self.finalize_calls = []

    def reset_runtime_state(self) -> None:
        self.reset_calls += 1

    def submit_frame(self, packet) -> bool:
        self.packets.append(packet)
        return True

    def process_available(self, limit: int = 128):
        del limit
        return []

    def finalize_camera(self, camera_id: str, *, final_seq=None, final_ts=None):
        self.finalize_calls.append((camera_id, final_seq, final_ts))
        event = PlateEvent(
            camera_id=camera_id,
            frame_seq=int(final_seq),
            ts=float(final_ts),
            text="12ب34567",
            confidence=0.93,
            bbox=(10, 4, 40, 16),
            quality=0.88,
            track_id="7",
            episode_id=f"{camera_id}:7",
            observations=3,
        )
        # Returning and invoking the callback exercises adapter de-duplication.
        self.callback(event)
        return [event]


class _FakeModels:
    def summary(self):
        return {
            "detector": {
                "backend": "onnxruntime",
                "device": "CPU",
                "providers": ("CPUExecutionProvider",),
            },
            "ocr": {
                "backend": "onnxruntime",
                "device": "CPU",
                "providers": ("CPUExecutionProvider",),
            },
            "session_count": 2,
            "sessions_per_camera": 0,
        }


class _FakeBundle:
    def __init__(self, engine: _FakeEngine) -> None:
        self.engine = engine
        self.models = _FakeModels()
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1


def _fake_v2_adapter(tmp_path: Path):
    detector = tmp_path / "detector.onnx"
    ocr = tmp_path / "ocr.onnx"
    detector.write_bytes(b"detector-model")
    ocr.write_bytes(b"ocr-model")
    observed: dict[str, Any] = {"factory_calls": 0}

    def factory(model_config, engine_config, callback):
        observed["factory_calls"] += 1
        observed["model_config"] = model_config
        observed["engine_config"] = engine_config
        engine = _FakeEngine(callback)
        bundle = _FakeBundle(engine)
        observed["bundle"] = bundle
        return bundle

    adapter = EngineV2OfflineAccuracyAdapter(
        V2OfflineAccuracyConfig(
            detector_model=detector,
            ocr_model=ocr,
            detector_frame_size=(640, 360),
        ),
        _bundle_factory=factory,
    )
    return adapter, observed, detector, ocr


def test_v2_adapter_shares_bundle_resets_samples_derives_subframe_and_finalizes(
    tmp_path: Path,
) -> None:
    image = np.zeros((48, 96, 3), dtype=np.uint8)
    image[10:35, 20:80] = (30, 160, 240)
    encoded_ok, encoded = cv2.imencode(".png", image)
    assert encoded_ok
    media = tmp_path / "same-input.png"
    media.write_bytes(encoded.tobytes())
    adapter, observed, detector, ocr = _fake_v2_adapter(tmp_path)

    first = adapter.predict(_sample(media, media_type="image", sample_id="clear-01"))
    second = adapter.predict(_sample(media, media_type="image", sample_id="clear-02"))

    bundle = observed["bundle"]
    engine = bundle.engine
    assert observed["factory_calls"] == 1
    assert engine.reset_calls == 2
    assert len(engine.finalize_calls) == 2
    assert len(first["events"]) == 1
    assert len(second["events"]) == 1
    assert first["events"][0]["timestamp_ms"] == 0.0
    assert first["events"][0]["confidence"] == pytest.approx(0.93)
    real_packets = [
        packet
        for packet in engine.packets
        if packet.metadata.get("detector_frame_derived_from_main") is True
    ]
    baselines = [
        packet for packet in engine.packets if "accuracy_motion_baseline" in packet.metadata
    ]
    assert len(real_packets) == len(baselines) == 2
    assert all(packet.detector_frame.shape == (360, 640, 3) for packet in real_packets)
    assert all(packet.frame.shape == image.shape for packet in real_packets)
    assert all(packet.metadata["timestamp_source"] == "image-zero" for packet in real_packets)
    metadata = adapter.reproducibility_metadata()
    assert metadata["sessions"] == {"service_total": 2, "per_camera": 0}
    assert metadata["selected_shared_model_runtime"]["detector"]["providers"] == [
        "CPUExecutionProvider"
    ]
    assert metadata["models"][0]["sha256"] == hashlib.sha256(detector.read_bytes()).hexdigest()
    assert metadata["models"][1]["sha256"] == hashlib.sha256(ocr.read_bytes()).hexdigest()
    assert metadata["decode"]["observed_runs"][0]["timestamp_sources"] == {
        "image-zero": 1
    }

    adapter.close()
    adapter.close()
    assert bundle.close_calls == 1
    with pytest.raises(RuntimeError, match="closed"):
        adapter.predict(_sample(media, media_type="image"))


def test_v2_video_decode_uses_pts_then_deterministic_frame_fallback(tmp_path: Path) -> None:
    adapter, _observed, _detector, _ocr = _fake_v2_adapter(tmp_path)
    media = tmp_path / "clip.mp4"
    media.write_bytes(b"fake-capture-input")

    class FakeCapture:
        def __init__(self) -> None:
            self.frames = [
                np.full((8, 12, 3), value, dtype=np.uint8) for value in (10, 20, 30)
            ]
            self.index = 0
            self.released = False

        def get(self, prop: int) -> float:
            if prop == cv2.CAP_PROP_FPS:
                return 10.0
            if prop == cv2.CAP_PROP_POS_MSEC:
                return (0.0, 0.0, 250.0)[max(0, self.index - 1)]
            return 0.0

        def read(self):
            if self.index >= len(self.frames):
                return False, None
            frame = self.frames[self.index]
            self.index += 1
            return True, frame

        def release(self) -> None:
            self.released = True

    capture = FakeCapture()
    adapter._open_video_capture = lambda _path: (  # type: ignore[method-assign]
        capture,
        {"decode_backend": "fake-ffmpeg"},
    )
    metadata: dict[str, Any] = {}

    decoded = list(adapter._decode(media, "video", metadata))

    assert [frame.timestamp_seconds for frame in decoded] == pytest.approx([0.0, 0.1, 0.25])
    assert [frame.timestamp_source for frame in decoded] == [
        "opencv-pos-msec-pts",
        "frame-index/fps-fallback",
        "opencv-pos-msec-pts",
    ]
    assert capture.released is True
    adapter.close()


def test_builtin_adapters_fail_closed_on_unimplemented_input_time_window(
    tmp_path: Path,
) -> None:
    media = tmp_path / "content-addressed.mp4"
    media.write_bytes(b"pre-clipping-required")
    sample = _sample(media)
    sample["input"].update({"start_ms": 1000, "end_ms": 2500})
    legacy = LegacyVideoAccuracyAdapter(_process_video=lambda **_kwargs: ({}, []))
    v2, _observed, _detector, _ocr = _fake_v2_adapter(tmp_path)

    for adapter in (legacy, v2):
        with pytest.raises(ValueError, match="pre-clipped content-addressed"):
            adapter.predict(sample)

    v2.close()


def test_cli_keeps_generic_adapters_and_builds_explicit_builtin_configs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    detector = tmp_path / "detector.onnx"
    ocr = tmp_path / "ocr.onnx"
    detector.write_bytes(b"detector")
    ocr.write_bytes(b"ocr")
    args = benchmark_cli.build_parser().parse_args(
        [
            "compare-accuracy",
            "--manifest",
            str(tmp_path / "manifest.json"),
            "--output-dir",
            str(tmp_path / "results"),
            "--v1-builtin",
            "legacy-video",
            "--v2-builtin",
            "engine-v2-offline",
            "--v2-detector-model",
            str(detector),
            "--v2-ocr-model",
            str(ocr),
            "--v2-detector-frame-size",
            "800x450",
            "--v2-backend",
            "onnxruntime",
            "--v1-frame-step",
            "2",
            "--v2-frame-step",
            "2",
        ]
    )
    created: dict[str, Any] = {}

    def fake_v2(config, *, name):
        created["config"] = config
        created["name"] = name
        return SimpleNamespace(adapter_name=name)

    monkeypatch.setattr(adapter_module, "EngineV2OfflineAccuracyAdapter", fake_v2)
    benchmark_cli._validate_builtin_accuracy_comparison(args)
    v1 = benchmark_cli._make_accuracy_adapter(args, "v1")
    v2 = benchmark_cli._make_accuracy_adapter(args, "v2")

    assert isinstance(v1, LegacyVideoAccuracyAdapter)
    assert v1.config.frame_step == 2
    assert v2.adapter_name == "v2"
    assert created["config"].detector_frame_size == (800, 450)
    assert created["config"].backend == "onnxruntime"
    assert created["config"].frame_step == 2
    # Existing generic modes remain mutually exclusive alternatives.
    generic = benchmark_cli.build_parser().parse_args(
        [
            "compare-accuracy",
            "--manifest",
            "manifest.json",
            "--output-dir",
            "results",
            "--v1-callable",
            "module:predict_v1",
            "--v2-command",
            "python adapter.py",
        ]
    )
    assert generic.v1_callable == "module:predict_v1"
    assert generic.v2_command == "python adapter.py"


@pytest.mark.parametrize(
    ("extra_flags", "message"),
    [
        (["--v2-max-frames", "10"], "--v2-max-frames is forbidden"),
        (
            ["--v1-frame-step", "2", "--v2-frame-step", "1"],
            "--v1-frame-step and --v2-frame-step must match",
        ),
        (["--v1-roi", "0,0,80,80"], "--v1-roi is forbidden"),
    ],
)
def test_cli_builtin_compare_rejects_asymmetric_effective_input(
    tmp_path: Path,
    extra_flags: list[str],
    message: str,
) -> None:
    args = benchmark_cli.build_parser().parse_args(
        [
            "compare-accuracy",
            "--manifest",
            str(tmp_path / "manifest.json"),
            "--output-dir",
            str(tmp_path / "results"),
            "--v1-builtin",
            "legacy-video",
            "--v2-builtin",
            "engine-v2-offline",
            "--v2-detector-model",
            str(tmp_path / "detector.onnx"),
            "--v2-ocr-model",
            str(tmp_path / "ocr.onnx"),
            *extra_flags,
        ]
    )

    with pytest.raises(ValueError, match=message):
        benchmark_cli._run_accuracy(args)


def test_cli_partial_coverage_help_mentions_known_positive_scope(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exit_info:
        benchmark_cli.build_parser().parse_args(["compare-accuracy", "--help"])

    assert exit_info.value.code == 0
    assert "label_scope=known_positives" in capsys.readouterr().out


def test_accuracy_manifest_schema_closes_inference_boundary() -> None:
    schema = json.loads(
        Path("docs/ANPR_ENGINE_V2_BENCHMARK_MANIFEST.schema.json").read_text(
            encoding="utf-8"
        )
    )
    sample = schema["$defs"]["sample"]
    input_schema = sample["properties"]["input"]
    event_schema = sample["properties"]["expected_events"]["items"]

    assert sample["additionalProperties"] is False
    assert input_schema["additionalProperties"] is False
    assert event_schema["additionalProperties"] is False
    assert "adapter_input" not in sample["properties"]
    assert sample["allOf"][0]["if"]["properties"]["label_scope"] == {
        "const": "known_positives"
    }
    positive_options = sample["allOf"][0]["then"]["anyOf"]
    expected_plate_option = next(
        option for option in positive_options if "expected_plate" in option["required"]
    )
    assert expected_plate_option["properties"]["expected_plate"]["type"] == "string"
    assert "Category completeness" in schema["description"]


def test_accuracy_manifest_template_demonstrates_non_null_known_positive() -> None:
    template = json.loads(
        Path("tests/fixtures/engine_v2_accuracy_manifest.template.json").read_text(
            encoding="utf-8"
        )
    )
    examples = [
        sample
        for sample in template["samples"]
        if sample.get("label_scope") == "known_positives"
    ]

    assert all(sample["id"].startswith("sample-") for sample in template["samples"])
    assert len(examples) == 1
    example = examples[0]
    assert example["enabled"] is False
    assert example["expected_events"]
    assert all(event["plate"] for event in example["expected_events"])


@pytest.mark.parametrize(
    ("extra_flags", "require_all_categories", "require_negative_sample"),
    [
        ([], True, True),
        (["--allow-partial-coverage"], False, True),
        (["--allow-no-negative"], True, False),
        (["--allow-partial-coverage", "--allow-no-negative"], False, False),
    ],
)
def test_cli_partial_accuracy_overrides_are_explicit_and_fail_closed_by_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    extra_flags: list[str],
    require_all_categories: bool,
    require_negative_sample: bool,
) -> None:
    args = benchmark_cli.build_parser().parse_args(
        [
            "compare-accuracy",
            "--manifest",
            str(tmp_path / "manifest.json"),
            "--output-dir",
            str(tmp_path / "results"),
            "--v1-callable",
            "unused:v1",
            "--v2-callable",
            "unused:v2",
            *extra_flags,
        ]
    )
    observed: dict[str, Any] = {}

    def fake_load(path, **kwargs):
        observed["path"] = path
        observed.update(kwargs)
        return object()

    adapters = {
        "v1": SimpleNamespace(adapter_name="v1", close=lambda: None),
        "v2": SimpleNamespace(adapter_name="v2", close=lambda: None),
    }
    monkeypatch.setattr(benchmark_cli, "load_accuracy_manifest", fake_load)
    monkeypatch.setattr(
        benchmark_cli,
        "_make_accuracy_adapter",
        lambda _args, prefix: adapters[prefix],
    )
    monkeypatch.setattr(
        benchmark_cli,
        "compare_accuracy_adapters",
        lambda *_args: {"schema": "test"},
    )
    monkeypatch.setattr(benchmark_cli, "write_accuracy_outputs", lambda *_args, **_kwargs: None)

    assert benchmark_cli._run_accuracy(args) == 0
    assert observed["require_all_categories"] is require_all_categories
    assert observed["require_negative_sample"] is require_negative_sample
    assert observed["require_input_files"] is True
