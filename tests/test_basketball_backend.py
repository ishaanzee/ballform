import sys
from types import SimpleNamespace

import numpy as np
import pytest

import app.basketball as basketball_module
from app.basketball import BasketballDetector
from scripts.benchmark_ball_backends import compare


def test_coreml_backend_requests_neural_engine_and_content_specific_cache(monkeypatch, tmp_path):
    model = tmp_path / "ball.onnx"
    model.write_bytes(b"model-a")
    created = []

    class Session:
        def __init__(self, path, sess_options, providers):
            created.append((path, providers))

        def get_providers(self):
            return ["CoreMLExecutionProvider", "CPUExecutionProvider"]

    monkeypatch.setitem(sys.modules, "onnxruntime", SimpleNamespace(
        disable_telemetry_events=lambda: None,
        SessionOptions=lambda: SimpleNamespace(intra_op_num_threads=None),
        get_available_providers=lambda: ["CoreMLExecutionProvider", "CPUExecutionProvider"],
        InferenceSession=Session,
    ))
    detector = BasketballDetector(model, backend="coreml")
    assert detector.backend == "coreml"
    assert created[0][1][0][0] == "CoreMLExecutionProvider"
    settings = created[0][1][0][1]
    assert settings["MLComputeUnits"] == "CPUAndNeuralEngine"
    assert settings["ModelFormat"] == "MLProgram"
    assert settings["RequireStaticInputShapes"] == "1"
    assert settings["ModelCacheDirectory"].startswith(str(tmp_path / "coreml-cache"))
    assert created[0][1][1] == "CPUExecutionProvider"
    first_cache = settings["ModelCacheDirectory"]
    model.write_bytes(b"model-b")
    BasketballDetector(model, backend="coreml")
    assert created[1][1][0][1]["ModelCacheDirectory"] != first_cache


def test_coreml_backend_fails_loudly_when_provider_is_missing(monkeypatch, tmp_path):
    model = tmp_path / "ball.onnx"
    model.write_bytes(b"model")
    monkeypatch.setitem(sys.modules, "onnxruntime", SimpleNamespace(
        disable_telemetry_events=lambda: None,
        SessionOptions=lambda: SimpleNamespace(intra_op_num_threads=None),
        get_available_providers=lambda: ["CPUExecutionProvider"],
    ))
    with pytest.raises(RuntimeError, match="Core ML provider"):
        BasketballDetector(model, backend="coreml")


def test_auto_backend_falls_back_to_cpu_and_records_reason(monkeypatch, tmp_path):
    model = tmp_path / "ball.onnx"
    model.write_bytes(b"model")
    calls = []

    class Session:
        def __init__(self, path, sess_options, providers):
            calls.append(providers)
            if isinstance(providers[0], tuple):
                raise RuntimeError("Neural Engine unavailable")
            self.providers = providers

        def get_providers(self):
            return self.providers

    monkeypatch.setattr(basketball_module.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(basketball_module.platform, "machine", lambda: "arm64")
    monkeypatch.setitem(sys.modules, "onnxruntime", SimpleNamespace(
        disable_telemetry_events=lambda: None,
        SessionOptions=lambda: SimpleNamespace(intra_op_num_threads=None),
        get_available_providers=lambda: ["CoreMLExecutionProvider", "CPUExecutionProvider"],
        InferenceSession=Session,
    ))
    detector = BasketballDetector(model)
    assert detector.requested_backend == "auto"
    assert detector.backend == "cpu"
    assert "Neural Engine unavailable" in detector.fallback_reason
    assert len(calls) == 2
    assert calls[1] == ["CPUExecutionProvider"]


def test_auto_backend_falls_back_if_coreml_inference_fails(monkeypatch, tmp_path):
    model = tmp_path / "ball.onnx"
    model.write_bytes(b"model")

    class Session:
        def __init__(self, path, sess_options, providers):
            self.providers = [item[0] if isinstance(item, tuple) else item for item in providers]

        def get_providers(self):
            return self.providers

        def get_inputs(self):
            return [SimpleNamespace(name="input")]

        def run(self, outputs, feed):
            if "CoreMLExecutionProvider" in self.providers:
                raise RuntimeError("Core ML execution failed")
            return [np.zeros((1, 300, 4), dtype=np.float32),
                    np.full((1, 300, 11), -20, dtype=np.float32)]

    monkeypatch.setattr(basketball_module.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(basketball_module.platform, "machine", lambda: "arm64")
    monkeypatch.setitem(sys.modules, "onnxruntime", SimpleNamespace(
        disable_telemetry_events=lambda: None,
        SessionOptions=lambda: SimpleNamespace(intra_op_num_threads=None),
        get_available_providers=lambda: ["CoreMLExecutionProvider", "CPUExecutionProvider"],
        InferenceSession=Session,
    ))
    detector = BasketballDetector(model)
    assert detector.detect(np.zeros((10, 10, 3), dtype=np.uint8)) == []
    assert detector.backend == "cpu"
    assert "Core ML execution failed" in detector.fallback_reason


def test_backend_comparison_flags_missing_ball_without_matching_player():
    cpu = [(1, .8, (.1, .1, .2, .2)), (4, .9, (.4, .2, .6, .8))]
    coreml = [(4, .88, (.4, .2, .6, .8))]
    matched, missed, added, minimum_iou, confidence_delta = compare(cpu, coreml)
    assert (matched, missed, added) == (1, 1, 0)
    assert minimum_iou == pytest.approx(1)
    assert confidence_delta == pytest.approx(.02)
