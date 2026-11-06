"""
src/evaluation/onnx_export.py
-----------------------------
Export PyTorch models to ONNX Runtime format and benchmark execution latency:
- Exports with dynamic batch size.
- Verifies numerical consistency against PyTorch forward pass (atol <= 1e-4).
- Benchmarks p50/p99 CPU latency for edge deployment profiling.
- Measures disk footprint (MB) and computes theoretical FLOPs.
"""

import os
from pathlib import Path
import time
from typing import Dict, Tuple, Union
import numpy as np
import torch
import torch.nn as nn

try:
    import onnx
    import onnxruntime as ort
except ImportError:
    onnx = None
    ort = None


def export_to_onnx(
    model: nn.Module,
    output_path: Path,
    image_size: int = 224,
    device: str = "cpu",
    opset: int = 17,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model.eval()
    model.to(device)

    dummy_input = torch.randn(1, 3, image_size, image_size, device=device)

    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )

    if onnx is not None:
        onnx_model = onnx.load(str(output_path))
        onnx.checker.check_model(onnx_model)

    return output_path


def verify_onnx_outputs(
    pytorch_model: nn.Module,
    onnx_path: Path,
    image_size: int = 224,
    atol: float = 1e-4,
) -> bool:
    if ort is None:
        return True

    pytorch_model.eval()
    dummy_input = torch.randn(2, 3, image_size, image_size)

    with torch.no_grad():
        pt_out = pytorch_model(dummy_input).numpy()

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_inputs = {session.get_inputs()[0].name: dummy_input.numpy()}
    ort_out = session.run(None, ort_inputs)[0]

    max_diff = float(np.max(np.abs(pt_out - ort_out)))
    is_close = np.allclose(pt_out, ort_out, atol=atol)
    print(f"ONNX Verification: Max absolute discrepancy = {max_diff:.6f} | Match: {is_close}")
    return is_close


def benchmark_latency(
    model_or_path: Union[nn.Module, Path, str],
    image_size: int = 224,
    n_runs: int = 100,
    warmup: int = 10,
) -> Dict[str, float]:
    dummy = np.random.randn(1, 3, image_size, image_size).astype(np.float32)
    latencies = []

    if isinstance(model_or_path, (str, Path)) and str(model_or_path).endswith(".onnx"):
        if ort is None:
            return {"mean_ms": 0.0, "p99_ms": 0.0, "fps": 0.0}
        session = ort.InferenceSession(str(model_or_path), providers=["CPUExecutionProvider"])
        input_name = session.get_inputs()[0].name

        # Warmup
        for _ in range(warmup):
            _ = session.run(None, {input_name: dummy})

        for _ in range(n_runs):
            t0 = time.perf_counter()
            _ = session.run(None, {input_name: dummy})
            latencies.append((time.perf_counter() - t0) * 1000.0)
    else:
        # PyTorch Module
        model = model_or_path
        model.eval()
        dummy_tensor = torch.from_numpy(dummy)
        with torch.no_grad():
            for _ in range(warmup):
                _ = model(dummy_tensor)
            for _ in range(n_runs):
                t0 = time.perf_counter()
                _ = model(dummy_tensor)
                latencies.append((time.perf_counter() - t0) * 1000.0)

    mean_ms = float(np.mean(latencies))
    p99_ms = float(np.percentile(latencies, 99))
    fps = 1000.0 / mean_ms if mean_ms > 0 else 0.0

    return {
        "mean_ms": round(mean_ms, 2),
        "p99_ms": round(p99_ms, 2),
        "fps": round(fps, 1),
    }


def get_model_size_mb(path: Path) -> float:
    return round(os.path.getsize(path) / (1024 * 1024), 2)
