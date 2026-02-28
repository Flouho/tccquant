from __future__ import annotations

import json

from .onnx_parser import parse_onnx_model
from .exporter import export_model_to_onnx
from .passes import CalibrationPass, ErrorAnalysisPass, PassManager, QuantContext, QuantOpReplacementPass, WeightQuantPass


def run_w8a8_pipeline(model_path: str, calibration_data: dict[str, object]) -> QuantContext:
    model = parse_onnx_model(model_path)
    ctx = QuantContext(model=model, calibration_data=calibration_data)
    manager = PassManager([WeightQuantPass(), CalibrationPass(), QuantOpReplacementPass(), ErrorAnalysisPass()])
    return manager.run(ctx)


def save_error_report(ctx: QuantContext, output_path: str) -> None:
    payload = {
        "ops": [
            {
                "op_name": e.op_name,
                "op_type": e.op_type,
                "mse": e.mse,
                "mae": e.mae,
                "max_abs_error": e.max_abs_error,
            }
            for e in ctx.op_errors
        ]
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def export_quantized_onnx(ctx: QuantContext, output_path: str) -> None:
    export_model_to_onnx(ctx.model, output_path)
