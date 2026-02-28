#!/usr/bin/env python3
"""End-to-end quantization test script.

Flow:
1) load ONNX model
2) load calibration data JSON
3) run W8A8 quant pipeline
4) export per-op error report
5) export quantized ONNX model
"""

from __future__ import annotations

import argparse
import json
import sys

from tccquant.onnx_parser import OnnxDependencyError
from tccquant.pipeline import export_quantized_onnx, run_w8a8_pipeline, save_error_report


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="TCCQuant E2E quantization test")
    p.add_argument("--model", required=True, help="input ONNX model path")
    p.add_argument("--calibration-json", required=True, help="calibration data json path")
    p.add_argument("--error-report", required=True, help="output error report json path")
    p.add_argument("--quantized-onnx", required=True, help="output quantized onnx path")
    return p


def main() -> int:
    args = build_parser().parse_args()

    with open(args.calibration_json, "r", encoding="utf-8") as f:
        calibration_data = json.load(f)

    try:
        ctx = run_w8a8_pipeline(args.model, calibration_data)
        save_error_report(ctx, args.error_report)
        export_quantized_onnx(ctx, args.quantized_onnx)
    except OnnxDependencyError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        print("Hint: pip install -e .[onnx]", file=sys.stderr)
        return 2

    print("[OK] E2E quantization finished")
    print(f"[OK] quantized ops: {len(ctx.op_quant_configs)}")
    print(f"[OK] op error entries: {len(ctx.op_errors)}")
    print(f"[OK] report: {args.error_report}")
    print(f"[OK] quantized onnx: {args.quantized_onnx}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
