#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random

from tccquant.config import QuantGranularity, QuantScheme, QuantSpec
from tccquant.graph_editor import NodeAnchor, QDQGraphEditor
from tccquant.pipeline import run_w8a8_pipeline, save_error_report, export_quantized_onnx
from tccquant.ppq_cleanup import prune_ppq_tree


def _random_tensor(shape: list[int]):
    if len(shape) == 1:
        return [random.uniform(-1, 1) for _ in range(shape[0])]
    if len(shape) == 2:
        return [[random.uniform(-1, 1) for _ in range(shape[1])] for _ in range(shape[0])]
    raise ValueError("Only 1D/2D calib-shape is supported in minimal CLI")


def cmd_prune(args: argparse.Namespace) -> None:
    removed = prune_ppq_tree(args.ppq_root, dry_run=args.dry_run)
    print(f"matched {len(removed)} paths")
    for p in removed:
        print(p)


def cmd_insert(args: argparse.Namespace) -> None:
    editor = QDQGraphEditor.from_path(args.model)
    spec = QuantSpec(
        bits=args.bits,
        granularity=QuantGranularity(args.granularity),
        scheme=QuantScheme(args.scheme),
        axis=args.axis,
        group_size=args.group_size,
    )
    calib = _random_tensor(args.calib_shape)
    editor.insert_qdq_after(NodeAnchor(args.node, args.output_index), calib, spec, prefix=args.prefix)
    editor.save(args.output)
    print(f"saved to {args.output}")


def cmd_w8a8(args: argparse.Namespace) -> None:
    with open(args.calibration_json, "r", encoding="utf-8") as f:
        calibration_data = json.load(f)
    ctx = run_w8a8_pipeline(args.model, calibration_data)
    save_error_report(ctx, args.report)
    print(f"quantized tensors: {len(ctx.quant_states)}")
    print(f"op error entries: {len(ctx.op_errors)}")
    print(f"report saved to {args.report}")
    if args.export_onnx:
        export_quantized_onnx(ctx, args.export_onnx)
        print(f"quantized onnx exported to {args.export_onnx}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="tccquant CLI")
    sub = p.add_subparsers(required=True)

    prune = sub.add_parser("prune-ppq")
    prune.add_argument("ppq_root")
    prune.add_argument("--dry-run", action="store_true")
    prune.set_defaults(func=cmd_prune)

    ins = sub.add_parser("insert-qdq")
    ins.add_argument("--model", required=True)
    ins.add_argument("--output", required=True)
    ins.add_argument("--node", required=True)
    ins.add_argument("--output-index", type=int, default=0)
    ins.add_argument("--bits", type=int, default=8)
    ins.add_argument("--granularity", default="per-tensor", choices=[g.value for g in QuantGranularity])
    ins.add_argument("--scheme", default="symmetric", choices=[s.value for s in QuantScheme])
    ins.add_argument("--axis", type=int, default=0)
    ins.add_argument("--group-size", type=int)
    ins.add_argument("--calib-shape", nargs="+", type=int, default=[1, 8])
    ins.add_argument("--prefix")
    ins.set_defaults(func=cmd_insert)

    w8a8 = sub.add_parser("run-w8a8")
    w8a8.add_argument("--model", required=True, help="onnx model path")
    w8a8.add_argument("--calibration-json", required=True, help="json file: tensor_name -> tensor values")
    w8a8.add_argument("--report", required=True, help="output error report json")
    w8a8.add_argument("--export-onnx", help="optional: export quantized model ONNX with QuantLinear/QuantRMSNorm nodes")
    w8a8.set_defaults(func=cmd_w8a8)

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
