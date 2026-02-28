from __future__ import annotations

from dataclasses import dataclass, field

from .config import QuantGranularity, QuantScheme, QuantSpec
from .ir import Model, OpError, OpQuantConfig, QuantParam
from .quant_math import calc_scale_zero_point, fake_quant_dequant_per_tensor, flatten, tensor_error


LINEAR_OPS = {"Linear", "MatMul", "Gemm"}
RMSNORM_OPS = {"RMSNorm"}


@dataclass
class QuantContext:
    model: Model
    calibration_data: dict[str, object] = field(default_factory=dict)
    activation_stats: dict[str, tuple[float, float]] = field(default_factory=dict)
    tensor_qparams: dict[str, QuantParam] = field(default_factory=dict)
    op_quant_configs: dict[str, OpQuantConfig] = field(default_factory=dict)
    op_errors: list[OpError] = field(default_factory=list)


class QuantPass:
    name = "QuantPass"

    def apply(self, ctx: QuantContext) -> None:
        raise NotImplementedError


class WeightQuantPass(QuantPass):
    """First pass: quantize weights per-op (W8A8 requires weight per-channel)."""

    name = "WeightQuantPass"

    def apply(self, ctx: QuantContext) -> None:
        w_spec = QuantSpec(bits=8, granularity=QuantGranularity.PER_CHANNEL, scheme=QuantScheme.SYMMETRIC, axis=0)
        for node in ctx.model.graph.nodes:
            cfg = OpQuantConfig(op_name=node.name, op_type=node.op_type)
            if node.op_type in LINEAR_OPS and len(node.inputs) >= 2:
                w_name = node.inputs[1]
                if w_name in ctx.model.graph.initializers:
                    w = ctx.model.graph.initializers[w_name]
                    s, z = calc_scale_zero_point(w, w_spec)
                    cfg.weight_q = QuantParam(s, z, 8, w_spec.granularity.value, w_spec.scheme.value)
                if len(node.inputs) >= 3 and node.inputs[2] in ctx.model.graph.initializers:
                    # Bias quant param is finalized in calibration pass after input scale is known.
                    pass
            ctx.op_quant_configs[node.name] = cfg


class CalibrationPass(QuantPass):
    """Second pass: layer-wise activation calibration, based on weight-quantized graph."""

    name = "CalibrationPass"

    def apply(self, ctx: QuantContext) -> None:
        a_spec = QuantSpec(bits=8, granularity=QuantGranularity.PER_TENSOR, scheme=QuantScheme.ASYMMETRIC)

        for tensor_name, value in ctx.calibration_data.items():
            vals = flatten(value)
            if vals:
                ctx.activation_stats[tensor_name] = (min(vals), max(vals))

        for node in ctx.model.graph.nodes:
            cfg = ctx.op_quant_configs[node.name]
            input_name = node.inputs[0] if node.inputs else ""
            output_name = node.outputs[0] if node.outputs else ""

            # continuity guarantee: consumer input qparam equals producer output qparam.
            if input_name in ctx.tensor_qparams:
                in_q = ctx.tensor_qparams[input_name]
            else:
                mn, mx = ctx.activation_stats.get(input_name, (-1.0, 1.0))
                s, z = calc_scale_zero_point([mn, mx], a_spec)
                in_q = QuantParam(s, z, 8, a_spec.granularity.value, a_spec.scheme.value)
                ctx.tensor_qparams[input_name] = in_q
            cfg.input_q = in_q

            out_mn, out_mx = ctx.activation_stats.get(output_name, (ctx.activation_stats.get(input_name, (-1.0, 1.0))))
            s, z = calc_scale_zero_point([out_mn, out_mx], a_spec)
            out_q = QuantParam(s, z, 8, a_spec.granularity.value, a_spec.scheme.value)
            cfg.output_q = out_q
            ctx.tensor_qparams[output_name] = out_q

            # bias quantization depends on input + weight quantization
            if node.op_type in LINEAR_OPS and len(node.inputs) >= 3 and cfg.weight_q is not None:
                w_scale = cfg.weight_q.scale[0] if cfg.weight_q.scale else 1.0
                in_scale = cfg.input_q.scale[0] if cfg.input_q and cfg.input_q.scale else 1.0
                b_scale = [max(in_scale * w_scale, 1e-8)]
                cfg.bias_q = QuantParam(b_scale, [0], 32, "per-tensor", QuantScheme.SYMMETRIC.value)


class QuantOpReplacementPass(QuantPass):
    """Third pass: replace FP ops with op-centric quant ops and carry quant params in attrs."""

    name = "QuantOpReplacementPass"

    def apply(self, ctx: QuantContext) -> None:
        for node in ctx.model.graph.nodes:
            cfg = ctx.op_quant_configs.get(node.name)
            if cfg is None:
                continue
            if node.op_type in LINEAR_OPS:
                node.op_type = "QuantLinear"
                node.attrs.update(
                    {
                        "simulation": "QDQ",
                        "input_scale": cfg.input_q.scale if cfg.input_q else [],
                        "input_zp": cfg.input_q.zero_point if cfg.input_q else [],
                        "weight_scale": cfg.weight_q.scale if cfg.weight_q else [],
                        "weight_zp": cfg.weight_q.zero_point if cfg.weight_q else [],
                        "bias_scale": cfg.bias_q.scale if cfg.bias_q else [],
                        "bias_zp": cfg.bias_q.zero_point if cfg.bias_q else [],
                        "output_scale": cfg.output_q.scale if cfg.output_q else [],
                        "output_zp": cfg.output_q.zero_point if cfg.output_q else [],
                    }
                )
            elif node.op_type in RMSNORM_OPS:
                node.op_type = "QuantRMSNorm"
                node.attrs.update(
                    {
                        "input_scale": cfg.input_q.scale if cfg.input_q else [],
                        "input_zp": cfg.input_q.zero_point if cfg.input_q else [],
                        "output_scale": cfg.output_q.scale if cfg.output_q else [],
                        "output_zp": cfg.output_q.zero_point if cfg.output_q else [],
                    }
                )


class ErrorAnalysisPass(QuantPass):
    """Fourth pass: per-op error analysis using output quant params."""

    name = "ErrorAnalysisPass"

    def apply(self, ctx: QuantContext) -> None:
        for node in ctx.model.graph.nodes:
            if not node.outputs:
                continue
            out = node.outputs[0]
            if out not in ctx.calibration_data:
                continue
            cfg = ctx.op_quant_configs.get(node.name)
            if cfg is None or cfg.output_q is None:
                continue
            src = ctx.calibration_data[out]
            oq = cfg.output_q
            qdq = fake_quant_dequant_per_tensor(
                src,
                scale=oq.scale[0],
                zp=oq.zero_point[0],
                bits=oq.bitwidth,
                symmetric=(oq.scheme == QuantScheme.SYMMETRIC.value),
            )
            mse, mae, max_abs = tensor_error(src, qdq)
            ctx.op_errors.append(OpError(op_name=node.name, op_type=node.op_type, mse=mse, mae=mae, max_abs_error=max_abs))


class PassManager:
    def __init__(self, passes: list[QuantPass]):
        self.passes = passes

    def run(self, ctx: QuantContext) -> QuantContext:
        for p in self.passes:
            p.apply(ctx)
        return ctx
