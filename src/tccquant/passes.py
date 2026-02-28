from __future__ import annotations

from dataclasses import dataclass, field

from .config import QuantGranularity, QuantScheme, QuantSpec
from .ir import Model, TensorQuantState, OpError
from .quant_math import calc_scale_zero_point, fake_quant_dequant_per_tensor, tensor_error


@dataclass
class QuantContext:
    model: Model
    calibration_data: dict[str, object] = field(default_factory=dict)
    activation_stats: dict[str, tuple[float, float]] = field(default_factory=dict)
    quant_states: dict[str, TensorQuantState] = field(default_factory=dict)
    op_errors: list[OpError] = field(default_factory=list)


class QuantPass:
    name = "QuantPass"

    def apply(self, ctx: QuantContext) -> None:
        raise NotImplementedError


class CalibrationPass(QuantPass):
    name = "CalibrationPass"

    def apply(self, ctx: QuantContext) -> None:
        for tensor_name, value in ctx.calibration_data.items():
            flat = []
            if isinstance(value, list):
                from .quant_math import flatten

                flat = flatten(value)
            elif isinstance(value, (int, float)):
                flat = [float(value)]
            if not flat:
                continue
            ctx.activation_stats[tensor_name] = (min(flat), max(flat))


class W8A8QuantPass(QuantPass):
    """W8A8 full-network quant pass.

    - Weight: per-channel symmetric 8-bit
    - Activation: per-tensor asymmetric 8-bit
    """

    name = "W8A8QuantPass"

    def apply(self, ctx: QuantContext) -> None:
        w_spec = QuantSpec(bits=8, granularity=QuantGranularity.PER_CHANNEL, scheme=QuantScheme.SYMMETRIC, axis=0)
        a_spec = QuantSpec(bits=8, granularity=QuantGranularity.PER_TENSOR, scheme=QuantScheme.ASYMMETRIC)

        # Quantize initializers (weights)
        for init_name, tensor in ctx.model.graph.initializers.items():
            if not isinstance(tensor, list):
                continue
            scale, zp = calc_scale_zero_point(tensor, w_spec)
            ctx.quant_states[init_name] = TensorQuantState(
                scale=scale,
                zero_point=zp,
                bitwidth=8,
                granularity=w_spec.granularity.value,
                scheme=w_spec.scheme.value,
            )

        # Quantize activation tensors observed by calibration
        for tensor_name, (mn, mx) in ctx.activation_stats.items():
            scale, zp = calc_scale_zero_point([mn, mx], a_spec)
            ctx.quant_states[tensor_name] = TensorQuantState(
                scale=scale,
                zero_point=zp,
                bitwidth=8,
                granularity=a_spec.granularity.value,
                scheme=a_spec.scheme.value,
            )


class ErrorAnalysisPass(QuantPass):
    name = "ErrorAnalysisPass"

    def apply(self, ctx: QuantContext) -> None:
        for node in ctx.model.graph.nodes:
            if not node.outputs:
                continue
            out = node.outputs[0]
            if out not in ctx.calibration_data or out not in ctx.quant_states:
                continue
            src = ctx.calibration_data[out]
            st = ctx.quant_states[out]
            qdq = fake_quant_dequant_per_tensor(
                src,
                scale=st.scale[0],
                zp=st.zero_point[0],
                bits=st.bitwidth,
                symmetric=(st.scheme == QuantScheme.SYMMETRIC.value),
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
