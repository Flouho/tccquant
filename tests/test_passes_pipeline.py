import json

from tccquant.ir import Graph, Model, Node
from tccquant.passes import (
    CalibrationPass,
    ErrorAnalysisPass,
    PassManager,
    QuantContext,
    QuantOpReplacementPass,
    WeightQuantPass,
)
from tccquant.pipeline import save_error_report


def _build_model() -> Model:
    return Model(
        graph=Graph(
            nodes=[
                Node(name="Linear_0", op_type="Linear", inputs=["x", "w0", "b0"], outputs=["h"]),
                Node(name="RMSNorm_0", op_type="RMSNorm", inputs=["h"], outputs=["y"]),
            ],
            initializers={
                "w0": [[0.1, -0.2], [0.3, -0.4]],
                "b0": [0.01, -0.02],
            },
        )
    )


def test_op_centric_pipeline_replaces_ops_and_keeps_adjacent_qparams_consistent(tmp_path):
    model = _build_model()
    calib = {
        "x": [[0.0, 0.1], [0.2, 0.3]],
        "h": [[0.2, -0.1], [0.05, 0.9]],
        "y": [[0.19, -0.08], [0.04, 0.88]],
    }

    ctx = QuantContext(model=model, calibration_data=calib)
    mgr = PassManager([WeightQuantPass(), CalibrationPass(), QuantOpReplacementPass(), ErrorAnalysisPass()])
    mgr.run(ctx)

    linear = next(n for n in ctx.model.graph.nodes if n.name == "Linear_0")
    rms = next(n for n in ctx.model.graph.nodes if n.name == "RMSNorm_0")

    assert linear.op_type == "QuantLinear"
    assert rms.op_type == "QuantRMSNorm"

    # continuity: previous op output quant params == next op input quant params
    assert linear.attrs["output_scale"] == rms.attrs["input_scale"]
    assert linear.attrs["output_zp"] == rms.attrs["input_zp"]

    # QuantLinear carries full requested qparams
    for key in [
        "input_scale",
        "input_zp",
        "weight_scale",
        "weight_zp",
        "bias_scale",
        "bias_zp",
        "output_scale",
        "output_zp",
    ]:
        assert key in linear.attrs

    assert len(ctx.op_errors) >= 1

    out = tmp_path / "report.json"
    save_error_report(ctx, str(out))
    payload = json.loads(out.read_text())
    assert "ops" in payload
    assert len(payload["ops"]) >= 1
