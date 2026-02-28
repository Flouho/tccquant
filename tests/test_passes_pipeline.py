import json

from tccquant.ir import Graph, Model, Node
from tccquant.passes import CalibrationPass, ErrorAnalysisPass, PassManager, QuantContext, W8A8QuantPass
from tccquant.pipeline import save_error_report


def _build_model() -> Model:
    return Model(
        graph=Graph(
            nodes=[
                Node(name="MatMul_0", op_type="MatMul", inputs=["x", "w"], outputs=["mm_out"]),
                Node(name="Relu_0", op_type="Relu", inputs=["mm_out"], outputs=["y"]),
            ],
            initializers={"w": [[0.1, -0.2], [0.3, -0.4]]},
        )
    )


def test_w8a8_pipeline_generates_quant_state_and_error_report(tmp_path):
    model = _build_model()
    calib = {
        "mm_out": [[0.2, -0.1], [0.05, 0.9]],
        "y": [[0.19, -0.08], [0.04, 0.88]],
    }

    ctx = QuantContext(model=model, calibration_data=calib)
    mgr = PassManager([CalibrationPass(), W8A8QuantPass(), ErrorAnalysisPass()])
    mgr.run(ctx)

    assert "w" in ctx.quant_states
    assert "mm_out" in ctx.quant_states
    assert len(ctx.op_errors) >= 1

    out = tmp_path / "report.json"
    save_error_report(ctx, str(out))
    payload = json.loads(out.read_text())
    assert "ops" in payload
    assert len(payload["ops"]) >= 1
