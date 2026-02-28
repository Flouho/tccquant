# TCCQuant

面向客户交付的最小可用量化工程（基于 PPQ 工作流抽象），支持：

- 删除 PPQ 原工程中的非必要文件（可配置）
- 在模型中间表示图（IR）指定位置插入/删除 Q-DQ 节点（可映射到 ONNX/PPQ 流程）
- 量化粒度：per-tensor / per-channel / per-group / per-block
- 对称 / 非对称量化
- 预设策略：W4A16、W4A8、W8A8、W8A16
- 提供最基础端到端测试（传统小模型 + LLM 风格线性层）

## 快速开始

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[test]
pytest -q
```

## CLI

- 删除 PPQ 冗余文件（演练）：

```bash
python scripts/tccquant_cli.py prune-ppq /path/to/ppq --dry-run
```

- 指定节点后插入 Q-DQ：

```bash
python scripts/tccquant_cli.py insert-qdq \
  --model model.json \
  --output model_qdq.json \
  --node MatMul_0 \
  --bits 8 \
  --granularity per-channel \
  --scheme symmetric \
  --axis 0
```

## 目录

- `src/tccquant/config.py`：量化配置
- `src/tccquant/policies.py`：预设量化策略
- `src/tccquant/quant_math.py`：scale/zero-point 计算
- `src/tccquant/graph_editor.py`：Q-DQ 图编辑
- `src/tccquant/ppq_cleanup.py`：PPQ 冗余文件清理
- `tests/`：基础与端到端测试
