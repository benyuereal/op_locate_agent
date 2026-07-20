# OpLocate Agent

> 把"vLLM 输出和 transformers 对不齐，定位到具体算子"——从天级压到小时级。
> 适用任意 HF/ModelScope 模型（MoE / dense）在 vLLM 上的精度异常定位。

## 前提

- vLLM + transformers + torch 已装
- 模型已下载到本地，目录含 `config.json`
- **卡用 `HIP_VISIBLE_DEVICES` 前置指定**，脚本不自动选（避免误占在线服务）

## 四步定位

### 1. 探测模型结构

从模型真实结构反射 + 源码交叉验证，不武断推断属性名：

```bash
python3 examples/probe_model.py --model <模型路径> --workdir /tmp/probe_xxx
```

### 2. 逐层定位发散层

比每层输入 hidden_states（层间残差真值），看误差从哪层开始：

```bash
HIP_VISIBLE_DEVICES=<空闲卡> python3 examples/compare_layers.py \
    --model <模型路径> --probe-dir /tmp/probe_xxx
```

默认全模型采样（32 层 → [0,1,2,8,16,24,28,30,31]），打印逐层 cos 衰减表 + 首发散点。

### 3. 算子级细化

定位到发散层后，在该层内做子算子 drill-down：

```bash
# MoE 模型：逐算子对比（router → attn → mlp）
HIP_VISIBLE_DEVICES=<空闲卡> python3 examples/compare_layers.py \
    --model <模型路径> --probe-dir /tmp/probe_xxx \
    --op router --layers <发散层号>
HIP_VISIBLE_DEVICES=<空闲卡> python3 examples/compare_layers.py \
    --model <模型路径> --probe-dir /tmp/probe_xxx \
    --op mlp --layers <发散层号>

# Dense 模型：逐算子对比（attn → mlp → rmsnorm 等）
HIP_VISIBLE_DEVICES=<空闲卡> python3 examples/compare_layers.py \
    --model <模型路径> --probe-dir /tmp/probe_xxx \
    --op <算子属性名> --layers <发散层号>
```

`--op` 支持任意属性名（如 `rmsnorm`/`input_layernorm`/`self_attn`），不限于 `attn/mlp/router`。
`--env` 设模型专属环境变量做对照（如 `VLLM_ENABLE_MOE_FUSED_GATE=0`，**默认不设**——排查工具不预设结论）。

### 4. 出报告

定位结论 + vLLM 源码调用链落盘成可人核的报告：

```bash
python3 examples/generate_report.py \
    --model <模型路径> \
    --compare-dir /tmp/compare_layers_xxx \   # 可多次传入聚合多轮细化
    --probe-dir /tmp/probe_xxx \
    --symptom "vLLM 输出异常描述"
```

产出 `reports/<model>_<date>/`：`report.md` + `verdict.json` + `evidence/`。

> **工作流**：探测结构 → 逐层定位 → 算子细化 → 出报告。

## 快速开始示例

```bash
# 0. 卡空闲检查
rocm-smi

# 1. 探测结构（一次性）
python3 examples/probe_model.py --model /path/to/model --workdir /tmp/probe_mymodel

# 2. 逐层对比（定位发散层）
HIP_VISIBLE_DEVICES=2,3,4,5 python3 examples/compare_layers.py \
    --model /path/to/model --probe-dir /tmp/probe_mymodel

# 3. 算子 drill-down（在发散层细化）
HIP_VISIBLE_DEVICES=2,3,4,5 python3 examples/compare_layers.py \
    --model /path/to/model --probe-dir /tmp/probe_mymodel \
    --op mlp --layers <发散层号>

# 4. 出报告
python3 examples/generate_report.py \
    --model /path/to/model \
    --compare-dir /tmp/compare_layers_xxx \
    --probe-dir /tmp/probe_mymodel \
    --symptom "vLLM 输出异常描述"
```

## 用 agent 自动定位

```bash
claude -p "用 op-locate skill 定位 <模型路径> 的 vLLM 精度问题" \
  --allowedTools "Bash,Read,Write,Edit,Glob,Grep,WebFetch,Skill"
```

不用 LLM 也能只用工具库：`from lib import load_model_profile, resolve_code_paths`。

## 单独调试

```bash
HIP_VISIBLE_DEVICES=<空闲卡> TP=<n> ./run/run_vllm.sh <模型路径> "你好"
HIP_VISIBLE_DEVICES=<空闲卡> ./run/run_hf.sh <模型路径> "你好"
```

## 测试

```bash
python3 -m pytest tests/ -q          # 单测
python3 lib/config_loader.py <模型路径>   # lib 自检
```

## 目录

```
op-locate-agent/
├── examples/      # 入口脚本：probe_model / compare_layers / generate_report
├── lib/           # 工具库：config/hook/compare/probe/path
├── knowledge/     # 知识库：arch→路径、已知坑、平台
├── skills/op-locate/  # Claude Code skill
├── run/           # 标准启动脚本
├── reports/       # 定位报告输出
└── tests/         # 单测
```

## 设计要点

- **运行时探针为准**：根因以 hook 抓取为准，不以静态分析为准。
- **不预设结论**：默认不设修复环境变量，需对照时显式开 `--env`。
- **不自动回写知识库**：人工 review 是质量门。
- **泛化设计**：`--op` 支持任意属性名，不限于 MoE 三件套；dense/MoE 共用同一套探针体系。
