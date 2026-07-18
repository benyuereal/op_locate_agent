# OpLocate Agent

> 把"vLLM 输出和 transformers 对不齐，定位到具体算子"从**天级别**压到**小时级别**。
> 适用任意 HF/ModelScope 模型在 vLLM 上的精度异常定位。设计见 [ARCHITECTURE.md](./ARCHITECTURE.md)。

## 前提

- 已装 vLLM + transformers + torch
- 模型已下载到本地，目录含 `config.json`
- **卡用 `HIP_VISIBLE_DEVICES` 前置指定**，脚本不自动选（避免误占在线服务）。先用 `rocminfo`/`rocm-smi` 确认指定的卡空闲

## 三步定位

### 1. 端到端对比（先看 vLLM 和 HF 对不对齐）

```bash
HIP_VISIBLE_DEVICES=<空闲卡> python3 examples/quickstart_antangelmed.py \
    --model <模型路径> --max-tokens 32
```

`--model` **必填**。同时跑 HF(基准)+vLLM，对比输出 token：✅ 一致率≥90% 且无 NULL / ❌ vLLM 全 NULL / ⚠️ 部分不一致。
`--env` 显式设 `VLLM_ENABLE_MOE_FUSED_GATE=0` 做对照（**默认不设**——排查工具不预设结论）。

### 2. 逐层 / 逐算子对比（定位到发散的层/算子）

输出不一致时，比每一层的中间张量，打印首个发散点：

```bash
HIP_VISIBLE_DEVICES=<空闲卡> python3 examples/compare_layers.py \
    --model <模型路径> --layers 0,1,2,3 --probe-dir /tmp/probe_xxx
```

`--probe-dir` 读探测结果（见第 3 步），跳过运行时探测；`--layers` 只比指定层（省显存）；`--only router` 只比 MoE 选专家；`--env` 对照验证。

### 3. 探测模型结构（先跑这个，落盘供第 2 步用）

不武断推断 attn/mlp/router 属性名——从模型真实结构反射 + 源码交叉验证，落盘可人核：

```bash
python3 examples/probe_model.py --model <模型路径> --workdir /tmp/probe_xxx
```

输出 `/tmp/probe_xxx/model_probe.json` + `.md`，每项带来源与置信度（high=两路一致）。不加载权重，只 from_config 建空结构，快且省显存。

> **工作流**：第 3 步探测结构 → 第 1 步看是否对齐 → 不一致则第 2 步定位发散层 → op-locate skill 在层内 sub-op 细化。

## 用 agent 自动定位

把 skill 软链后，在 Claude Code 里 `/op-locate <模型路径>`；或 headless：

```bash
claude -p "用 op-locate skill 定位 <模型路径> 的 vLLM 精度问题" \
  --allowedTools "Bash,Read,Write,Edit,Glob,Grep,WebFetch,Skill"
```

不用 LLM 也能只用工具库：`from lib import load_model_profile, resolve_code_paths`。

## 单独跑启动脚本（调试）

```bash
HIP_VISIBLE_DEVICES=<空闲卡> TP=<n> ./run/run_vllm.sh <模型路径> "你好"
HIP_VISIBLE_DEVICES=<空闲卡> ./run/run_hf.sh <模型路径> "你好"
```

## 测试

```bash
python3 -m pytest tests/ -q          # 单测（48 个）
python3 lib/config_loader.py <模型路径>   # lib 自检
```

## 目录

```
op_locate_agent/
├── examples/      # 入口脚本：quickstart / compare_layers / probe_model
├── lib/           # 工具库（agent 的"手"）：config/hook/compare/probe
├── knowledge/     # 知识库（agent 的"脑"）：arch→路径、已知坑、平台
├── skills/op-locate/  # Claude Code skill（流程编排）
├── run/           # 标准启动脚本
├── reports/       # 定位报告输出（.gitignore）
└── tests/         # 单测
```

## 设计要点

- **运行时探针为准**：根因以 hook 抓取为准，不以静态分析为准（曾致误判）。
- **不预设结论**：排查工具默认不设修复环境变量，需对照时显式开 `--env`。
- **不自动回写知识库**：人工 review 是质量门，防错误结论污染。
- **脱离 CLI 但能力不打折**：headless `claude -p` 推理仍满血，工具仍是本库。
