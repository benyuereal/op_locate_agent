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

### 2. 逐层中间值对比（定位到发散的层）

输出不一致时，比每层入口的 hidden_states（层间残差真值），看误差从哪层开始、如何累积：

```bash
HIP_VISIBLE_DEVICES=<空闲卡> python3 examples/compare_layers.py \
    --model <模型路径> --probe-dir /tmp/probe_xxx
```

默认全模型采样层（32 层 → [0,1,2,8,16,24,28,30,31]，浅+中+深），打印逐层 cos 衰减表 + 首发散点。定位到发散层后，用 `--only attn/mlp/router --layers N` 在该层内做算子级细化。

`--probe-dir` 读探测结果（见第 3 步）；`--layers` 指定层；`--env` 设 `VLLM_ENABLE_MOE_FUSED_GATE=0` 对照（fused_gate 历史已排除，仅备）。

> 口径：比 **layer input**（`register_forward_pre_hook` + 立即 clone），不是 attn/mlp 输出——layer body 的 in-place 残差操作会污染 post-hook 抓取值，曾致误判。embedding 用 monkey-patch（CustomOp 的 register_forward_hook 不可靠）。源自历史验证过的 `test_stage_compare.py`。

### 3. 探测模型结构（先跑这个，落盘供第 2 步用）

不武断推断 attn/mlp/router 属性名——从模型真实结构反射 + 源码交叉验证，落盘可人核：

```bash
python3 examples/probe_model.py --model <模型路径> --workdir /tmp/probe_xxx
```

输出 `/tmp/probe_xxx/model_probe.json` + `.md`，每项带来源与置信度（high=两路一致）。不加载权重，只 from_config 建空结构，快且省显存。

### 4. 出报告（把定位结论 + vLLM 源码调用链落盘）

定位到算子后，用 `generate_report.py` 生成可人核的报告——从 compare_layers 落盘重算 cos 表，**实时 grep vLLM 源码**抽算子 forward + 调用链，套模板写 `report.md` / `verdict.json`：

```bash
python3 examples/generate_report.py \
    --model <模型路径> \
    --compare-dir /tmp/compare_layers_xxx \   # 可多次传入聚合多轮（粗粒度+router+mlp）
    --probe-dir /tmp/probe_xxx \
    --symptom "vLLM 输出异常描述"
```

产出 `reports/<model>_<date>/`：
- `report.md`：现象、逐 stage cos 表、定位过程、**vLLM 源码调用链**（实时抽取，标 `file:line`）、根因、被证伪假设。
- `verdict.json`：`bug_operator` / `bug_file` / `root_cause` / `confidence`。
- `evidence/stage_cos.csv` + `evidence/source_snippets.md`。

> 多轮聚合：粗粒度逐层 + `--only router` + `--only mlp` 各跑一次 compare_layers，`--compare-dir` 多次传入，被证伪的 router 假设会自动进报告。

> **工作流**：第 3 步探测结构 → 第 1 步看是否对齐 → 不一致则第 2 步定位发散层 → op-locate skill 在层内 sub-op 细化 → 第 4 步出报告。

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
op-locate-agent/
├── examples/      # 入口脚本：quickstart / compare_layers / probe_model / generate_report
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
