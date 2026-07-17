# OpLocate Agent

> 把"vLLM 输出和 transformers 对不齐，定位到具体算子"从**天级别**压到**小时级别**。
> 详见 [ARCHITECTURE.md](./ARCHITECTURE.md)。

## 5 分钟上手

### 前提
- vLLM + transformers + torch 已装（本机 vLLM 0.15.1, torch 2.9.0）
- 模型本地目录含 config.json
- **卡由用户用 `HIP_VISIBLE_DEVICES` 前置指定**，脚本不自动判定（避免误占在线服务）

## 快速验证（完整 vLLM + HF 启动例子）

入门先看 [`examples/quickstart_antangelmed.py`](./examples/quickstart_antangelmed.py)——
它是这个 repo 的"门面"脚本：探测平台 → 跑 HF(基准) → 跑 vLLM → 对比输出 token，
一次给出"vLLM 和 HF 对不对齐"的结论。也是 clone 下来第一个该跑的脚本。

### 一次精度对比测试怎么跑

```bash
cd op_locate_agent

HIP_VISIBLE_DEVICES=<空闲卡> python3 examples/quickstart_antangelmed.py \
    --model <模型路径> \
    --prompt "你好，请介绍一下你自己" \
    --max-tokens 32
```

**参数说明**

| 参数 | 默认 | 说明 |
|---|---|---|
| `HIP_VISIBLE_DEVICES` | 必填（前置） | 用哪些卡，**用户自己指定**，脚本不自动选 |
| `--model` | `/models/AntAngelMed` | 模型本地目录（含 config.json） |
| `--tp` | =可见卡数 | vLLM tensor parallel，默认等于暴露的卡数 |
| `--prompt` | 内置 | 对比用的提示词 |
| `--max-tokens` | 32 | 生成 token 数 |
| `--skip-hf` / `--skip-vllm` | off | 只跑一边 |
| `--fix-env` | off | 显式设 `VLLM_ENABLE_MOE_FUSED_GATE=0` 做对照。**默认不设**——排查工具不预设结论 |

**几种典型用法**

```bash
# 默认模型、4 卡：vLLM TP=4 + HF device_map 铺 4 卡
HIP_VISIBLE_DEVICES=2,3,4,5 python3 examples/quickstart_antangelmed.py

# 指定别的模型（大 MoE 同样需要多卡）
HIP_VISIBLE_DEVICES=2,3,4,5 python3 examples/quickstart_antangelmed.py --model /path/to/other_moe

# 只跑 vLLM 一边（先确认 vLLM 自己能出正常输出）
HIP_VISIBLE_DEVICES=2,3,4,5 python3 examples/quickstart_antangelmed.py --skip-hf

# 已定位到需绕过 fused_gate 做对照验证时（默认不设，排查工具不预设结论）
HIP_VISIBLE_DEVICES=2,3,4,5 python3 examples/quickstart_antangelmed.py --fix-env
```

**选卡注意**

- 大模型（如 AntAngelMed ~192GB）**HF 单卡装不下**，靠 `device_map="auto"` 铺多卡，
  所以 `HIP_VISIBLE_DEVICES` 要暴露**≥4 张空闲卡**（每张 ~64GB）。
- vLLM 用 TP，`--tp` 必须 ≤ 暴露卡数（默认相等）。
- 先用 `rocminfo` / `rocm-smi` 确认指定的卡**确实空闲**，别占用在线服务的卡。
- `HIP_VISIBLE_DEVICES=0,1,6,7` 这种写法只是示例，**实际请填本机空闲卡号**。

**跑起来会看到什么（日志阶段）**

脚本分阶段实时打印进度（不会卡几分钟没反应）：

```
[platform] Hygon DCU BW100 (gfx936), HIP 6.3.26113, 4 devices, is_cuda=False is_rocm=True
[platform] HIP_VISIBLE_DEVICES=2,3,4,5 (4 卡)
============================================================
[HF] 加载 transformers ...
============================================================
  [HF] 1/4 加载 config...
  [HF] config patch: ['add_router_probs', ...]
  [HF] 2/4 加载模型权重 (device_map=auto, bf16, eager)...     ← 大模型这里最慢
  [HF] 3/4 加载 tokenizer...
  [HF] 4/4 generate...
  [HF] token_ids: [198, 198, 7018, ...]
============================================================
[vLLM] 加载 vLLM (tp=4, VLLM_ENABLE_MOE_FUSED_GATE=0)...
============================================================
  [vLLM] 1/3 构造 LLM (tp=4, bf16, eager, gmu=0.9)...        ← 这里最慢：加载权重+编译图+KV cache
  [vLLM] 2/3 generate...
  [vLLM] token_ids: [...]
============================================================
[compare] HF vs vLLM
============================================================
[compare] token 前缀一致率: 95.8% (23/24)
[compare] vLLM 含 NULL(188): False
[compare] HF text:    '...'
[compare] vLLM text:  '...'

[verdict] ✅ vLLM 与 HF 基本一致 — 精度正常
```

**verdict 判定**

- ✅ 一致率 ≥ 90% 且无 NULL → 精度正常
- ❌ vLLM 输出全 NULL(188) → 精度问题未修复，接着用 agent 定位
- ⚠️ 一致率 < 90% 且非全 NULL → 部分对齐，需进一步定位（用 op-locate skill）

### 逐层 / 逐算子中间值对比（定位到具体层/算子）

quickstart 只比最终 token；当输出不一致时，用
[`examples/compare_layers.py`](./examples/compare_layers.py) 比每一层、
每个关键算子的中间张量，**直接打印哪一层开始发散、哪个算子开始发散**：

```bash
cd op_locate_agent

# 默认比所有层的 attn_out / mlp_out（+ MoE 的 router_logits/topk）
HIP_VISIBLE_DEVICES=2,3,4,5 python3 examples/compare_layers.py

# 大模型省显存：只比几层
HIP_VISIBLE_DEVICES=2,3,4,5 python3 examples/compare_layers.py --layers 0,1,2,3,15,30

# 只比 MoE router（选专家那一步）
HIP_VISIBLE_DEVICES=2,3,4,5 python3 examples/compare_layers.py --only router

# 对照：让 vLLM 绕过 fused_gate，看中间值是否恢复一致
HIP_VISIBLE_DEVICES=2,3,4,5 python3 examples/compare_layers.py --fix-env
```

输出形如（逐层逐算子一行，标出首个发散点）：

```
[compare] 逐层 / 逐算子中间值对比
---------------------------------------------------------------------------
layer0_attn_out                1.000000   0.0000e+00   0.0000e+00 ✅
layer0_mlp_out                 0.999998   1.2e-03      3.4e-05    ✅
layer1_attn_out                1.000000   0.0000e+00   0.0000e+00 ✅
layer1_mlp_out                 0.872341   4.7e+01      1.2e+00    ❌
---------------------------------------------------------------------------
[verdict] ❌ 首个发散点: layer1_mlp_out
  → 该层/算子之前的层都一致，从这里开始 HF 与 vLLM 分叉。
  → 用 op-locate skill 在此层内继续细化（sub-op 级 hook）。
```

> 工作流：quickstart 发现不一致 → compare_layers 找到首个发散层/算子 →
> op-locate skill 在该层内挂 sub-op hook 进一步缩小到具体算子。

### 单独跑启动脚本（调试用）

不想端到端、只想单独看一边的启动行为：
```bash
HIP_VISIBLE_DEVICES=2,3,4,5 TP=4 ./run/run_vllm.sh /models/AntAngelMed "你好"
HIP_VISIBLE_DEVICES=2,3,4,5 ./run/run_hf.sh /models/AntAngelMed "你好"
```

## 使用 agent 定位

### 方式 A：Claude Code 交互式

把 skill 软链到 Claude 的 skills 目录：
```bash
mkdir -p ~/.claude/skills
ln -s $(pwd)/skills/op-locate ~/.claude/skills/op-locate
```
然后在 Claude Code 里：
```
/op-locate /models/AntAngelMed
```

### 方式 B：Claude Code headless（脱离交互式 CLI，可塞 cron/CI）

```bash
cd op_locate_agent
claude -p "用 op-locate skill 定位 /models/NewMoE 模型的 vLLM 精度问题" \
  --allowedTools "Bash,Read,Write,Edit,Glob,Grep,WebFetch,Skill"
```

### 方式 C：只用工具库（不要 LLM，自己跑）

```python
from lib import load_model_profile, resolve_code_paths

profile = load_model_profile("/models/AntAngelMed")
print(profile.moe_summary())

cp = resolve_code_paths(profile)
for k in cp.key_ops:
    print(k.name, k.known_issue)
```

## 目录

```
op_locate_agent/
├── ARCHITECTURE.md          # 设计文档（先读这个）
├── README.md                # 本文件
├── skills/op-locate/        # Claude Code skill（流程编排）
│   ├── SKILL.md  probe.md  compare.md  report.md
├── lib/                     # 工具库（agent 的"手"）
├── knowledge/               # 知识库（agent 的"脑"）
├── run/                     # 标准启动脚本
├── reports/                 # 定位报告输出
└── tests/                   # 单测 + 黄金回归
```

## 测试

```bash
cd op_locate_agent
python3 -m pytest tests/ -v          # 全部单测（32 个，含 AntAngelMed fixture）
python3 -m pytest tests/test_e2e_antangelmed.py -v   # 黄金回归
```

## 验证 lib 自检

```bash
python3 lib/config_loader.py /models/AntAngelMed    # 解析配置
python3 -m lib.path_resolver /models/AntAngelMed    # 解析代码路径
python3 lib/tensor_compare.py                       # 对比器自检
```

## 当前已沉淀的模型

| 模型 | 状态 | 根因 | 报告 |
|---|---|---|---|
| AntAngelMed (BailingMoeV2) | ✅ 已定位 | `ops.moe_fused_gate` 在 gfx936 选错专家 | `precision_compare/`（待迁入 reports/） |

对照验证：`VLLM_ENABLE_MOE_FUSED_GATE=0`（走 Python grouped_topk）。**这是该案例的绕过手段，不是 gfx936 通用补丁**——其他模型需独立排查。

## 设计要点（一句话版）

- **不依赖 langchain**：Claude 本身就是最好的任务编排器，工具层+知识层先沉淀，编排是最后一公里。
- **脱离 CLI 但能力不打折**：用 `claude -p` headless，推理仍满血 Claude，工具仍是本库。
- **运行时探针为准**：根因以运行时 hook 抓取为准，不以代码静态分析为准（曾导致误判）。
- **不自动回写知识库**：人工 review 是质量门，防错误结论污染。
