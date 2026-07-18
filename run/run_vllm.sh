#!/usr/bin/env bash
# run_vllm.sh — 标准化 vLLM 离线推理启动
#
# 卡用 HIP_VISIBLE_DEVICES 前置指定（用户负责确认空闲）。
#
# 用法：
#   HIP_VISIBLE_DEVICES=2 ./run_vllm.sh /models/AntAngelMed
#   HIP_VISIBLE_DEVICES=0,1,6,7 TP=4 ./run_vllm.sh /models/AntAngelMed "你好"
#   HIP_VISIBLE_DEVICES=2 ENV=1 ./run_vllm.sh /models/AntAngelMed   # 显式绕过 fused_gate
#
# 环境变量：
#   HIP_VISIBLE_DEVICES=<ids>      # 必填，卡
#   TP=<n>                          # tensor parallel，默认=可见卡数
#   ENV=1                           # 显式设 VLLM_ENABLE_MOE_FUSED_GATE=0 做对照（排查工具默认不设）
#   GPU_MEM=0.9  MAX_MODEL_LEN=2048  DTYPE=auto

set -euo pipefail

MODEL="${1:?usage: HIP_VISIBLE_DEVICES=<ids> $0 <model_path> [prompt]}"
PROMPT="${2:-你好，我是一名医生}"

if [ -z "${HIP_VISIBLE_DEVICES:-}" ]; then
  echo "错误：请用 HIP_VISIBLE_DEVICES 前置指定卡" >&2
  echo "  例：HIP_VISIBLE_DEVICES=2 $0 $MODEL" >&2
  exit 2
fi

N_GPU=$(awk -F, '{print NF}' <<<"$HIP_VISIBLE_DEVICES")
TP="${TP:-$N_GPU}"
if [ "$TP" != "$N_GPU" ]; then
  echo "错误：TP=$TP 与 HIP_VISIBLE_DEVICES 卡数($N_GPU) 不匹配" >&2
  exit 2
fi

if [ -n "${ENV:-}" ]; then
  export VLLM_ENABLE_MOE_FUSED_GATE=0
  echo "[run_vllm] 已显式设 VLLM_ENABLE_MOE_FUSED_GATE=0（对照验证）"
else
  echo "[run_vllm] 未设修复 env（排查工具默认不预设结论）"
fi

cat > /tmp/_run_vllm.py <<PY
import os
from vllm import LLM, SamplingParams
llm = LLM(
    model="$MODEL", tensor_parallel_size=int("${TP}"),
    enforce_eager=True,
    gpu_memory_utilization=float(os.getenv("GPU_MEM", "0.9")),
    max_model_len=int(os.getenv("MAX_MODEL_LEN", "2048")),
    dtype=os.getenv("DTYPE", "auto"),
    trust_remote_code=True,
)
sp = SamplingParams(temperature=0.0, max_tokens=32)
outs = llm.generate(["$PROMPT"], sp)
print("=== vLLM output ===")
print(repr(outs[0].outputs[0].text))
toks = outs[0].outputs[0].token_ids
print("token_ids:", toks)
print("contains NULL(188)?", 188 in toks)
PY

python3 /tmp/_run_vllm.py
