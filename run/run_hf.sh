#!/usr/bin/env bash
# run_hf.sh — 标准化 transformers(HF) 推理启动（正确性对照基准）
#
# 卡用 HIP_VISIBLE_DEVICES 前置指定。大模型用多卡 device_map=auto 铺开。
#
# 用法：
#   HIP_VISIBLE_DEVICES=3 ./run_hf.sh /models/AntAngelMed
#   HIP_VISIBLE_DEVICES=2,3,4,5 ./run_hf.sh /models/AntAngelMed "你好"   # 大模型多卡
#
# 默认：bf16，greedy，max_new_tokens=32，attn_implementation=eager，
#       自动补 BailingMoeV2 等 config 缺失字段。

set -euo pipefail

MODEL="${1:?usage: HIP_VISIBLE_DEVICES=<ids> $0 <model_path> [prompt]}"
PROMPT="${2:-你好，我是一名医生}"

if [ -z "${HIP_VISIBLE_DEVICES:-}" ]; then
  echo "错误：请用 HIP_VISIBLE_DEVICES 前置指定卡" >&2
  echo "  例：HIP_VISIBLE_DEVICES=2,3,4,5 $0 $MODEL" >&2
  exit 2
fi

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$HERE")"

cat > /tmp/_run_hf.py <<PY
import sys
sys.path.insert(0, "$ROOT")
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from lib import config_patch

model_path = "$MODEL"
cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
patched = config_patch.patch_config(cfg)
if patched:
    print(f"[HF] config patch: {patched}")

model = AutoModelForCausalLM.from_pretrained(
    model_path, config=cfg, torch_dtype=torch.bfloat16,
    trust_remote_code=True, local_files_only=True,
    attn_implementation="eager", device_map="auto",
).eval()
tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)

first_dev = next(model.parameters()).device
ids = tok("$PROMPT", return_tensors="pt").input_ids.to(first_dev)
with torch.no_grad():
    out = model.generate(ids, max_new_tokens=32, do_sample=False)
print("=== HF output ===")
print(repr(tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)))
print("token_ids:", out[0][ids.shape[1]:].tolist())
PY

python3 /tmp/_run_hf.py
