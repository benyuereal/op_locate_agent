"""
quickstart_antangelmed.py — 完整的 vLLM + HF 启动与对比示例

repo 的"门面"例子：clone 下来第一个跑的脚本。
端到端验证：探测平台 → 跑 HF(基准) → 跑 vLLM(带修复) → 对比输出。

== 卡的指定 ==
卡由用户用 HIP_VISIBLE_DEVICES 前置指定，脚本继承、不覆盖：
    HIP_VISIBLE_DEVICES=0,1,6,7 python3 examples/quickstart_antangelmed.py

vLLM 的 TP 默认 = HIP_VISIBLE_DEVICES 的卡数（可用 --tp 覆盖）。
HF 用 device_map="auto" 自动铺在暴露的卡上（大模型多卡）。
vLLM 和 HF 各起独立子进程，避免同进程显存互占。

== 用法 ==
    # 4 卡（vLLM TP=4 + HF device_map 4 卡）
    HIP_VISIBLE_DEVICES=0,1,6,7 python3 examples/quickstart_antangelmed.py

    # 单卡（小模型）
    HIP_VISIBLE_DEVICES=2 python3 examples/quickstart_antangelmed.py

    # vLLM TP 覆盖
    HIP_VISIBLE_DEVICES=0,1,6,7 python3 examples/quickstart_antangelmed.py --tp 2

    # 只跑某一边
    HIP_VISIBLE_DEVICES=2,3,4,5 python3 examples/quickstart_antangelmed.py --skip-vllm
    HIP_VISIBLE_DEVICES=2 python3 examples/quickstart_antangelmed.py --skip-hf

== 前置 ==
    - vLLM + transformers + torch 已装
    - 模型已下载到本地（默认 /models/AntAngelMed）
    - 用户已确认指定卡空闲（rocminfo 查显存）

== 说明 ==
    - gfx936 (BW100) 上 AntAngelMed 需 VLLM_ENABLE_MOE_FUSED_GATE=0 才正确（已知坑）
    - 本脚本默认带上该环境变量；--no-fix-env 关闭；其他模型按定位结果调整
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)


def parse_args():
    ap = argparse.ArgumentParser(
        description="vLLM + HF quickstart 对比",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--model", default="/models/AntAngelMed",
                    help="模型本地路径")
    ap.add_argument("--tp", type=int, default=None,
                    help="vLLM tensor parallel size，默认=HIP_VISIBLE_DEVICES 卡数")
    ap.add_argument("--prompt", default="你好，请介绍一下你自己，你叫什么名字")
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--skip-hf", action="store_true", help="跳过 HF 基准")
    ap.add_argument("--skip-vllm", action="store_true", help="跳过 vLLM")
    ap.add_argument("--no-fix-env", action="store_true",
                    help="不设 VLLM_ENABLE_MOE_FUSED_GATE=0（默认会设，gfx936 MoE 需要）")
    return ap.parse_args()


def n_visible_gpus() -> int:
    """HIP_VISIBLE_DEVICES 暴露的卡数"""
    v = os.environ.get("HIP_VISIBLE_DEVICES", "")
    if not v:
        return 0
    return len([x for x in v.split(",") if x.strip() != ""])


def probe():
    """打印平台信息（不自动选卡，仅展示）"""
    from lib import probe_platform
    info = probe_platform()
    print(f"[platform] {info.summary()}")
    vis = os.environ.get("HIP_VISIBLE_DEVICES", "(未设置)")
    print(f"[platform] HIP_VISIBLE_DEVICES={vis} ({n_visible_gpus()} 卡)")
    print("[platform] 卡由用户前置指定，脚本继承不覆盖")


# 子进程脚本：结果以 RESULT_JSON 行打印
_HF_SCRIPT = r"""
import os, sys, json
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
sys.path.insert(0, %r)
from lib import config_patch

model_path = %r
prompt = %r
max_tokens = %r

cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
patched = config_patch.patch_config(cfg)
if patched:
    print(f"[HF] config patch: {patched}", flush=True)

model = AutoModelForCausalLM.from_pretrained(
    model_path, config=cfg, torch_dtype=torch.bfloat16,
    trust_remote_code=True, local_files_only=True,
    attn_implementation="eager", device_map="auto",
).eval()
tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)

first_dev = next(model.parameters()).device
ids = tok(prompt, return_tensors="pt").input_ids.to(first_dev)
with torch.no_grad():
    out = model.generate(ids, max_new_tokens=max_tokens, do_sample=False)
new_ids = out[0][ids.shape[1]:].tolist()
text = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
print("RESULT_JSON:", json.dumps({"token_ids": new_ids, "text": text}), flush=True)
"""

_VLLM_SCRIPT = r"""
import os, sys, json
from vllm import LLM, SamplingParams
model_path = %r
prompt = %r
max_tokens = %r
tp = %r

llm = LLM(
    model=model_path, tensor_parallel_size=tp, dtype="bfloat16",
    trust_remote_code=True, enforce_eager=True,
    max_model_len=2048, gpu_memory_utilization=0.9,
)
sp = SamplingParams(temperature=0.0, max_tokens=max_tokens)
outs = llm.generate([prompt], sp)
new_ids = list(outs[0].outputs[0].token_ids)
text = outs[0].outputs[0].text
print("RESULT_JSON:", json.dumps({"token_ids": new_ids, "text": text}), flush=True)
"""


def run_subprocess(script: str, extra_env: dict = None) -> dict:
    """起子进程跑 script，继承父环境（含 HIP_VISIBLE_DEVICES），解析 RESULT_JSON"""
    env = dict(os.environ)
    if extra_env:
        env.update(extra_env)
    proc = subprocess.run(
        [sys.executable, "-c", script],
        env=env, capture_output=True, text=True,
    )
    if proc.stderr:
        for line in proc.stderr.splitlines():
            if any(k in line.lower() for k in ("error", "traceback", "oom", "exception")):
                print(f"  [sub stderr] {line}")
    result = None
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT_JSON:"):
            result = json.loads(line[len("RESULT_JSON:"):].strip())
        else:
            print(f"  [sub] {line}")
    if proc.returncode != 0 and result is None:
        print(proc.stdout[-2000:])
        print(proc.stderr[-2000:])
        raise RuntimeError(f"子进程失败 returncode={proc.returncode}")
    return result


def run_hf(args):
    print("\n" + "=" * 60)
    print(f"[HF] 加载 transformers (HIP_VISIBLE_DEVICES="
          f"{os.environ.get('HIP_VISIBLE_DEVICES','?')})...")
    print("=" * 60)
    script = _HF_SCRIPT % (_ROOT, args.model, args.prompt, args.max_tokens)
    return run_subprocess(script)


def run_vllm(args, tp):
    print("\n" + "=" * 60)
    tag = "VLLM_ENABLE_MOE_FUSED_GATE=0" if not args.no_fix_env else "no fix env"
    print(f"[vLLM] 加载 vLLM (tp={tp}, {tag})...")
    print("=" * 60)
    extra = {}
    if not args.no_fix_env:
        extra["VLLM_ENABLE_MOE_FUSED_GATE"] = "0"
    script = _VLLM_SCRIPT % (args.model, args.prompt, args.max_tokens, tp)
    return run_subprocess(script, extra_env=extra)


def compare(hf: dict, vllm: dict):
    print("\n" + "=" * 60)
    print("[compare] HF vs vLLM")
    print("=" * 60)
    hf_ids, vllm_ids = hf["token_ids"], vllm["token_ids"]
    has_null = 188 in vllm_ids
    n = min(len(hf_ids), len(vllm_ids))
    match = sum(1 for a, b in zip(hf_ids[:n], vllm_ids[:n]) if a == b)
    rate = match / n if n else 0
    print(f"[compare] token 前缀一致率: {rate*100:.1f}% ({match}/{n})")
    print(f"[compare] vLLM 含 NULL(188): {has_null}")
    print(f"[compare] HF text:    {hf['text']!r}")
    print(f"[compare] vLLM text:  {vllm['text']!r}")

    if has_null:
        print("\n[verdict] ❌ vLLM 输出全 NULL — 精度问题未修复")
        return False
    if rate >= 0.9:
        print("\n[verdict] ✅ vLLM 与 HF 基本一致 — 精度正常")
        return True
    print("\n[verdict] ⚠️ vLLM 与 HF 输出不一致 — 需进一步定位（用 op-locate skill）")
    return False


def main():
    args = parse_args()
    probe()

    ngpu = n_visible_gpus()
    if ngpu == 0:
        print("错误：请用 HIP_VISIBLE_DEVICES 前置指定卡，例如：\n"
              "  HIP_VISIBLE_DEVICES=0,1,6,7 python3 examples/quickstart_antangelmed.py",
              file=sys.stderr)
        sys.exit(2)

    tp = args.tp if args.tp is not None else ngpu
    if not args.skip_vllm and tp > ngpu:
        print(f"错误：--tp={tp} > 可见卡数({ngpu})", file=sys.stderr)
        sys.exit(2)

    hf_res = vllm_res = None
    if not args.skip_hf:
        hf_res = run_hf(args)
        print(f"[HF] token_ids: {hf_res['token_ids']}")

    if not args.skip_vllm:
        vllm_res = run_vllm(args, tp)
        print(f"[vLLM] token_ids: {vllm_res['token_ids']}")
        print(f"[vLLM] contains NULL(188)? {188 in vllm_res['token_ids']}")

    if hf_res and vllm_res:
        ok = compare(hf_res, vllm_res)
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
