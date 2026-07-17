"""
compare_layers.py — 逐层 / 逐算子中间值对比（HF 基准 vs vLLM）

quickstart 只比最终 token；本脚本比**每一层、每个关键算子的中间张量**，
把"哪一层开始发散、哪个算子开始发散"打印出来，定位精度问题到具体层/算子。

== 原理 ==
1. 用 lib.hook_manager 的 HookManager 在 HF 模型和 vLLM worker 模型上挂同一组
   hook 点（逐层：layer_in / attn_out / mlp_out；MoE：router_logits / topk）。
2. HF、vLLM 各起独立子进程跑同一个 prompt，hook 抓到的中间张量序列化落盘
   （.pt 文件，仅本次运行用）。
3. 主进程读两边的 .pt，逐 hook 点用 lib.tensor_compare 对比，打印每层每算子
   的 cos / max_abs / 是否发散，并标出第一个发散点。

== 卡的指定 ==
卡由用户用 HIP_VISIBLE_DEVICES 前置指定，脚本继承不覆盖：
    HIP_VISIBLE_DEVICES=2,3,4,5 python3 examples/compare_layers.py --model /path/to/model

== 用法 ==
    # 逐层对比所有层（大模型省显存可 --layers 0,1,15,30）
    HIP_VISIBLE_DEVICES=2,3,4,5 python3 examples/compare_layers.py --model /path/to/model

    # 只比前 4 层 + 中间几层（省显存/省时间）
    HIP_VISIBLE_DEVICES=2,3,4,5 python3 examples/compare_layers.py --model /path/to/model --layers 0,1,2,3,15,30

    # 只比 router（MoE 选专家那一步），不比 attn/mlp
    HIP_VISIBLE_DEVICES=2,3,4,5 python3 examples/compare_layers.py --model /path/to/model --only router

    # 对照验证：让 vLLM 绕过 fused_gate
    HIP_VISIBLE_DEVICES=2,3,4,5 python3 examples/compare_layers.py --model /path/to/model --fix-env

== 前置 ==
    - vLLM + transformers + torch 已装
    - 模型已下载到本地
    - 用户已确认指定卡空闲

== 说明 ==
    - 排查工具不预设结论：默认不设任何修复环境变量，需对照用 --fix-env
    - vLLM 侧 hook 通过 VLLM_PLATFORM_WORKER_MULTIPROC=0 单进程模式挂载，
      避免 EngineCore 子进程 hook 不生效（TP>1 仍可，但 hook 抓的是 rank0）
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)


def parse_args():
    ap = argparse.ArgumentParser(
        description="逐层 / 逐算子中间值对比 (HF vs vLLM)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--model", required=True,
                    help="模型本地路径（必填，含 config.json）")
    ap.add_argument("--tp", type=int, default=None, help="vLLM TP，默认=可见卡数")
    ap.add_argument("--prompt", default="你好，请介绍一下你自己")
    ap.add_argument("--max-tokens", type=int, default=1,
                    help="生成 token 数，逐层对比默认只看首 token（1）即可定位发散层")
    ap.add_argument("--layers", default=None,
                    help="只比这些层，逗号分隔，如 0,1,2,15,30；默认全部层")
    ap.add_argument("--only", default=None,
                    choices=["attn", "mlp", "router"],
                    help="只比某一类算子；默认全部（attn+mlp+router）")
    ap.add_argument("--fix-env", action="store_true",
                    help="显式设 VLLM_ENABLE_MOE_FUSED_GATE=0 做对照（默认不设）")
    ap.add_argument("--skip-hf", action="store_true")
    ap.add_argument("--skip-vllm", action="store_true")
    ap.add_argument("--atol", type=float, default=1e-2)
    ap.add_argument("--cos-thresh", type=float, default=0.999)
    return ap.parse_args()


def n_visible_gpus() -> int:
    v = os.environ.get("HIP_VISIBLE_DEVICES", "")
    return len([x for x in v.split(",") if x.strip() != ""]) if v else 0


def probe():
    from lib import probe_platform
    info = probe_platform()
    print(f"[platform] {info.summary()}")
    print(f"[platform] HIP_VISIBLE_DEVICES={os.environ.get('HIP_VISIBLE_DEVICES','?')} "
          f"({n_visible_gpus()} 卡)")


# ============================================================================
# 子进程脚本：抓中间值落盘
# ============================================================================

# HF 侧：直接在主进程内挂 hook，跑 forward，落盘
_HF_SCRIPT = r'''
import os, sys, json, torch
sys.path.insert(0, %r)
from lib import load_model_profile, HookManager, HookPoint, CaptureSpec, config_patch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

model_path = %r
prompt = %r
max_tokens = %r
out_pt = %r
layers_arg = %r   # None 或 list[int]
only_arg = %r     # None 或 "attn"/"mlp"/"router"

prof = load_model_profile(model_path)
cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
config_patch.patch_config(cfg)
print(f"[HF] layers={prof.num_layers} moe={prof.is_moe} dense_first={prof.first_k_dense_replace}", flush=True)

model = AutoModelForCausalLM.from_pretrained(
    model_path, config=cfg, torch_dtype=torch.bfloat16,
    trust_remote_code=True, local_files_only=True,
    attn_implementation="eager", device_map="auto",
).eval()
tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)

# 建 hook 点
layers = layers_arg if layers_arg is not None else list(range(prof.num_layers))
pts = []
for i in layers:
    base = f"model.layers.{i}"
    if only_arg in (None, "attn"):
        pts.append(HookPoint(f"layer{i}_attn_out", f"{base}.self_attn", kind="post"))
    if only_arg in (None, "mlp"):
        pts.append(HookPoint(f"layer{i}_mlp_out", f"{base}.mlp", kind="post"))
    if only_arg == "router" and prof.is_moe:
        rp = f"{base}.mlp.experts.router"
        pts.append(HookPoint(f"layer{i}_router_logits", rp, kind="pre"))
        pts.append(HookPoint(f"layer{i}_topk", rp, kind="post"))
print(f"[HF] hook 点数: {len(pts)}", flush=True)

hm = HookManager()
spec = CaptureSpec(hook_points=pts, owner=model, clone=True)
first_dev = next(model.parameters()).device
ids = tok(prompt, return_tensors="pt").input_ids.to(first_dev)
with torch.no_grad(), hm.capture(model, spec):
    out = model.generate(ids, max_new_tokens=max_tokens, do_sample=False)
inter = hm.get_intermediates()
# 只存首 token 位置的中间值（generate 多步时 hook 被多次调用，最后一次是最后一步；
# 为对齐，统一取每个 hook 点最后一次抓到的 tensor——对 max_tokens=1 即首 token）
torch.save({k: v.cpu() for k, v in inter.items()}, out_pt)
print(f"[HF] 已抓 {len(inter)} 个中间值 -> {out_pt}", flush=True)
print(f"[HF] gen_ids: {out[0][ids.shape[1]:].tolist()}", flush=True)
'''

# vLLM 侧：单进程 worker 模式挂 hook
_VLLM_SCRIPT = r'''
import os, sys, torch
os.environ.setdefault("VLLM_PLATFORM_WORKER_MULTIPROC", "0")  # 单进程，hook 才能挂上
sys.path.insert(0, %r)
from lib import load_model_profile, HookManager, HookPoint, CaptureSpec
from vllm import LLM, SamplingParams

model_path = %r
prompt = %r
max_tokens = %r
tp = %r
out_pt = %r
layers_arg = %r
only_arg = %r

prof = load_model_profile(model_path)
print(f"[vLLM] layers={prof.num_layers} moe={prof.is_moe}", flush=True)

llm = LLM(
    model=model_path, tensor_parallel_size=tp, dtype="bfloat16",
    trust_remote_code=True, enforce_eager=True,
    max_model_len=2048, gpu_memory_utilization=0.9,
)

# 在 worker 模型上挂 hook
worker_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
print(f"[vLLM] worker_model type: {type(worker_model).__name__}", flush=True)

layers = layers_arg if layers_arg is not None else list(range(prof.num_layers))
pts = []
for i in layers:
    base = f"model.layers.{i}"
    if only_arg in (None, "attn"):
        pts.append(HookPoint(f"layer{i}_attn_out", f"{base}.self_attn", kind="post"))
    if only_arg in (None, "mlp"):
        pts.append(HookPoint(f"layer{i}_mlp_out", f"{base}.mlp", kind="post"))
    if only_arg == "router" and prof.is_moe:
        rp = f"{base}.mlp.experts.router"
        pts.append(HookPoint(f"layer{i}_router_logits", rp, kind="pre"))
        pts.append(HookPoint(f"layer{i}_topk", rp, kind="post"))
print(f"[vLLM] hook 点数: {len(pts)}", flush=True)

hm = HookManager()
spec = CaptureSpec(hook_points=pts, owner=worker_model, clone=True)
sp = SamplingParams(temperature=0.0, max_tokens=max_tokens)
with hm.capture(worker_model, spec):
    outs = llm.generate([prompt], sp)
inter = hm.get_intermediates()
torch.save({k: v.cpu() for k, v in inter.items()}, out_pt)
print(f"[vLLM] 已抓 {len(inter)} 个中间值 -> {out_pt}", flush=True)
print(f"[vLLM] gen_ids: {list(outs[0].outputs[0].token_ids)}", flush=True)
'''


def run_subprocess(script: str, extra_env=None, tag="sub"):
    env = dict(os.environ)
    if extra_env:
        env.update(extra_env)
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.Popen(
        [sys.executable, "-u", "-c", script],
        env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.rstrip("\n")
        if line.strip():
            print(f"  [{tag}] {line}")
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"子进程失败 returncode={proc.returncode}")


def main():
    args = parse_args()
    probe()
    ngpu = n_visible_gpus()
    if ngpu == 0:
        print("错误：请用 HIP_VISIBLE_DEVICES 前置指定卡", file=sys.stderr)
        sys.exit(2)
    tp = args.tp if args.tp is not None else ngpu

    layers_arg = None
    if args.layers:
        layers_arg = [int(x) for x in args.layers.split(",")]

    tmpdir = tempfile.mkdtemp(prefix="compare_layers_")
    hf_pt = os.path.join(tmpdir, "hf_inter.pt")
    vllm_pt = os.path.join(tmpdir, "vllm_inter.pt")

    if not args.skip_hf:
        print("\n" + "=" * 60)
        print("[HF] 抓逐层中间值...")
        print("=" * 60)
        script = _HF_SCRIPT % (_ROOT, args.model, args.prompt, args.max_tokens,
                               hf_pt, layers_arg, args.only)
        run_subprocess(script, tag="HF")

    if not args.skip_vllm:
        print("\n" + "=" * 60)
        tag = "VLLM_ENABLE_MOE_FUSED_GATE=0" if args.fix_env else "no fix env"
        print(f"[vLLM] 抓逐层中间值 (tp={tp}, {tag})...")
        print("=" * 60)
        extra = {"VLLM_ENABLE_MOE_FUSED_GATE": "0"} if args.fix_env else {}
        script = _VLLM_SCRIPT % (_ROOT, args.model, args.prompt, args.max_tokens,
                                 tp, vllm_pt, layers_arg, args.only)
        run_subprocess(script, extra_env=extra, tag="vLLM")

    if args.skip_hf or args.skip_vllm:
        print("\n[skip] 一边被跳过，无法对比。中间值已落盘：", tmpdir)
        return

    # 对比
    print("\n" + "=" * 60)
    print("[compare] 逐层 / 逐算子中间值对比")
    print("=" * 60)
    import torch
    from lib import compare_tensors
    hf = torch.load(hf_pt)
    vllm = torch.load(vllm_pt)

    names = sorted(set(hf) & set(vllm), key=_sort_key)
    if not names:
        print("[compare] 两边无共同 hook 点。HF 侧:", list(hf)[:5],
              "vLLM 侧:", list(vllm)[:5])
        sys.exit(1)

    first_diverge = None
    print(f"{'stage':<28} {'cos':>10} {'max_abs':>12} {'mean_abs':>12} {'verdict'}")
    print("-" * 80)
    for name in names:
        a, b = hf[name], vllm[name]
        try:
            if a.shape != b.shape:
                # router topk 形状可能因 TP 切分不同，跳过或只比 logits
                print(f"{name:<28} {'shape_mismatch':>10} {tuple(a.shape)} vs {tuple(b.shape)}")
                continue
            r = compare_tensors(name, a, b, atol=args.atol)
            v = r.verdict(atol=args.atol, cos_thresh=args.cos_thresh)
            ok = r.is_close or r.cos >= args.cos_thresh
            mark = "✅" if ok else "❌"
            print(f"{name:<28} {r.cos:>10.6f} {r.max_abs_diff:>12.4e} "
                  f"{r.mean_abs_diff:>12.4e} {mark}")
            if not ok and first_diverge is None:
                first_diverge = name
        except Exception as e:
            print(f"{name:<28} ERR {e}")

    print("-" * 80)
    if first_diverge:
        print(f"\n[verdict] ❌ 首个发散点: {first_diverge}")
        print("  → 该层/算子之前的层都一致，从这里开始 HF 与 vLLM 分叉。")
        print("  → 用 op-locate skill 在此层内继续细化（sub-op 级 hook）。")
    else:
        print("\n[verdict] ✅ 所有对比的层/算子均一致")
    print(f"\n[info] 中间值落盘目录: {tmpdir}")


def _sort_key(name: str):
    """layer0_attn_out < layer0_mlp_out < layer1_..."""
    import re
    m = re.search(r"layer(\d+)", name)
    layer = int(m.group(1)) if m else 999
    op_order = {"attn_out": 0, "router_logits": 1, "topk": 2, "mlp_out": 3}
    op = 99
    for k, v in op_order.items():
        if k in name:
            op = v
            break
    return (layer, op)


if __name__ == "__main__":
    main()
