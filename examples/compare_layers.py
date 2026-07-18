"""
compare_layers.py — 逐层 / 逐算子中间值对比（HF 基准 vs vLLM）

quickstart 只比最终 token；本脚本比**每一层、每个关键算子的中间张量**，
把"哪一层开始发散、哪个算子开始发散"打印出来，定位精度问题到具体层/算子。

== 原理 ==
1. 用 lib.hook_manager 的 HookManager 在 HF 模型和 vLLM worker 模型上挂同一组
   hook 点（逐层：attn_out / mlp_out；MoE：router_logits）。
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
    - vLLM 侧 TP>1 时 worker 在 spawn 子进程，主进程拿不到 model 对象，
      故用 collective_rpc 把 hook 函数 RPC 进 worker 子进程注册（需
      VLLM_ALLOW_INSECURE_SERIALIZATION=1）。捕获结果存 worker 内 model._cap
      （持久对象），不能存模块级 dict——collective_rpc 每次序列化新副本会读不到。
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
    ap.add_argument("--layer-prefix", default="model.layers",
                    help="decoder 层在模型里的点路径，默认 model.layers；"
                         "GPT 系可能是 transformer.h，falcon 可能是 transformer.h")
    ap.add_argument("--probe-dir", default=None,
                    help="读取 probe_model.py 落盘的探测结果（model_probe.json），"
                         "用其中的 layer_prefix/attn/mlp/router，跳过运行时探测。"
                         "推荐：先跑 probe_model.py 再用此参数")
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

# 探测层属性名的 helper，注入到两个子进程脚本。直接复用 lib.model_probe 的别名表
# 与探测函数，避免重复定义导致两处发散（不同架构 attn/mlp/router 命名不同）。
_DETECT_HELPER = r'''
from lib.model_probe import (
    _ATTN_ALIASES, _MLP_ALIASES, _ROUTER_ALIASES, _detect_attr, _detect_router,
)

def _detect_layer_attrs(model, layer_prefix="model.layers"):
    """探测真实层的 attn/mlp/router 属性名。返回 (attn_attr, mlp_attr, router_relpath)。
    router_relpath 相对 layer，如 'mlp.gate'；找不到返回 None。
    attn/mlp 用第一层即可；router 要找第一个含 MoE 结构的层（前若干层可能是 dense，
    没有 gate/experts）。"""
    parts = layer_prefix.split(".")
    obj = model
    for p in parts:
        obj = getattr(obj, p) if not p.isdigit() else obj[int(p)]
    layers = obj
    layer0 = layers[0]
    attn = _detect_attr(layer0, _ATTN_ALIASES)
    mlp = _detect_attr(layer0, _MLP_ALIASES)
    router = None
    if mlp:
        for layer in layers:
            router = _detect_router(layer, mlp)
            if router:
                break
    return attn, mlp, router
'''

# HF 侧：直接在主进程内挂 hook，跑 forward，落盘
_HF_SCRIPT = _DETECT_HELPER + r'''
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
layer_prefix = %r
probe_attrs = %r  # None 或 dict(layer_prefix/attn/mlp/router)，来自 probe_model.py

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

# 属性名：优先用 probe_attrs（probe_model.py 落盘的探测结果），否则运行时反射探测
if probe_attrs and probe_attrs.get("attn"):
    attn_attr = probe_attrs["attn"]; mlp_attr = probe_attrs["mlp"]; router_rel = probe_attrs["router"]
    print(f"[HF] 用 probe 结果: attn={attn_attr} mlp={mlp_attr} router={router_rel}", flush=True)
else:
    attn_attr, mlp_attr, router_rel = _detect_layer_attrs(model, layer_prefix)
    print(f"[HF] 运行时探测: attn={attn_attr} mlp={mlp_attr} router={router_rel}", flush=True)
if only_arg == "attn" and not attn_attr:
    print("[HF] 警告：未探测到 attn 属性，无法挂 attn hook", flush=True)
if only_arg == "mlp" and not mlp_attr:
    print("[HF] 警告：未探测到 mlp 属性，无法挂 mlp hook", flush=True)

# 建 hook 点
layers = layers_arg if layers_arg is not None else list(range(prof.num_layers))
pts = []
for i in layers:
    base = f"{layer_prefix}.{i}"
    if only_arg in (None, "attn") and attn_attr:
        pts.append(HookPoint(f"layer{i}_attn_out", f"{base}.{attn_attr}", kind="post"))
    if only_arg in (None, "mlp") and mlp_attr:
        pts.append(HookPoint(f"layer{i}_mlp_out", f"{base}.{mlp_attr}", kind="post"))
    if only_arg == "router" and prof.is_moe and router_rel:
        rp = f"{base}.{router_rel}"
        # gate.forward 返回 tuple，如 BailingMoeV2 (topk_idx, topk_weight, logits)。
        # pre-hook 抓的是 gate 输入(hidden_states)不是 logits，post-hook 抓的是
        # 整个 tuple——都不能直接拿 logits。改用 patch + output_index 取 tuple 里
        # 的 logits（通常在最后，index=-1）。topk_idx/weight 的 index 因模型而异，
        # 这里只可靠对比 router_logits；topk 对比需按模型适配 output_index。
        pts.append(HookPoint(f"layer{i}_router_logits", rp, kind="patch", output_index=-1))
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

# vLLM 侧：TP>1 时 worker 在 spawn 子进程，主进程拿不到 model 对象，
# 用 collective_rpc 把 hook 函数 RPC 进 worker 子进程注册。捕获结果存
# model._cap（model 在子进程单一持久；不能存模块级 dict——collective_rpc
# 每次 cloudpickle 序列化新副本，会读不到）。详见 docs/ 里 collective_rpc 陷阱。
_VLLM_SCRIPT = _DETECT_HELPER + r'''
import os, sys, torch
# collective_rpc 用 cloudpickle 传函数进 worker 子进程，必须开此项
os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
sys.path.insert(0, %r)
from lib import load_model_profile
from vllm import LLM, SamplingParams

model_path = %r
prompt = %r
max_tokens = %r
tp = %r
out_pt = %r
layers_arg = %r
only_arg = %r
layer_prefix = %r
probe_attrs = %r  # None 或 dict，来自 probe_model.py

prof = load_model_profile(model_path)
print(f"[vLLM] layers={prof.num_layers} moe={prof.is_moe}", flush=True)

# ============================================================================
# 模块级 RPC 函数（collective_rpc 要求 top-level，pickle 序列化）
# hook 描述 hook_specs = list[(name, module_path, kind, output_index)]
# kind: "post"=register_forward_hook 抓输出；"patch"=monkey-patch forward
#       抓返回 tuple 的某一项（router gate 返回 (topk_idx,topk_weight,logits)）
# 捕获统一存 worker.model_runner.model._cap（持久对象，规避 cloudpickle 副本陷阱）
# ============================================================================

def _resolve(model, path):
    """点路径取子模块，支持数字索引（layers.1）"""
    obj = model
    for part in path.split("."):
        obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
    return obj

def _attach_hooks(worker, hook_specs):
    """在 worker 子进程挂 hook。hook_specs: list[(name, path, kind, output_index)]"""
    import torch
    model = worker.model_runner.model
    # 先摘旧 hook（幂等）
    for h in getattr(model, "_cmp_hooks", []) or []:
        try: h.remove()
        except Exception: pass
    for mod, orig in getattr(model, "_cmp_patches", []) or []:
        try: mod.forward = orig
        except Exception: pass
    model._cmp_hooks = []
    model._cmp_patches = []
    model._cap = {}
    fired = {"n": 0}

    def make_post(name, output_index):
        def hook(_m, _a, output):
            # 只留第一次触发（prefill）；后续 decode 步是单 token，不覆盖
            if name in model._cap:
                return
            try:
                t = output[output_index] if isinstance(output, tuple) else output
                if torch.is_tensor(t):
                    model._cap[name] = t.detach().to(torch.float32).cpu()
                    fired["n"] += 1
            except Exception as e:
                model._cap.setdefault("_err_" + name, repr(e))
        return hook

    def make_patch(name, module, output_index):
        orig = module.forward
        def patched(*a, **kw):
            out = orig(*a, **kw)
            if name not in model._cap:
                try:
                    t = out[output_index] if isinstance(out, tuple) else out
                    if torch.is_tensor(t):
                        model._cap[name] = t.detach().to(torch.float32).cpu()
                        fired["n"] += 1
                except Exception as e:
                    model._cap.setdefault("_err_" + name, repr(e))
            return out
        module.forward = patched
        model._cmp_patches.append((module, orig))

    attached = 0
    for name, path, kind, output_index in hook_specs:
        try:
            mod = _resolve(model, path)
            if kind == "patch":
                make_patch(name, mod, output_index)
            else:
                h = mod.register_forward_hook(make_post(name, output_index))
                model._cmp_hooks.append(h)
            attached += 1
        except Exception as e:
            model._cap.setdefault("_err_" + name, f"attach: {e!r}")
    model._cap["_fired"] = 0  # 占位，fetched 时读 fired["n"]
    model._fired_ref = fired
    return {"attached": attached, "total": len(hook_specs)}

def _fetch_capture(worker):
    model = worker.model_runner.model
    cap = getattr(model, "_cap", None) or {}
    fired = getattr(model, "_fired_ref", None)
    out = {k: v for k, v in cap.items() if not k.startswith("_")}
    out["_fired"] = fired["n"] if fired else cap.get("_fired", 0)
    out["_errors"] = {k: v for k, v in cap.items() if k.startswith("_err_")}
    return out

def _detach_hooks(worker):
    model = worker.model_runner.model
    for h in getattr(model, "_cmp_hooks", []) or []:
        try: h.remove()
        except Exception: pass
    for mod, orig in getattr(model, "_cmp_patches", []) or []:
        try: mod.forward = orig
        except Exception: pass
    model._cmp_hooks = []
    model._cmp_patches = []
    return {"detached": True}

# ============================================================================
# 先用一个轻量 RPC 探测 worker 内模型的真实属性名（attn/mlp/router）
# probe_attrs 若已给则跳过
# ============================================================================
def _probe_layer_attrs(worker, layer_prefix):
    model = worker.model_runner.model
    return _detect_layer_attrs(model, layer_prefix)

llm = LLM(
    model=model_path, tensor_parallel_size=tp, dtype="bfloat16",
    trust_remote_code=True, enforce_eager=True,   # 禁 compile，hook 才挂得上
    max_model_len=2048, gpu_memory_utilization=0.9,
)
print(f"[vLLM] engine started (tp={tp})", flush=True)

# 属性名：优先 probe_attrs，否则 RPC 进 worker 反射探测
if probe_attrs and probe_attrs.get("attn"):
    attn_attr = probe_attrs["attn"]; mlp_attr = probe_attrs["mlp"]; router_rel = probe_attrs["router"]
    print(f"[vLLM] 用 probe 结果: attn={attn_attr} mlp={mlp_attr} router={router_rel}", flush=True)
else:
    attn_attr, mlp_attr, router_rel = llm.collective_rpc(_probe_layer_attrs, args=(layer_prefix,))[0]
    print(f"[vLLM] worker 内探测: attn={attn_attr} mlp={mlp_attr} router={router_rel}", flush=True)

# 生成 hook 描述（与 HF 侧点路径一致）
layers = layers_arg if layers_arg is not None else list(range(prof.num_layers))
specs = []  # (name, path, kind, output_index)
for i in layers:
    base = f"{layer_prefix}.{i}"
    if only_arg in (None, "attn") and attn_attr:
        specs.append((f"layer{i}_attn_out", f"{base}.{attn_attr}", "post", 0))
    if only_arg in (None, "mlp") and mlp_attr:
        specs.append((f"layer{i}_mlp_out", f"{base}.{mlp_attr}", "post", 0))
    if only_arg == "router" and prof.is_moe and router_rel:
        # gate.forward 返回 tuple (topk_idx, topk_weight, logits)；post-hook 抓
        # 整个 tuple 错、pre-hook 抓输入错。用 patch 取 output_index=-1 即 logits。
        specs.append((f"layer{i}_router_logits", f"{base}.{router_rel}", "patch", -1))
print(f"[vLLM] hook 点数: {len(specs)}", flush=True)

# 注入 hook → generate 触发 → 取回 → 摘 hook
res = llm.collective_rpc(_attach_hooks, args=(specs,))
print(f"[vLLM] attach: {res[0]}", flush=True)

sp = SamplingParams(temperature=0.0, max_tokens=max_tokens)
outs = llm.generate([prompt], sp)
gen_ids = list(outs[0].outputs[0].token_ids)
print(f"[vLLM] gen_ids: {gen_ids}", flush=True)

cap = llm.collective_rpc(_fetch_capture)[0]
print(f"[vLLM] hook fired: {cap.get('_fired')}", flush=True)
if cap.get("_errors"):
    print(f"[vLLM] hook errors: {cap['_errors']}", flush=True)
llm.collective_rpc(_detach_hooks)

# 取回捕获（已是 CPU fp32），剔除元数据键
inter = {k: v for k, v in cap.items() if not k.startswith("_") and torch.is_tensor(v)}
print(f"[vLLM] 已抓 {len(inter)} 个中间值", flush=True)
for k, v in sorted(inter.items()):
    print(f"  {k}: {tuple(v.shape)}", flush=True)
torch.save(inter, out_pt)
print(f"[vLLM] -> {out_pt}", flush=True)
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

    # 读取 probe_model.py 落盘的探测结果（若提供）
    probe_attrs = None
    if args.probe_dir:
        import json
        pj = os.path.join(args.probe_dir, "model_probe.json")
        if not os.path.isfile(pj):
            print(f"错误：--probe-dir 下无 model_probe.json: {pj}", file=sys.stderr)
            sys.exit(2)
        with open(pj) as f:
            pd = json.load(f)
        probe_attrs = {
            "layer_prefix": pd["layer_prefix"]["value"],
            "attn": pd["attn_attr"]["value"],
            "mlp": pd["mlp_attr"]["value"],
            "router": pd["router_attr"]["value"],
        }
        # probe 的 layer_prefix 优先于 --layer-prefix
        if probe_attrs["layer_prefix"]:
            args.layer_prefix = probe_attrs["layer_prefix"]
        print(f"[probe] 读取探测结果 {pj}")
        print(f"[probe] layer_prefix={probe_attrs['layer_prefix']} "
              f"attn={probe_attrs['attn']} mlp={probe_attrs['mlp']} "
              f"router={probe_attrs['router']}")

    tmpdir = tempfile.mkdtemp(prefix="compare_layers_")
    hf_pt = os.path.join(tmpdir, "hf_inter.pt")
    vllm_pt = os.path.join(tmpdir, "vllm_inter.pt")

    if not args.skip_hf:
        print("\n" + "=" * 60)
        print("[HF] 抓逐层中间值...")
        print("=" * 60)
        script = _HF_SCRIPT % (_ROOT, args.model, args.prompt, args.max_tokens,
                               hf_pt, layers_arg, args.only, args.layer_prefix, probe_attrs)
        run_subprocess(script, tag="HF")

    if not args.skip_vllm:
        print("\n" + "=" * 60)
        tag = "VLLM_ENABLE_MOE_FUSED_GATE=0" if args.fix_env else "no fix env"
        print(f"[vLLM] 抓逐层中间值 (tp={tp}, {tag})...")
        print("=" * 60)
        extra = {"VLLM_ENABLE_MOE_FUSED_GATE": "0"} if args.fix_env else {}
        script = _VLLM_SCRIPT % (_ROOT, args.model, args.prompt, args.max_tokens,
                                 tp, vllm_pt, layers_arg, args.only, args.layer_prefix, probe_attrs)
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
    skipped = []
    print(f"{'stage':<28} {'cos':>10} {'max_abs':>12} {'mean_abs':>12} {'verdict'}")
    print("-" * 80)
    for name in names:
        a, b = hf[name], vllm[name]
        try:
            a, b = _align(a, b)
            if a.shape != b.shape:
                # 形状对不齐（如 router topk 因 TP 切分不同），跳过
                print(f"{name:<28} {'shape_skip':>10} {tuple(a.shape)} vs {tuple(b.shape)}")
                skipped.append(name)
                continue
            r = compare_tensors(name, a, b, atol=args.atol)
            ok = r.is_close or r.cos >= args.cos_thresh
            mark = "✅" if ok else "❌"
            print(f"{name:<28} {r.cos:>10.6f} {r.max_abs_diff:>12.4e} "
                  f"{r.mean_abs_diff:>12.4e} {mark}")
            if not ok and first_diverge is None:
                first_diverge = name
        except Exception as e:
            print(f"{name:<28} ERR {e}")
            skipped.append(name)

    print("-" * 80)
    if first_diverge:
        print(f"\n[verdict] ❌ 首个发散点: {first_diverge}")
        print("  → 该层/算子之前的层都一致，从这里开始 HF 与 vLLM 分叉。")
        print("  → 用 op-locate skill 在此层内继续细化（sub-op 级 hook）。")
    elif skipped:
        print(f"\n[verdict] ⚠️  已对比的层/算子均一致，但 {len(skipped)} 个因形状未对齐被跳过：{skipped}")
    else:
        print("\n[verdict] ✅ 所有对比的层/算子均一致")
    print(f"\n[info] 中间值落盘目录: {tmpdir}")


def _align(a, b):
    """对齐 HF 与 vLLM 中间张量的 batch 维差异。
    HF 输出常为 [1, seq, H]（带 batch 维），vLLM 为 [seq, H]（无 batch 维，PLAN
    layer_compare §3 提到）。squeeze 掉大小为 1 的前导维使两者可比。"""
    import torch
    while a.dim() > b.dim() and a.shape[0] == 1:
        a = a.squeeze(0)
    while b.dim() > a.dim() and b.shape[0] == 1:
        b = b.squeeze(0)
    return a, b


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
