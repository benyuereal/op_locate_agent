"""
compare_layers.py — 逐层中间值对比（HF 基准 vs vLLM），定位精度发散点

比 quickstart（只比最终 token）更进一步：hook 每层入口的 hidden_states（残差，
层间流动的真值），逐层对比 cos/max_diff，看误差从哪层开始、如何累积放大。

== 口径（关键，源自历史验证过的 test_stage_compare.py）==
1. **比 layer INPUT hidden_states**，不是 attn_out/mlp_out。原因：vLLM
   BailingMoeBlock.forward 返回 (hidden_states, residual) 分开，"层输出"对下一层
   就是它的 input；hook 每层 input 给出一致跨层口径，且就是层间真实流动值。
2. **用 register_forward_PRE_hook + 立即 clone**，不用 post-hook。原因：layer
   body 做 `residual = hidden_states` 后有 in-place 操作，会改 args[0]，post-hook
   抓到的是被污染的脏值（历史曾因此误判 layer0 发散，实际是 hook 口径 bug）。
3. **embedding 用 monkey-patch embed_input_ids**，不用 register_forward_hook。
   原因：VocabParallelEmbedding 是 CustomOp，register_forward_hook 对 CustomOp
   不可靠（抓的不是真正用于 forward 的输出）。
4. **采样层覆盖全模型** [0,1,2,8,16,24,31]（浅+中+深），不只前几层——误差常在
   深层累积放大。`--layers` 可覆盖。
5. **prefill + decode1 两阶段**，用 call counter 切换——发散可能在 decode 才暴露。

== 子算子细化（可选）==
默认比 layer input（残差）。定位到发散层后，用 `--only attn/mlp/router` 在该层
内做算子级对比（attn_out/mlp_out/router_logits）。注意子模块输出用 post-hook，
口径不如 layer input 可靠，仅作 drill-down 参考。

== 卡的指定 ==
卡由用户用 HIP_VISIBLE_DEVICES 前置指定，脚本继承不覆盖：
    HIP_VISIBLE_DEVICES=2,3,4,5 python3 examples/compare_layers.py --model /path/to/model

== 用法 ==
    # 默认：全模型采样层 [0,1,2,8,16,24,31] 的 layer input 对比
    HIP_VISIBLE_DEVICES=2,3,4,5 python3 examples/compare_layers.py --model /path/to/model

    # 指定层（省时间/聚焦）
    HIP_VISIBLE_DEVICES=2,3,4,5 python3 examples/compare_layers.py --model /path/to/model --layers 0,1,2,27,30,31

    # 发散层内算子级细化（定位是 attn 还是 mlp 发散）
    HIP_VISIBLE_DEVICES=2,3,4,5 python3 examples/compare_layers.py --model /path/to/model --layers 27 --only mlp

    # 对照验证：让 vLLM 绕过 fused_gate（历史已排除 fused_gate，仅备对照）
    HIP_VISIBLE_DEVICES=2,3,4,5 python3 examples/compare_layers.py --model /path/to/model --env

== 前置 ==
    - 先跑 probe_model.py 探测架构（attn/mlp/router 属性名），用 --probe-dir 传入
    - vLLM + transformers + torch 已装，模型已下载，指定卡空闲

== 说明 ==
    - 排查工具不预设结论：默认不设任何修复环境变量，需对照用 --env
    - vLLM 侧 TP>1 时 worker 在 spawn 子进程，主进程拿不到 model 对象，用
      collective_rpc 把 hook 函数 RPC 进 worker 子进程注册（需
      VLLM_ALLOW_INSECURE_SERIALIZATION=1）。捕获存 worker 内 model._cap（持久
      对象），不能存模块级 dict——collective_rpc 每次序列化新副本会读不到。
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
        description="逐层中间值对比 (HF vs vLLM)，定位精度发散点",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--model", required=True,
                    help="模型本地路径（必填，含 config.json）")
    ap.add_argument("--tp", type=int, default=None, help="vLLM TP，默认=可见卡数")
    ap.add_argument("--prompt", default="你好，我是一名医生")
    ap.add_argument("--max-tokens", type=int, default=1,
                    help="vLLM 生成 token 数；默认 1（prefill+1 decode）。"
                         "逐层定位看 prefill 即可，decode1 由阶段切换自动捕获")
    ap.add_argument("--layers", default=None,
                    help="采样层，逗号分隔；默认自动选 [0,1,2,8,16,24,31] 按总层数缩放")
    ap.add_argument("--only", default=None,
                    choices=["attn", "mlp", "router"],
                    help="只比某一类子算子（attn_out/mlp_out/router_logits），"
                         "用于发散层内 drill-down；默认比 layer input 残差")
    ap.add_argument("--env", action="store_true",
                    help="显式设 VLLM_ENABLE_MOE_FUSED_GATE=0 做对照（绕过 fused_gate，"
                         "默认不设——排查工具不预设结论）")
    ap.add_argument("--skip-hf", action="store_true")
    ap.add_argument("--skip-vllm", action="store_true")
    ap.add_argument("--layer-prefix", default="model.layers",
                    help="decoder 层在模型里的点路径，默认 model.layers")
    ap.add_argument("--probe-dir", default=None,
                    help="读取 probe_model.py 落盘的 model_probe.json，用其中的 "
                         "layer_prefix/attn/mlp/router。推荐先跑 probe_model.py")
    ap.add_argument("--atol", type=float, default=1e-2)
    ap.add_argument("--cos-thresh", type=float, default=0.999)
    return ap.parse_args()


def n_visible_gpus() -> int:
    v = os.environ.get("HIP_VISIBLE_DEVICES", "")
    return len([x for x in v.split(",") if x.strip() != ""]) if v else 0


def sample_layers(num_layers: int) -> list[int]:
    """全模型采样层：浅+中+深，按总层数缩放。32 层 → [0,1,2,8,16,24,31]。"""
    if num_layers <= 8:
        return list(range(num_layers))
    n = num_layers
    return sorted(set([
        0, 1, 2,
        n // 4, n // 2, (3 * n) // 4,
        n - 4, n - 2, n - 1,
    ]))


def probe():
    from lib import probe_platform
    info = probe_platform()
    print(f"[platform] {info.summary()}")
    print(f"[platform] HIP_VISIBLE_DEVICES={os.environ.get('HIP_VISIBLE_DEVICES','?')} "
          f"({n_visible_gpus()} 卡)")


# ============================================================================
# 子进程脚本：抓中间值落盘
# ============================================================================

# 探测层属性名的 helper，注入到两个子进程脚本。复用 lib.model_probe 的别名表。
_DETECT_HELPER = r'''
from lib.model_probe import (
    _ATTN_ALIASES, _MLP_ALIASES, _ROUTER_ALIASES, _detect_attr, _detect_router,
)

def _detect_layer_attrs(model, layer_prefix="model.layers"):
    """探测真实层的 attn/mlp/router 属性名。返回 (attn_attr, mlp_attr, router_relpath)。"""
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

# HF 侧：主进程内挂 hook，跑 forward（prefill + decode1），落盘
_HF_SCRIPT = _DETECT_HELPER + r'''
import os, sys, torch
sys.path.insert(0, %r)
from lib import load_model_profile, config_patch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

model_path = %r
prompt = %r
out_pt = %r
layers_arg = %r   # None 或 list[int]
only_arg = %r     # None 或 "attn"/"mlp"/"router"
layer_prefix = %r
probe_attrs = %r  # None 或 dict，来自 probe_model.py

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
hf = model.model  # BailingMoeModel 等骨干

# 属性名：优先 probe_attrs，否则反射探测
if probe_attrs and probe_attrs.get("attn"):
    attn_attr = probe_attrs["attn"]; mlp_attr = probe_attrs["mlp"]; router_rel = probe_attrs["router"]
    print(f"[HF] 用 probe 结果: attn={attn_attr} mlp={mlp_attr} router={router_rel}", flush=True)
else:
    attn_attr, mlp_attr, router_rel = _detect_layer_attrs(model, layer_prefix)
    print(f"[HF] 运行时探测: attn={attn_attr} mlp={mlp_attr} router={router_rel}", flush=True)

layers = layers_arg if layers_arg is not None else list(range(prof.num_layers))
cap = {}
phase = {"v": "prefill"}

# ---- 默认口径：layer INPUT hidden_states（pre-hook + 立即 clone）----
# 用 pre-hook 是因为 layer body 做 residual=hidden_states 后有 in-place，post-hook
# 抓到的 args[0] 已被污染。pre-hook 在 body 执行前抓，立即 clone 锁定原始值。
hooks = []
if only_arg is None:
    for li in layers:
        layer = hf.layers[li]
        def make_pre(idx):
            def ph(module, args, kwargs):
                key = f"{phase['v']}:layer{idx}_in"
                if key not in cap:
                    hs = args[0] if len(args) > 0 else kwargs.get("hidden_states")
                    cap[key] = hs.detach().to(torch.float32).cpu().clone()
            return ph
        hooks.append(layer.register_forward_pre_hook(make_pre(li), with_kwargs=True))

    # embedding 输出（HF word_embeddings 是普通 nn.Embedding，post-hook 可靠）
    def emb_hook(module, args, kwargs, output):
        key = f"{phase['v']}:embedding"
        if key not in cap:
            cap[key] = output.detach().to(torch.float32).cpu()
    hooks.append(hf.word_embeddings.register_forward_hook(emb_hook, with_kwargs=True))

    # final norm 输出 + logits
    def norm_hook(module, args, kwargs, output):
        key = f"{phase['v']}:final_norm"
        if key not in cap:
            out = output[0] if isinstance(output, tuple) else output
            cap[key] = out.detach().to(torch.float32).cpu()
    hooks.append(hf.norm.register_forward_hook(norm_hook, with_kwargs=True))
    def logits_hook(module, args, kwargs, output):
        key = f"{phase['v']}:logits"
        if key not in cap:
            cap[key] = output.detach().to(torch.float32).cpu()
    hooks.append(model.lm_head.register_forward_hook(logits_hook, with_kwargs=True))
else:
    # ---- 子算子细化（drill-down）：只抓 prefill，key 不带 phase ----
    router_patches = []  # (mod, orig) 还原用
    for li in layers:
        base = hf.layers[li]
        if only_arg == "attn" and attn_attr:
            mod = getattr(base, attn_attr)
            def make_post(idx, name):
                def ph(module, args, kwargs, output):
                    key = f"layer{idx}_{name}"
                    if key not in cap:
                        t = output[0] if isinstance(output, tuple) else output
                        cap[key] = t.detach().to(torch.float32).cpu()
                return ph
            hooks.append(mod.register_forward_hook(make_post(li, "attn_out"), with_kwargs=True))
        if only_arg == "mlp" and mlp_attr:
            mod = getattr(base, mlp_attr)
            def make_post2(idx, name):
                def ph(module, args, kwargs, output):
                    key = f"layer{idx}_{name}"
                    if key not in cap:
                        t = output[0] if isinstance(output, tuple) else output
                        cap[key] = t.detach().to(torch.float32).cpu()
                return ph
            hooks.append(mod.register_forward_hook(make_post2(li, "mlp_out"), with_kwargs=True))
        if only_arg == "router" and prof.is_moe and router_rel:
            # gate.forward 返回 tuple (topk_idx, topk_weight, logits)；monkey-patch
            # forward 取 output[-1] 即 logits（post-hook 抓 tuple 不便取项）
            mod = base
            for p in router_rel.split("."):
                mod = getattr(mod, p)
            orig = mod.forward
            def make_router(idx, orig_fn):
                def patched(*a, **kw):
                    out = orig_fn(*a, **kw)
                    key = f"layer{idx}_router_logits"
                    if key not in cap:
                        t = out[-1] if isinstance(out, tuple) else out
                        cap[key] = t.detach().to(torch.float32).cpu()
                    return out
                return patched
            mod.forward = make_router(li, orig)
            router_patches.append((mod, orig))
            hooks.append("router_patch")  # 计数占位
print(f"[HF] hook 点数: {len(hooks)}", flush=True)

# ---- prefill forward ----
first_dev = next(model.parameters()).device
ids = tok(prompt, return_tensors="pt").input_ids.to(first_dev)
seq_len = ids.shape[1]
pos_ids = torch.arange(0, seq_len, dtype=torch.long).unsqueeze(0).to(first_dev)
print(f"[HF] prefill ({seq_len} tokens)...", flush=True)
with torch.no_grad():
    out_pre = model(input_ids=ids, position_ids=pos_ids, use_cache=True)
print(f"[HF] prefill 抓到 {len(cap)} 个", flush=True)

# ---- decode step 1（仅默认口径需要；--only 子算子模式只看 prefill）----
if only_arg is None:
    phase["v"] = "decode1"
    next_tok = out_pre.logits[0, -1, :].argmax(dim=-1, keepdim=True).unsqueeze(0)
    next_pos = torch.tensor([[seq_len]], dtype=torch.long, device=first_dev)
    print(f"[HF] decode1 (next_tok={next_tok.item()})...", flush=True)
    with torch.no_grad():
        _ = model(input_ids=next_tok, position_ids=next_pos,
                  use_cache=True, past_key_values=out_pre.past_key_values)
    print(f"[HF] decode1 抓到 {sum(1 for k in cap if k.startswith('decode1'))} 个", flush=True)

for h in hooks:
    if isinstance(h, str): continue
    try: h.remove()
    except Exception: pass
for mod, orig in router_patches:
    try: mod.forward = orig
    except Exception: pass

torch.save(cap, out_pt)
print(f"[HF] -> {out_pt}", flush=True)
'''

# vLLM 侧：collective_rpc 注入 hook，generate 触发，取回落盘
_VLLM_SCRIPT = _DETECT_HELPER + r'''
import os, sys, torch
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
# 捕获统一存 worker.model_runner.model._cap（持久对象，规避 cloudpickle 副本陷阱）
# 阶段切换：用 call counter，prefill 采样层都触发过一次后翻到 decode1
# ============================================================================

def _attach_stage_hooks(worker, layer_indices, only_arg, attn_attr, mlp_attr, router_rel, is_moe):
    """挂 stage hook。only_arg=None 比 layer input 残差 + emb + final_norm + logits；
    否则比指定子算子（attn_out/mlp_out/router_logits）。"""
    import torch
    model = worker.model_runner.model
    for h in getattr(model, "_st_hooks", []) or []:
        try: h.remove()
        except Exception: pass
    for mod, orig in getattr(model, "_st_patches", []) or []:
        try: mod.forward = orig
        except Exception: pass
    model._st_hooks = []
    model._st_patches = []
    model._cap = {}
    model._st_phase = "prefill"
    model._st_prefill_fired = 0
    inner = model.model

    if only_arg is None:
        # ---- embedding：monkey-patch embed_input_ids（CustomOp 的 register_forward_hook 不可靠）----
        orig_embed = inner.embed_input_ids
        def patched_embed(input_ids):
            out = orig_embed(input_ids)
            key = f"{model._st_phase}:embedding"
            if key not in model._cap:
                model._cap[key] = out.detach().to(torch.float32).cpu().clone()
            return out
        inner.embed_input_ids = patched_embed
        model._st_orig_embed = orig_embed
        model._st_patches.append(("__embed__", None))  # 标记需还原

        # ---- per-layer INPUT hidden_states：pre-hook + 立即 clone ----
        for li in layer_indices:
            layer = inner.layers[li]
            def make_pre(idx):
                def ph(module, args, kwargs):
                    phase = model._st_phase
                    key = f"{phase}:layer{idx}_in"
                    if key not in model._cap:
                        hs = args[0] if len(args) > 0 else kwargs.get("hidden_states")
                        model._cap[key] = hs.detach().to(torch.float32).cpu().clone()
                        if phase == "prefill":
                            model._st_prefill_fired += 1
                            if model._st_prefill_fired >= len(layer_indices):
                                model._st_phase = "decode1"
                return ph
            model._st_hooks.append(layer.register_forward_pre_hook(make_pre(li), with_kwargs=True))

        # ---- final norm 输出 ----
        def norm_hook(module, args, kwargs, output):
            phase = model._st_phase
            key = f"{phase}:final_norm"
            if key not in model._cap:
                out = output[0] if isinstance(output, tuple) else output
                model._cap[key] = out.detach().to(torch.float32).cpu().clone()
        model._st_hooks.append(inner.norm.register_forward_hook(norm_hook, with_kwargs=True))

        # ---- logits ----
        def logits_hook(module, args, kwargs, output):
            phase = model._st_phase
            key = f"{phase}:logits"
            if key not in model._cap:
                model._cap[key] = output.detach().to(torch.float32).cpu().clone()
        model._st_hooks.append(model.lm_head.register_forward_hook(logits_hook, with_kwargs=True))
    else:
        # ---- 子算子细化（drill-down）：只抓 prefill（第一次触发），不区分阶段 ----
        # key 不带 phase 前缀；if key not in cap 保证只记 prefill 那次。
        # 避免阶段翻转错位（--layers 单层时 prefill 一触发就翻阶段会把 prefill 值记成 decode1）。
        for li in layer_indices:
            base = inner.layers[li]
            if only_arg == "attn" and attn_attr:
                mod = getattr(base, attn_attr)
                def make_post(idx):
                    def ph(module, args, kwargs, output):
                        key = f"layer{idx}_attn_out"
                        if key not in model._cap:
                            t = output[0] if isinstance(output, tuple) else output
                            model._cap[key] = t.detach().to(torch.float32).cpu()
                    return ph
                model._st_hooks.append(mod.register_forward_hook(make_post(li), with_kwargs=True))
            if only_arg == "mlp" and mlp_attr:
                mod = getattr(base, mlp_attr)
                def make_post2(idx):
                    def ph(module, args, kwargs, output):
                        key = f"layer{idx}_mlp_out"
                        if key not in model._cap:
                            t = output[0] if isinstance(output, tuple) else output
                            model._cap[key] = t.detach().to(torch.float32).cpu()
                    return ph
                model._st_hooks.append(mod.register_forward_hook(make_post2(li), with_kwargs=True))
            if only_arg == "router" and is_moe and router_rel:
                mod = base
                for p in router_rel.split("."):
                    mod = getattr(mod, p)
                orig_fn = mod.forward
                def make_router(idx, orig_fn):
                    def patched(*a, **kw):
                        out = orig_fn(*a, **kw)
                        key = f"layer{idx}_router_logits"
                        if key not in model._cap:
                            t = out[-1] if isinstance(out, tuple) else out
                            model._cap[key] = t.detach().to(torch.float32).cpu()
                        return out
                    return patched
                mod.forward = make_router(li, orig_fn)
                model._st_patches.append((mod, orig_fn))
    return {"attached": len(model._st_hooks), "layers": layer_indices, "only": only_arg}

def _fetch_cap(worker):
    model = worker.model_runner.model
    return dict(getattr(model, "_cap", {}) or {})

def _detach_stage_hooks(worker):
    model = worker.model_runner.model
    for h in getattr(model, "_st_hooks", []) or []:
        try: h.remove()
        except Exception: pass
    # 还原 embed_input_ids monkey-patch
    if hasattr(model, "_st_orig_embed"):
        try: model.model.embed_input_ids = model._st_orig_embed
        except Exception: pass
    for mod, orig in getattr(model, "_st_patches", []) or []:
        if mod == "__embed__": continue
        try: mod.forward = orig
        except Exception: pass
    model._st_hooks = []
    model._st_patches = []
    return {"detached": True}

def _probe_layer_attrs(worker, layer_prefix):
    model = worker.model_runner.model
    return _detect_layer_attrs(model, layer_prefix)

llm = LLM(
    model=model_path, tensor_parallel_size=tp, dtype="bfloat16",
    trust_remote_code=True, enforce_eager=True,
    max_model_len=2048, gpu_memory_utilization=0.9,
)
print(f"[vLLM] engine started (tp={tp})", flush=True)

# 属性名：优先 probe_attrs，否则 RPC 进 worker 探测
if probe_attrs and probe_attrs.get("attn"):
    attn_attr = probe_attrs["attn"]; mlp_attr = probe_attrs["mlp"]; router_rel = probe_attrs["router"]
    print(f"[vLLM] 用 probe 结果: attn={attn_attr} mlp={mlp_attr} router={router_rel}", flush=True)
else:
    attn_attr, mlp_attr, router_rel = llm.collective_rpc(_probe_layer_attrs, args=(layer_prefix,))[0]
    print(f"[vLLM] worker 内探测: attn={attn_attr} mlp={mlp_attr} router={router_rel}", flush=True)

layers = layers_arg if layers_arg is not None else list(range(prof.num_layers))
res = llm.collective_rpc(_attach_stage_hooks,
                         args=(layers, only_arg, attn_attr, mlp_attr, router_rel, prof.is_moe))
print(f"[vLLM] attach: {res[0]}", flush=True)

sp = SamplingParams(temperature=0.0, max_tokens=max_tokens)
outs = llm.generate([prompt], sp)
gen_ids = list(outs[0].outputs[0].token_ids)
print(f"[vLLM] gen_ids: {gen_ids}", flush=True)

cap = llm.collective_rpc(_fetch_cap)[0]
print(f"[vLLM] 抓到 {len(cap)} 个 stage 值", flush=True)
for k in sorted(cap):
    v = cap[k]
    print(f"  {k}: {tuple(v.shape) if torch.is_tensor(v) else type(v).__name__}", flush=True)
llm.collective_rpc(_detach_stage_hooks)

inter = {k: v for k, v in cap.items() if torch.is_tensor(v)}
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
        if probe_attrs["layer_prefix"]:
            args.layer_prefix = probe_attrs["layer_prefix"]
        print(f"[probe] 读取探测结果 {pj}")
        print(f"[probe] layer_prefix={probe_attrs['layer_prefix']} "
              f"attn={probe_attrs['attn']} mlp={probe_attrs['mlp']} "
              f"router={probe_attrs['router']}")

    # 默认采样层（仅在未指定 --layers 时用，需总层数；从 probe 或 config 推断）
    if layers_arg is None:
        try:
            from lib import load_model_profile
            nl = load_model_profile(args.model).num_layers
            layers_arg = sample_layers(nl)
            print(f"[layers] 自动采样 {nl} 层 → {layers_arg}")
        except Exception as e:
            print(f"[layers] 无法推断总层数({e})，请用 --layers 显式指定", file=sys.stderr)
            sys.exit(2)

    tmpdir = tempfile.mkdtemp(prefix="compare_layers_")
    hf_pt = os.path.join(tmpdir, "hf_inter.pt")
    vllm_pt = os.path.join(tmpdir, "vllm_inter.pt")

    if not args.skip_hf:
        print("\n" + "=" * 60)
        print("[HF] 抓逐层中间值...")
        print("=" * 60)
        script = _HF_SCRIPT % (_ROOT, args.model, args.prompt,
                               hf_pt, layers_arg, args.only, args.layer_prefix, probe_attrs)
        run_subprocess(script, tag="HF")

    if not args.skip_vllm:
        print("\n" + "=" * 60)
        tag = "VLLM_ENABLE_MOE_FUSED_GATE=0" if args.env else "no env (默认)"
        print(f"[vLLM] 抓逐层中间值 (tp={tp}, {tag})...")
        print("=" * 60)
        extra = {"VLLM_ENABLE_MOE_FUSED_GATE": "0"} if args.env else {}
        script = _VLLM_SCRIPT % (_ROOT, args.model, args.prompt, args.max_tokens,
                                 tp, vllm_pt, layers_arg, args.only, args.layer_prefix, probe_attrs)
        run_subprocess(script, extra_env=extra, tag="vLLM")

    if args.skip_hf or args.skip_vllm:
        print("\n[skip] 一边被跳过，无法对比。中间值已落盘：", tmpdir)
        return

    # 对比
    print("\n" + "=" * 60)
    print("[compare] 逐层中间值对比（cos decay）")
    print("=" * 60)
    import torch
    from lib import compare_tensors
    hf = torch.load(hf_pt)
    vllm = torch.load(vllm_pt)

    # stage 顺序：默认口径 embedding → 各层 layer_in → final_norm → logits，带 phase 前缀
    # 子算子口径（--only）：各层 attn_out/mlp_out/router_logits，只比 prefill，key 无 phase 前缀
    if args.only is None:
        stage_order = ["embedding"] + [f"layer{i}_in" for i in layers_arg] + ["final_norm", "logits"]
        phases = ["prefill", "decode1"] if args.max_tokens >= 1 else ["prefill"]
        key_of = lambda phase, st: f"{phase}:{st}"
    else:
        suffix = {"attn": "attn_out", "mlp": "mlp_out", "router": "router_logits"}[args.only]
        stage_order = [f"layer{i}_{suffix}" for i in layers_arg]
        phases = ["prefill"]
        key_of = lambda phase, st: st  # --only 模式 key 无 phase 前缀

    first_diverge = None
    for phase in phases:
        print(f"\n----- phase: {phase} -----")
        print(f"{'stage':<24} {'cos':>10} {'max_abs':>12} {'mean_abs':>12} {'verdict'}")
        print("-" * 76)
        for st in stage_order:
            key = key_of(phase, st)
            a = hf.get(key); b = vllm.get(key)
            if a is None or b is None:
                print(f"{st:<24} {'MISSING':>10}  hf={'no' if a is None else 'ok'} vllm={'no' if b is None else 'ok'}")
                continue
            a, b = _align(a, b)
            if a.shape != b.shape:
                print(f"{st:<24} {'shape_skip':>10} {tuple(a.shape)} vs {tuple(b.shape)}")
                continue
            try:
                r = compare_tensors(st, a, b, atol=args.atol)
                ok = r.is_close or r.cos >= args.cos_thresh
                mark = "✅" if ok else "❌"
                print(f"{st:<24} {r.cos:>10.6f} {r.max_abs_diff:>12.4e} "
                      f"{r.mean_abs_diff:>12.4e} {mark}")
                if not ok and first_diverge is None:
                    first_diverge = key
            except Exception as e:
                print(f"{st:<24} ERR {e}")

    print("-" * 76)
    if first_diverge:
        print(f"\n[verdict] ❌ 首个发散点: {first_diverge}")
        print("  → 该 stage 之前都一致，从这里开始 HF 与 vLLM 分叉。")
        print("  → 若是 layerN_in：误差在 L{N-1} 的 attention 或 MoE 累积产生。")
        print("  → 用 --only attn/mlp/router --layers N 在该层内 sub-op 细化。")
    else:
        print("\n[verdict] ✅ 所有对比的 stage 均一致")
    print(f"\n[info] 中间值落盘目录: {tmpdir}")


def _align(a, b):
    """对齐 HF 与 vLLM 中间张量的 batch 维差异。
    HF 常为 [1, seq, H]，vLLM 为 [seq, H]（无 batch 维）。squeeze 掉大小为 1 的前导维。"""
    import torch
    while a.dim() > b.dim() and a.shape[0] == 1:
        a = a.squeeze(0)
    while b.dim() > a.dim() and b.shape[0] == 1:
        b = b.squeeze(0)
    return a, b


if __name__ == "__main__":
    main()
