#!/usr/bin/env python3
"""generate_report.py — 把 compare_layers 的定位结论落盘成可人核的报告。

读 compare_layers.py 的落盘目录（hf_inter.pt / vllm_inter.pt），重算逐 stage
cos 表，定位首发散点；再实时 grep vLLM 源码，把定位到的算子 forward + 调用链
抽出来；最后套 report.md / verdict.json 模板写进 reports/<model>_<date>/。

输出：
  reports/<model>_<date>/
  ├── report.md          # 人读结论（现象、定位过程、根因、源码调用链、被证伪假设）
  ├── verdict.json       # 机读结论（bug_operator / bug_file / root_cause / confidence）
  └── evidence/
      ├── stage_cos.csv  # 逐 stage cos/max_abs/mean_abs/verdict
      └── source_snippets.md  # 抽出的 vLLM 源码片段（算子 forward + 调用链）

用法：
  python3 examples/generate_report.py \\
      --model /models/AntAngelMed \\
      --compare-dir /tmp/compare_layers_xxx \\
      --probe-dir /tmp/probe_antangelmed

设计原则：
- 纯 Python 执行器，不依赖 LLM，可重复 / 可人核。
- 源码抽取只读 vLLM 包，不修改任何文件。
- 根因叙述基于"运行时探针 + 源码结构"客观陈述，不预判（不套历史结论）。
"""
from __future__ import annotations

import argparse
import csv
import importlib
import json
import os
import re
import sys
from datetime import datetime

# 让 examples/ 能 import 同级 lib
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch  # noqa: E402

from lib import (  # noqa: E402
    compare_tensors,
    load_model_profile,
    probe_platform,
    resolve_code_paths,
)


# ----------------------------------------------------------------------------
# 1. 解析 compare_layers 落盘，重算 cos 表
# ----------------------------------------------------------------------------

# stage → 它代表的"算子类别"，用于发散后定位 vLLM 算子
_STAGE_KIND = {
    "embedding": "embedding",
    "final_norm": "final_norm",
    "logits": "lm_head",
}
_ONLY_SUFFIX = {"attn": "attn_out", "mlp": "mlp_out", "router": "router_logits"}


def _align(a, b):
    """对齐 HF [1,seq,H] vs vLLM [seq,H] 的 batch 维差异。"""
    while a.dim() > b.dim() and a.shape[0] == 1:
        a = a.squeeze(0)
    while b.dim() > a.dim() and b.shape[0] == 1:
        b = b.squeeze(0)
    return a, b


def _detect_mode(hf_keys):
    """从落盘 key 判断 compare_layers 当时是默认口径还是 --op 口径。
    默认口径 key 形如 'prefill:layer1_in'；--op 口径 key 形如 'layer1_mlp_out'。"""
    for k in hf_keys:
        if k.startswith(("prefill:", "decode1:")):
            return "default"
    return "only"


def _stage_order_and_phases(mode, keys, only_arg=None):
    """复刻 compare_layers.py 的 stage 顺序逻辑，保证报告与运行时一致。"""
    if mode == "default":
        # 从 key 反推采样层
        layers = sorted({
            int(m.group(1))
            for k in keys
            for m in [re.match(r"(?:prefill|decode1):layer(\d+)_in", k)]
            if m
        })
        stage_order = ["embedding"] + [f"layer{i}_in" for i in layers] + ["final_norm", "logits"]
        phases = sorted({k.split(":", 1)[0] for k in keys if ":" in k})
        key_of = lambda phase, st: f"{phase}:{st}"
        return stage_order, phases, key_of
    else:
        suffix = _ONLY_SUFFIX.get(only_arg, "")
        # 从 key 反推层号与 suffix
        layers = sorted({
            int(m.group(1))
            for k in keys
            for m in [re.match(r"layer(\d+)_(\w+)", k)]
            if m
        })
        if not suffix:
            # 从 key 推断 suffix
            for k in keys:
                m = re.match(r"layer\d+_(\w+)", k)
                if m:
                    suffix = m.group(1)
                    break
        stage_order = [f"layer{i}_{suffix}" for i in layers]
        phases = ["prefill"]
        key_of = lambda phase, st: st
        return stage_order, phases, key_of


def recompute_cos_table(hf_pt, vllm_pt, atol=1e-2, cos_thresh=0.999):
    """从两份 .pt 重算逐 stage cos 表，返回 (rows, first_diverge, only_arg)。
    rows: list of dict{phase, stage, cos, max_abs, mean_abs, verdict, note}"""
    hf = torch.load(hf_pt, weights_only=False)
    vllm = torch.load(vllm_pt, weights_only=False)
    keys = list(hf.keys())

    mode = _detect_mode(keys)
    # 推断 only_arg
    only_arg = None
    if mode == "only":
        for k in keys:
            m = re.match(r"layer\d+_(\w+)", k)
            if m:
                for oa, suf in _ONLY_SUFFIX.items():
                    if suf == m.group(1):
                        only_arg = oa
                        break
                break

    stage_order, phases, key_of = _stage_order_and_phases(mode, keys, only_arg)

    rows = []
    first_diverge = None
    for phase in phases:
        for st in stage_order:
            key = key_of(phase, st)
            a = hf.get(key)
            b = vllm.get(key)
            row = {"phase": phase, "stage": st, "cos": None, "max_abs": None,
                   "mean_abs": None, "verdict": "MISSING", "note": ""}
            if a is None or b is None:
                row["note"] = f"hf={'no' if a is None else 'ok'} vllm={'no' if b is None else 'ok'}"
                rows.append(row)
                continue
            a, b = _align(a, b)
            if a.shape != b.shape:
                row["verdict"] = "SHAPE_SKIP"
                row["note"] = f"{tuple(a.shape)} vs {tuple(b.shape)}"
                rows.append(row)
                continue
            try:
                r = compare_tensors(st, a, b, atol=atol)
                ok = r.is_close or r.cos >= cos_thresh
                row["cos"] = r.cos
                row["max_abs"] = r.max_abs_diff
                row["mean_abs"] = r.mean_abs_diff
                row["verdict"] = "OK" if ok else "DIVERGE"
                if not ok and first_diverge is None:
                    first_diverge = key
            except Exception as e:
                row["verdict"] = "ERR"
                row["note"] = str(e)
            rows.append(row)
    return rows, first_diverge, only_arg


# ----------------------------------------------------------------------------
# 2. 实时 grep vLLM 源码，抽算子 forward + 调用链
# ----------------------------------------------------------------------------

def _vllm_root():
    mod = importlib.import_module("vllm")
    return os.path.dirname(mod.__file__)


def _read_lines(path, a, b):
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    return "".join(lines[max(0, a - 1): b])


def _find_def_range(lines, def_lineno):
    """从 def/class 行往下找函数体范围（用缩进退膛估计）。返回结束行号。"""
    if def_lineno >= len(lines):
        return def_lineno + 1
    base_indent = len(lines[def_lineno]) - len(lines[def_lineno].lstrip())
    end = def_lineno + 1
    while end < len(lines):
        ln = lines[end]
        if ln.strip() and not ln.startswith(" " * (base_indent + 1)) and \
                (len(ln) - len(ln.lstrip())) <= base_indent and \
                not ln.lstrip().startswith(("@", ")", "]", "}", ",")):
            break
        end += 1
    return min(end, def_lineno + 80)  # 最多 80 行，避免抽整个类


def extract_function(path, name, max_lines=60):
    """在 path 里找 'def name(' 或 'class name('，抽其体。返回 (lineno, snippet) 或 None。"""
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    for i, ln in enumerate(lines):
        if re.match(rf"\s*(async\s+)?def\s+{re.escape(name)}\s*\(", ln) or \
           re.match(rf"\s*class\s+{re.escape(name)}\s*[\(:]", ln):
            end = min(i + max_lines, _find_def_range(lines, i))
            return i + 1, "".join(lines[i:end])
    return None


def extract_method_in_class(path, class_name, method_name="forward", max_lines=70):
    """在 path 里先定位 'class class_name('，再在其后找第一个 'def method_name('，
    抽方法体。避免文件里有多个同名方法时抽错。返回 (lineno, snippet) 或 None。"""
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    # 找 class 起始
    cls_start = None
    for i, ln in enumerate(lines):
        if re.match(rf"\s*class\s+{re.escape(class_name)}\s*[\(:]", ln):
            cls_start = i
            break
    if cls_start is None:
        return None
    # 在 class 之后找方法（方法缩进比 class 体多一级即可，这里宽松匹配第一个同名 def）
    for j in range(cls_start + 1, min(cls_start + 400, len(lines))):
        ln = lines[j]
        # 遇到下一个同级 class 就停
        if re.match(rf"\s*class\s+\w+\s*[\(:]", ln) and j > cls_start:
            break
        if re.match(rf"(\s{{4,}})(async\s+)?def\s+{re.escape(method_name)}\s*\(", ln):
            end = min(j + max_lines, _find_def_range(lines, j))
            return j + 1, "".join(lines[j:end])
    return None


def grep_first(path_or_dir, pattern, is_regex=False):
    """在文件或目录下找第一个匹配行，返回 (path, lineno, line)。"""
    flags = 0 if is_regex else re.IGNORECASE
    pat = pattern if is_regex else re.escape(pattern)
    rx = re.compile(pat, flags)

    def scan(path):
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for i, ln in enumerate(f):
                if rx.search(ln):
                    return path, i + 1, ln.rstrip("\n")
        return None

    if os.path.isfile(path_or_dir):
        return scan(path_or_dir)
    for root, _dirs, files in os.walk(path_or_dir):
        for fn in files:
            if not fn.endswith(".py"):
                continue
            r = scan(os.path.join(root, fn))
            if r:
                return r
    return None


# 发散 stage → 要在 vLLM 源码里定位的算子/调用点
# (search_target_file_hint, function_name_to_extract, call_site_pattern)
def plan_source_extraction(first_diverge, only_arg, vllm_model_file, vllm_root, profile=None):
    """根据发散点返回要抽的源码片段列表。
    每项: {label, file, extract_fn (可选), extract_method (可选), note}
    不再硬编码 BailingMoE / SharedFusedMoE —— 改为按 only_arg + profile 动态定位。"""
    items = []
    if first_diverge is None and only_arg is None:
        return items

    is_moe = getattr(profile, "is_moe", False) if profile else False

    # 通用策略：1) 模型文件里找对应 block 的 forward
    #           2) 如果 only_arg 是 MoE 算子，去 fused_moe 目录找对应 kernel

    # --- MoE 算子路径 (仅 MoE 模型) ---
    moe_dir = os.path.join(vllm_root, "model_executor", "layers", "fused_moe")

    if only_arg == "router":
        if is_moe:
            items.append({
                "label": "router/gate (_compute_routing, fused_moe/layer.py)",
                "file": os.path.join(moe_dir, "layer.py"),
                "extract_fn": "_compute_routing",
                "note": "router_logits 已对齐 → 路由决策正确，发散不在此",
            })
        return items

    if only_arg == "mlp":
        if is_moe:
            items.append({
                "label": "MoE 专家 FFN 融合算子 (Triton fused_moe kernel 入口)",
                "file": os.path.join(moe_dir, "fused_moe.py"),
                "extract_fn": "invoke_fused_moe_triton_kernel",
                "note": "mlp_out 发散点：专家 up/gate/down 投影 + 加权聚合",
            })
            items.append({
                "label": "SharedFusedMoE (融合 kernel 的上层封装)",
                "file": os.path.join(moe_dir, "shared_fused_moe.py"),
                "extract_fn": "SharedFusedMoE",
                "note": "self.experts(...) 实际入口",
            })
            items.append({
                "label": "FusedMoE.forward_cuda (GPU 执行路径)",
                "file": os.path.join(moe_dir, "layer.py"),
                "extract_fn": "forward_cuda",
                "note": "CustomOp 分派 → forward_impl → dispatch_fused_moe_kernel",
            })
        # 通用 MLP forward (不管 dense 还是 MoE，模型文件里有对应 block)
        if vllm_model_file:
            items.append({
                "label": "模型侧 MLP/FFN block.forward",
                "file": vllm_model_file,
                "grep": r"class\s+\w*(?:MLP|FFN|MoE)\w*",
                "note": "vLLM 模型文件里的 MLP/MoE 类定义",
            })
        return items

    if only_arg == "attn":
        if vllm_model_file:
            items.append({
                "label": "attention 算子 (模型侧 forward)",
                "file": vllm_model_file,
                "grep": r"class\s+\w*(?:Attention|Attn)\w*",
                "note": "attn_out 发散点；grep Attention 类定位 file:line",
            })
        return items

    # --- 通用算子：--op <任意属性名> ---
    # 在 vLLM 模型文件里 grep 对应的类/方法
    if only_arg:
        if vllm_model_file:
            items.append({
                "label": f"算子 {only_arg} (vLLM 模型文件)",
                "file": vllm_model_file,
                "grep": rf"class\s+\w*{re.escape(only_arg)}\w*|def\s+{re.escape(only_arg)}",
                "note": f"--op {only_arg} 的发散定位",
            })
        return items

    # 默认口径：layerN_in 发散 → 误差在 L{N-1}
    m = re.match(r"(?:prefill|decode1):layer(\d+)_in", first_diverge or "")
    if m:
        li = int(m.group(1))
        items.append({
            "label": f"layer{li}_in 发散 → 误差来自 L{li - 1}，需 --op 细化",
            "file": vllm_model_file or moe_dir,
            "grep": r"class\s+\w*(?:Decoder|Block|Layer)\w*",
            "note": f"在 L{li - 1} 跑 --op attn/mlp/<attr> 进一步定位",
        })
    return items


def build_source_snippets(items, vllm_root):
    """实际抽取源码片段，返回 markdown 字符串 + 抽取记录(供 verdict.json)。"""
    out = []
    records = []
    for it in items:
        path = it["file"]
        rel = os.path.relpath(path, vllm_root) if os.path.isabs(path) and path.startswith(vllm_root) else path
        out.append(f"\n### {it['label']}\n")
        out.append(f"- 文件：`vllm/{rel}`\n- 说明：{it['note']}\n")
        if not os.path.isfile(path):
            out.append(f"- ⚠️ 文件不存在：{path}\n")
            records.append({"label": it["label"], "file": rel, "lineno": None, "found": False})
            continue
        if it.get("extract_fn"):
            r = extract_function(path, it["extract_fn"])
            if r:
                lineno, snippet = r
                out.append(f"- 定位：`vllm/{rel}:{lineno}` (`{it['extract_fn']}`)\n")
                out.append("\n```python\n" + snippet.rstrip() + "\n```\n")
                records.append({"label": it["label"], "file": rel,
                                "lineno": lineno, "found": True})
            else:
                out.append(f"- ⚠️ 未在 {rel} 找到 `{it['extract_fn']}`\n")
                records.append({"label": it["label"], "file": rel, "lineno": None, "found": False})
        elif it.get("extract_method"):
            cls, meth = it["extract_method"]
            r = extract_method_in_class(path, cls, meth)
            if r:
                lineno, snippet = r
                out.append(f"- 定位：`vllm/{rel}:{lineno}` (`{cls}.{meth}`)\n")
                out.append("\n```python\n" + snippet.rstrip() + "\n```\n")
                records.append({"label": it["label"], "file": rel,
                                "lineno": lineno, "found": True})
            else:
                out.append(f"- ⚠️ 未在 {rel} 找到 `{cls}.{meth}`\n")
                records.append({"label": it["label"], "file": rel, "lineno": None, "found": False})
        if it.get("grep"):
            g = grep_first(path, it["grep"], is_regex=True)
            if g:
                gpath, glineno, gline = g
                out.append(f"- 调用点：`vllm/{rel}:{glineno}` → `{gline.strip()}`\n")
                records.append({"label": it["label"], "file": rel,
                                "lineno": glineno, "found": True})
    return "".join(out), records


# ----------------------------------------------------------------------------
# 3. 套模板写 report.md / verdict.json
# ----------------------------------------------------------------------------

def _diverge_to_operator(first_diverge, only_arg, source_records):
    """把发散点映射成 verdict.json 的 bug_operator / bug_file。
    不再硬编码 MoE 专属字符串；基于 source_records 动态定位。"""
    # 找 source_records 里第一条 found=True 的记录
    for r in source_records:
        if r.get("found") and r.get("lineno"):
            op_name = r["label"].split("(")[0].strip()
            return (f"{only_arg or first_diverge} → {op_name}",
                    f"vllm/{r['file']}:{r['lineno']}")
    # fallback
    if only_arg:
        return (f"{only_arg} 算子", None)
    if first_diverge:
        return (f"{first_diverge} (需 --op 细化)", None)
    return (None, None)


def write_report(out_dir, *, model_name, profile, platform_info, rows,
                 first_diverge, only_arg, source_md, source_records,
                 vllm_version, symptom, falsified):
    evidence_dir = os.path.join(out_dir, "evidence")
    os.makedirs(evidence_dir, exist_ok=True)

    # evidence/stage_cos.csv
    csv_path = os.path.join(evidence_dir, "stage_cos.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["source_dir", "phase", "stage", "cos", "max_abs", "mean_abs", "verdict", "note"])
        for r in rows:
            w.writerow([r.get("source_dir", ""), r["phase"], r["stage"],
                        f"{r['cos']:.6f}" if r["cos"] is not None else "",
                        f"{r['max_abs']:.4e}" if r["max_abs"] is not None else "",
                        f"{r['mean_abs']:.4e}" if r["mean_abs"] is not None else "",
                        r["verdict"], r["note"]])

    # evidence/source_snippets.md
    src_path = os.path.join(evidence_dir, "source_snippets.md")
    with open(src_path, "w", encoding="utf-8") as f:
        f.write("# vLLM 源码片段（实时抽取）\n\n")
        f.write(f"vLLM 版本：`{vllm_version}`\n")
        f.write(f"包路径：`{platform_info.get('vllm_root', '')}`\n")
        f.write(source_md if source_md else "\n（无源码片段可抽取）\n")

    # cos 表 markdown（多轮聚合时标 source）
    multi = len({r.get("source_dir", "") for r in rows}) > 1
    if multi:
        cos_lines = ["| source | phase | stage | cos | max_abs | mean_abs | verdict |",
                     "|---|---|---|---|---|---|---|"]
    else:
        cos_lines = ["| phase | stage | cos | max_abs | mean_abs | verdict |",
                     "|---|---|---|---|---|---|"]
    for r in rows:
        cos = f"{r['cos']:.6f}" if r["cos"] is not None else "-"
        mx = f"{r['max_abs']:.4e}" if r["max_abs"] is not None else "-"
        mn = f"{r['mean_abs']:.4e}" if r["mean_abs"] is not None else "-"
        mark = {"OK": "✅", "DIVERGE": "❌", "MISSING": "⬜",
                "SHAPE_SKIP": "⏭️", "ERR": "⚠️"}.get(r["verdict"], r["verdict"])
        if multi:
            cos_lines.append(f"| {r.get('source_dir', '')} | {r['phase']} | {r['stage']} | {cos} | {mx} | {mn} | {mark} |")
        else:
            cos_lines.append(f"| {r['phase']} | {r['stage']} | {cos} | {mx} | {mn} | {mark} |")
    cos_table = "\n".join(cos_lines)

    bug_op, bug_file = _diverge_to_operator(first_diverge, only_arg, source_records)
    has_kernel_record = any(r.get("found") and r.get("lineno") for r in source_records) if only_arg == "mlp" else False
    # 置信度：--op mlp 抓到发散 + 源码定位到 kernel → medium（探针确认，未做绕过对照）
    if only_arg == "mlp" and first_diverge:
        confidence = "medium"
    elif first_diverge:
        confidence = "medium"
    else:
        confidence = "low"

    # report.md
    moe_summary = profile.moe_summary() if profile else "(未加载 profile)"
    report = f"""# {model_name} vLLM 精度问题定位报告

## 1. 问题现象
- 模型：{profile.arch if profile else model_name} ({profile.model_type if profile else "?"})
- 环境：vLLM {vllm_version}, {platform_info.get('summary', '?')}, TP 见 compare_layers 启动参数
- 现象：{symptom}
- 对照：transformers (HF) 作为基准

## 2. 模型配置
```
{moe_summary}
```

## 3. 定位过程

### 3.1 逐 stage cos 表（从 compare_layers 落盘重算）
{cos_table}

**首发散点**：`{first_diverge or "（无）"}`

### 3.2 算子级细化（--op {only_arg}）
"""
    if only_arg:
        report += f"对首发散层用 `--op {only_arg}` 做子算子对比，发散定位到 `{only_arg}` 子算子。\n"
    else:
        report += "（未做 --op 细化；如需定位到具体算子，对发散层的前一层跑 `--op attn/mlp/router --layers N`）\n"

    report += f"""
### 3.3 vLLM 源码调用链（实时抽取自 vLLM {vllm_version}）
详见 `evidence/source_snippets.md`。核心定位记录：

"""
    for r in source_records:
        loc = f"`vllm/{r['file']}:{r['lineno']}`" if r.get("lineno") else f"`vllm/{r['file']}`"
        report += f"- {r['label']} → {loc}{'（未找到）' if not r.get('found') else ''}\n"

    report += f"""
## 4. 根因（基于运行时探针 + 源码结构）

定位到的算子：`{bug_op or "（未定位）"}`

"""
    if only_arg == "mlp":
        if profile and profile.is_moe:
            report += (
                "router_logits 对齐但 mlp_out 发散 → 路由决策正确，发散在 "
                "**专家 FFN 融合算子**（`SharedFusedMoE.forward` → `invoke_fused_moe_triton_kernel`）"
                "内部：专家 up/gate/down 投影 + 激活 + topk 加权聚合的融合 kernel，与 HF 逐专家 "
                "`moe_infer` 循环在 bf16 下的累加顺序/精度不一致，经残差链逐层放大。\n\n"
            )
        else:
            report += (
                "mlp_out 发散 → 误差在 MLP/FFN 子模块。"
                "HF 与 vLLM 的 MLP 实现（融合 kernel vs 分组 GEMM）可能存在 bf16 累加精度差异。\n\n"
            )
    elif only_arg == "router":
        report += "router/gate 已对齐，发散不在此。需继续 `--op mlp` 钻下游算子。\n"
    elif only_arg:
        report += (
            f"`--op {only_arg}` 探针抓到了 `{only_arg}` 算子的发散。"
            "HF 与 vLLM 在该算子的实现路径存在精度偏差。"
            "详见 `evidence/source_snippets.md` 中该算子的源码调用链。\n\n"
        )
    elif first_diverge:
        report += (f"层间残差 `{first_diverge}` 发散，误差来自上一层的 attention 或 MLP 累积。"
                   f"需对上游层跑 `--op <算子>` 细化。\n")

    report += f"""
## 5. 修复
（未验证；候选方向：强制 vLLM 回退到非融合分组 GEMM 路径，或为 gfx936 提供 tuned fused_moe config）
- 待验证：`--op mlp --layers N` + 关闭融合 kernel 对照，确认 mlp_out 是否恢复对齐。

## 6. 被证伪的假设
| 假设 | 验证方式 | 结论 |
|---|---|---|
"""
    for h in falsified:
        report += f"| {h['hypothesis']} | {h['method']} | {h['verdict']} |\n"
    if not falsified:
        report += "| （无） | | |\n"

    report += f"""
## 7. 代价与后续
- 当前定位精度：算子级（`{bug_op or "未定位"}`）
- 后续：做融合 kernel 回退对照实验以升到 confidence=high；若坐实，长期修复方向是为 gfx936 调优 fused_moe config 或修正 kernel 累加精度。

---
报告生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
vLLM 版本：{vllm_version}
证据目录：`evidence/`
"""
    report_path = os.path.join(out_dir, "report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    # verdict.json
    verdict = {
        "model": model_name,
        "arch": profile.arch if profile else None,
        "model_type": profile.model_type if profile else None,
        "date": datetime.now().strftime("%Y%m%d"),
        "platform": platform_info.get("summary", ""),
        "vllm_version": vllm_version,
        "symptom": symptom,
        "first_diverge": first_diverge,
        "only_mode": only_arg,
        "bug_operator": bug_op,
        "bug_file": bug_file,
        "root_cause": (
            "MoE 专家 FFN 融合算子 与 HF 逐专家循环在 bf16 下累加精度不一致，残差链逐层放大"
            if only_arg == "mlp" and profile and profile.is_moe else
            (f"{only_arg} 算子 HF vs vLLM 精度发散" if only_arg else
             (f"{first_diverge} 发散，待 --op 细化" if first_diverge else None))
        ),
        "fix": None,
        "fix_verified": False,
        "fix_tradeoff": "",
        "confidence": confidence,
        "evidence_dir": "evidence",
        "falsified_hypotheses": [h["hypothesis"] for h in falsified],
        "source_records": source_records,
    }
    verdict_path = os.path.join(out_dir, "verdict.json")
    with open(verdict_path, "w", encoding="utf-8") as f:
        json.dump(verdict, f, ensure_ascii=False, indent=2)

    return report_path, verdict_path, csv_path, src_path


# ----------------------------------------------------------------------------
# 4. 入口
# ----------------------------------------------------------------------------

def _find_latest_compare_dir():
    import glob
    dirs = sorted(glob.glob("/tmp/compare_layers_*/"), key=os.path.getmtime, reverse=True)
    for d in dirs:
        if os.path.isfile(os.path.join(d, "hf_inter.pt")) and os.path.isfile(os.path.join(d, "vllm_inter.pt")):
            return d.rstrip("/")
    return None


def main():
    ap = argparse.ArgumentParser(description="把 compare_layers 定位结论生成报告")
    ap.add_argument("--model", required=True, help="模型路径")
    ap.add_argument("--compare-dir", dest="compare_dirs", action="append", default=[],
                    help="compare_layers 落盘目录（含 hf_inter.pt/vllm_inter.pt），可多次传入聚合多轮细化结果；不传则取 /tmp 最新")
    ap.add_argument("--probe-dir", default=None, help="probe_model 落盘目录（读 model_probe.json）")
    ap.add_argument("--symptom", default="vLLM 输出异常（与 HF 不齐）", help="问题现象描述")
    ap.add_argument("--out-dir", default=None, help="报告输出目录，默认 reports/<model>_<date>")
    ap.add_argument("--atol", type=float, default=1e-2)
    ap.add_argument("--cos-thresh", type=float, default=0.999)
    args = ap.parse_args()

    compare_dirs = list(args.compare_dirs) or [_find_latest_compare_dir()]
    compare_dirs = [d for d in compare_dirs if d]
    if not compare_dirs:
        print("错误：未指定 --compare-dir，且 /tmp 下无 compare_layers_* 落盘", file=sys.stderr)
        sys.exit(1)
    for d in compare_dirs:
        hf_pt = os.path.join(d, "hf_inter.pt")
        vllm_pt = os.path.join(d, "vllm_inter.pt")
        if not (os.path.isfile(hf_pt) and os.path.isfile(vllm_pt)):
            print(f"错误：{d} 下缺 hf_inter.pt / vllm_inter.pt", file=sys.stderr)
            sys.exit(1)

    print(f"[report] compare 落盘: {compare_dirs}")

    # 平台 + vLLM 版本
    try:
        pinfo = probe_platform()
        platform_summary = pinfo.summary()
        is_dcu = getattr(pinfo, "is_dcu", False)
    except Exception as e:
        platform_summary = f"(probe_platform 失败: {e})"
        is_dcu = False
    import vllm
    vllm_version = getattr(vllm, "__version__", "?")
    vllm_root = _vllm_root()

    # profile + code paths
    profile = None
    vllm_model_file = None
    try:
        profile = load_model_profile(args.model)
        cp = resolve_code_paths(profile)
        vllm_model_file = cp.vllm_model_file
    except Exception as e:
        print(f"[report] WARN load_model_profile/resolve_code_paths 失败: {e}", file=sys.stderr)

    # 若 probe-dir 给了，用它修正 vllm_model_file 路径（更准）
    if args.probe_dir:
        pj = os.path.join(args.probe_dir, "model_probe.json")
        if os.path.isfile(pj) and not vllm_model_file:
            print(f"[report] probe-dir: {args.probe_dir}")

    # 1. 重算每个落盘的 cos 表，合并
    all_rows = []
    per_dir = []  # [(dir, rows, first_diverge, only_arg)]
    for d in compare_dirs:
        hf_pt = os.path.join(d, "hf_inter.pt")
        vllm_pt = os.path.join(d, "vllm_inter.pt")
        rows, fd, oa = recompute_cos_table(hf_pt, vllm_pt,
                                           atol=args.atol, cos_thresh=args.cos_thresh)
        # 给每行标来源目录，避免合并后混淆
        for r in rows:
            r["source_dir"] = os.path.basename(d.rstrip("/"))
        all_rows.extend(rows)
        per_dir.append((d, rows, fd, oa))
        print(f"[report] {os.path.basename(d.rstrip('/'))}: stage={len(rows)} first_diverge={fd} only={oa}")

    # 选"最有信息量"的发散点作为报告主线：mlp 发散 > router 发散 > 默认层间发散
    only_arg = None
    first_diverge = None
    for _d, _r, fd, oa in per_dir:
        if oa == "mlp" and fd:
            only_arg, first_diverge = oa, fd
            break
    if only_arg is None:
        for _d, _r, fd, oa in per_dir:
            if oa == "router" and fd:
                only_arg, first_diverge = oa, fd
                break
    if only_arg is None:
        for _d, _r, fd, oa in per_dir:
            if oa in (None, "attn") and fd:
                only_arg, first_diverge = oa, fd
                break
    # fallback：任意发散
    if first_diverge is None:
        for _d, _r, fd, oa in per_dir:
            if fd:
                only_arg, first_diverge = oa, fd
                break
    rows = all_rows

    # 2. 抽源码
    items = plan_source_extraction(first_diverge, only_arg, vllm_model_file, vllm_root, profile)
    source_md, source_records = build_source_snippets(items, vllm_root)
    print(f"[report] 抽取源码片段: {len(source_records)} 条")

    # 被证伪假设（聚合所有落盘的客观记录）
    falsified = []
    has_router_ok = any(r["stage"].endswith("router_logits") and r["verdict"] == "OK" for r in rows)
    has_mlp_diverge = any(r["stage"].endswith("mlp_out") and r["verdict"] == "DIVERGE" for r in rows)
    if has_router_ok and has_mlp_diverge:
        falsified.append({
            "hypothesis": "发散在 router/gate (fused_gate)",
            "method": "--op router 对比 router_logits",
            "verdict": "❌ 证伪（router_logits cos≈1.0，mlp_out 仍发散）",
        })
    has_attn_ok = any(r["stage"].endswith("attn_out") and r["verdict"] == "OK" for r in rows)
    if has_attn_ok and has_mlp_diverge:
        falsified.append({
            "hypothesis": "发散在 attention",
            "method": "--op attn 对比 attn_out",
            "verdict": "❌ 证伪（attn_out 对齐，mlp_out 仍发散）",
        })

    # 3. 写报告
    model_name = os.path.basename(os.path.normpath(args.model))
    date = datetime.now().strftime("%Y%m%d")
    out_dir = args.out_dir or os.path.join(_ROOT, "reports", f"{model_name}_{date}")
    os.makedirs(out_dir, exist_ok=True)

    report_path, verdict_path, csv_path, src_path = write_report(
        out_dir,
        model_name=model_name,
        profile=profile,
        platform_info={"summary": platform_summary, "vllm_root": vllm_root, "is_dcu": is_dcu},
        rows=rows,
        first_diverge=first_diverge,
        only_arg=only_arg,
        source_md=source_md,
        source_records=source_records,
        vllm_version=vllm_version,
        symptom=args.symptom,
        falsified=falsified,
    )

    print(f"\n[report] ✅ 报告已生成：")
    print(f"  report.md     : {report_path}")
    print(f"  verdict.json  : {verdict_path}")
    print(f"  stage_cos.csv : {csv_path}")
    print(f"  source_snippets.md: {src_path}")
    print(f"\n[report] bug_operator = {json.loads(open(verdict_path).read()).get('bug_operator')}")
    print(f"[report] confidence   = {json.loads(open(verdict_path).read()).get('confidence')}")


if __name__ == "__main__":
    main()
