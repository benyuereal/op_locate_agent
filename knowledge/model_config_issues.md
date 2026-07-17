# Model Config Issues — 模型 config 缺失字段坑

> 某些模型的 modeling 代码读取 config.json 未定义的字段，直接 `from_pretrained` 会
> AttributeError。本文件记录已知坑与补丁。补丁逻辑在 `lib/config_patch.py`。

---

## BailingMoeV2 — 已确认

- **model_type**: `bailing_moe_v2` / `bailing_moe`
- **现象**: `AutoModelForCausalLM.from_pretrained` 报
  `AttributeError: 'BailingMoeV2Config' object has no attribute 'num_nextn_predict_layers'`
- **根因**: modeling 代码（`modeling_bailing_moe_v2.py`）读 MTP/NextN 相关字段，
  但 config.json 未定义。
- **修复**: 加载前补默认值：
  ```python
  from lib import config_patch
  cfg = AutoConfig.from_pretrained(path, trust_remote_code=True, local_files_only=True)
  config_patch.patch_config(cfg)   # 补 add_router_probs/mtp_loss_scaling_factor/num_nextn_predict_layers/n_layers/n_positions
  model = AutoModelForCausalLM.from_pretrained(path, config=cfg, ...)
  ```
  其中 `num_nextn_predict_layers=0` → 不建 MTP 层，只跑 32 层主模型。
- **验证**: AntAngelMed, 2026-07-17, `examples/quickstart_antangelmed.py` HF 分支跑通。
- **状态**: 已确认

---

## 通用提醒

- `lib/config_patch.patch_config(cfg)` 按 model_type 分派，**未知 model_type 不动**
  （不越权改 config）。新模型遇到类似 AttributeError，先确认是 config 缺字段，
  再往 `_PATCH_DEFAULTS` 加条目。
- 补字段前先判断 `not hasattr(config, k)`，避免覆盖用户 config 已有值。
