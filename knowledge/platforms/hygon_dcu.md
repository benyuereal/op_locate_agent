# Platform: Hygon DCU 家族

> Hygon DCU（海光协处理器）环境必查。ROCm/HIP 栈，但 device name 仍报 "cuda"。
> 本知识文件覆盖整个 DCU 家族；具体型号见下表。

## 硬件型号 → gfx 架构映射

| 型号 (Marketing Name) | gfx 架构 | 说明 |
|---|---|---|
| BW100 | gfx936 | 当前 AntAngelMed 调查所用机型 |
| BW150 | gfx936 | |
| BW1000 | gfx936 | |
| K100AI | gfx938 | |
| BW1100 | gfx938 | |

> **重要**：gfx936 和 gfx938 是**不同架构**，kernel 兼容性、aiter 支持度可能不同。
> 排查时必须先确认当前机器的 gfx 型号，不要假设。

## 如何探测当前硬件

最可靠：`rocminfo`（DCU 上在 `/opt/dtk/bin/rocminfo`）。

```bash
# 提取 GPU 的 gfx 架构与 Marketing Name
/opt/dtk/bin/rocminfo 2>&1 | grep -iE "^  Name:|^  Marketing Name:" | grep -A1 -iE "gfx" 
# 或直接看 gfx
/opt/dtk/bin/rocminfo 2>&1 | grep -iE "gfx[0-9]+|BW[0-9]+|K100"
```

torch 侧：
```python
import torch
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))   # 如 "BW100"
```

DTK 环境：
```bash
hipconfig                  # HIP 版本（本机 6.3.26113）
echo $ROCM_PATH            # 通常 /opt/dtk
```

> `rocm-info` 命令在 DCU 上通常不在 PATH，用 `/opt/dtk/bin/rocminfo` 代替。

## 平台身份（所有 DCU 共性）

- **软件栈**: ROCm/HIP（DTK，通常装在 `/opt/dtk`），非 CUDA
- **vLLM 平台判定**: `current_platform` 是 `RocmPlatform`
  - `is_rocm() = True`
  - `is_cuda() = False`  ← **关键**：很多 vLLM 代码用 `is_cuda()` 守卫，DCU 上不进入
- **device name**: 仍是 `"cuda"`（vLLM/torch 在 ROCm 下用 cuda 命名）

## 关键影响：is_cuda()=False 改变调度

vLLM 多处用 `current_platform.is_cuda()` 做平台守卫。DCU 上 `is_cuda()=False`，导致：

| 代码位置 | 守卫 | DCU 上的结果 |
|---|---|---|
| `grouped_topk_router.py:82` | `if (... and current_platform.is_cuda() ...)` | **不进入** fused `grouped_topk` 分支 |
| `use_fused_gate` 路径 | 不依赖 is_cuda | 仍走 `ops.moe_fused_gate` |

> 这就是为什么 `ops.grouped_topk` 在 DCU 上**从未被调用**——它被 is_cuda 守卫挡在外面。
> 真正执行的是 `ops.moe_fused_gate`（经 `use_fused_gate=True`，与 is_cuda 无关）。
> 详见 `moe_known_issues.md` 的"易混淆点"。

## 设备可见性

- **环境变量**: `HIP_VISIBLE_DEVICES`（不是 `CUDA_VISIBLE_DEVICES`）
- **device name**: 仍是 `"cuda"`
- 多卡环境注意避开在线服务占用的卡。本 BW100 机器有 8 卡，在线服务常占 0,1,6,7；空闲 2,3 可调试。

## aiter 状态（gfx936 / BW100 实测）

- `rocm_aiter_ops.is_fused_moe_enabled() = False`
- aiter fused MoE 加速路径**未启用**，router 走 Python `grouped_topk`。
- 未深挖原因（aiter 初始化？DCU 兼容？gfx938 上可能不同，需单独探测）。

## MoE 配置缺失（性能警告，非性能问题）

vLLM 启动警告（BW100 示例）：
```
Using default MoE config. Config file not found at
.../fused_moe/configs/E=256,N=256,device_name=gfx936_64cu_nn.json
```
- **影响**: 仅性能（fused kernel 用默认配置），**不影响正确性**。
- AntAngelMed 调查中一度怀疑此为精度问题根因，**已证伪**（fused vs native 结果一致）。
- gfx938 机型配置文件名不同，但同理：缺失只影响性能。

## 推荐调试环境变量

| 变量 | 值 | 作用 |
|---|---|---|
| `HIP_VISIBLE_DEVICES` | `2` 或 `3` | 选空闲卡，避免抢在线服务 |
| `VLLM_ENABLE_MOE_FUSED_GATE` | `0` | **绕过 fused_gate bug**（gfx936 MoE sigmoid+bias 用）；正确但慢 |
| `enforce_eager` | True | 禁用 CUDA graph，便于 hook |
| `gpu_memory_utilization` | 0.9 | 调试时适当调低 |

> **注意**: `VLLM_ENABLE_MOE_FUSED_GATE=0` 是当前已验证的 gfx936 正确性修复，代价是速度。
> gfx938 机型上 fused_gate 是否有同样 bug **未验证**——不能假设，需单独探测。

## 重要：不要把单一案例当普适

> AntAngelMed (BailingMoeV2) 在 gfx936 上的 `ops.moe_fused_gate` 选错专家问题，
> 是**一个具体案例**，不代表其他模型/其他场景/其他 gfx 都是这个问题。
> 精度问题的根因可能是：router 算子、attention、norm、量化、dtype 转换、
> 模型加载、tokenizer……每个案例都要独立用运行时探针定位，不要预判。
