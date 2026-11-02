# User Guide: How to Run a LLMCompass Simulation

## Multi-format Tensor Core precision-ratio exploration

The opt-in reference backend keeps a fixed M/N PE grid and changes the per-PE
dot length along K for the 16-bit, 8/6-bit, and 4-bit compute classes. It supports
FP16/BF16, MXFP8/6, MXFP4/NVFP4/HiF4 format descriptors. A profile such as
`{"base_dot_length": 16, "ratios": [1, 2, 6]}` selects dot lengths 16/32/96.
Attach `precision_profile` to `SystolicArray` or its JSON configuration to opt in;
existing devices keep the historical scalar backend.

```bash
.venv/bin/python scripts/study_precision_ratios.py \
  --config configs/precision_ratio_study.json \
  --ppa configs/precision_ppa_template.json \
  --output-dir artifacts/precision-ratios/my-run --trace
```

The bundled four shape proxies produce 112 format/profile/workload rows. They are
not actual model benchmarks. Replace `workloads[].ops` with target GEMMs and
supply `non_gemm_seconds` for the same workload scope. All GEMMs in one workload
use one fixed operand format. `count` repeats an operator; `batch_count` denotes
independent matrix products. An exporter connects the existing dense block:

```bash
.venv/bin/python scripts/export_transformer_gemms.py \
  --hidden 4096 --heads 32 --phase prefill --sequence 512 \
  --output artifacts/precision-ratios/legacy-prefill.json
.venv/bin/python scripts/study_precision_ratios.py \
  --config artifacts/precision-ratios/legacy-prefill.json \
  --output-dir artifacts/precision-ratios/legacy-prefill
```

`result.json` includes operator schedules, native SCALE-Sim provenance and source
hashes; `summary.csv` contains workload comparisons. Optional NPZ files contain
**vector tokens, not physical SRAM requests**. Current validation uses
SCALE-Sim 2.0.2 compute components; this change does not upgrade to v3.

The backend supports OS with one array step per dot/accumulator update and link
advance, ready operands/scales, and FP32 resident accumulation. It retains
SCALE-Sim skew/drain and reports inclusive allocated slots, K/MN/wave/pipeline
efficiencies. WS/IS require different mappings and are explicitly unsupported.
The operator latency is an ideal overlap envelope of this compute schedule and
compulsory HBM/L2 transfers. Local-memory stalls, scale/format conversion compute,
vendor timing and quantization accuracy remain outside this implementation.

PPA input is optional; unknown candidate area/power stays null. Relative TC area
and activity-energy proxies are not full-chip area or tokens/J. The legacy FPU
area function rejects precision profiles instead of silently reporting old area.
See the [architecture, scope and experiment design report](reports/precision-ratio-design/report.html).

## Reproducible CPU baseline

The repository now includes a small CPU-only baseline runner. It exercises the
original analytical, lookup-table, and actual SCALE-Sim fallback paths; a complete
GEMM mapping search; small Transformer prefill/decode blocks; the JSON hardware
adapter and area model; and GPT-3-sized roofline estimates. It does not run a
numerical LLM or measure GPU/NPU performance.

From the repository root, create an isolated Python 3.9 environment using `uv`
and install the pinned dependencies:

```bash
UV_PYTHON_INSTALL_DIR="$PWD/.runtime/python" uv venv --python 3.9 .venv
uv pip sync --python .venv/bin/python requirements-cpu.lock.txt \
  --index https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match
git submodule update --init --recursive
.venv/bin/python scripts/run_baseline.py
```

The tested baseline uses Python 3.9.25, CPU PyTorch 2.0.0, and SCALE-Sim 2.0.2.
`requirements-cpu.txt` lists direct dependencies; `requirements-cpu.lock.txt`
records all resolved versions. This environment preserves the historical
simulator interface; it does not claim SCALE-Sim v3 compatibility.

Each run creates a fresh `artifacts/llmcompass-baseline/<timestamp>/` directory
containing `result.json`, `manifest.json`, `run.log`, and copied lookup tables.
SCALE-Sim cache misses write into that run directory, including its `temp/`
directory. The original lookup tables and historical AE result files remain
unchanged. Use `--output-dir PATH` to select a new output directory. A nonzero exit
status indicates a failed check; `result.json` includes the error details.

For a separate inspection of the installed SCALE-Sim 2.0.2 OS compute and memory
components, run:

```bash
.venv/bin/python scripts/inspect_scalesim.py --output /tmp/scalesim-diagnostics.json
```

This diagnostic compares demand-trace length, cycle timestamps, mapping
efficiency, overall utilization, and buffer-limited execution. It also records
two existing 2.0.2 boundary behaviors: one USER-mode case reports negative stalls,
and the reported OS output-request count differs from the number of valid output
addresses. These are observations to resolve before reusing those reports for a
new bandwidth or energy model; they are not hardware validation results.

The built-in A100 performance profile and `configs/GA100.json` have different
core-count and buffer parameters. The runner labels these separately and retains
the existing modeling assumptions. Passing this baseline establishes execution
and internal consistency, not hardware accuracy or full paper reproduction.

## Step 1: Build a Hardware Configuration

Follow the [NVIDIA GA100 example](../configs/GA100.json). This is a 4-GA100 node connected with NVLinks.

### Explanations on the Knobs
Most of the attributes are self-explained:

```json
{
    "name": "NVIDIA A100(80GB)x4", 
    "device_count": 4, # how many devices in a node
    "interconnect": {
        "link": {
            "name": "NVLink3",
            "bandwidth_per_direction_byte": 25e9,
            "bandwidth_both_directions_byte": 50e9,
            "latency_second": 8.92e-6,
            "flit_size_byte": 16,
            "header_size_byte": 16,
            "max_payload_size_byte": 256
        },
        "link_count_per_device": 12,
        "topology": "FC" # currently support FC (fully-connected) and RING
    },
    "device": {
        "frequency_Hz": 1410e6,
        "compute_chiplet_count": 1,
        "compute_chiplet": {
            "physical_core_count": 128, # used for area model
            "core_count": 128, # used for performance model
            "process_node": "7nm", # currently support 7nm, 6nm, 5nm
            "core": {
                "sublane_count": 4,
                "systolic_array": {
                    "array_width": 16,
                    "array_height": 16,
                    "data_type": "fp16",
                    "mac_per_cycle": 1
                },
                "vector_unit": {
                    "vector_width": 32,
                    "flop_per_cycle": 4, # 32*4=128 flops per cycle per vector unit
                    "data_type": "fp16",
                    "int32_count": 16, # the number of int32 ALUs, used for area model
                    "fp16_count": 0,
                    "fp32_count": 16,
                    "fp64_count": 8
                },
                "register_file": {
                    "num_reg_files": 1,
                    "num_registers": 16384,
                    "register_bitwidth":32,
                    "num_rdwr_ports":4
                },
                "SRAM_KB": 192
            }
        },
        "memory_protocol": "HBM2e",
        "_memory_protocol_list": [
            "HBM2e",
            "DDR4",
            "DDR5",
            "PCIe4",
            "PCIe5"
        ],
        "io": {
            "process_node": "7nm",
            "global_buffer_MB": 48,
            "physical_global_buffer_MB": 48,
            "global_buffer_bandwidth_per_cycle_byte": 5120,
            "memory_channel_physical_count": 6, # used for area model
            "memory_channel_active_count": 5, # used for performance model
            "pin_count_per_channel": 1024,
            "bandwidth_per_pin_bit": 3.2e9
        },
        "memory": {
            "total_capacity_GB": 80
        }
    }
}

```

## Step 2: Build a LLM Computational Graph

Transformer blocks have been provided as in [`transformer.py`](../software_model/transformer.py), including Initial Computation (also called Prefill or Context stage) and Auto Regression (also called Decoding or Generation stage), with Tensor Parallelism support (automatically turned of if the system only has 1 device).

The user needs to provide these parameter:
* `d_model`: the hidden dimension, 12288 for GPT3
* `n_heads`: the number of heads, 96 for GPT3
* `device_count`: tensor parallelism
* `data_type`: `int8`, `fp16`, or `fp32`

### Build Your Own LLM

The user can also build their own computational graph following the [`transformer.py`](../software_model/transformer.py) example using provided operators: [`matmul`](../software_model/matmul.py), [`softmax`](../software_model/softmax.py), [`layernorm`](../software_model/layernorm.py), [`gelu`](../software_model/gelu.py), and [`allreduce`](../software_model/communication_primitives.py).

The user needs to define a new `class` by inheriting `Operator` the class and configure these fields:
* `__init__`: define the needed operators in the initial function
* `__call__`: build the computational graph. The shape of Tensors will be automatically calculated and used for simulation.
* `compile_and_simulate`: simulate all the operators and get the total latency as well as other runtimes.
* `roofline_model` (optional): a roofline model analysis.
* `run_on_gpu` (optional): run the computational graph on real-world GPUs with PyTorch.

## Step 3: Run a LLMCompass Simulation

First, read the hardware configuration and parse it to LLMCompass:
```python
from design_space_exploration.dse import template_to_system, read_architecture_template

specs = read_architecture_template("PATH/TO/YOUR/JSON")
system = template_to_system(specs)

```

Next, initiate and instantiate an LLM as in this example:
```python
model_auto_regression = TransformerBlockAutoRegressionTP(
        d_model=12288,
        n_heads=96,
        device_count=1,
        data_type=data_type_dict["fp16"],
    )
_ = model_auto_regression(
	Tensor([bs, 1, 12288], data_type_dict["fp16"]),
	seq_len,
)

```

Finally, run the simulation
```
auto_regression_latency_simulated = model_auto_regression.compile_and_simulate(
	system, "heuristic-GPU"
)
```

## CACTI：阵列本地操作数 SRAM 的带宽与代价模型

`cost_model/local_sram.py` 建模的是直接向计算阵列供数的 A/B 操作数 SRAM，
对应 L0A／L0B **这一类功能层次**。它不是对某款昇腾芯片物理实现的还原，也不把
原来的 `Core.SRAM_size` 或 GPU shared memory／L1 当作这里的存储。
模块作为独立 API 使用，没有把其面积自动叠加到旧成本模型，避免重复计数。

### 构建与运行

```bash
.venv/bin/python scripts/build_cacti.py
.venv/bin/python scripts/model_local_sram.py --plot
# 同一个已安装组织，原负载只有一半：验证带宽余量
.venv/bin/python scripts/model_local_sram.py --demand-scale 0.5 --plot
.venv/bin/python -m unittest discover -s tests -v
```

构建器使用官方 `HewlettPackard/cacti`，固定 revision
`1ffd8dfb10303d306ecd8d215320aea07651e878`。需要 git、make 和 C++ 编译器；
已有其他版本或有修改的目录不会被覆盖。默认源码在 `.runtime/cacti/`，
CACTI 调用缓存位于 `.runtime/cacti-cache/`。运行输出为
`artifacts/local-sram/<timestamp>/result.json`、`sweep.csv`，以及可选 `sweep.png`。
通过 `--config`、`--cacti-dir`、`--cache-dir`、`--output-dir` 可以调整路径。

每个缓存条目包含完整 cfg、CACTI stdout/stderr、原始 CSV 和归一化结果。
缓存键包含宏参数、配置、可执行文件、技术文件的 SHA-256 和源码 revision。
失败、超时、缺失输出和非法数值会被记录；不能只依赖 CACTI 的返回码，
因为某些非法配置会以状态 0 退出。

### 示例与可修改输入

`configs/local_sram_example.json` 是说明用的 22nm 设计点，不是厂商规格：

- A/B 各 32 KiB **总物理容量**、1 GHz、300 K。
- 每路两个独立物理 ping-pong slot，各 16 KiB。
- 已安装组织：每 slot 两个独立 256-bit、1RW SRAM 宏。
- 每路读供给为 64 Byte/cycle；后台写由另一个 slot 服务。
- BF16／FP8／FP4 吞吐由 256／512／1024 改为 256／512／2048 MAC/cycle。
- 原需求为每路读 64、写 16 Byte/cycle；只将 FP4 的速率乘二。
- 搜索每 slot 1／2／4／8 个 bank，64／128／256／512-bit 字宽，以及
  1RW 或 1R1W 端口。选择的是这个候选集中的最小可行面积，不是物理设计全局最优。

主要字段：

| 字段 | 单位与口径 |
|---|---|
| `capacity_bytes` / `capacity_basis` | `total` 为总物理容量；`per_slot` 明确表示每个 slot 容量 |
| `slots` | 1 或 2；不是吞吐倍数 |
| `layout` | `separate_slots` 为独立物理宏；`shared_macro` 为同一宏的地址分区 |
| `frequency_hz` | 首版假设阵列与本地 SRAM 同频；不建模跨时钟、多泵访问 |
| `bank_efficiency` | 0–1 的有效并行度折扣；默认无 bank 冲突，未自动推导地址冲突 |
| `live_tile_bytes` | 当前 tile 必须放入一个 slot |
| `installed` | 明确指定原 SRAM 组织；推荐用真实已有设计数据 |
| `baseline_headroom_factor` | 未指定 installed 时，为原需求预留倍率再选择 baseline |
| `baseline_demands` | 每个精度、每个操作数的物理读写 Byte/cycle，包含打包、补齐与元数据 |
| `baseline/target_throughput_mac_per_cycle` | 同一基准下的绝对吞吐；模式内复用、频率、数据表示须保持一致 |
| `refill_bytes_per_tile` / `compute_window_cycles` | 可选，检查后台填充是否能在计算窗口内完成 |
| `interface_area_mm2` / `interface_leakage_mw` | 已知的额外接口代价；默认 0，不代表实际硬件没有这部分成本 |

原需求和新需求分开。新需求仍能由已安装组织满足时，保持原组织，
`status=retained`、面积增量为 0；功率仍按新访问率计算。超过供给后才搜索升级；
无可行组织时返回 `infeasible`，不增加总容量、不伪造可实现功率。
所有互斥精度逐一验证，同一物理设计必须支持所有模式；不会把不同模式的读写峰值
相加，也不会在精度切换时更换一套 SRAM。`sweep` 只改变指定模式，其他模式保持基线需求。

### 双缓冲、端口与时序

每次 CACTI 调用使用 tagless `ram`、UCA bank count=1、无 ECC、line 等于
输出 word。外层 `banks` 表示有独立接口且可并行访问的物理宏，不能把 CACTI
内部 bank 数无条件当成外部并行带宽。

独立 ping-pong 时：

```text
总容量 = slots × banks_per_slot × macro_capacity
面积与泄漏 = 所有已部署宏的总和 + 显式接口项
计算读带宽 = 当前读 slot 的 bank 数 × 每 bank 可持续读带宽
后台填充 = 另一个 slot 的写接口
```

固定总容量 C、双缓冲时每个 slot 只有 C/2。双缓冲不会让读带宽乘二。
如果两个 slot 只是同一个 SRAM 的逻辑分区，读写仍要共享其端口。
一个 1RW 口不能同时满速读取和写入；1R1W 或独立物理 slot 可以提供对应并行性。

同步发射间隔使用 `ceil(CACTI cycle_time × frequency)`，至少一拍。
`access_time` 单独转换为首访问延迟，不能与 `cycle_time` 混用。
后台填充检查使用扣除首访问延迟后的窗口，是保守的速率／deadline 近似，
不是逐请求离散调度。用户还应提供足够的上一级搬运带宽；本模块不模拟 L1/HBM。

### 功耗、能量与统计边界

CACTI 返回的是每次宏访问能量和漏电，工作负载功率由 wrapper 计算：

```text
read_accesses/cycle  = physical_read_bytes/cycle / word_bytes
write_accesses/cycle = physical_write_bytes/cycle / word_bytes
P_dynamic[W] = frequency[Hz] × (read_accesses × E_read[nJ]
                               + write_accesses × E_write[nJ]) × 1e-9
P_leakage[W] = sum(all macro subthreshold + gate leakage[mW]) × 1e-3
```

假设完整物理 word 的访问可合并，不把阵列利用率当成电路翻转率。
无可行供给时不会输出该速率下的“已实现功率”。
`workload_energy()` 接受实际宏事务次数与运行时间，计算动态／泄漏能量。
对同一个 GEMM，算力提高可能使相同字节在更短时间内读完；瞬时功率可以增加，
总动态访问能量不一定同比增加。

报告面积含 CACTI 宏内部外围电路。独立宏之间新增的 mux、布线、控制、时钟动态功耗，
以及上游存储和 Tensor Core 本体，都不包含在默认宏总和中，需提供额外模型或校准。
更高带宽改变宏数量／字宽／端口时，不能把默认 0 的外部开销误认为已经完成物理布线评估。

### Python API

```python
from pathlib import Path
from cost_model.cacti import CactiRunner
from cost_model.local_sram import BufferSpec, Organization, Demand, plan_upgrade

runner = CactiRunner(Path('.runtime/cacti'), Path('.runtime/cacti-cache'))
spec = BufferSpec('A', capacity_bytes=32768, frequency_hz=1e9, slots=2)
installed = Organization(banks=2, width_bits=256)
choices = [Organization(banks=b, width_bits=w)
           for b in (1, 2, 4, 8) for w in (64, 128, 256, 512)]
result = plan_upgrade(spec, Demand(32, 8), Demand(64, 16), choices, runner,
                      installed=installed)
assert result['status'] == 'retained'
assert result['delta_area_mm2'] == 0
```

`plan_upgrade` 也接受 `{mode: Demand(...)}`，用于一套物理 SRAM 的多模式共同选型。
运行前必须明确各 mode 的物理访存需求；仅有 Tensor Core 峰值比例不能推导所有
layout、复用和 padding 开销。

### 工艺与验证边界

当前适配器支持官方实现中可用的 22／32／45／65／90nm 参数点。
12nm 不会被静默替换成 22nm，也不会按节点平方缩放；需要另行接入有来源的
技术参数或校准模型。CACTI 通用技术表不是某厂商 SRAM compiler／PDK 的 sign-off 数据。

测试包含数学约束及受控的失败进程／解析用例；其中 fake CACTI 数值明确标记为
测试数据。完整示例另外运行真实 CACTI。参考：
[官方 CACTI](https://github.com/HewlettPackard/cacti)、
[固定版本配置](https://github.com/HewlettPackard/cacti/blob/1ffd8dfb10303d306ecd8d215320aea07651e878/cache.cfg)、
[工艺分支](https://github.com/HewlettPackard/cacti/blob/1ffd8dfb10303d306ecd8d215320aea07651e878/parameter.cc)。

## 从 H16Q32E64 点积 PE 推导 SRAM lane 需求

**数据流适用范围更新（2026-10-29）：** 以下实现固定采用 M/N 空间映射、A/B 每点积组供数。
`resident` / `l0c_rmw` 是累加存储策略，不能分别视为完整 OS / WS 数据流。
驻留操作数多周期预装与 OS 结果排空是否允许保持 SRAM lane 不变，必须检查复用窗口、
端口、队列与依赖。新的调研、K/M/N 扩展对照及证据在
[点积阵列数据流报告](reports/dot-product-dataflow-2026-10-29/report.html)。
此前 CACTI 数字仍只对应原有流量假设；本次调研未替换该实现。

`cost_model/dot_product_sram.py` 在上述 CACTI 宏模型之前增加一个结构化流量前端：
从 PE 阵列形状、各精度的点积长度和累加策略生成 A/B/C 访问事件，再将每个事件
量化为候选 SRAM 宏的完整 word 事务。`scripts/model_dot_product_sram.py` 负责
比较同容量候选、检查已有组织是否仍够用，并输出每种精度的访存和成本。

这里的 L0A-like／L0B-like／L0C-like 表示直接连接计算单元的操作数／累加存储层次。
[昇腾 Cube 官方说明](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/ascendcopapi/docs/en/api/SIMD-API/basic_api/cube_compute_ISASI/overview/matrix_computation_unit.md)
明确区分左矩阵 L0A、右矩阵 L0B、初始化累加值与计算结果 L0C。
本实现借用这种功能划分，**所有 lane 数、映射、端口和缓冲组织都是自定义设计空间，
不是对任何真实昇腾芯片 bank 结构的复原**。H16Q32E64 同样是本研究输入的 PE 定义，
不能据此断言某款 NVIDIA／昇腾芯片采用该内部排布。

### 配置、运行与结果文件

```bash
.venv/bin/python scripts/build_cacti.py
.venv/bin/python scripts/model_dot_product_sram.py
.venv/bin/python scripts/model_dot_product_sram.py --config configs/dot_product_sram_example.json --output-dir artifacts/dot-product-example
.venv/bin/python -m unittest discover -s tests -p 'test_dot_product_sram.py' -v
```

`--output-dir` 必须是新目录或空目录；不指定时使用
`artifacts/local-sram/dot-product-<timestamp>/`。`--cacti-dir`、`--cache-dir`
复用前述 CACTI 构建与缓存。输出包括：

| 文件 | 用途 |
|---|---|
| `input.json` | 本次运行的完整配置快照 |
| `result.json` | 代码 SHA-256、CACTI 宏目录、逻辑事件、逐模式物理事务、不可行原因和选型结果 |
| `logical_traffic.csv` | dot length、issue 窗口、有效字节、逻辑峰值／平均带宽 |
| `lane_cost.csv` | 最终组织、物理事务、lane 活动比例、面积、动态能量与功率、泄漏 |
| `failure.json` | 运行异常时保存错误；正常完成不会生成 |

程序按每个阵列形状、累加策略和 gating 策略，分别评估四种配比。
每个配比中的 BF16／FP8／FP4 是互斥模式：一套选定的物理 SRAM 必须支持全部模式，
不会把三种精度同时相加，也不会按精度临时更换 SRAM。原组织能满足新需求时标记
`retained`；否则在候选集中选最小可行面积组织，标记 `upgraded`。
`baseline_infeasible` 表示指定的原组织本身不满足基线；`infeasible` 表示候选集
不能支持目标。失败不会通过偷偷扩大容量来解决。最终存在不可行 study 时退出码为 2。

`configs/dot_product_sram_example.json` 的说明用设计点如下，均不是厂商实测规格：

- 阵列形状为 16×16 和 32×8；每个输出 tile 的 K=256，点积发射间隔 II=1。
- A/B 各 **128 KiB 总物理容量**，两组独立 ping-pong slot，每 slot 64 KiB。
- C 为 **64 KiB 总物理容量**，单 slot、同宏存储，配置独立 1R1W 端口。
- 阵列和全部 SRAM 同频 1 GHz，工艺参数点 22nm；沿用默认温度 300 K。
- 每个 slice 是一个独立 128-bit word 宏；A/B 初始每个 group 部署 2 个 slice。
- 每 group 候选 slice 数为 1／2／4／8，A/B 使用 1RW 宏；容量随宏数重新分割，
  总容量始终固定。C 初始 slice 数按一行 FP32 输出所需 word 数向上取 2 的幂。
- 同时评估 `resident`／`l0c_rmw` 和 `slice`／`full_group`；初始 C 为零，
  因而默认不读取已有 C。设置 `initial_c=true` 可加入已有 C 的读取。

最初的 A/B 各 64 KiB、C 16 KiB 示例会产生 256-Byte、128-bit word 的小宏，
当前固定 CACTI 版本对相关 1RW／1R1W 组织返回 `no valid data array`。
因此默认配置明确改为上述较大容量，以演示完整的形状／策略组合；原始失败记录保留。
这是公开修改示例输入，不是运行时自动扩容。每次实验内部的各配比和形状仍使用相同
总容量。**CACTI 搜索不到某个小宏组织不等于该 SRAM 硬件无法实现**；应保留为模型
适用范围限制，或接入有来源的 SRAM compiler／综合数据。

### 配比如何变成 A/B 带宽

用 H×W 表示物理 PE 行列数，h×w 表示该 tile 的有效行列数，L 表示某精度
每个 PE 一次发射的点积长度。A 沿 PE 行广播，B 沿 PE 列广播。一次完整发射的
逻辑需求为：

```text
A_bytes_per_issue = H × L × a_bits / 8
B_bytes_per_issue = W × L × b_bits / 8
peak_MAC_per_cycle = H × W × L / II
```

这里从 bit 转 Byte 必须除以 8。用户示例 `64 × 16 × 4` 是 4096 bit，
即 A 每次发射 512 Byte，已经只计算每行的一份 A，不能再除以广播复用次数。
B 同理。II>1 时，完整发射的长期平均需求还要除以 II，但单次发射的 burst 不变。

在 16×16、无 M/N 尾块、II=1 下，四种配置对应：

| 吞吐配比 BF16:FP8:FP4 | BF16／FP8／FP4 点积长度 | A 每完整 issue 的 BF16／FP8／FP4 字节 | B 每完整 issue 的 BF16／FP8／FP4 字节 |
|---|---|---|---|
| 1:2:4 | 16／32／64 | 512／512／512 | 512／512／512 |
| 1:2:6 | 16／32／96 | 512／512／768 | 512／512／768 |
| 1:2:8 | 16／32／128 | 512／512／1024 | 512／512／1024 |
| 1:4:8 | 16／64／128 | 512／1024／1024 | 512／1024／1024 |

改变 H/W 比例会改变两侧的压力。32×8 仍有 256 个 PE，基准模式每次发射
A=1024 Byte、B=256 Byte；FP4 点积长度增至 128 时变为 A=2048、B=512。
两侧不能只用一个总带宽数替代，否则会隐藏单侧供给瓶颈。
`ArrayTile.active_rows`／`active_cols` 支持 M/N 尾块，物理阵列和已部署宏保持不变，
只有对应的有效 group 和有效 C 数据减少；示例 runner 当前使用完整 M/N tile。

### K 尾块、计算利用率和带宽利用率

发射组数 Q=`ceil(K/L)`，末组只消费 `K-(Q-1)L` 个有效元素。模型逐组记录
有效 bit，之后才按 SRAM word 向上取整。因此完整 issue 的理论带宽、实际 peak、
tile 平均带宽以及物理访问带宽有明确区别。

例如 1:2:6 的 FP4、16×16、K=256、II=1，K 分成 96／96／64 三组。
A/B 每路逻辑读取为 768／768／512 Byte，总计 2048 Byte，平均为
2048/3≈682.67 Byte/cycle，而不是始终 768 Byte/cycle。
同精度、同 tile 的有效输入字节不因 L 增大而增加；更大的 L 主要将这些字节
压缩到更短的 issue 窗口中。更宽物理访问造成的 padding 则可能额外增加事务。

以下指标各有含义，不能互相代替：

| 指标 | 定义与限制 |
|---|---|
| `issue_window_compute_utilization` | `h×w×K / (H×W×Q×L)`；只反映 M/N/K 尾块的计算槽利用率，不含流水线或带宽停顿 |
| `read_payload_efficiency` | 有效读字节／物理读字节；衡量 word 和 lane padding 浪费 |
| `peak_read_lane_activity` | 单次最大实际访问 lane 数／已部署 lane 数 |
| `average_read_lane_activity` | 实际读事务／已部署 lane 数／issue 窗口周期；同时反映尾块、gating 和 II |
| `read_bandwidth_utilization` | 平均物理读 Byte/cycle／CACTI 组织的可持续读供给；写侧有对应指标 |

软件同时检查单次 burst 是否有足够 lane、每个固定 lane 的平均访问是否超过
端口可持续速率、共享 RW 口的读写总预算，以及所有模式的容量约束。
平均带宽低并不保证能在一次发射中提供足够数据。

### Slice gating、整体访问与固定地址映射

一个 group 对应一行 A、一列 B 或一行 C；每个 group 下部署若干独立 SRAM slice。
每次访问需要 `ceil(valid_bits_per_group / slice_bits)` 个完整宏事务。

`slice` 策略只访问所需的 slice；`full_group` 策略访问有效 group 下的全部已部署
slice。两者都保留全部物理面积和泄漏；未用 group 不发出事务。
这描述的是访问使能粒度，**不表示已经实现或计入了电源门控**。
把有效 512 Byte 除以 1024 Byte 总字宽所得的 0.5，不能直接视为“半次宏访问”。

例如 FP4 L=96 时，每个 A/B group 需要 96×4=384 bit，即 3 个 128-bit slice。
当前候选数按 2 的幂量化，实际须部署 4 个 slice。完整 issue 在 `slice` 策略下
访问 3/4 的 lane；BF16 L=16 只需 256 bit，在同一组织下访问 2/4。
在 `full_group` 策略下，两种模式每个有效 group 都访问 4 个 slice。
上述 K=256 的 E96 例子在 `slice` 策略下三次分别访问 3／3／2 个 slice，
平均 lane 活动比例为 8/12，峰值为 3/4。

当前地址映射是 `fixed_active_prefix`：每个 group 总是从第 0 个 slice 开始放置，
每个 K 组在被访问的 slice 中占一个 word，不跨 issue 合并，也不轮转到空闲 slice。
因此检查 **总 tile footprint** 仍不够，还必须检查每个宏中每个 slot 的 footprint。
报告同时给出 `physical_tile_footprint_bytes`、`per_macro_required_bytes_max` 和
`per_macro_slot_capacity_bytes`。在 `shared_macro` 双逻辑 slot 情况下，每 slot 只能
使用宏容量的一半；独立 slot 则每个宏本来就只属于一个 slot。

例如 A 总容量 32 KiB、双 slot、16 group×4 slice、128-bit word，独立 slot 中
每宏只有 256 Byte。BF16 L=16、K=512 时每行只用前两个 slice，但每个活跃宏要放
32×16=512 Byte。即使有效 tile 总量 16 KiB 恰好等于 slot 总容量，该固定映射仍
不可行。空闲 slice 不会自动借出容量；若研究轮转布局，应另行建立布局和互连代价模型。

### A/B 双缓冲与 FP32 C 的两种策略

A/B 的每个输入 tile 默认只填充一次，写入采用与读取相同的物理布局。
refill 事务均匀摊到 compute issue 窗口内，示例由另一个物理 slot 承接。
128 KiB 总容量分为两个 64 KiB slot，读带宽只由当前读 slot 提供；双缓冲不使读带宽翻倍。
当前前端只检查 refill 平均速率，不调度其到达时间，不验证上游 L1/HBM 的实际搬运能力
和 tile 切换 deadline。即使各项检查通过，也不是完整访存时序可行性的证明。

FP32 C 每次更新的有效数据量为 h×w×4 Byte，与点积长度 L 无关：

| 累加策略 | L0C 读取 | L0C 写入 | 常驻状态 |
|---|---|---|---|
| `resident` | `initial_c=true` 时读取初值一次，否则零次 | 所有 K 组完成后写回一次 | 额外 ACC 保持中间累加值；L0C 存一份输出 tile |
| `l0c_rmw` | Q−1 次；有初始 C 时为 Q 次 | Q 次 | 每次更新覆盖同一份 L0C 部分和 |

`resident` 在更大的 L 下平均写回带宽可能提高，因为归一化的 compute 窗口变短；
但写回总量和一次写回 burst 不变。16×16 的 FP32 tile 为 1024 Byte。
额外 ACC 的物理状态大小单独报告为 `resident_accumulator_state_bytes`，
**ACC 电路面积和能量没有算进 L0C 的 CACTI 宏面积**。

`l0c_rmw` 的单次 C 访问不随 L 成倍增加，Q 下降反而减少总访问次数。
容量仍只计算一份 C 状态，不能把 Q 次写入字节之和当作所需容量。
同一部分和存在读后更新依赖，不能直接套用 A/B 的独立读／写 ping-pong slot 来
绕过端口约束；当前模型拒绝这种组织。示例 C 采用单 slot、1R1W。
对于同一 issue 的 RMW，前端保守要求同一 lane 能接纳读和写，未实现将 1RW 端口
在 II>1 时分时调度；`resident` 初值读取和最终写回分别检查 burst，不要求二者同拍。

### Python API、功率口径与后续验证

```python
from cost_model.dot_product_sram import ArrayTile, PrecisionMode, generate_traffic

tile = ArrayTile(rows=16, cols=16, k=256,
                 accumulator_policy="resident", initial_c=False)
mode = PrecisionMode("fp4", dot_length=96, a_bits=4, b_bits=4)
traffic = generate_traffic(tile, mode)
assert traffic["compute_issue_cycles"] == 3
assert traffic["operands"]["A"]["useful_read_bytes"] == 2048
```

`generate_traffic()` 不调用 CACTI，输出逻辑事件；将 `characterize()` 得到的宏组织和
该 traffic 交给 `evaluate_lane(spec, characterization, traffic, operand,
slices_per_group, physical_groups, gating)`，才会获得候选相关的物理事务、约束检查
和动态能量。`physical_groups×slices_per_group` 必须等于组织的 `banks`。
API 允许 A/B 位宽不同；当前 JSON runner 使用对称位宽。

对每个精度，动态能量先由整数事务数计算：

```text
E_dynamic[J] = (N_read × E_macro_read[nJ] + N_write × E_macro_write[nJ]) × 1e-9
T_issue[s] = compute_issue_cycles / frequency_hz
P_dynamic[W] = E_dynamic / T_issue
P_total[W] = P_dynamic + leakage_of_all_deployed_macros
```

`modeled_dynamic_energy_j` 是该模型所请求事务的能量；不可行候选的
`dynamic_power_w`／`total_power_w` 返回空值，避免当成已实现功率。
减少未用 slice 的访问可以减少动态能量，但不会减少已部署宏的面积和泄漏。
同精度同 GEMM 在不同算力下可以有相同有效字节、相近动态访问能量和更高窗口平均功率。
若要计算运行总能量，还需真实或时序模型得到的持续时间，以计入 `P_leakage×T`。

`compute_issue_cycles=ceil(K/L)×II` **不是实际 GEMM 周期数**：它不含阵列填充／排空、
dot pipeline 延迟、累加反馈延迟、SRAM 首访问、传输等待和最终写回延迟。
C 的事件时间也是结构化发射阶段，不是经验证的 RTL 因果时序。CACTI 的 cycle time
用于供给约束，其 access time 另列；当前前端未将这些时间与计算图拼成完整调度。
动态功率按该 issue 窗口归一化，不应标成芯片实测或完整 kernel 平均功率。

验证应分为三层：用数学测试核对广播、尾块、word 量化、端口和容量；用
[官方 CACTI](https://github.com/HewlettPackard/cacti) 的原始 cfg／输出核对宏成本；
再用所设计阵列的 RTL／综合／SRAM compiler 数据校准事务能量、物理面积和端口时序。
现有 12nm／22nm 自有数据可作为同节点校准依据，但不能未经校准地跨节点混加。
本前端没有修改 SCALE-Sim，也没有把自定义 dot-product PE 假装成普通标量 MAC PE；
完整计算周期仍需后续接入能够表达该 PE、累加策略和供数时序的模拟器。

### 已保存的集成验证记录

本次已完成 57 项单元测试和两组真实 CACTI 实验，各含 32 个 shape／配比／策略
study。原始记录可用于复核，而非硬件正确性验证：

- [默认配置结果](../artifacts/local-sram/dot-product-128k-verified/result.json)：
  A/B 每 group 初始 2 slice，超过供给后搜索升级。
- [有带宽余量的配置结果](../artifacts/local-sram/dot-product-headroom-verified/result.json)：
  A/B 每 group 初始 4 slice，32 个 study 的 96 个 buffer 均保留原组织，面积增量为零。
- [验证清单](../artifacts/local-sram/dot-product-128k-verified/validation.json)：
  核对容量固定、多精度共享面积／泄漏、E96 尾块、C 策略以及事务使能粒度。

默认实验中，16×16 的 A 单路由每 group 2 slice 升级到 4 slice，预测宏面积由
0.21438592 增至 0.31006592 mm²；总容量始终为 128 KiB。这只是此工艺参数、容量、
候选宏结构下的 CACTI 结果。增加宏数同时改变了每宏容量和每次访问能量，不能由
吞吐配比直接断言功率单调增加。例如 E96 的尾块和宏组织变化会同时影响该 issue
窗口内的平均功率，应结合 CSV 中的事务数、访问能量和窗口长度逐项解释。
这些验证证明前端与 CACTI 的计算口径一致，不替代 RTL、PDK 或芯片测量校准。
