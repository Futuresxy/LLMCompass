#!/usr/bin/env python3
"""Published path rates and explicit analytical feed budgets; no hardware run.

Run with Python 3.9+. Only device_feed_budget.json next to this script is
written. Values marked unknown are deliberately not inferred from capacity.
"""
import hashlib
import json
from pathlib import Path


def budget(m, n, k, a_bytes, b_bytes, peak_flops_per_cycle):
    cycles = 2 * m * n * k / peak_flops_per_cycle
    a, b = m * k * a_bytes, k * n * b_bytes
    return {"m": m, "n": n, "k": k,
            "peak_flops_per_cycle": peak_flops_per_cycle,
            "ideal_compute_cycles": cycles,
            "a_payload_bytes": a, "b_payload_bytes": b,
            "a_required_bytes_per_cycle": a / cycles,
            "b_required_bytes_per_cycle": b / cycles,
            "ab_required_bytes_per_cycle": (a + b) / cycles,
            "fp32_accumulator_bytes": 4 * m * n,
            "equivalent_fp32_bytes_per_cycle_per_direction_if_one_update": 4 * m * n / cycles}


def main():
    sources = {
        "ascend_2201": "https://asc.gitcode.com/guide/programming_guide/advanced_programming/hardware_implementation/architecture_spec/npu_arch_2201.html",
        "ascend_3510": "https://asc.gitcode.com/guide/programming_guide/advanced_programming/hardware_implementation/architecture_spec/npu_arch_3510.html",
        "ascend_mmad": "https://asc.gitcode.com/api/SIMD-API/c_api/cube_compute/mmad_compute/asc_mmad.html",
        "ascend_counters": "https://github.com/Ascend/msopprof/blob/master/docs/en/user_guide/msopprof_performance_data.md",
        "flashattention4": "https://arxiv.org/html/2603.05451v1",
        "ptx": "https://docs.nvidia.com/cuda/parallel-thread-execution/",
        "blackwell_ultra": "https://developer.nvidia.com/blog/inside-nvidia-blackwell-ultra-the-chip-powering-the-ai-factory-era/",
    }
    public_paths = [
        {"architecture": "Ascend NPU_ARCH 2201 (A2/A3)", "scope": "per AIC",
         "capacity_bytes": {"L0A": 65536, "L0B": 65536, "L0C": 131072},
         "path_bytes_per_cycle": {"L1_to_L0A": 256, "L1_to_L0B": 128,
                                   "L0A_to_Cube": None, "L0B_to_Cube": None,
                                   "Cube_L0C_read": None, "Cube_L0C_write": None,
                                   "L0C_to_Fixpipe": None},
         "source": sources["ascend_2201"]},
        {"architecture": "Ascend NPU_ARCH 3510 (950-family profile must match device)",
         "scope": "per AIC",
         "capacity_bytes": {"L0A": 65536, "L0B": 65536, "L0C": 262144},
         "path_bytes_per_cycle": {"L1_to_L0A": 256, "L1_to_L0B": 256,
                                   "L0C_to_UB_via_Fixpipe": 128,
                                   "L0C_to_L1_via_Fixpipe": 128,
                                   "L0A_to_Cube": None, "L0B_to_Cube": None,
                                   "Cube_L0C_read": None, "Cube_L0C_write": None},
         "source": sources["ascend_3510"]},
        {"architecture": "Hopper H100", "scope": "per SM",
         "capacity_bytes": {"TMEM": 0},
         "path_bytes_per_cycle": {"SMEM_read_reference": 128},
         "evidence_kind": "Research microbenchmark reference, not a vendor guarantee for all paths",
         "source": sources["flashattention4"]},
        {"architecture": "Blackwell B200", "scope": "per SM",
         "capacity_bytes": {"TMEM": 262144},
         "path_bytes_per_cycle": {"SMEM_read_reference": 128,
                                   "TMEM_to_register": None, "register_to_TMEM": None,
                                   "TMEM_to_MMA": None, "MMA_to_TMEM": None,
                                   "SMEM_to_TMEM": None},
         "evidence_kind": "SMEM measured in cited research; no single verified TMEM peak imported",
         "source": sources["flashattention4"]},
        {"architecture": "Blackwell Ultra B300", "scope": "per SM",
         "capacity_bytes": {"TMEM": 262144},
         "path_bytes_per_cycle": {"SMEM_read": None, "TMEM_paths": None},
         "sensitivity_baseline_not_measurement": {"SMEM_read_bytes_per_cycle": 128},
         "source": sources["blackwell_ultra"]},
    ]
    # Reference FP16 Cube: one 16x16x16 matmul per issue, 8192 FLOPs.
    # The 3510 architecture page states this throughput explicitly.
    npu = []
    for m, n, k in [(16, 16, 16), (128, 128, 64), (128, 256, 64)]:
        row = budget(m, n, k, 2, 2, 8192)
        for arch, ba, bb in [("2201", 256, 128), ("3510", 256, 256)]:
            ta = row["a_payload_bytes"] / ba
            tb = row["b_payload_bytes"] / bb
            row[arch + "_preload"] = {
                "a_service_cycles": ta, "b_service_cycles": tb,
                "parallel_independent_lower_bound_cycles": max(ta, tb),
                "conservative_serial_service_cycles": ta + tb,
                "serial_budget_fits_compute_window": ta + tb <= row["ideal_compute_cycles"],
                "warning": "Engine arbitration/latency, conversion and RAW hazards are not simulated"}
        npu.append(row)
    assert npu[1]["ideal_compute_cycles"] == 256
    assert npu[1]["2201_preload"]["conservative_serial_service_cycles"] == 192
    assert npu[0]["2201_preload"]["conservative_serial_service_cycles"] == 6
    gpu = []
    for name, m, n, k, size, rate in [
        ("H100 BF16 example", 64, 128, 64, 2, 4096),
        ("B200 BF16 example", 128, 128, 64, 2, 8192),
        ("B200 FP8 normalized example", 128, 128, 64, 1, 16384),
        ("B200 FP4 1:2:4 packed-payload example", 128, 128, 64, .5, 32768),
        ("Hypothetical FP4 1:2:6", 128, 128, 64, .5, 49152),
        ("Hypothetical FP4 1:2:8", 128, 128, 64, .5, 65536),
    ]:
        row = budget(m, n, k, size, size, rate)
        row.update(name=name, smem_read_reference_bytes_per_cycle=128,
                   read_service_cycles=(row["a_payload_bytes"] + row["b_payload_bytes"]) / 128,
                   smem_budget_fraction=row["ab_required_bytes_per_cycle"] / 128,
                   maximum_compute_utilization_from_this_smem_bound=min(1, 128 / row["ab_required_bytes_per_cycle"]),
                   scope="Each A/B value read once at the modeled boundary; pure payload; no extra reuse, metadata or other consumers")
        gpu.append(row)
    assert gpu[0]["ab_required_bytes_per_cycle"] == 96
    assert gpu[1]["ab_required_bytes_per_cycle"] == 128
    assert gpu[4]["ab_required_bytes_per_cycle"] == 192
    assert gpu[5]["ab_required_bytes_per_cycle"] == 256
    drains = []
    for k in [64, 256, 4096]:
        row = budget(128, 128, k, 2, 2, 8192)
        row.update(output_path_bytes_per_cycle=128,
                   fp32_final_drain_service_cycles=row["fp32_accumulator_bytes"] / 128,
                   final_output_rate_bytes_per_compute_cycle=row["fp32_accumulator_bytes"] / row["ideal_compute_cycles"],
                   scope="Final FP32 output only; full K kept in accumulator; output route independent and overlap assumed")
        drains.append(row)
    assert [x["final_output_rate_bytes_per_compute_cycle"] for x in drains] == [256, 64, 4]
    result = {"status": "passed", "date": "2026-10-29", "sources": sources,
              "kind": "Published path references plus analytical estimates; no device benchmark",
              "unknown_policy": "null means no verified numeric peak for this exact path; capacity and row width are not bandwidth",
              "units": "Byte/cycle per AIC or SM; FLOP counts multiply and add separately; GHz times Byte/cycle gives decimal GB/s",
              "public_paths": public_paths, "ascend_fp16_reference_budgets": npu,
              "nvidia_feed_budgets": gpu, "ascend_3510_final_output_budgets": drains,
              "assumptions": ["Ideal full-tile compute at reference peak, not measured instruction latency",
                              "No bank conflicts, startup, synchronization, pipeline dependency, scale metadata or HBM limit",
                              "MMA updates do not prove physical TMEM read/write transaction counts",
                              "A 512-byte L0 row or TMEM column does not imply 512 bytes/cycle",
                              "Do not transfer a 910/A2/A3/950 or B200/B300 profile without target verification"],
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    output = Path(__file__).with_suffix('.json')
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print('passed: ' + str(output))


if __name__ == '__main__':
    main()
