"""Read-only SCALE-Sim 2.0.2 component diagnostics; writes only explicit JSON output.

Run from LLMCompass root:
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/inspect_scalesim.py \
    --output /tmp/llmcompass-scalesim-inspect.json
No SCALE-Sim source, LLMCompass source or lookup CSV is changed.
"""
import argparse
import hashlib
import importlib.metadata
import inspect
import json
import math
from pathlib import Path

import numpy as np
from scalesim.compute.systolic_compute_os import systolic_compute_os
from scalesim.memory.double_buffered_scratchpad_mem import double_buffered_scratchpad


class Config:
    def get_array_dims(self):
        return 16, 16


def simulate(M, N, K, calc=True, bw=16, capacity=1048576):
    comp = systolic_compute_os()
    comp.set_params(
        config_obj=Config(),
        ifmap_op_mat=np.arange(M * K).reshape(M, K),
        filter_op_mat=np.arange(K * N).reshape(K, N) + 10000000,
        ofmap_op_mat=np.arange(M * N).reshape(M, N) + 20000000,
    )
    pre = comp.get_prefetch_matrices()
    a, b, c = comp.get_demand_matrices()
    mem = double_buffered_scratchpad()
    mem.set_params(
        verbose=False,
        estimate_bandwidth_mode=calc,
        word_size=1,
        ifmap_buf_size_bytes=capacity,
        filter_buf_size_bytes=capacity,
        ofmap_buf_size_bytes=capacity,
        rd_buf_active_frac=0.5,
        wr_buf_active_frac=0.5,
        ifmap_backing_buf_bw=bw,
        filter_backing_buf_bw=bw,
        ofmap_backing_buf_bw=16 if calc else bw,
    )
    if not calc:
        mem.set_read_buf_prefetch_matrices(*pre)
    mem.service_memory_requests(a, b, c)
    cycles = mem.get_total_compute_cycles()
    return {
        "MNK": [M, N, K],
        "array_hw": [16, 16],
        "mode": "CALC" if calc else "USER",
        "capacity_bytes_per_operand": capacity,
        "word_bytes": 1,
        "backing_bw_elements_per_cycle": bw,
        "trace_rows": a.shape[0],
        "cycles": cycles,
        "stalls": mem.get_stall_cycles(),
        "overall_util_pct": 100 * M * N * K / (16 * 16 * cycles),
        "mapping_eff_pct": 100 * comp.get_avg_mapping_efficiency(),
        "compute_util_pct": 100 * comp.get_avg_compute_utilization(),
        "analytic_no_stall_last_timestamp": math.ceil(M / 16) * math.ceil(N / 16) * (K + 30) - 1,
        "input_reads": [comp.get_ifmap_requests(), comp.get_filter_requests()],
        "output_report_count": comp.get_ofmap_requests(),
        "actual_output_addresses": int(np.count_nonzero(c >= 0)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("/tmp/llmcompass-scalesim-inspect.json"))
    args = parser.parse_args()
    if importlib.metadata.version("scalesim") != "2.0.2":
        raise RuntimeError("This diagnostic is pinned to SCALE-Sim 2.0.2")
    samples = [simulate(*shape) for shape in [(32, 16, 256), (17, 19, 31), (8, 8, 16), (16, 16, 1)]]
    samples += [simulate(32, 16, 256, False, bw, 1024) for bw in [1, 16, 64]]
    source_hashes = {}
    for cls in (systolic_compute_os, double_buffered_scratchpad):
        path = Path(inspect.getsourcefile(cls))
        source_hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    result = {
        "scalesim_version": importlib.metadata.version("scalesim"),
        "source_sha256": source_hashes,
        "scope": "Original OS and scratchpad classes called in memory; no simulator/package/repository source modification; no RTL or silicon measurement.",
        "samples": samples,
        "observations": [
            "CALC sample cycles match final 0-based output service timestamps, trace_rows - 1; do not conflate timestamps with inclusive cycle counts.",
            "USER 1024-byte buffers and bw=64 reproduce negative reported stalls in this unmodified version; this is a diagnostic of an existing edge case, not a calibrated hardware result.",
            "OS reported output requests include extra shape-dependent counts; compare to valid output demand addresses before using this report for energy accounting.",
        ],
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
