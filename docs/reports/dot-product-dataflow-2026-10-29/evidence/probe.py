#!/usr/bin/env python3
"""Read-only SCALE-Sim 2.0.2 scalar-PE demand-matrix probe.

Run from the repository root:
  .venv/bin/python -B docs/reports/dot-product-dataflow-2026-10-29/evidence/probe.py \
    > docs/reports/dot-product-dataflow-2026-10-29/evidence/probe.json

This constructs operand ADDRESS matrices and calls the OS/WS/IS demand
generators. It does not run memory/DRAM service, numerical GEMM, or dot-product
PE simulation. A demand-matrix row is a nominal schedule step, not a measured
hardware cycle. No input files or installed package files are modified.
"""

import hashlib
import importlib.metadata
import inspect
import json
from collections import Counter
from pathlib import Path

import numpy as np

from scalesim.compute.systolic_compute_is import systolic_compute_is
from scalesim.compute.systolic_compute_os import systolic_compute_os
from scalesim.compute.systolic_compute_ws import systolic_compute_ws
from scalesim.scale_config import scale_config


def summarize(matrix):
    valid_mask = matrix != -1
    values = matrix[valid_mask].astype(np.int64)
    address_counts = Counter(int(value) for value in values)
    per_step = np.count_nonzero(valid_mask, axis=1)
    return {
        "shape": list(matrix.shape),
        "valid_request_count": int(len(values)),
        "unique_address_count": len(address_counts),
        "peak_requests_per_nominal_cycle": int(per_step.max()),
        "mean_requests_per_nominal_cycle": float(per_step.mean()),
        "per_cycle_request_counts": per_step.tolist(),
        "address_occurrence_histogram": dict(
            sorted(Counter(address_counts.values()).items())
        ),
        "address_counts": dict(sorted(address_counts.items())),
    }


def run_case(m, n, k):
    a = np.arange(m * k, dtype=np.int64).reshape(m, k)
    b = (10_000_000 + np.arange(k * n, dtype=np.int64)).reshape(k, n)
    c = (20_000_000 + np.arange(m * n, dtype=np.int64)).reshape(m, n)
    result = {"M": m, "N": n, "K": k, "array_rows": 2, "array_cols": 2, "flows": {}}
    for flow, cls in (
        ("os", systolic_compute_os),
        ("ws", systolic_compute_ws),
        ("is", systolic_compute_is),
    ):
        config = scale_config()
        config.set_arr_dims(rows=2, cols=2)
        config.set_dataflow(flow)
        # set_arr_dims does not mark a config loaded from a file as valid.
        # This probe explicitly supplies the only config values used here.
        config.valid_conf_flag = True
        compute = cls()
        compute.set_params(
            config_obj=config, ifmap_op_mat=a, filter_op_mat=b, ofmap_op_mat=c
        )
        demands = compute.get_demand_matrices()
        reports = (
            compute.get_ifmap_requests(),
            compute.get_filter_requests(),
            compute.get_ofmap_requests(),
        )
        channels = {}
        for name, matrix, reported in zip(
            ("A_read", "B_read", "C_write"), demands, reports
        ):
            channel = summarize(matrix)
            channel["library_reported_request_count"] = int(reported)
            channel["reported_count_matches_non_null_trace"] = (
                int(reported) == channel["valid_request_count"]
            )
            channels[name] = channel
        result["flows"][flow] = {
            "nominal_schedule_rows": int(demands[0].shape[0]),
            "channels": channels,
            "C_read_channel": "not provided by this demand-generator interface",
        }
    return result


def main():
    version = importlib.metadata.version("scalesim")
    if version != "2.0.2":
        raise RuntimeError(f"This evidence probe targets scalesim 2.0.2, got {version}")
    sources = {}
    for cls in (systolic_compute_os, systolic_compute_ws, systolic_compute_is):
        path = Path(inspect.getsourcefile(cls)).resolve()
        sources[path.name] = {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    document = {
        "scope": "scalar-PE OS/WS/IS demand-generation only",
        "scalesim_version": version,
        "scalesim_git_commit": None,
        "commit_note": "Installed wheel metadata has no Git commit provenance.",
        "source_files": sources,
        "units": {
            "addresses": "logical scalar operand element IDs, not physical byte addresses",
            "bandwidth": "element requests per nominal demand-matrix row",
        },
        "limitations": [
            "Not a dot-product PE model or a validation of its timing.",
            "No DRAM/buffer service or stall simulation was run.",
            "No numerical MAC, partial-sum value, or numerical correctness simulation.",
            "No physical SRAM read-modify-write or bank/port implementation was modeled.",
            "The absence of a C-read channel is an interface observation, not a claim that hardware never reads partial sums.",
            "Library request counters may differ from counts of actual non-null demand entries; both are recorded.",
        ],
        "cases": [run_case(4, 4, 4), run_case(4, 4, 8)],
    }
    print(json.dumps(document, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
