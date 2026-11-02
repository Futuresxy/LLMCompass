#!/usr/bin/env python3
"""Reproducible uniform-GEMM ratio sweep. No numerical LLM or silicon execution.

Input workloads are explicit operator lists; the bundled lists are shape proxies.
PPA values are optional and never synthesized from precision width/peak MACs.
"""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hardware_model.compute_module import ComputeModule, Core, SystolicArray, vector_unit_dict
from hardware_model.device import Device
from hardware_model.io_module import IOModule
from hardware_model.memory_module import MemoryModule
from hardware_model.precision import KExpansionProfile
from software_model.matmul import BatchedMatmul, Matmul
from software_model.utils import Tensor, data_type_dict
from systolic_array_model.precision_k import provenance, token_trace


def _positive(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(name + " must be finite and positive")
    return value


def make_device(spec, profile):
    for name in ("rows", "cols", "core_count", "arrays_per_core"):
        value = spec[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(name + " must be a positive integer")
    for name in ("frequency_hz", "hbm_bytes_per_second", "l2_bytes_per_cycle"):
        _positive(spec[name], name)
    array = SystolicArray(spec["rows"], spec["cols"], profile.base_dot_length,
                          2, 4, precision_profile=profile)
    core = Core(vector_unit_dict["A100_fp16"], array, spec["arrays_per_core"], 192 * 1024)
    cm = ComputeModule(core, spec["core_count"], spec["frequency_hz"],
                       40 * 1024 ** 2, spec["l2_bytes_per_cycle"])
    return Device(cm, IOModule(spec["hbm_bytes_per_second"], 0), MemoryModule(float("inf")))


def _count(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(name + " must be a positive integer")
    return value


def evaluate_workload(workload, dtype, device):
    entries = []
    other = workload.get("non_gemm_seconds", 0)
    if isinstance(other, bool) or not isinstance(other, (int, float)) or not math.isfinite(other) or other < 0:
        raise ValueError("non_gemm_seconds must be finite and nonnegative")
    if not workload.get("ops"):
        raise ValueError("A workload must contain at least one GEMM")
    seconds = compute = memory = active_cycles = 0
    for op in workload["ops"]:
        m, n, k = (_count(op[x], x) for x in ("m", "n", "k"))
        batch = _count(op.get("batch_count", 1), "batch_count")
        count = _count(op.get("count", 1), "count")
        if batch == 1:
            gemm = Matmul(dtype)
            gemm(Tensor([m, k], dtype), Tensor([k, n], dtype))
        else:
            gemm = BatchedMatmul(dtype)
            gemm(Tensor([batch, m, k], dtype), Tensor([batch, k, n], dtype))
        gemm.compile_and_simulate(device, "precision-k")
        result = gemm.precision_result
        seconds += count * result["latency_seconds"]
        compute += count * result["compute_seconds"]
        memory += count * result["compulsory_memory_seconds"]
        active_cycles += count * result["total_array_active_cycles"]
        entries.append({"name": op["name"], "count": count, "result": result})
    return {"estimate_seconds": seconds + other, "compute_seconds": compute,
            "compulsory_memory_seconds": memory, "non_gemm_seconds": other,
            "total_array_active_cycles": active_cycles, "operators": entries}


def _ppa_value(ppa, ratio, key, precision_class=None, format_name=None):
    profile = (ppa or {}).get("profiles", {}).get(ratio, {})
    value = profile.get(key)
    if precision_class is not None:
        value = (value or {}).get(str(precision_class))
        override = profile.get("active_power_relative_by_format", {}).get(format_name)
        if override is not None:
            value = override
    if value is not None:
        _positive(value, "PPA " + key)
    return value


def run_study(config, ppa=None):
    profiles = [KExpansionProfile(config.get("base_dot_length", 16), tuple(p))
                for p in config["profiles"]]
    names = [p.name for p in profiles]
    if len(names) != len(set(names)) or config["baseline"] not in names:
        raise ValueError("Profiles must be unique and contain the baseline")
    if not config.get("formats") or len(config["formats"]) != len(set(config["formats"])):
        raise ValueError("formats must be nonempty and unique")
    workload_names = [w["name"] for w in config["workloads"]]
    if not workload_names or len(workload_names) != len(set(workload_names)):
        raise ValueError("workload names must be nonempty and unique")
    for w in config["workloads"]:
        if w["split"] not in ("design", "holdout"):
            raise ValueError("split must be design or holdout")
    if ppa:
        if ppa.get("scope") != "one_tensor_core":
            raise ValueError("PPA scope must be one_tensor_core")
        if ppa.get("frequency_hz") is not None and ppa["frequency_hz"] != config["hardware"]["frequency_hz"]:
            raise ValueError("PPA frequency differs from the simulated fixed frequency")
    rows = []
    for workload in config["workloads"]:
        for name in config["formats"]:
            dtype = data_type_dict[name]
            measurements = {}
            for profile in profiles:
                device = make_device(config["hardware"], profile)
                measurements[profile.name] = evaluate_workload(workload, dtype, device)
            baseline = measurements[config["baseline"]]
            base_area = _ppa_value(ppa, config["baseline"], "area_relative")
            base_power = _ppa_value(ppa, config["baseline"], "active_power_relative", dtype.compute_class, name)
            for profile in profiles:
                result = measurements[profile.name]
                area = _ppa_value(ppa, profile.name, "area_relative")
                power = _ppa_value(ppa, profile.name, "active_power_relative", dtype.compute_class, name)
                speedup = baseline["estimate_seconds"] / result["estimate_seconds"]
                rows.append({
                    "workload": workload["name"], "family": workload.get("family", "unspecified"),
                    "split": workload["split"], "format": name,
                    "compute_class": dtype.compute_class, "ratio": profile.name,
                    "speedup": speedup,
                    "compute_speedup": baseline["compute_seconds"] / result["compute_seconds"],
                    "tc_area_relative": area / base_area if area and base_area else None,
                    "throughput_per_tc_area_relative": speedup * base_area / area if area and base_area else None,
                    "tc_active_energy_proxy_relative": (
                        result["total_array_active_cycles"] * power
                        / (baseline["total_array_active_cycles"] * base_power)
                        if power and base_power else None),
                    **result})
    aggregates = []
    for name in config["formats"]:
        for profile in profiles:
            for split in ("design", "holdout"):
                selected = [r for r in rows if r["format"] == name and r["ratio"] == profile.name and r["split"] == split]
                if selected:
                    speeds = [r["speedup"] for r in selected]
                    aggregates.append({"format": name, "ratio": profile.name, "split": split,
                                       "equal_weight_geomean_speedup": math.exp(sum(math.log(s) for s in speeds) / len(speeds)),
                                       "worst_speedup": min(speeds),
                                       "regression_fraction": sum(s < 1 - 1e-12 for s in speeds) / len(speeds),
                                       "workload_count": len(speeds)})
    return {"schema_version": 1, "config": config, "ppa_input": ppa,
            "provenance": dict(provenance()), "results": rows, "aggregates": aggregates,
            "limitations": [
                "Bundled workloads are illustrative shape proxies, not actual model results.",
                "Uniform operand format per workload; FP32 accumulation; output in operand format.",
                "Scale/format conversion compute and numerical accuracy are not simulated.",
                "Ideal local operand supply; HBM/L2 compulsory-byte floor with ideal overlap only.",
                "No full-chip area/power or fixed-area/fixed-power result is inferred from Tensor Core PPA.",
                "Active energy proxy assumes per-array supplied active power for all allocated job slots; idle/leakage outside jobs and memory energy omitted.",
                "No hardware selection uses the holdout set; report raw per-workload results and freeze selection rules before real evaluation."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/precision_ratio_study.json")
    parser.add_argument("--ppa", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--trace", action="store_true", help="Save one full-grid native token trace per format/profile (not SRAM traces)")
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("Use a new/empty output directory to preserve prior results")
    config = json.loads(args.config.read_text())
    ppa = json.loads(args.ppa.read_text()) if args.ppa else None
    report = run_study(config, ppa)
    source_files = [ROOT / p for p in (
        "software_model/utils.py", "software_model/matmul.py", "hardware_model/compute_module.py",
        "hardware_model/precision.py", "systolic_array_model/precision_k.py",
        "scripts/study_precision_ratios.py", "cost_model/cost_model.py",
        "design_space_exploration/dse.py")]
    git = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True)
    report["manifest"] = {"commit": git.stdout.strip(), "source_sha256": {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files},
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "ppa_sha256": hashlib.sha256(args.ppa.read_bytes()).hexdigest() if args.ppa else None}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "result.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    flat = [{k: v for k, v in r.items() if k != "operators"} for r in report["results"]]
    with (args.output_dir / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    if args.trace:
        import numpy as np
        spec = config["hardware"]
        for p in config["profiles"]:
            profile = KExpansionProfile(config.get("base_dot_length", 16), tuple(p))
            for name in config["formats"]:
                k = 193
                groups = math.ceil(k / profile.dot_length(data_type_dict[name]))
                a, b, c = token_trace(spec["rows"], spec["cols"], groups, spec["rows"], spec["cols"])
                np.savez_compressed(args.output_dir / (name + "_" + profile.name.replace(":", "-") + "_tokens.npz"),
                                    a=a, b=b, c=c, original_k=k,
                                    dot_length=profile.dot_length(data_type_dict[name]))
    print(json.dumps({"results": len(report["results"]), "output": str(args.output_dir),
                      "scope": "SCALE-Sim compute schedule / ideal memory overlap envelope; not hardware validation"}))


if __name__ == "__main__":
    main()
