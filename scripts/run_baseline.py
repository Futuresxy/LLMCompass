#!/usr/bin/env python3
"""Run the original CPU performance models with an isolated writable cache.

This is a smoke run, not numerical inference or a paper/hardware reproduction.
Run from any directory with the project's prepared Python environment.
"""

import argparse
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import traceback


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_output(*args):
    completed = subprocess.run(
        ["git", "-C", str(ROOT), *args], capture_output=True, text=True
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def json_default(value):
    if hasattr(value, "item"):
        return value.item()
    raise TypeError("Cannot serialize {}".format(type(value).__name__))


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False,
                   default=json_default) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def positive(value, label):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise AssertionError("{} must be finite and positive, got {}".format(label, value))
    return value


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
        return len(text)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def execute(output):
    started = time.perf_counter()
    source_cache = ROOT / "systolic_array_model"
    cache_files = sorted(source_cache.glob("look_up_table_[0-9]*_[0-9]*.csv"))
    source_hashes = {str(p.relative_to(ROOT)): sha256(p) for p in cache_files}
    versions = {}
    for package in ("torch", "numpy", "pandas", "scalesim", "scipy"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    source_files = sorted(
        list((ROOT / "software_model").glob("*.py"))
        + list((ROOT / "hardware_model").glob("*.py"))
        + [ROOT / "cost_model/cost_model.py", ROOT / "design_space_exploration/dse.py",
           ROOT / "configs/GA100.json", Path(__file__).resolve()]
    )
    manifest = {
        "schema_version": 1,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository": str(ROOT),
        "commit": git_output("rev-parse", "HEAD"),
        "submodules": git_output("submodule", "status"),
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "packages": versions,
        "output_directory": str(output),
        "units": {"latency": "seconds", "cycles": "clock cycles",
                  "utilization": "percent", "area": "mm^2", "wall_time": "seconds"},
        "source_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in source_files},
        "source_cache_sha256_before": source_hashes,
        "scope": [
            "Original LLMCompass CPU performance and area estimation paths.",
            "SCALE-Sim cache misses write only to a copied cache in this output directory.",
            "No GPU/NPU execution, numeric model correctness, or full paper reproduction.",
            "Legacy modeling assumptions and known limitations are preserved.",
            "Built-in A100 performance parameters differ from configs/GA100.json.",
        ],
    }
    report = {"schema_version": 1, "status": "running", "stages": []}
    write_json(output / "manifest.json", manifest)
    write_json(output / "result.json", report)

    def stage(name, function):
        print("[start] " + name, flush=True)
        tick = time.perf_counter()
        entry = {"name": name}
        try:
            entry["result"] = function()
            entry["status"] = "passed"
        except Exception as exc:
            entry.update(status="failed", error=repr(exc), traceback=traceback.format_exc())
            traceback.print_exc()
        entry["wall_time_seconds"] = time.perf_counter() - tick
        report["stages"].append(entry)
        write_json(output / "result.json", report)
        print("[{}] {} ({:.3f}s)".format(entry["status"], name, entry["wall_time_seconds"]),
              flush=True)

    previous_cwd = Path.cwd()
    try:
        if not cache_files:
            raise FileNotFoundError("No dimension-specific source lookup tables found")
        run_cache = output / "systolic_array_model"
        (run_cache / "temp").mkdir(parents=True)
        for source in cache_files:
            shutil.copy2(source, run_cache / source.name)
        os.chdir(output)

        import pandas as pd
        from software_model.matmul import Matmul
        from software_model.utils import Tensor, data_type_dict
        from software_model.transformer import (
            TransformerBlockInitComputationTP, TransformerBlockAutoRegressionTP,
        )
        from hardware_model.device import device_dict
        from hardware_model.system import system_dict

        dtype = data_type_dict["fp16"]
        device = device_dict["A100_80GB_fp16"]
        system = system_dict["A100_4_fp16"]
        manifest["builtin_hardware"] = {
            "device_id": "A100_80GB_fp16", "system_id": "A100_4_fp16",
            "core_count": device.compute_module.core_count,
            "arrays_per_core": device.compute_module.core.systolic_array_count,
            "array": vars(device.compute_module.core.systolic_array),
            "clock_hz": device.compute_module.clock_freq,
            "local_sram_bytes": device.compute_module.core.SRAM_size,
            "global_buffer_bytes": device.compute_module.l2_size,
            "global_buffer_bytes_per_cycle": device.compute_module.l2_bandwidth_per_cycle,
            "external_memory_bytes_per_second": device.io_module.bandwidth,
        }
        write_json(output / "manifest.json", manifest)

        columns = ["M", "N", "K", "ArrayHeight", "ArrayWidth", "Dataflow",
                   "cycle_count", "util_rate"]
        keys = columns[:6]

        def load_table():
            table = pd.read_csv(run_cache / "look_up_table_16_16.csv",
                                header=None, names=columns)
            return table.drop_duplicates(subset=keys).set_index(keys)

        table = load_table()

        def array_case(shape, kind):
            m, n, k = shape
            key = (m, n, k, 16, 16, "os")
            if kind == "fallback":
                if key in table.index or (n, m, k, 16, 16, "os") in table.index:
                    raise AssertionError("Fallback test key already exists in source cache")
            cycles = Matmul.simulate_systolic_array_cycle_count(
                table, m, n, k, 16, 16, 1, "os"
            )
            positive(cycles, "array cycles")
            result = {"shape_mnk": list(shape), "array_hw": [16, 16],
                      "mac_per_pe_per_cycle": 1, "dataflow": "os", "cycles": int(cycles)}
            if kind == "cache" and cycles != 571:
                raise AssertionError("Expected legacy cached cycle count 571")
            if kind == "analytic":
                expected = math.ceil(m * n * k / 16 / 16 / 0.99)
                if cycles != expected:
                    raise AssertionError("Analytic path disagrees with its documented formula")
                result["expected_formula_cycles"] = expected
            else:
                row = table.loc[key]
                utilization = float(row["util_rate"])
                expected_utilization = 100 * m * n * k / (16 * 16 * cycles)
                if not math.isclose(utilization, expected_utilization, abs_tol=0.001):
                    raise AssertionError("Utilization is inconsistent with useful MAC count")
                result.update(utilization_percent=utilization,
                              expected_utilization_percent=expected_utilization)
                if kind == "fallback":
                    persisted = load_table().loc[key]
                    if int(persisted["cycle_count"]) != cycles:
                        raise AssertionError("SCALE-Sim result was not appended to copied cache")
                    result["persisted_utilization_percent"] = float(persisted["util_rate"])
                    result["generated_by_actual_scalesim_call"] = True
            return result

        stage("array_analytic", lambda: array_case((256, 256, 4096), "analytic"))
        stage("array_cache", lambda: array_case((32, 16, 256), "cache"))
        stage("array_scalesim_fallback", lambda: array_case((7, 11, 13), "fallback"))

        def gemm():
            model = Matmul(dtype)
            result = model(Tensor([256, 256], dtype), Tensor([256, 256], dtype))
            if result.shape != [256, 256]:
                raise AssertionError("Unexpected GEMM output shape")
            latency = positive(model.compile_and_simulate(device, "heuristic-GPU"), "GEMM latency")
            positive(model.best_cycle_count, "GEMM cycles")
            if not math.isclose(latency, model.best_cycle_count / device.compute_module.clock_freq):
                raise AssertionError("GEMM seconds/cycles conversion mismatch")
            return {"shape_mnk": [256, 256, 256], "compile_mode": "heuristic-GPU",
                    "dtype": "fp16", "latency_seconds": latency,
                    "cycles": int(model.best_cycle_count), "best_mapping": vars(model.best_mapping)}

        stage("gemm_full_mapping", gemm)

        def transformer(phase, roofline=False):
            batch, sequence, hidden, heads = (8, 2048, 12288, 96) if roofline else (2, 32, 512, 8)
            constructor = (TransformerBlockInitComputationTP if phase == "prefill"
                           else TransformerBlockAutoRegressionTP)
            model = constructor(hidden, heads, 4, dtype)
            shape = [batch, sequence if phase == "prefill" else 1, hidden]
            history = 3072 if roofline else sequence
            output_tensor = (model(Tensor(shape, dtype)) if phase == "prefill"
                             else model(Tensor(shape, dtype), history))
            if output_tensor.shape != shape:
                raise AssertionError("Unexpected Transformer output shape")
            latency = positive(model.roofline_model(system) if roofline else
                               model.compile_and_simulate(system, "heuristic-GPU"), "block latency")
            return {"phase": phase, "input_shape": shape, "hidden": hidden, "heads": heads,
                    "tensor_parallel_ranks": 4, "dtype": "fp16", "block_count": 1,
                    "historical_kv_length": history if phase == "decode" else None,
                    "model": "legacy_roofline" if roofline else "legacy_tile_simulation",
                    "latency_seconds": latency,
                    "breakdown_seconds_csv": model.roofline_log if roofline else model.simluate_log}

        stage("transformer_small_prefill", lambda: transformer("prefill"))
        stage("transformer_small_decode", lambda: transformer("decode"))

        def json_hardware_and_area():
            from design_space_exploration.dse import read_architecture_template, template_to_system
            from cost_model.cost_model import calc_compute_chiplet_area_mm2, calc_io_die_area_mm2
            specs = read_architecture_template(str(ROOT / "configs/GA100.json"))
            parsed = template_to_system(specs)
            compute_area, core_breakdown, compute_breakdown = calc_compute_chiplet_area_mm2(specs, verbose=True)
            io_area, io_breakdown = calc_io_die_area_mm2(specs, verbose=True)
            positive(compute_area, "compute chiplet area")
            positive(io_area, "IO area")
            return {"config": "configs/GA100.json", "expanded_source_config": specs,
                    "parsed_core_count": parsed.device.compute_module.core_count,
                    "parsed_clock_hz": parsed.device.compute_module.clock_freq,
                    "parsed_matrix_peak_ops_per_second": positive(
                        parsed.device.compute_module.total_systolic_array_flops, "matrix peak"),
                    "compute_chiplet_area_mm2": float(compute_area),
                    "io_area_mm2": float(io_area), "core_breakdown_mm2": core_breakdown,
                    "compute_breakdown_mm2": compute_breakdown, "io_breakdown_mm2": io_breakdown}

        stage("ga100_json_and_area", json_hardware_and_area)
        stage("gpt3_prefill_roofline", lambda: transformer("prefill", roofline=True))
        stage("gpt3_decode_roofline", lambda: transformer("decode", roofline=True))
    except BaseException as exc:
        report["runner_error"] = {"error": repr(exc), "traceback": traceback.format_exc()}
        traceback.print_exc()
    finally:
        os.chdir(previous_cwd)
        after_hashes = {str(p.relative_to(ROOT)): sha256(p) for p in cache_files}
        manifest["source_cache_sha256_after"] = after_hashes
        report["source_cache_unchanged"] = source_hashes == after_hashes
        report["wall_time_seconds"] = time.perf_counter() - started
        passed = (not report.get("runner_error") and len(report["stages"]) == 9
                  and all(item["status"] == "passed" for item in report["stages"])
                  and report["source_cache_unchanged"])
        report["status"] = "passed" if passed else "failed"
        manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(output / "manifest.json", manifest)
        write_json(output / "result.json", report)
        print("[{}] Baseline results: {}".format(report["status"], output / "result.json"), flush=True)
    return 0 if passed else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, help="New/empty directory for logs and copied caches")
    args = parser.parse_args()
    output = (args.output_dir.expanduser().resolve() if args.output_dir else
              ROOT / "artifacts" / "llmcompass-baseline" /
              datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ"))
    if output.exists() and any(output.iterdir()):
        parser.error("Output directory is not empty; use a new directory to preserve prior runs")
    output.mkdir(parents=True, exist_ok=True)
    with (output / "run.log").open("w", encoding="utf-8", buffering=1) as log:
        with redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)):
            return execute(output)


if __name__ == "__main__":
    raise SystemExit(main())
