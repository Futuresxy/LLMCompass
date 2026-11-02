#!/usr/bin/env python3
"""Search fixed-capacity, array-local operand SRAM organizations using CACTI."""
import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import itertools
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

from cost_model.cacti import CactiRunner
from cost_model.local_sram import (BufferSpec, Demand, Organization, evaluate, evaluate_modes,
                                   plan_upgrade, scale_mode_demands)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/local_sram_example.json")
    parser.add_argument("--cacti-dir", type=Path, default=ROOT / ".runtime/cacti")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / ".runtime/cacti-cache")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--demand-scale", type=float, default=1.0,
                        help="Scale both baseline/target workload rates; 0.5 demonstrates installed headroom")
    parser.add_argument("--plot", action="store_true", help="Save a sampled area/power sweep PNG (requires matplotlib)")
    args = parser.parse_args()
    output = (args.output_dir or ROOT / "artifacts/local-sram" /
              datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        parser.error("Output directory must be new or empty")
    output.mkdir(parents=True, exist_ok=True)
    report = {"status": "running", "schema_version": 1, "buffers": {}, "sweep": []}
    try:
        config = json.loads(args.config.read_text())
        if config.get("schema_version") != 1:
            raise ValueError("Expected config schema_version=1")
        specs = [BufferSpec(**data) for data in config["buffers"]]
        names = {s.name for s in specs}
        if len(names) != len(specs):
            raise ValueError("Buffer names must be unique")
        base = {mode: {name: Demand(**data).scaled(args.demand_scale)
                       for name, data in operands.items()}
                for mode, operands in config["baseline_demands"].items()}
        if any(set(operands) != names for operands in base.values()):
            raise ValueError("Every precision mode must specify every operand buffer")
        target = scale_mode_demands(config["baseline_throughput_mac_per_cycle"],
                                    config["target_throughput_mac_per_cycle"], base)
        search = config["search"]
        organizations = [Organization(banks, width, **ports) for banks, width, ports in
                         itertools.product(search["banks"], search["width_bits"], search["ports"])]
        runner = CactiRunner(args.cacti_dir, args.cache_dir)
        installed = config.get("installed", {})
        report.update(config=config, demand_scale=args.demand_scale,
                      source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                     for p in [ROOT / "cost_model/cacti.py", ROOT / "cost_model/local_sram.py",
                                               Path(__file__).resolve()]},
                      scope="Macro-level predictive SRAM cost model; no proprietary PDK, RTL, or chip validation; no Tensor Core area/power included")
        write_json(output / "input.json", config)
        for spec in specs:
            print("[CACTI] {}: {} candidate organizations".format(spec.name, len(organizations)), flush=True)
            old = {mode: operands[spec.name] for mode, operands in base.items()}
            new = {mode: operands[spec.name] for mode, operands in target.items()}
            plan = plan_upgrade(spec, old, new, organizations, runner,
                                installed=Organization(**installed[spec.name]) if spec.name in installed else None,
                                baseline_headroom=config.get("baseline_headroom_factor", 1.0))
            plan["spec"] = asdict(spec)
            plan["modes"] = {}
            if plan["baseline"]:
                for mode in base:
                    plan["modes"][mode] = {
                        "baseline": evaluate(spec, plan["baseline"], base[mode][spec.name]),
                        "target_on_installed": evaluate(spec, plan["baseline"], target[mode][spec.name]),
                        "target_on_selected": evaluate(spec, plan["target"], target[mode][spec.name]) if plan["target"] else None,
                    }
            report["buffers"][spec.name] = plan
            print("[{}] {} delta_area={} mm^2".format(plan["status"], spec.name, plan["delta_area_mm2"]), flush=True)
            write_json(output / "result.json", report)

        mode = config.get("sweep_mode", "fp4")
        for multiplier in config.get("sweep_multipliers", []):
            row = {"multiplier": multiplier, "mode": mode, "buffers": {}}
            for spec in specs:
                original = report["buffers"][spec.name]
                if original["baseline"] is None:
                    row["buffers"][spec.name] = {"status": "infeasible"}
                    continue
                demand = base[mode][spec.name].scaled(multiplier)
                all_modes = {name: operands[spec.name] for name, operands in base.items()}
                all_modes[mode] = demand
                on_old = dict(original["baseline"], **evaluate_modes(spec, original["baseline"], all_modes))
                if on_old["feasible"]:
                    selected, state = on_old, "retained"
                else:
                    options = [dict(c, **evaluate_modes(spec, c, all_modes)) for c in original["candidates"]]
                    options = [c for c in options if c["feasible"]]
                    selected = min(options, key=lambda c: (c["area_mm2"], c["total_power_w"])) if options else None
                    state = "upgraded" if selected else "infeasible"
                row["buffers"][spec.name] = {
                    "status": state, "read_bytes_per_cycle": demand.read_bytes_per_cycle,
                    "area_mm2": selected["area_mm2"] if selected else None,
                    "delta_area_mm2": selected["area_mm2"] - original["baseline"]["area_mm2"] if selected else None,
                    "total_power_w": evaluate(spec, selected, demand)["total_power_w"] if selected else None,
                    "organization": selected["organization"] if selected else None,
                }
            report["sweep"].append(row)
        report["status"] = "passed" if all(p["target"] for p in report["buffers"].values()) else "infeasible"
        report["total_baseline_area_mm2"] = sum(p["baseline"]["area_mm2"] for p in report["buffers"].values() if p["baseline"])
        report["total_target_area_mm2"] = sum(p["target"]["area_mm2"] for p in report["buffers"].values()) if report["status"] == "passed" else None
        report["total_delta_area_mm2"] = report["total_target_area_mm2"] - report["total_baseline_area_mm2"] if report["status"] == "passed" else None
        write_json(output / "result.json", report)
        with (output / "sweep.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["buffer", "demand_multiplier", "read_bytes_per_cycle", "status", "area_mm2", "delta_area_mm2", "total_power_w"])
            for row in report["sweep"]:
                for name, value in row["buffers"].items():
                    writer.writerow([name, row["multiplier"], value.get("read_bytes_per_cycle"), value["status"],
                                     value.get("area_mm2"), value.get("delta_area_mm2"), value.get("total_power_w")])
        if args.plot:
            os.environ.setdefault("MPLCONFIGDIR", str(output / "matplotlib"))
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
            for spec in specs:
                rows = [(r["multiplier"], r["buffers"][spec.name]) for r in report["sweep"]
                        if r["buffers"][spec.name].get("area_mm2") is not None]
                for ax, field in zip(axes, ("area_mm2", "total_power_w")):
                    ax.scatter([r[0] for r in rows], [r[1][field] for r in rows], label=spec.name,
                               marker="o" if spec == specs[0] else "x", s=55)
            axes[0].set_ylabel("SRAM macro area (mm2)")
            axes[1].set_ylabel("Workload SRAM power (W)")
            for ax in axes:
                ax.set_xlabel("Demand multiplier relative to baseline")
                ax.grid(alpha=0.25)
                ax.legend()
            nodes = ", ".join(str(node) for node in sorted({s.technology_nm for s in specs}))
            fig.suptitle("Fixed capacity, sampled CACTI design search ({} nm)".format(nodes))
            fig.savefig(output / "sweep.png", dpi=160)
            plt.close(fig)
        print("Results: {}".format(output / "result.json"))
        return 0 if report["status"] == "passed" else 2
    except Exception as exc:
        report.update(status="failed", error=repr(exc))
        write_json(output / "result.json", report)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
