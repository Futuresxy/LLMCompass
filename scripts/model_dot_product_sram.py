#!/usr/bin/env python3
"""Map dot-product PE traffic to fixed-capacity CACTI SRAM slice organizations."""
import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

from cost_model.cacti import CactiError, CactiRunner
from cost_model.dot_product_sram import ArrayTile, PrecisionMode, evaluate_lane, generate_traffic
from cost_model.local_sram import BufferSpec, Organization, characterize


def next_power_two(value):
    return 1 << (max(1, value) - 1).bit_length()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def write_csv(path, rows):
    if rows:
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def run(config, runner, output):
    if config.get("schema_version") != 1:
        raise ValueError("Expected config schema_version=1")
    specs = {name: BufferSpec(**data) for name, data in config["buffers"].items()}
    if set(specs) != {"A", "B", "C"}:
        raise ValueError("Exactly A, B and C buffers must be specified")
    if len({spec.frequency_hz for spec in specs.values()}) != 1:
        raise ValueError("This frontend requires the array and all buffers to share one clock")
    profile_modes = {
        profile: [PrecisionMode(name, length, config["precision_bits"][name],
                                config["precision_bits"][name], config["issue_interval_cycles"])
                  for name, length in modes.items()]
        for profile, modes in config["profiles"].items()
    }
    baseline_profile = config["baseline_profile"]
    if baseline_profile not in profile_modes or not all(profile_modes.values()):
        raise ValueError("A nonempty baseline and nonempty target profiles are required")
    if any(set(m.name for m in modes) != set(config["profiles"][baseline_profile])
           for modes in profile_modes.values()):
        raise ValueError("Every profile must include the same precision modes")
    if not config["shapes"] or not config["accumulator_policies"] or not config["gating_policies"]:
        raise ValueError("Shape and policy lists must be nonempty")
    report = {
        "schema_version": 1, "status": "running", "config": config,
        "source_sha256": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [ROOT / "cost_model/cacti.py", ROOT / "cost_model/local_sram.py",
                      ROOT / "cost_model/dot_product_sram.py", Path(__file__).resolve()]
        },
        "scope": [
            "Structural dot issue and SRAM macro cost model, not a SCALE-Sim or RTL timing model",
            "Independently accessed SRAM slices with configured word width, fixed row/column-to-prefix-slice placement",
            "Equal total capacity across candidates; slices partition capacity rather than replicate it",
            "One A/B refill per tile; slot capacity and physical port sharing follow each buffer specification",
            "Upstream refill transport bandwidth is an assumption, not an independently checked link",
            "Power is normalized to the compute issue window and excludes fill/drain and accumulator latency",
            "ACC registers, Tensor Core logic, inter-macro wiring and clock/control dynamic power excluded",
            "CACTI predictive technology parameters, not foundry-qualified SRAM or chip measurements",
        ],
        "catalog": {}, "traffic": {}, "studies": [], "candidate_errors": [],
    }
    logical_rows, physical_rows = [], []
    for rows, cols in config["shapes"]:
        # Each A group belongs to a PE row; each B group to a PE column.
        # C groups hold one row of FP32 output state.
        groups = {"A": next_power_two(rows), "B": next_power_two(cols), "C": next_power_two(rows)}
        installed_slices = {
            "A": config["installed_input_slices_per_group"],
            "B": config["installed_input_slices_per_group"],
            "C": next_power_two(math.ceil(cols * config["accumulator_bits"] / config["slice_bits"])),
        }
        catalog_ids = {}
        print("[CACTI] {}x{}: characterize fixed-capacity slice candidates".format(rows, cols), flush=True)
        for operand, spec in specs.items():
            catalog_ids[operand] = {}
            for slices in sorted(set(config["candidate_slices_per_group"]) | {installed_slices[operand]}):
                key = "{}x{}/{}/{}".format(rows, cols, operand, slices)
                try:
                    org = Organization(groups[operand] * slices, config["slice_bits"], **config["ports"][operand])
                    record = characterize(spec, org, runner)
                except (CactiError, ValueError) as exc:
                    report["candidate_errors"].append({"id": key, "error": str(exc)})
                    continue
                report["catalog"][key] = {"spec": asdict(spec), "characterization": record,
                                          "physical_groups": groups[operand], "slices_per_group": slices}
                catalog_ids[operand][slices] = key
        for policy in config["accumulator_policies"]:
            tile = ArrayTile(rows, cols, config["k"], accumulator_bits=config["accumulator_bits"],
                             accumulator_policy=policy, initial_c=config["initial_c"])
            traffic_by_profile = {}
            for profile, modes in profile_modes.items():
                traffic_key = "{}x{}/{}/{}".format(rows, cols, policy, profile)
                traffic_by_profile[profile] = {mode.name: generate_traffic(tile, mode) for mode in modes}
                report["traffic"][traffic_key] = traffic_by_profile[profile]
                for mode, traffic in traffic_by_profile[profile].items():
                    for operand, event in traffic["operands"].items():
                        logical_rows.append({
                            "rows": rows, "cols": cols, "k": config["k"], "accumulator_policy": policy,
                            "profile": profile, "mode": mode, "operand": operand,
                            "dot_length": traffic["mode"]["dot_length"],
                            "issue_window_cycles": traffic["compute_issue_cycles"],
                            "compute_slot_utilization": traffic["issue_window_compute_utilization"],
                            "nominal_read_bytes_per_issue": event["nominal_read_bytes_per_issue"],
                            "peak_read_bytes_per_issue": event["peak_read_bytes_per_issue"],
                            "average_read_bytes_per_cycle": event["average_read_bytes_per_cycle"],
                            "peak_write_bytes_per_issue": event["peak_write_bytes_per_issue"],
                            "average_write_bytes_per_cycle": event["average_write_bytes_per_cycle"],
                            "useful_read_bytes": event["useful_read_bytes"],
                            "useful_write_bytes": event["useful_write_bytes"],
                        })
            for gating in config["gating_policies"]:
                for profile in profile_modes:
                    study = {"rows": rows, "cols": cols, "k": config["k"], "accumulator_policy": policy,
                             "gating": gating, "profile": profile, "buffers": {}}
                    for operand, spec in specs.items():
                        def evaluate_candidate(key, traffic_set):
                            candidate = report["catalog"][key]
                            return {mode: evaluate_lane(spec, candidate["characterization"], traffic, operand,
                                                        candidate["slices_per_group"], candidate["physical_groups"], gating)
                                    for mode, traffic in traffic_set.items()}

                        installed_key = catalog_ids[operand].get(installed_slices[operand])
                        baseline = evaluate_candidate(installed_key, traffic_by_profile[baseline_profile]) if installed_key else {}
                        on_installed = evaluate_candidate(installed_key, traffic_by_profile[profile]) if installed_key else {}
                        baseline_ok = bool(baseline) and all(v["feasible"] for v in baseline.values())
                        installed_ok = baseline_ok and all(v["feasible"] for v in on_installed.values())
                        selected_key, selected = (installed_key, on_installed) if installed_ok else (None, None)
                        evaluations = {}
                        if baseline_ok and not installed_ok:
                            for key in catalog_ids[operand].values():
                                evaluated = evaluate_candidate(key, traffic_by_profile[profile])
                                evaluations[key] = evaluated
                                if all(v["feasible"] for v in evaluated.values()):
                                    rank = (report["catalog"][key]["characterization"]["area_mm2"],
                                            max(v["total_power_w"] for v in evaluated.values()))
                                    if selected is None or rank < selected_rank:
                                        selected_key, selected, selected_rank = key, evaluated, rank
                        status = ("baseline_infeasible" if not baseline_ok else
                                  "retained" if installed_ok else "upgraded" if selected else "infeasible")
                        base_area = report["catalog"][installed_key]["characterization"]["area_mm2"] if installed_key else None
                        area = report["catalog"][selected_key]["characterization"]["area_mm2"] if selected_key else None
                        study["buffers"][operand] = {
                            "status": status, "installed_catalog_id": installed_key, "selected_catalog_id": selected_key,
                            "baseline_modes": baseline, "target_on_installed": on_installed,
                            "selected_modes": selected, "candidate_evaluations": evaluations,
                            "area_mm2": area, "baseline_area_mm2": base_area,
                            "delta_area_mm2": area - base_area if area is not None else None,
                        }
                        for mode in profile_modes[profile]:
                            value = selected[mode.name] if selected else {}
                            physical_rows.append({
                                "rows": rows, "cols": cols, "accumulator_policy": policy, "gating": gating,
                                "profile": profile, "mode": mode.name, "operand": operand, "status": status,
                                "selected_catalog_id": selected_key, "area_mm2": area,
                                "delta_area_mm2": study["buffers"][operand]["delta_area_mm2"],
                                **{key: value.get(key) for key in (
                                    "slices_per_group", "physical_groups", "read_transactions", "write_transactions",
                                    "physical_read_bytes", "physical_write_bytes", "read_payload_efficiency",
                                    "peak_read_lane_activity", "average_read_lane_activity",
                                    "per_macro_required_bytes_max", "modeled_dynamic_energy_j", "dynamic_power_w",
                                    "leakage_w", "total_power_w")},
                            })
                    study["feasible"] = all(v["selected_modes"] for v in study["buffers"].values())
                    study["total_sram_area_mm2"] = sum(v["area_mm2"] for v in study["buffers"].values()) if study["feasible"] else None
                    report["studies"].append(study)
                print("[traffic] {}x{} {} {}: profiles evaluated".format(rows, cols, policy, gating), flush=True)
        write_json(output / "result.json", report)
    report["status"] = "passed" if all(s["feasible"] for s in report["studies"]) else "infeasible"
    write_json(output / "result.json", report)
    write_csv(output / "logical_traffic.csv", logical_rows)
    write_csv(output / "lane_cost.csv", physical_rows)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/dot_product_sram_example.json")
    parser.add_argument("--cacti-dir", type=Path, default=ROOT / ".runtime/cacti")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / ".runtime/cacti-cache")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output = (args.output_dir or ROOT / "artifacts/local-sram" /
              ("dot-product-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ"))).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        parser.error("Output directory must be new or empty")
    output.mkdir(parents=True, exist_ok=True)
    try:
        config = json.loads(args.config.read_text())
        write_json(output / "input.json", config)
        report = run(config, CactiRunner(args.cacti_dir, args.cache_dir), output)
    except Exception as exc:
        write_json(output / "failure.json", {"status": "failed", "error": repr(exc)})
        raise
    print("{}: {} ({} studies, {} rejected CACTI candidates)".format(
        report["status"], output / "result.json", len(report["studies"]), len(report["candidate_errors"])))
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
