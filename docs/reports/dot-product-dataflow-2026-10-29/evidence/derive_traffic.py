#!/usr/bin/env python3
"""Independent analytical OS/WS/IS traffic derivation; not a timing simulator.

Run from any directory. Only derived.json next to this file is written. No
LLMCompass/SCALE-Sim model is imported. The tiny integer reference validates
MAC coverage and results; large examples use exact closed-form byte counts.
"""

from collections import Counter
from fractions import Fraction
import hashlib
import json
from pathlib import Path


def ceil_div(value, divisor):
    return (value + divisor - 1) // divisor


def number(value):
    value = Fraction(value)
    return value.numerator if value.denominator == 1 else float(value)


def bytes_for(elements, bits):
    return number(Fraction(elements * bits, 8))


def ceil_fraction(value):
    value = Fraction(value)
    return ceil_div(value.numerator, value.denominator)


def derive(m, n, k, rows, cols, dot_length, a_bits, b_bits,
           preload_bytes_per_cycle=512):
    """Count useful payload crossing the SRAM/array boundary, without packing.

    II=1, beta=0, FP32 outputs. No cross-output-tile operand retention for OS.
    WS holds a weight tile while streaming all M; IS is the transposed analogue.
    Preload transfers and steady streaming are separate phases.
    """
    assert all(isinstance(x, int) and x > 0 for x in
               (m, n, k, rows, cols, dot_length, a_bits, b_bits,
                preload_bytes_per_cycle))
    r, c, length = rows, cols, dot_length
    mf, nf = ceil_div(m, r), ceil_div(n, c)
    k_dot_groups = ceil_div(k, length)
    k_folds = ceil_div(k, r * length)
    k_sizes = [min(r * length, k - f * r * length) for f in range(k_folds)]
    n_sizes = [min(c, n - f * c) for f in range(nf)]
    is_m_sizes = [min(c, m - f * c) for f in range(ceil_div(m, c))]
    fp32_output = m * n * 4
    macs = m * n * k
    peak_macs = r * c * length

    def make(flow, issues, a_elements, b_elements, a_preload, b_preload,
             preload_phases, active_groups, loop_order, accumulator_working_set):
        a_read, b_read = bytes_for(a_elements, a_bits), bytes_for(b_elements, b_bits)
        a_load, b_load = bytes_for(a_preload, a_bits), bytes_for(b_preload, b_bits)
        folds = 1 if flow == "os" else k_folds
        result = {
            "dataflow": flow,
            "loop_order": loop_order,
            "ideal_compute_issues": issues,
            "useful_macs": macs,
            "physical_peak_macs_per_issue": peak_macs,
            "mapping_utilization": number(Fraction(macs, issues * peak_macs)),
            "a_total_array_input_bytes": a_read,
            "b_total_array_input_bytes": b_read,
            "a_preload_bytes": a_load,
            "b_preload_bytes": b_load,
            "a_stream_bytes": a_read - a_load,
            "b_stream_bytes": b_read - b_load,
            "a_average_stream_bytes_per_compute_issue": number(Fraction(a_read - a_load) / issues),
            "b_average_stream_bytes_per_compute_issue": number(Fraction(b_read - b_load) / issues),
            "preload_interface_bytes_per_cycle_assumed": preload_bytes_per_cycle,
            "preload_transfer_cycles_lower_bound": sum(
                ceil_fraction(Fraction(payload) / preload_bytes_per_cycle) for payload in preload_phases),
            "preload_is_excluded_from_compute_issues": True,
            "preload_phase_count": len(preload_phases),
            "array_result_emissions": folds * m * n,
            "array_result_emission_bytes": folds * fp32_output,
            "average_array_output_bytes_per_compute_issue": number(Fraction(folds * fp32_output, issues)),
            "final_d_elements": m * n,
            "final_d_bytes": fp32_output,
            "initial_c_read_bytes": 0,
            "sram_accumulator_if_every_emission_is_materialized": {
                "read_bytes": (folds - 1) * fp32_output,
                "write_bytes": folds * fp32_output,
                "final_d_aliases_accumulator": True,
                "extra_final_d_copy_bytes": 0,
            },
            "resident_accumulator_policy": {
                "sram_initial_c_read_bytes": 0,
                "sram_final_d_write_bytes": fp32_output,
                "cross_k_state_working_set_bytes": accumulator_working_set if k_folds > 1 and flow != "os" else 0,
                "logical_result_working_set_bytes": accumulator_working_set,
                "where": "PE-local FP32 accumulators" if flow == "os" else "external result accumulator",
                "state_and_update_cost_is_additional_to_operand_sram": True,
            },
            "active_reduction_rows_by_k_fold": active_groups,
        }
        assert result["useful_macs"] <= issues * peak_macs
        assert result["final_d_bytes"] == m * n * 32 // 8
        assert a_read == bytes_for(a_elements, a_bits)
        assert b_read == bytes_for(b_elements, b_bits)
        assert result["array_result_emissions"] * 32 // 8 == result["array_result_emission_bytes"]
        return result

    os = make("os", mf * nf * k_dot_groups,
              nf * m * k, mf * k * n, 0, 0, [], [],
              ["m_tile(rows)", "n_tile(cols)", "k_dot_group(dot_length)"],
              min(r, m) * min(c, n) * 4)
    os.update({
        "spatial_mapping": {"pe_row": "m", "pe_col": "n", "dot_lane": "k"},
        "stationary_operand": "FP32 partial sum for one output tile",
        "nominal_full_issue_a_bytes": bytes_for(r * length, a_bits),
        "nominal_full_issue_b_bytes": bytes_for(c * length, b_bits),
        "whole_output_tile_bytes": min(r, m) * min(c, n) * 4,
        "writeback_burst_shape_is_not_scheduled": True,
        "pe_accumulator_physical_bytes": r * c * 4,
    })
    ws = make("ws", nf * k_folds * m,
              nf * m * k, k * n, 0, k * n,
              [bytes_for(ks * ns, b_bits) for ns in n_sizes for ks in k_sizes],
              [ceil_div(ks, length) for ks in k_sizes],
              ["n_tile(cols)", "k_tile(rows*dot_length)", "m_stream"],
              m * min(c, n) * 4)
    ws.update({
        "spatial_mapping": {"pe_row": "k_dot_group", "pe_col": "n", "dot_lane": "k_within_dot_group"},
        "stationary_operand": "B weight vector in each PE",
        "full_stream_a_bytes_per_issue_by_k_fold": [bytes_for(ks, a_bits) for ks in k_sizes],
        "nominal_full_hardware_a_bytes_per_issue": bytes_for(r * length, a_bits),
        "steady_array_output_upper_bound_bytes_per_issue": min(c, n) * 4,
        "pe_weight_storage_capacity_bytes": bytes_for(r * c * length, b_bits),
        "preload_payload_per_n_tile_bytes": [bytes_for(k * ns, b_bits) for ns in n_sizes],
        "useful_pe_to_pe_partial_sum_hop_bytes": sum(
            (ceil_div(ks, length) - 1) * m * n * 4 for ks in k_sizes),
    })
    is_flow = make("is", ceil_div(m, c) * k_folds * n,
                   m * k, ceil_div(m, c) * k * n, m * k, 0,
                   [bytes_for(ms * ks, a_bits) for ms in is_m_sizes for ks in k_sizes],
                   [ceil_div(ks, length) for ks in k_sizes],
                   ["m_tile(cols)", "k_tile(rows*dot_length)", "n_stream"],
                   n * min(c, m) * 4)
    is_flow.update({
        "spatial_mapping": {"pe_row": "k_dot_group", "pe_col": "m", "dot_lane": "k_within_dot_group"},
        "stationary_operand": "A input vector in each PE",
        "full_stream_b_bytes_per_issue_by_k_fold": [bytes_for(ks, b_bits) for ks in k_sizes],
        "steady_array_output_upper_bound_bytes_per_issue": min(c, m) * 4,
        "pe_input_storage_capacity_bytes": bytes_for(r * c * length, a_bits),
    })
    assert ws["a_total_array_input_bytes"] == os["a_total_array_input_bytes"]
    assert ws["array_result_emissions"] == k_folds * m * n
    return {
        "gemm": {"m": m, "n": n, "k": k, "beta": 0},
        "array": {"rows": r, "cols": c, "dot_length": length, "issue_interval": 1,
                  "a_bits": a_bits, "b_bits": b_bits, "accumulator_bits": 32},
        "k_fold_span": r * length,
        "k_fold_count": k_folds,
        "k_fold_valid_lengths": k_sizes,
        "flows": {"os": os, "ws": ws, "is": is_flow},
    }


def verify_mac_coverage(m=7, n=11, k=13, rows=4, cols=3, dot_length=2):
    """Enumerate individual MACs, including partial rows/columns/dot groups."""
    a = [[(i * 3 + j * 2) % 7 - 3 for j in range(k)] for i in range(m)]
    b = [[(i * 5 + j) % 11 - 5 for j in range(n)] for i in range(k)]
    golden = [[sum(a[i][q] * b[q][j] for q in range(k)) for j in range(n)] for i in range(m)]
    results = []
    for flow in ("os", "ws", "is"):
        output = [[0] * n for _ in range(m)]
        coverage = Counter()

        def mac(i, j, q):
            coverage[i, j, q] += 1
            output[i][j] += a[i][q] * b[q][j]

        if flow == "os":
            for m0 in range(0, m, rows):
                for n0 in range(0, n, cols):
                    for k0 in range(0, k, dot_length):
                        for rr in range(min(rows, m - m0)):
                            for cc in range(min(cols, n - n0)):
                                for ll in range(min(dot_length, k - k0)):
                                    mac(m0 + rr, n0 + cc, k0 + ll)
        else:
            free_extent = n if flow == "ws" else m
            stream_extent = m if flow == "ws" else n
            for c0 in range(0, free_extent, cols):
                for k0 in range(0, k, rows * dot_length):
                    for stream in range(stream_extent):
                        for cc in range(min(cols, free_extent - c0)):
                            for rr in range(rows):
                                for ll in range(dot_length):
                                    q = k0 + rr * dot_length + ll
                                    if q < k:
                                        i, j = (stream, c0 + cc) if flow == "ws" else (c0 + cc, stream)
                                        mac(i, j, q)
        assert output == golden
        assert len(coverage) == m * n * k
        assert all(value == 1 for value in coverage.values())
        assert sum(coverage.values()) == m * n * k
        results.append({"dataflow": flow, "exact_integer_gemm": True,
                        "every_mac_covered_exactly_once": True,
                        "macs_checked": m * n * k})
    return {"m": m, "n": n, "k": k, "rows": rows, "cols": cols,
            "dot_length": dot_length, "checks": results}


def derive_growth_examples():
    """Hold dtype and GEMM fixed while changing physical spatial dimensions."""
    configurations = {
        "baseline": (16, 16, 64),
        "double_dot_k_length": (16, 16, 128),
        "double_pe_rows": (32, 16, 64),
        "double_pe_columns": (16, 32, 64),
        "double_all_three": (32, 32, 128),
    }
    cases = {name: derive(256, 256, 4096, r, c, length, 4, 4)
             for name, (r, c, length) in configurations.items()}
    baseline = cases["baseline"]
    for case in cases.values():
        ratios = {}
        for flow in ("os", "ws"):
            original, changed = baseline["flows"][flow], case["flows"][flow]
            ratios[flow] = {key: number(Fraction(changed[key]) / original[key]) for key in (
                "physical_peak_macs_per_issue", "ideal_compute_issues",
                "a_total_array_input_bytes", "b_total_array_input_bytes",
                "array_result_emission_bytes", "final_d_bytes",
                "average_array_output_bytes_per_compute_issue")}
        ratios["os"]["nominal_a_stream_bandwidth"] = number(Fraction(
            case["flows"]["os"]["nominal_full_issue_a_bytes"],
            baseline["flows"]["os"]["nominal_full_issue_a_bytes"]))
        ratios["os"]["nominal_b_stream_bandwidth"] = number(Fraction(
            case["flows"]["os"]["nominal_full_issue_b_bytes"],
            baseline["flows"]["os"]["nominal_full_issue_b_bytes"]))
        ratios["ws"]["nominal_a_stream_bandwidth"] = number(Fraction(
            case["flows"]["ws"]["nominal_full_hardware_a_bytes_per_issue"],
            baseline["flows"]["ws"]["nominal_full_hardware_a_bytes_per_issue"]))
        ratios["ws"]["pe_weight_capacity"] = number(Fraction(
            case["flows"]["ws"]["pe_weight_storage_capacity_bytes"],
            baseline["flows"]["ws"]["pe_weight_storage_capacity_bytes"]))
        case["ratios_to_baseline"] = ratios
    for name in ("double_dot_k_length", "double_pe_rows", "double_pe_columns"):
        assert cases[name]["flows"]["os"]["ideal_compute_issues"] == 8192
        assert cases[name]["flows"]["ws"]["ideal_compute_issues"] == 8192
    assert cases["double_dot_k_length"]["flows"]["ws"]["array_result_emission_bytes"] == 524288
    assert cases["double_pe_columns"]["flows"]["ws"]["array_result_emission_bytes"] == 1048576

    # True M parallelism consists of separate reduction groups, each streaming
    # different M rows. It is not achieved by relabeling a WS reduction row.
    per_replica = derive(128, 256, 4096, 16, 16, 64, 4, 4)["flows"]["ws"]
    replicas = 2
    replicated = {
        "replicas": replicas,
        "mapping": "Two independent 16x16 WS groups, each assigned 128 distinct M rows",
        "dtype": "fp4",
        "dot_length": 64,
        "ideal_concurrent_compute_issues": per_replica["ideal_compute_issues"],
        "aggregate_peak_macs_per_issue": replicas * per_replica["physical_peak_macs_per_issue"],
        "aggregate_a_stream_bytes_per_issue": replicas * per_replica["nominal_full_hardware_a_bytes_per_issue"],
        "aggregate_array_partial_output_bytes_per_issue": replicas * per_replica["steady_array_output_upper_bound_bytes_per_issue"],
        "a_total_array_input_bytes": replicas * per_replica["a_total_array_input_bytes"],
        "b_delivered_to_all_pe_copies_bytes": replicas * per_replica["b_preload_bytes"],
        "b_sram_read_bytes_with_perfect_multicast": per_replica["b_preload_bytes"],
        "b_sram_read_bytes_without_multicast": replicas * per_replica["b_preload_bytes"],
        "k_folds_per_replica": 4,
        "array_result_emission_bytes": replicas * per_replica["array_result_emission_bytes"],
        "final_d_bytes": replicas * per_replica["final_d_bytes"],
        "sram_accumulator_rmw_read_bytes": replicas * per_replica["sram_accumulator_if_every_emission_is_materialized"]["read_bytes"],
        "sram_accumulator_rmw_write_bytes": replicas * per_replica["sram_accumulator_if_every_emission_is_materialized"]["write_bytes"],
        "cross_k_retained_acc_state_aggregate_bytes": replicas * per_replica["resident_accumulator_policy"]["cross_k_state_working_set_bytes"],
        "hardware_requirements": [
            "Independent reduction-chain starts/ends and result ports for every WS group",
            "Independent activation input streams, plus per-group routing and control",
            "Replicated PE weight storage even if SRAM reads use multicast",
            "Enough simultaneous SRAM/ACC port bandwidth; multicast network cost is not zero",
        ],
    }
    assert replicated["ideal_concurrent_compute_issues"] == 8192
    assert replicated["final_d_bytes"] == baseline["flows"]["ws"]["final_d_bytes"]
    assert replicated["array_result_emission_bytes"] == baseline["flows"]["ws"]["array_result_emission_bytes"]
    assert replicated["b_sram_read_bytes_with_perfect_multicast"] == 524288
    assert replicated["b_sram_read_bytes_without_multicast"] == 1048576
    core_ws_rows, core_os_rows = [], []
    for name in ("baseline", "double_dot_k_length", "double_pe_columns"):
        ws = cases[name]["flows"]["ws"]
        core_ws_rows.append({
            "variant": name,
            "ideal_compute_issues": ws["ideal_compute_issues"],
            "a_stream_bytes_per_issue": ws["nominal_full_hardware_a_bytes_per_issue"],
            "array_partial_output_bytes_per_issue": ws["steady_array_output_upper_bound_bytes_per_issue"],
            "k_folds": cases[name]["k_fold_count"],
            "accumulator_sram_write_bytes": ws["sram_accumulator_if_every_emission_is_materialized"]["write_bytes"],
            "accumulator_sram_read_bytes": ws["sram_accumulator_if_every_emission_is_materialized"]["read_bytes"],
            "final_d_bytes": ws["final_d_bytes"],
            "final_d_average_bytes_per_compute_issue": number(Fraction(ws["final_d_bytes"], ws["ideal_compute_issues"])),
            "b_sram_preload_read_bytes_min": ws["b_preload_bytes"],
            "b_sram_preload_read_bytes_max": ws["b_preload_bytes"],
        })
    core_ws_rows.append({
        "variant": "double_m_via_independent_ws_groups",
        "ideal_compute_issues": replicated["ideal_concurrent_compute_issues"],
        "a_stream_bytes_per_issue": replicated["aggregate_a_stream_bytes_per_issue"],
        "array_partial_output_bytes_per_issue": replicated["aggregate_array_partial_output_bytes_per_issue"],
        "k_folds": replicated["k_folds_per_replica"],
        "accumulator_sram_write_bytes": replicated["sram_accumulator_rmw_write_bytes"],
        "accumulator_sram_read_bytes": replicated["sram_accumulator_rmw_read_bytes"],
        "final_d_bytes": replicated["final_d_bytes"],
        "final_d_average_bytes_per_compute_issue": number(Fraction(replicated["final_d_bytes"], replicated["ideal_concurrent_compute_issues"])),
        "b_sram_preload_read_bytes_min": replicated["b_sram_read_bytes_with_perfect_multicast"],
        "b_sram_preload_read_bytes_max": replicated["b_sram_read_bytes_without_multicast"],
    })
    for name in ("baseline", "double_dot_k_length", "double_pe_columns", "double_pe_rows"):
        os = cases[name]["flows"]["os"]
        core_os_rows.append({
            "variant": name,
            "ideal_compute_issues": os["ideal_compute_issues"],
            "a_stream_bytes_per_issue": os["nominal_full_issue_a_bytes"],
            "b_stream_bytes_per_issue": os["nominal_full_issue_b_bytes"],
            "pe_fp32_accumulator_bytes": os["pe_accumulator_physical_bytes"],
            "final_d_bytes": os["final_d_bytes"],
            "final_d_average_bytes_per_compute_issue": os["average_array_output_bytes_per_compute_issue"],
        })
    assert [(x["ideal_compute_issues"], x["a_stream_bytes_per_issue"],
             x["array_partial_output_bytes_per_issue"], x["k_folds"])
            for x in core_ws_rows] == [(16384, 512, 64, 4), (8192, 1024, 64, 2),
                                      (8192, 512, 128, 4), (8192, 1024, 128, 4)]
    assert [x["final_d_average_bytes_per_compute_issue"] for x in core_os_rows] == [16, 32, 32, 32]
    return {
        "scope": "Same FP4 dtype, M=N=256,K=4096; full mapping, II=1; no preload overlap/timing",
        "core_comparison_ws": core_ws_rows,
        "core_comparison_os": core_os_rows,
        "symbolic_growth": {
            "definition": "(R,C,L) -> (uR,vC,wL), fixed dtype and clock",
            "physical_peak_mac_factor": "u*v*w",
            "os_a_full_issue_bandwidth_factor": "u*w",
            "os_b_full_issue_bandwidth_factor": "v*w",
            "os_pe_partial_sum_state_factor": "u*v",
            "os_no_spill_whole_gemm_final_d_byte_factor": "1",
            "os_ideal_average_final_d_bandwidth_factor_when_fully_mapped": "u*v*w",
            "ws_reduction_k_coverage_factor": "u*w",
            "ws_a_full_issue_bandwidth_factor": "u*w",
            "ws_pe_weight_storage_capacity_factor": "u*v*w",
            "ws_b_preload_bandwidth_if_loading_one_PE_row_per_issue_factor": "v*w; loader-dependent, not automatically u*v*w",
            "ws_array_partial_output_width_factor": "v",
            "ws_k_fold_count": "F=ceil(K/(R*L)); after growth F_new=ceil(K/(u*w*R*L))",
            "ws_accumulator_sram_reads_bytes_beta0": "(F-1)*M*N*4",
            "ws_accumulator_sram_writes_bytes": "F*M*N*4",
            "ws_final_d_bytes": "M*N*4 regardless of F; D aliases accumulator in these counts",
            "ws_total_partial_emissions_factor": "F_new/F; approximate 1/(u*w) only without ceil/tail effects",
            "ws_global_B_preload_bytes_with_full_M_reuse": "K*N*b_bits/8; constant, despite PE capacity growth",
            "ws_R_semantics": "R extends K reduction, not M; true M parallelism needs independent/replicated WS groups",
            "ws_underfilled_k_limit": "useful A bytes/issue=min(K_remaining,R*L)*a_bits/8; cannot scale beyond available K",
            "ws_final_vs_partial_bandwidth": "Array-edge partial-sum width scales with v. Whole-GEMM final-D bytes are fixed, so final-D average over a shortened full-utilization window can scale by u*v*w; these are different metrics.",
        },
        "cases": cases,
        "true_ws_m_parallelism_two_groups": replicated,
    }


def main():
    modes = {"bf16": (16, 16), "fp8": (32, 8), "fp4": (64, 4)}
    small_k = {name: derive(256, 256, 256, 16, 16, length, bits, bits)
               for name, (length, bits) in modes.items()}
    large_k = {name: derive(256, 256, 4096, 16, 16, length, bits, bits)
               for name, (length, bits) in modes.items()}
    matched = derive(64, 16, 1024, 16, 16, 64, 4, 4)
    os, ws = matched["flows"]["os"], matched["flows"]["ws"]
    assert os["ideal_compute_issues"] == ws["ideal_compute_issues"] == 64
    assert os["a_total_array_input_bytes"] == os["b_total_array_input_bytes"] == 32768
    assert ws["a_total_array_input_bytes"] == 32768
    assert ws["b_preload_bytes"] == 8192
    assert os["final_d_bytes"] == ws["final_d_bytes"] == 4096
    assert ws["preload_transfer_cycles_lower_bound"] == 16
    assert small_k["bf16"]["flows"]["os"]["average_array_output_bytes_per_compute_issue"] == 64
    assert small_k["fp8"]["flows"]["os"]["average_array_output_bytes_per_compute_issue"] == 128
    assert small_k["fp4"]["flows"]["os"]["average_array_output_bytes_per_compute_issue"] == 256
    assert all(x["flows"]["ws"]["average_array_output_bytes_per_compute_issue"] == 64 for x in small_k.values())
    profiles = {"1:2:4": (16, 32, 64), "1:2:6": (16, 32, 96),
                "1:2:8": (16, 32, 128), "1:4:8": (16, 64, 128)}
    underfill = []
    for profile, lengths in profiles.items():
        for name, length, bits in zip(("bf16", "fp8", "fp4"), lengths, (16, 8, 4)):
            row = derive(256, 256, 256, 16, 16, length, bits, bits)
            underfill.append({"profile": profile, "precision": name,
                              "dot_length": length, "active_reduction_rows": ceil_div(256, length),
                              "last_active_row_valid_dot_lanes": 256 - (ceil_div(256, length) - 1) * length,
                              "ws_mapping_utilization": row["flows"]["ws"]["mapping_utilization"],
                              "os_mapping_utilization": row["flows"]["os"]["mapping_utilization"],
                              "ws_compute_issues": row["flows"]["ws"]["ideal_compute_issues"],
                              "os_compute_issues": row["flows"]["os"]["ideal_compute_issues"],
                              "ws_stream_a_bytes_per_issue": row["flows"]["ws"]["full_stream_a_bytes_per_issue_by_k_fold"][0],
                              "nominal_full_hardware_a_bytes_per_issue": bytes_for(16 * length, bits)})
    e96 = next(x for x in underfill if x["profile"] == "1:2:6" and x["precision"] == "fp4")
    assert e96["active_reduction_rows"] == 3 and e96["last_active_row_valid_dot_lanes"] == 64
    assert e96["ws_mapping_utilization"] == float(Fraction(1, 6))
    report = {
        "schema_version": 1,
        "status": "passed",
        "kind": "analytical derivation and tiny integer MAC-coverage checks; NOT simulation",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "units": {"bytes": "logical useful bits / 8; no SRAM word padding or metadata",
                  "cycle": "ideal compute issue, II=1; no latency, skew, stalls, preload or drain",
                  "mac": "one scalar multiply-accumulate, not two operations"},
        "assumptions": [
            "D=A*B with beta=0 and FP32 C/partial-sum/D paths; no initial C reads",
            "Same R*C dot PEs and same L per compared precision; this does not assert equal hardware area",
            "A reconfigurable fabric must support the listed spatial mappings, stationary-vector registers and partial-sum routing",
            "No independent partitioning of unused WS/IS reduction rows; unused rows are masked",
            "Physical FP32 pass-through of inactive rows and PE pipeline latency are not counted",
            "OS keeps one R*C result tile in PE accumulators across all K",
            "WS loop order n_tile,k_tile,m retains each weight tile across all M",
            "IS loop order m_tile,k_tile,n retains each input tile across all N",
            "WS/IS array-edge partial sums may go to registers, SRAM accumulator, or a forwarding path; only actual SRAM crossings are CACTI accesses",
            "When materializing every partial sum in one SRAM accumulator, final D aliases that accumulator; separate D copies need explicit extra events",
            "512 B/cycle preload is an assumed packed transfer lower bound, not a simulated PE loading network",
            "Compute issues omit preload; additive preload+compute is still not true elapsed time",
            "Operand SRAM refill from higher levels, cache reuse and cross-tile operand retention beyond stated loops are excluded",
        ],
        "gemm_256_cubed": small_k,
        "gemm_256_256_4096": large_k,
        "matched_fp4_64_16_1024": matched,
        "four_profiles_k256_underfill": underfill,
        "fixed_dtype_growth": derive_growth_examples(),
        "mac_coverage_checks": [verify_mac_coverage(dot_length=2), verify_mac_coverage(dot_length=5)],
        "loop_order_capacity_tradeoff": {
            "ws_full_m_residency_bytes": "M * min(C,N) * 4 when retaining partial sums across K folds",
            "ws_register_acc_block_m": "floor(accumulator_capacity_bytes / (min(C,N)*4))",
            "ws_blocked_loop_order": ["n_tile", "m_block", "k_tile", "m_in_block"],
            "ws_b_preload_with_blocked_m_bytes": "ceil(M/m_block) * K * N * b_bits / 8",
            "example_m256_n256_k4096": {
                "full_m_acc_state_per_n_tile_bytes": 16384,
                "if_acc_state_only_1024_bytes_m_block": 16,
                "weight_preload_multiplier_for_m_block16": 16,
                "consequence": "Weight reuse advantage over OS disappears for this schedule; A traffic remains unchanged",
            },
            "alternative": "Keep n_tile,k_tile,m and spill partial sums to external SRAM; count its RMW traffic instead of reloading B",
        },
        "trace_schema_recommendations": [
            "Separate logical tensors A,B,C_initial,partial_sum,D from physical buffers and alias groups",
            "Record spatial axis mapping, temporal loop order, tile sizes, partitions, phase and precision",
            "Each transfer records source/destination storage levels and instances; PE forwarding is not SRAM traffic",
            "Record logical indices/ranges, valid masks, useful bits, packed bits, address/bank/lane and transaction width",
            "Represent preload, streaming, initial seed, reduction forwarding, spill/reload and final drain as separate phases",
            "Track partial-sum version, accumulation dependency and final-result marker, including beta initialization",
            "Represent accumulator location/capacity/ports/latency and D aliasing explicitly",
            "Attach issue and completion times plus stalls only when a scheduler actually computes them",
            "Validate MAC coverage, exactly one final D per m,n, value dependencies, storage lifetimes and per-port constraints",
        ],
        "primary_sources": [
            {"url": "https://arxiv.org/html/2610.19645v1", "section": "II-A",
             "role": "Example of WS dot-product PEs, fixed FP32 inter-PE partial sums and external result accumulation; not a specification of this custom H16Q32E64 array"},
            {"url": "https://raw.githubusercontent.com/scalesim-project/SCALE-Sim/main/scalesim/compute/systolic_compute_ws.py",
             "role": "Primary code showing WS row/reduction and column/output mapping plus output emissions per K fold"},
            {"url": "https://raw.githubusercontent.com/scalesim-project/SCALE-Sim/main/scalesim/compute/systolic_compute_is.py",
             "role": "Primary code showing IS reduction rows and transposed free-axis mapping"},
        ],
    }
    destination = Path(__file__).with_name("derived.json")
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print("passed: {} (analytical, 6 exact integer schedule checks)".format(destination))


if __name__ == "__main__":
    main()
