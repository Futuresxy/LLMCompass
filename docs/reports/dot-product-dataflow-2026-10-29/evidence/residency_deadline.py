#!/usr/bin/env python3
"""Integer transfer-window examples; no CACTI, RTL or LLMCompass model changes.

The server transfers at most B bytes in each integer cycle. Startup reserves S
whole cycles before service, and a job must finish by the end of W cycles.
This is an exact check of this declared toy contract, not a hardware simulator.
"""

import hashlib
import json
from pathlib import Path


def ceil_div(value, divisor):
    return (value + divisor - 1) // divisor


def service_window(name, payload_bytes, bandwidth_bytes_per_cycle, window_cycles,
                   startup_cycles=0, separate_live_state_assumed=True):
    for value in (payload_bytes, bandwidth_bytes_per_cycle, window_cycles):
        assert isinstance(value, int) and value > 0
    assert isinstance(startup_cycles, int) and startup_cycles >= 0
    assert isinstance(separate_live_state_assumed, bool)
    remaining = payload_bytes
    trace = []
    for cycle in range(window_cycles):
        transferred = min(remaining, bandwidth_bytes_per_cycle) if cycle >= startup_cycles else 0
        trace.append(transferred)
        remaining -= transferred
    transfer_cycles = ceil_div(payload_bytes, bandwidth_bytes_per_cycle)
    completion_cycles = startup_cycles + transfer_cycles
    rate_fits = completion_cycles <= window_cycles
    assert (remaining == 0) == rate_fits
    assert sum(trace) + remaining == payload_bytes
    assert all(0 <= moved <= bandwidth_bytes_per_cycle for moved in trace)
    usable = window_cycles - startup_cycles
    return {
        "name": name,
        "payload_bytes": payload_bytes,
        "bandwidth_bytes_per_cycle": bandwidth_bytes_per_cycle,
        "window_cycles": window_cycles,
        "startup_cycles": startup_cycles,
        "transfer_cycles_without_startup": transfer_cycles,
        "completion_cycles_from_window_start": completion_cycles,
        "service_window_fits": rate_fits,
        "remaining_bytes_at_deadline": remaining,
        "minimum_integer_bytes_per_cycle_to_fit": ceil_div(payload_bytes, usable) if usable > 0 else None,
        "average_payload_bytes_per_window_cycle": payload_bytes / window_cycles,
        "separate_current_and_next_live_state_assumed": separate_live_state_assumed,
        "current_plus_next_payload_storage_bytes_if_double_buffered": 2 * payload_bytes,
        "can_hide_transfer_under_stated_assumptions": rate_fits and separate_live_state_assumed,
        "hardware_bank_conflicts_and_dependencies_verified": False,
        "sram_lane_area_estimated": False,
        "service_trace_bytes_per_cycle": trace,
    }


def output_backlog(payload_bytes, interval_cycles, bandwidth_bytes_per_cycle,
                   tiles=8, queue_capacity_bytes=4096):
    """Arrival at interval start, followed by interval_cycles of service.

    This is an unbounded queue ledger with an observed capacity threshold. It
    does not drop bytes or silently stall the producer after crossing capacity.
    """
    queue, history, first_overflow = 0, [], None
    for tile in range(1, tiles + 1):
        queue += payload_bytes
        peak = queue
        if queue > queue_capacity_bytes and first_overflow is None:
            first_overflow = tile
        sent = min(queue, interval_cycles * bandwidth_bytes_per_cycle)
        queue -= sent
        history.append({"tile": tile, "after_arrival_bytes": peak,
                        "served_bytes": sent, "after_interval_bytes": queue})
    assert queue == tiles * payload_bytes - sum(x["served_bytes"] for x in history)
    return {
        "payload_bytes_per_tile": payload_bytes,
        "tile_arrival_interval_cycles": interval_cycles,
        "service_bytes_per_cycle": bandwidth_bytes_per_cycle,
        "queue_capacity_bytes_observed": queue_capacity_bytes,
        "first_overflow_at_arrival_tile": first_overflow,
        "steady_backlog_growth_bytes_per_tile": max(0, payload_bytes - interval_cycles * bandwidth_bytes_per_cycle),
        "history": history,
        "scope": "Unbounded ledger shows why a finite FIFO cannot cure a sustained rate deficit; overflow means real hardware must stall or increase service",
    }


def main():
    a_cases = [
        service_window("A_base_8KiB_window256", 8192, 128, 256),
        service_window("A_grown_16KiB_window256", 16384, 128, 256),
        service_window("A_base_8KiB_window64", 8192, 128, 64),
        service_window("A_grown_16KiB_window64", 16384, 128, 64),
        service_window("A_enough_bandwidth_but_no_separate_live_state", 8192, 128, 256,
                       separate_live_state_assumed=False),
        service_window("A_startup4_breaks_exact64cycle_budget", 8192, 128, 64,
                       startup_cycles=4),
    ]
    assert [x["service_window_fits"] for x in a_cases[:4]] == [True, True, True, False]
    assert a_cases[3]["remaining_bytes_at_deadline"] == 8192
    assert a_cases[3]["minimum_integer_bytes_per_cycle_to_fit"] == 256
    assert a_cases[4]["service_window_fits"] and not a_cases[4]["can_hide_transfer_under_stated_assumptions"]
    assert a_cases[5]["completion_cycles_from_window_start"] == 68

    os_cases = []
    for name, k, length in (("OS_K4096_L64", 4096, 64),
                            ("OS_K4096_L128", 4096, 128),
                            ("OS_K512_L128", 512, 128)):
        issue_cycles = ceil_div(k, length)
        item = service_window(name, 16 * 16 * 32 // 8, 64, issue_cycles)
        item.update({"array_rows": 16, "array_cols": 16, "gemm_k": k,
                     "dot_length": length, "accumulator_bits": 32,
                     "compute_issue_cycles": issue_cycles,
                     "available_window_definition": "The following tile's compute window; assumes old output state remains separate and independently readable",
                     "single_tile_serial_compute_plus_drain_lower_bound_cycles": issue_cycles + item["completion_cycles_from_window_start"],
                     "ideal_repeated_tile_period_lower_bound_cycles": max(issue_cycles, item["completion_cycles_from_window_start"]),
                     "not_measured_kernel_elapsed_time": True})
        os_cases.append(item)
    assert [x["compute_issue_cycles"] for x in os_cases] == [64, 32, 4]
    assert [x["transfer_cycles_without_startup"] for x in os_cases] == [16, 16, 16]
    assert [x["average_payload_bytes_per_window_cycle"] for x in os_cases] == [16, 32, 256]
    assert [x["service_window_fits"] for x in os_cases] == [True, True, False]
    queue = output_backlog(1024, 4, 64)
    assert queue["steady_backlog_growth_bytes_per_tile"] == 768
    assert queue["first_overflow_at_arrival_tile"] == 6
    report = {
        "schema_version": 1,
        "status": "passed",
        "kind": "Exact integer service-window examples under declared constant-bandwidth assumptions; NOT hardware timing/PPA simulation",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "equations": {
            "load_or_drain_cycles": "S + ceil(V/B), with startup S defined as nonoverlapped cycles before byte service",
            "deadline": "S + ceil(V/B) <= W",
            "minimum_ideal_integer_bandwidth": "ceil(V/(W-S)), only when W>S",
            "output_mean_if_one_tile_every_Q_cycles": "V_D/Q",
            "lane_retention_condition": "The existing organization satisfies deadline, per-bank ports, storage lifetime/capacity and dependency constraints; then extra SRAM lane area may be zero",
        },
        "a_preload_examples": a_cases,
        "os_serialized_drain_examples": os_cases,
        "short_k_repeated_output_backlog": queue,
        "conditions_not_to_omit": [
            "A must actually be reused from a nearer resident store; staging through a register without reuse cannot reduce required long-run SRAM bytes",
            "A-stationary is IS at that storage boundary; a blocked/hierarchical design can have different stationary operands at different levels",
            "Overlap requires lifetime-safe current/next state separation, such as two register sets or independent banks; in-place reuse needs an explicit proof that old data is retired before overwrite",
            "The resident store and its distribution network must still sustain the PE-facing rate, even if the upstream SRAM lane count is unchanged",
            "Increasing dot K coverage may increase preload V or shorten W; residency is not a guarantee of constant bandwidth",
            "OS old partial sums must remain available during multicycle drain; two L0 slots alone do not establish safe overlap with a new PE computation",
            "A single isolated tile can always take longer to drain if latency is allowed; zero extra lanes does not imply zero performance cost",
            "A finite output FIFO smooths bursts but cannot cure a sustained arrival rate above service capacity",
            "V must include actual word packing, padding, metadata and required copies when connected to a physical-memory model",
            "B is achieved service at the relevant boundary, accounting for shared-port competition and bank conflicts; peak marketing bandwidth is not sufficient",
            "Any additional register, FIFO, mux, routing, clock and dependency-control area/energy remains payable even when extra SRAM lane area is zero",
            "The existing dot_product_sram.py full-tile output burst policy is not silently replaced by these independent drain examples",
        ],
        "primary_sources": [
            {"url": "https://github.com/ucb-bar/gemmini#systolic-array-and-transposer",
             "support": "Gemmini documents row-per-cycle inputs/outputs and retained preloads; its loop controllers use double buffering and access/execute overlap",
             "limit": "Mechanism example, not a proof of custom A residency or arbitrary SRAM width"},
            {"url": "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/efficient_gemm.html#pipelining",
             "support": "CUTLASS documents separate current/next shared-memory tiles and register fragments",
             "limit": "Software-visible GPU hierarchy; does not reveal physical Tensor Core SRAM lanes"},
            {"url": "https://docs.nvidia.com/cutlass/latest/media/docs/cpp/efficient_gemm.html#hopper-warp-specialization",
             "support": "Persistent ping-pong assigns different output tiles to two consumer warp groups, allowing epilogue/math overlap",
             "limit": "Overlap uses separate state and synchronization; it is not free or guaranteed at every shape"},
        ],
    }
    destination = Path(__file__).with_suffix(".json")
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print("passed: {} (6 preload windows, 3 output windows, periodic backlog)".format(destination))


if __name__ == "__main__":
    main()
