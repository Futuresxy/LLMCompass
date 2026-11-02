"""CACTI-backed, explicitly banked operand SRAM adjacent to a compute array.

This module does not reinterpret Core.SRAM_size as Ascend L0 or GPU shared memory.
One CACTI call represents one independently accessible, untagged SRAM macro.
Inter-macro wiring/control costs must be supplied separately when known.
"""

from dataclasses import asdict, dataclass, replace
import math
from typing import Dict, Iterable, Optional

from cost_model.cacti import MacroSpec


def _number(value, name, minimum=0, strict=False):
    if not math.isfinite(value) or value < minimum or (strict and value == minimum):
        raise ValueError("{} must be finite and {} {}".format(
            name, ">" if strict else ">=", minimum))


def _integer(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError("{} must be an integer >= {}".format(name, minimum))


@dataclass(frozen=True)
class BufferSpec:
    name: str
    capacity_bytes: int
    frequency_hz: float
    slots: int = 2
    layout: str = "separate_slots"
    capacity_basis: str = "total"
    technology_nm: float = 22
    temperature_k: int = 300
    bank_efficiency: float = 1.0
    interface_area_mm2: float = 0.0
    interface_leakage_mw: float = 0.0
    live_tile_bytes: int = 0

    def __post_init__(self):
        if not self.name:
            raise ValueError("Buffer name is required")
        _integer(self.capacity_bytes, "capacity_bytes")
        _integer(self.live_tile_bytes, "live_tile_bytes", 0)
        if self.slots not in (1, 2):
            raise ValueError("slots must be 1 or 2")
        if self.layout not in ("separate_slots", "shared_macro"):
            raise ValueError("layout must be separate_slots or shared_macro")
        if self.capacity_basis not in ("total", "per_slot"):
            raise ValueError("capacity_basis must be total or per_slot")
        _number(self.frequency_hz, "frequency_hz", strict=True)
        _number(self.bank_efficiency, "bank_efficiency", strict=True)
        if self.bank_efficiency > 1:
            raise ValueError("bank_efficiency must not exceed 1")
        _number(self.interface_area_mm2, "interface_area_mm2")
        _number(self.interface_leakage_mw, "interface_leakage_mw")
        if self.total_capacity_bytes % self.slots:
            raise ValueError("Total physical capacity must divide exactly into slots")
        if self.live_tile_bytes > self.total_capacity_bytes // self.slots:
            raise ValueError("live_tile_bytes exceeds capacity of one slot")

    @property
    def total_capacity_bytes(self):
        return self.capacity_bytes * (self.slots if self.capacity_basis == "per_slot" else 1)


@dataclass(frozen=True)
class Organization:
    banks: int
    width_bits: int
    rw_ports: int = 1
    read_ports: int = 0
    write_ports: int = 0

    def __post_init__(self):
        _integer(self.banks, "banks")
        _integer(self.width_bits, "width_bits", 64)
        if self.banks & (self.banks - 1):
            raise ValueError("banks must be a power of two")
        if self.width_bits % 8 or self.width_bits & (self.width_bits - 1):
            raise ValueError("width_bits must be a byte-aligned power of two >= 64")
        for name in ("rw_ports", "read_ports", "write_ports"):
            _integer(getattr(self, name), name, 0)
        if not (self.rw_ports + self.read_ports) or not (self.rw_ports + self.write_ports):
            raise ValueError("Operand SRAM must support both reads and refills")


@dataclass(frozen=True)
class Demand:
    # Physical transferred bytes, including packing/padding, averaged per array clock.
    read_bytes_per_cycle: float
    write_bytes_per_cycle: float = 0.0
    refill_bytes_per_tile: int = 0
    compute_window_cycles: float = 0.0

    def __post_init__(self):
        _number(self.read_bytes_per_cycle, "read_bytes_per_cycle")
        _number(self.write_bytes_per_cycle, "write_bytes_per_cycle")
        _integer(self.refill_bytes_per_tile, "refill_bytes_per_tile", 0)
        _number(self.compute_window_cycles, "compute_window_cycles")
        if bool(self.refill_bytes_per_tile) != bool(self.compute_window_cycles):
            raise ValueError("refill_bytes_per_tile and compute_window_cycles must be specified together")

    def scaled(self, factor):
        """Same workload bytes at a higher service rate, not more total bytes."""
        _number(factor, "demand scale", strict=True)
        return replace(self, read_bytes_per_cycle=self.read_bytes_per_cycle * factor,
                       write_bytes_per_cycle=self.write_bytes_per_cycle * factor,
                       compute_window_cycles=self.compute_window_cycles / factor)


def characterize(spec: BufferSpec, organization: Organization, runner) -> dict:
    independent_slots = spec.slots if spec.layout == "separate_slots" else 1
    macro_count = organization.banks * independent_slots
    total = spec.total_capacity_bytes
    if total % macro_count:
        raise ValueError("Physical capacity is not divisible by the number of macros")
    macro_capacity = total // macro_count
    word_bytes = organization.width_bits // 8
    if macro_capacity < 64 or macro_capacity % word_bytes:
        raise ValueError("Each macro must contain >=64 bytes and an integer number of words")
    macro_spec = MacroSpec(
        capacity_bytes=macro_capacity, width_bits=organization.width_bits,
        rw_ports=organization.rw_ports, read_ports=organization.read_ports,
        write_ports=organization.write_ports, technology_nm=spec.technology_nm,
        temperature_k=spec.temperature_k,
    )
    macro = runner.simulate(macro_spec)
    # Synchronous, whole-cycle issue model; access latency and cycle time differ.
    interval = max(1, math.ceil(macro.cycle_time_ns * spec.frequency_hz * 1e-9 - 1e-12))
    latency = max(1, math.ceil(macro.access_time_ns * spec.frequency_hz * 1e-9 - 1e-12))
    per_port = organization.banks * word_bytes * spec.bank_efficiency / interval
    return {
        "organization": asdict(organization),
        "total_capacity_bytes": total,
        "slot_capacity_bytes": total // spec.slots,
        "macro_count": macro_count,
        "macro_capacity_bytes": macro_capacity,
        "active_read_banks": organization.banks,
        "physical_slot_groups": independent_slots,
        "issue_interval_cycles": interval,
        "access_latency_cycles": latency,
        "read_bandwidth_bytes_per_cycle": per_port * (organization.read_ports + organization.rw_ports),
        "write_bandwidth_bytes_per_cycle": per_port * (organization.write_ports + organization.rw_ports),
        "dedicated_read_bandwidth_bytes_per_cycle": per_port * organization.read_ports,
        "dedicated_write_bandwidth_bytes_per_cycle": per_port * organization.write_ports,
        "shared_rw_capacity_bytes_per_cycle": per_port * organization.rw_ports,
        "read_write_use_distinct_macros": independent_slots == 2,
        "macro_area_mm2": macro.area_mm2 * macro_count,
        "area_mm2": macro.area_mm2 * macro_count + spec.interface_area_mm2,
        "leakage_w": ((macro.leakage_mw + macro.gate_leakage_mw) * macro_count
                      + spec.interface_leakage_mw) * 1e-3,
        "macro": asdict(macro),
        "macro_spec": asdict(macro_spec),
        "cost_boundary": "SRAM macros plus explicitly supplied interface area/leakage; excludes unspecified inter-macro wiring, mux/control, clock power and upstream memory",
    }


def evaluate(spec: BufferSpec, characterization: dict, demand: Demand) -> dict:
    c = characterization
    read = demand.read_bytes_per_cycle
    write = demand.write_bytes_per_cycle
    power_write = write
    reasons = []
    refill_deadline_write = 0.0
    if demand.refill_bytes_per_tile:
        power_write = max(write, demand.refill_bytes_per_tile / demand.compute_window_cycles)
        usable = demand.compute_window_cycles - c["access_latency_cycles"]
        if usable <= 0:
            reasons.append("Refill window is shorter than SRAM access latency")
        else:
            refill_deadline_write = demand.refill_bytes_per_tile / usable
            write = max(write, refill_deadline_write)
        if spec.slots != 2:
            reasons.append("Overlapped next-tile refill requires two slots in this model")
        if demand.refill_bytes_per_tile > c["slot_capacity_bytes"]:
            reasons.append("Refill tile exceeds the physical capacity of one slot")
    tolerance = 1e-9
    if read > c["read_bandwidth_bytes_per_cycle"] + tolerance:
        reasons.append("Array read bandwidth exceeds available supply")
    if write > c["write_bandwidth_bytes_per_cycle"] + tolerance:
        reasons.append("Refill write bandwidth exceeds available supply")
    if not c["read_write_use_distinct_macros"]:
        shared = (max(0.0, read - c["dedicated_read_bandwidth_bytes_per_cycle"])
                  + max(0.0, write - c["dedicated_write_bandwidth_bytes_per_cycle"]))
        if shared > c["shared_rw_capacity_bytes_per_cycle"] + tolerance:
            reasons.append("Read and write demand compete for shared RW ports")
    word_bytes = c["organization"]["width_bits"] / 8
    read_accesses = read / word_bytes
    write_accesses = power_write / word_bytes
    # Workload is assumed to coalesce complete physical words; no data-dependent switching model.
    requested_dynamic = spec.frequency_hz * 1e-9 * (
        read_accesses * c["macro"]["read_energy_nj"]
        + write_accesses * c["macro"]["write_energy_nj"])
    feasible = not reasons
    return {
        "feasible": feasible, "reasons": reasons, "demand": asdict(demand),
        "required_read_bytes_per_cycle": read,
        "required_write_bytes_per_cycle": write,
        "average_write_bytes_per_cycle": power_write,
        "read_margin_bytes_per_cycle": c["read_bandwidth_bytes_per_cycle"] - read,
        "write_margin_bytes_per_cycle": c["write_bandwidth_bytes_per_cycle"] - write,
        "refill_deadline_write_bytes_per_cycle": refill_deadline_write,
        "read_accesses_per_cycle": read_accesses,
        "write_accesses_per_cycle": write_accesses,
        "dynamic_power_w": requested_dynamic if feasible else None,
        "total_power_w": requested_dynamic + c["leakage_w"] if feasible else None,
        "power_boundary": "CACTI full-word access energy times requested service rate, plus leakage; infeasible rates have no achieved-power estimate",
    }


def workload_energy(characterization: dict, read_accesses: int, write_accesses: int,
                    duration_seconds: float) -> dict:
    """Energy uses actual macro transactions (not useful bytes or array utilization)."""
    _integer(read_accesses, "read_accesses", 0)
    _integer(write_accesses, "write_accesses", 0)
    _number(duration_seconds, "duration_seconds", strict=True)
    macro = characterization["macro"]
    dynamic = (read_accesses * macro["read_energy_nj"]
               + write_accesses * macro["write_energy_nj"]) * 1e-9
    leakage = characterization["leakage_w"] * duration_seconds
    return {"dynamic_energy_j": dynamic, "leakage_energy_j": leakage,
            "total_energy_j": dynamic + leakage,
            "average_power_w": (dynamic + leakage) / duration_seconds}


def _modes(demand):
    if isinstance(demand, Demand):
        return {"default": demand}
    if (not isinstance(demand, dict) or not demand
            or any(not isinstance(value, Demand) for value in demand.values())):
        raise ValueError("Expected Demand or a nonempty mapping of mode names to Demand")
    return demand


def evaluate_modes(spec: BufferSpec, characterization: dict, demands) -> dict:
    """Require each mutually exclusive mode to fit, without adding their traffic."""
    modes = {name: evaluate(spec, characterization, demand)
             for name, demand in _modes(demands).items()}
    if list(modes) == ["default"]:
        return modes["default"]
    feasible = all(m["feasible"] for m in modes.values())
    return {
        "feasible": feasible,
        "reasons": ["{}: {}".format(name, reason) for name, mode in modes.items()
                    for reason in mode["reasons"]],
        "mode_evaluations": modes,
        "dynamic_power_w": max(m["dynamic_power_w"] for m in modes.values()) if feasible else None,
        "total_power_w": max(m["total_power_w"] for m in modes.values()) if feasible else None,
        "power_aggregation": "Maximum over mutually exclusive modes; not simultaneous traffic",
    }


def plan_upgrade(spec: BufferSpec, baseline_demand, target_demand,
                 organizations: Iterable[Organization], runner,
                 installed: Optional[Organization] = None,
                 baseline_headroom: float = 1.0) -> dict:
    """Keep installed hardware when feasible; otherwise search the same-capacity pool.

    Demand arguments may be single Demand objects or mappings for mutually
    exclusive precision modes. One physical organization must support every mode.
    """
    _number(baseline_headroom, "baseline_headroom", 1)
    baseline_modes, target_modes = _modes(baseline_demand), _modes(target_demand)
    if set(baseline_modes) != set(target_modes):
        raise ValueError("Baseline and target mode sets must match")
    organizations = list(dict.fromkeys(organizations))
    if installed is not None and installed not in organizations:
        organizations.append(installed)
    candidates, failures = [], []
    for organization in organizations:
        try:
            candidates.append(characterize(spec, organization, runner))
        except (ValueError, RuntimeError, OSError) as error:
            failures.append({"organization": asdict(organization), "error": str(error)})

    def scored(c, demand):
        return dict(c, **evaluate_modes(spec, c, demand))

    def best(demand):
        feasible = [scored(c, demand) for c in candidates]
        feasible = [c for c in feasible if c["feasible"]]
        return min(feasible, key=lambda c: (c["area_mm2"], c["total_power_w"],
                                           c["access_latency_cycles"])) if feasible else None

    baseline = None
    if installed is not None:
        for candidate in candidates:
            if candidate["organization"] == asdict(installed):
                baseline = scored(candidate, baseline_demand)
                break
        if baseline is None or not baseline["feasible"]:
            raise ValueError("Installed organization cannot be characterized or serve baseline demand")
    else:
        chosen = best({name: demand.scaled(baseline_headroom)
                       for name, demand in baseline_modes.items()})
        if chosen is not None:
            baseline = scored(chosen, baseline_demand)
    if baseline is None:
        return {"status": "infeasible", "reason": "No baseline candidate satisfies the requested headroom",
                "baseline": None, "target": None, "delta_area_mm2": None,
                "candidates": candidates, "failures": failures}

    on_installed = scored(baseline, target_demand)
    target = on_installed if on_installed["feasible"] else best(target_demand)
    status = "retained" if on_installed["feasible"] else "upgraded" if target else "infeasible"
    return {
        "status": status,
        "baseline": baseline, "target": target,
        "target_on_installed": on_installed,
        "upgrade_required": not on_installed["feasible"],
        "delta_area_mm2": target["area_mm2"] - baseline["area_mm2"] if target else None,
        "delta_total_power_w": target["total_power_w"] - baseline["total_power_w"] if target else None,
        "baseline_headroom_factor": baseline_headroom if installed is None else None,
        "baseline_selection": "explicit_installed" if installed is not None else "minimum_area_at_headroom_demand",
        "candidates": candidates, "failures": failures,
    }


def scale_mode_demands(baseline_ratios: Dict[str, float], target_ratios: Dict[str, float],
                       baseline_demands: dict) -> dict:
    """Scale within the same precision, frequency, reuse and physical representation.

    A 1:2:4 -> 1:2:8 change scales FP4 only. Ratios must have the same absolute
    BF16 reference; changes in clock/reuse/layout require explicit input demands.
    """
    if set(baseline_ratios) != set(target_ratios) or set(baseline_demands) != set(baseline_ratios):
        raise ValueError("Mode sets must match exactly")
    result = {}
    for mode in baseline_ratios:
        _number(baseline_ratios[mode], "baseline mode throughput", strict=True)
        _number(target_ratios[mode], "target mode throughput", strict=True)
        factor = target_ratios[mode] / baseline_ratios[mode]
        result[mode] = {name: demand.scaled(factor) for name, demand in baseline_demands[mode].items()}
    return result
