"""Derive array-local A/B/C traffic from dot-product PE organization.

This is a structural, issue-window model, not a replacement for SCALE-Sim's
fill/drain/stall timeline. A is broadcast across each PE row; B across each
column. Each logical group has a fixed set of independently accessed SRAM
macros (slices). No implicit cross-cycle packing or lane remapping is assumed.
"""

from dataclasses import asdict, dataclass
import math

from cost_model.local_sram import BufferSpec, Demand, evaluate


def _integer(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError("{} must be an integer >= {}".format(name, minimum))


@dataclass(frozen=True)
class PrecisionMode:
    name: str
    dot_length: int
    a_bits: int
    b_bits: int
    issue_interval_cycles: int = 1

    def __post_init__(self):
        if not self.name:
            raise ValueError("Mode name is required")
        for field in ("dot_length", "a_bits", "b_bits", "issue_interval_cycles"):
            _integer(getattr(self, field), field)


@dataclass(frozen=True)
class ArrayTile:
    rows: int
    cols: int
    k: int
    active_rows: int = None
    active_cols: int = None
    accumulator_bits: int = 32
    accumulator_policy: str = "resident"
    initial_c: bool = False

    def __post_init__(self):
        for field in ("rows", "cols", "k", "accumulator_bits"):
            _integer(getattr(self, field), field)
        for field, limit in (("active_rows", self.rows), ("active_cols", self.cols)):
            value = getattr(self, field)
            if value is not None:
                _integer(value, field)
                if value > limit:
                    raise ValueError("{} exceeds physical array shape".format(field))
        if self.accumulator_policy not in ("resident", "l0c_rmw"):
            raise ValueError("accumulator_policy must be resident or l0c_rmw")
        if not isinstance(self.initial_c, bool):
            raise ValueError("initial_c must be boolean")


def _event(index, cycle, groups, bits, full_bits, phase):
    return {"issue_index": index, "cycle": cycle, "groups": groups,
            "bits_per_group": bits, "full_bits_per_group": full_bits,
            "phase": phase}


def _summarize(groups, required_groups, reads, writes, cycles, nominal_read, nominal_write):
    def total(events):
        return sum(e["groups"] * e["bits_per_group"] for e in events) / 8

    def peak(events):
        return max((e["groups"] * e["bits_per_group"] / 8 for e in events), default=0.0)

    return {
        "groups": groups, "physical_groups_required": required_groups,
        "read_events": reads, "write_events": writes,
        "useful_read_bytes": total(reads), "useful_write_bytes": total(writes),
        "peak_read_bytes_per_issue": peak(reads),
        "peak_write_bytes_per_issue": peak(writes),
        "average_read_bytes_per_cycle": total(reads) / cycles,
        "average_write_bytes_per_cycle": total(writes) / cycles,
        "nominal_read_bytes_per_issue": nominal_read,
        "nominal_write_bytes_per_issue": nominal_write,
    }


def generate_traffic(tile: ArrayTile, mode: PrecisionMode) -> dict:
    """Generate useful-bit events; physical word rounding is candidate-dependent.

    A/B refill each input tile once. Their refill requests are spread over the
    compute issue window, rather than forced to occur at the compute read bursts.
    C resident writeback is an end-of-tile burst. All averages are normalized to
    the issue window; they are not measured kernel bandwidth or total latency.
    """
    h = tile.rows if tile.active_rows is None else tile.active_rows
    w = tile.cols if tile.active_cols is None else tile.active_cols
    count = math.ceil(tile.k / mode.dot_length)
    cycles = count * mode.issue_interval_cycles
    operands = {}
    for name, groups, physical_groups, bits in (
            ("A", h, tile.rows, mode.a_bits), ("B", w, tile.cols, mode.b_bits)):
        reads, writes = [], []
        for index in range(count):
            valid_k = min(mode.dot_length, tile.k - index * mode.dot_length)
            reads.append(_event(index, index * mode.issue_interval_cycles, groups,
                                valid_k * bits, mode.dot_length * bits, "compute_read"))
            writes.append(_event(index, None, groups, valid_k * bits,
                                 mode.dot_length * bits, "uniform_background_refill"))
        operands[name] = _summarize(
            groups, physical_groups, reads, writes, cycles,
            physical_groups * mode.dot_length * bits / 8,
            physical_groups * mode.dot_length * bits / 8)
    c_bits = w * tile.accumulator_bits
    full_c_bits = tile.cols * tile.accumulator_bits
    reads, writes = [], []
    if tile.accumulator_policy == "resident":
        if tile.initial_c:
            reads.append(_event(0, 0, h, c_bits, full_c_bits, "initial_c_read"))
        writes.append(_event(count, cycles, h, c_bits, full_c_bits, "final_c_writeback"))
    else:
        for index in range(count):
            if index or tile.initial_c:
                reads.append(_event(index, index * mode.issue_interval_cycles,
                                    h, c_bits, full_c_bits, "partial_sum_read"))
            writes.append(_event(index, index * mode.issue_interval_cycles,
                                 h, c_bits, full_c_bits, "partial_sum_write"))
    operands["C"] = _summarize(h, tile.rows, reads, writes, cycles,
                               tile.rows * full_c_bits / 8,
                               tile.rows * full_c_bits / 8)
    useful_macs = h * w * tile.k
    peak_macs = tile.rows * tile.cols * mode.dot_length / mode.issue_interval_cycles
    return {
        **asdict(tile), "active_rows": h, "active_cols": w,
        "mode": asdict(mode), "k_groups": count,
        "last_group_valid_k": tile.k - (count - 1) * mode.dot_length,
        "compute_issue_cycles": cycles,
        "useful_macs": useful_macs,
        "issued_mac_slots": tile.rows * tile.cols * count * mode.dot_length,
        "peak_macs_per_cycle": peak_macs,
        "issue_window_compute_utilization": useful_macs / (peak_macs * cycles),
        "resident_accumulator_state_bytes": (
            tile.rows * tile.cols * tile.accumulator_bits / 8
            if tile.accumulator_policy == "resident" else 0),
        "active_accumulator_state_bytes": (
            h * w * tile.accumulator_bits / 8 if tile.accumulator_policy == "resident" else 0),
        "operands": operands,
        "assumptions": [
            "A row broadcast and B column broadcast; reuse is already counted exactly once",
            "One operand tile refill per compute tile; no reuse across output tiles is inferred",
            "Issue window excludes array fill/drain, dot latency, accumulator feedback and memory stalls",
            "C event cycles are structural issue phases, not a causal RTL pipeline schedule",
            "Resident ACC state is additional to the modeled L0C SRAM area",
        ],
    }


def evaluate_lane(spec: BufferSpec, characterization: dict, traffic: dict,
                  operand: str, slices_per_group: int, physical_groups: int,
                  gating: str = "slice") -> dict:
    """Quantize every issue into physical transactions on a fixed row/slice map.

    ``slice`` enables only slices needed by that event; ``full_group`` accesses
    every deployed slice in an active group. Unused groups never issue accesses.
    All deployed macros retain leakage. A/B refill writes are uniform over the
    issue window. C RMW reads/writes share the same state and port budget.
    """
    _integer(slices_per_group, "slices_per_group")
    _integer(physical_groups, "physical_groups")
    if operand not in ("A", "B", "C"):
        raise ValueError("operand must be A, B or C")
    if gating not in ("slice", "full_group"):
        raise ValueError("gating must be slice or full_group")
    c = characterization
    org = c["organization"]
    events = traffic["operands"][operand]
    if org["banks"] != physical_groups * slices_per_group:
        raise ValueError("Organization banks must equal physical_groups * slices_per_group")
    if physical_groups < events["physical_groups_required"]:
        raise ValueError("physical_groups cannot be smaller than the physical row/column count")
    word_bits, word_bytes = org["width_bits"], org["width_bits"] // 8
    cycles = traffic["compute_issue_cycles"]
    reasons = []
    if (operand == "C" and traffic["accumulator_policy"] == "l0c_rmw"
            and c["read_write_use_distinct_macros"]):
        reasons.append("L0C RMW updates the same partial sum; independent read/write slots require an explicit state protocol")

    def quantize(source):
        result = []
        per_slice_counts = [0] * slices_per_group
        for event in source:
            needed = math.ceil(event["bits_per_group"] / word_bits)
            accessed = needed if gating == "slice" else max(needed, slices_per_group)
            if needed > slices_per_group:
                reasons.append("Event payload needs {} slices per group but only {} are deployed".format(
                    needed, slices_per_group))
            for lane in range(min(accessed, slices_per_group)):
                per_slice_counts[lane] += 1
            result.append({**event, "needed_slices": needed, "accessed_slices": accessed,
                           "transactions": event["groups"] * accessed,
                           "physical_bytes": event["groups"] * accessed * word_bytes})
        return result, per_slice_counts

    reads, read_counts = quantize(events["read_events"])
    writes, write_counts = quantize(events["write_events"])
    nread = sum(e["transactions"] for e in reads)
    nwrite = sum(e["transactions"] for e in writes)
    pread, pwrite = nread * word_bytes, nwrite * word_bytes
    peak_read = max((e["physical_bytes"] for e in reads), default=0)
    peak_write = max((e["physical_bytes"] for e in writes), default=0)

    # Fixed group/slice address assignment. Idle slices do not lend capacity to
    # active ones; each K-group occupies one word in each accessed slice.
    if operand in ("A", "B"):
        footprint = pwrite
        max_macro_footprint = max(write_counts, default=0) * word_bytes
    else:
        # Repeated updates overwrite one state tile, not Q distinct tiles.
        state_slices = math.ceil(traffic["active_cols"] * traffic["accumulator_bits"] / word_bits)
        state_slices = state_slices if gating == "slice" else max(state_slices, slices_per_group)
        footprint = events["groups"] * state_slices * word_bytes
        max_macro_footprint = word_bytes
    if footprint > c["slot_capacity_bytes"]:
        reasons.append("Physical word-rounded tile footprint exceeds one slot capacity")
    # Logical ping-pong partitions inside one macro also partition each fixed
    # slice's address space. Two live slots cannot both claim its full capacity.
    macro_slot_capacity = c["macro_capacity_bytes"] // (
        spec.slots if spec.layout == "shared_macro" else 1)
    if max_macro_footprint > macro_slot_capacity:
        reasons.append("Fixed slice mapping exceeds per-macro capacity even if aggregate capacity fits")

    average = Demand(pread / cycles, pwrite / cycles)
    aggregate = evaluate(spec, c, average)
    reasons.extend(aggregate["reasons"])
    interval = c["issue_interval_cycles"]
    efficiency = spec.bank_efficiency
    r_ports = org["read_ports"] + org["rw_ports"]
    w_ports = org["write_ports"] + org["rw_ports"]
    # Peak issue admission is distinct from sustainable average bandwidth.
    read_burst_supply = physical_groups * slices_per_group * word_bytes * r_ports
    write_burst_supply = physical_groups * slices_per_group * word_bytes * w_ports
    if peak_read > read_burst_supply:
        reasons.append("Read issue burst exceeds parallel lane/port admission")
    if operand == "C" and peak_write > write_burst_supply:
        reasons.append("C write issue burst exceeds parallel lane/port admission")
    # Aggregate bandwidth alone misses concentration in a subset of fixed lanes.
    for r_count, w_count in zip(read_counts, write_counts):
        r_rate, w_rate = r_count / cycles, w_count / cycles
        if r_rate > r_ports * efficiency / interval + 1e-12:
            reasons.append("Active macro read rate exceeds its sustainable port rate")
        if w_rate > w_ports * efficiency / interval + 1e-12:
            reasons.append("Active macro write rate exceeds its sustainable port rate")
        if not c["read_write_use_distinct_macros"]:
            shared = (max(0, r_rate - org["read_ports"] * efficiency / interval)
                      + max(0, w_rate - org["write_ports"] * efficiency / interval))
            if shared > org["rw_ports"] * efficiency / interval + 1e-12:
                reasons.append("Active macro read/write rates compete for shared RW ports")
    if operand == "C" and traffic["accumulator_policy"] == "l0c_rmw":
        read_by_issue = {e["issue_index"]: e for e in reads}
        for write in writes:
            read = read_by_issue.get(write["issue_index"])
            if read is not None:
                # One transaction in both directions on each common active lane.
                simultaneous_shared = max(0, 1 - org["read_ports"]) + max(0, 1 - org["write_ports"])
                if simultaneous_shared > org["rw_ports"]:
                    reasons.append("L0C update requires simultaneous read/write admission on the same lane")
    reasons = list(dict.fromkeys(reasons))
    feasible = not reasons
    deployed = physical_groups * slices_per_group
    energy = (nread * c["macro"]["read_energy_nj"] + nwrite * c["macro"]["write_energy_nj"]) * 1e-9
    dynamic = energy / (cycles / spec.frequency_hz)
    result = {
        "operand": operand, "gating": gating, "slice_mapping": "fixed_active_prefix",
        "physical_groups": physical_groups, "slices_per_group": slices_per_group,
        "deployed_lanes_per_slot": deployed,
        "feasible": feasible, "reasons": reasons,
        "read_events": reads, "write_events": writes,
        "read_transactions": nread, "write_transactions": nwrite,
        "physical_read_bytes": pread, "physical_write_bytes": pwrite,
        "useful_read_bytes": events["useful_read_bytes"],
        "useful_write_bytes": events["useful_write_bytes"],
        "average_read_bytes_per_cycle": pread / cycles,
        "average_write_bytes_per_cycle": pwrite / cycles,
        "peak_read_bytes_per_issue": peak_read,
        "peak_write_bytes_per_issue": peak_write,
        "read_payload_efficiency": events["useful_read_bytes"] / pread if pread else None,
        "write_payload_efficiency": events["useful_write_bytes"] / pwrite if pwrite else None,
        "average_read_lane_activity": nread / (deployed * cycles),
        "average_write_lane_activity": nwrite / (deployed * cycles),
        "peak_read_lane_activity": peak_read / (deployed * word_bytes),
        "peak_write_lane_activity": peak_write / (deployed * word_bytes),
        "read_bandwidth_utilization": pread / cycles / c["read_bandwidth_bytes_per_cycle"],
        "write_bandwidth_utilization": pwrite / cycles / c["write_bandwidth_bytes_per_cycle"],
        "available_read_bytes_per_cycle": c["read_bandwidth_bytes_per_cycle"],
        "available_write_bytes_per_cycle": c["write_bandwidth_bytes_per_cycle"],
        "physical_tile_footprint_bytes": footprint,
        "per_macro_required_bytes_max": max_macro_footprint,
        "macro_capacity_bytes": c["macro_capacity_bytes"],
        "per_macro_slot_capacity_bytes": macro_slot_capacity,
        "slot_capacity_bytes": c["slot_capacity_bytes"],
        "area_mm2": c["area_mm2"],
        "modeled_dynamic_energy_j": energy,
        "dynamic_power_w": dynamic if feasible else None,
        "leakage_w": c["leakage_w"],
        "total_power_w": dynamic + c["leakage_w"] if feasible else None,
        "resident_accumulator_state_bytes": traffic["resident_accumulator_state_bytes"] if operand == "C" else 0,
        "power_normalization": "Physical SRAM transactions over compute issue window; excludes fill/drain and deferred-writeback time, not measured full-kernel power",
        "limitations": [
            "Fixed active-prefix slice mapping; unused lane capacity is not automatically redistributed",
            "Read bursts and same-lane rates are checked without a cycle-accurate SRAM scheduler",
            "A/B refill is uniformly spread over the compute window; upstream transport is not modeled",
            "ACC state area/energy and inter-macro broadcast/wiring/control are excluded",
        ],
    }
    return result
