"""Traffic/lane mathematics, using deliberately synthetic SRAM costs.

These are not physical SRAM estimates. Real CACTI scenarios are run separately.
"""

import unittest

from cost_model.cacti import MacroResult
from cost_model.dot_product_sram import (
    ArrayTile, PrecisionMode, generate_traffic, evaluate_lane,
)
from cost_model.local_sram import BufferSpec, Organization, characterize


class FakeCacti:
    def simulate(self, spec):
        return MacroResult(
            area_mm2=0.1 + spec.capacity_bytes * 1e-4,
            read_energy_nj=1.0, write_energy_nj=2.0,
            leakage_mw=4.0, gate_leakage_mw=1.0,
            access_time_ns=0.5, cycle_time_ns=0.5,
            artifact_dir="synthetic-unit-test", provenance={"source": "not-hardware"},
        )


def mode(name="fp4", length=64, bits=4, ii=1):
    return PrecisionMode(name=name, dot_length=length, a_bits=bits, b_bits=bits,
                         issue_interval_cycles=ii)


def operand_layout(groups=16, slices=4, width=128, operand="A", slots=None,
                   layout=None, read_ports=0, write_ports=0, rw_ports=1):
    is_c = operand == "C"
    spec = BufferSpec(
        name="L0" + operand, capacity_bytes=16384 if is_c else 32768,
        frequency_hz=1e9, slots=(1 if is_c else 2) if slots is None else slots,
        layout=("shared_macro" if is_c else "separate_slots") if layout is None else layout,
    )
    org = Organization(banks=groups * slices, width_bits=width, rw_ports=rw_ports,
                       read_ports=read_ports, write_ports=write_ports)
    return spec, characterize(spec, org, FakeCacti())


class DotProductTrafficTests(unittest.TestCase):
    def test_h16q32e64_equal_operand_bandwidth(self):
        for precision in [mode("bf16", 16, 16), mode("fp8", 32, 8), mode()]:
            traffic = generate_traffic(ArrayTile(16, 16, 256), precision)
            for operand in ("A", "B"):
                self.assertEqual(traffic["operands"][operand]["peak_read_bytes_per_issue"], 512)
                self.assertEqual(traffic["operands"][operand]["average_read_bytes_per_cycle"], 512)
            self.assertEqual(traffic["useful_macs"], 16 * 16 * 256)

    def test_four_precision_ratios_generate_required_bandwidths(self):
        # K=768 divides every tested dot length; no K tail obscures steady rate.
        for lengths, expected in [
            ((16, 32, 64), (512, 512, 512)),
            ((16, 32, 96), (512, 512, 768)),
            ((16, 32, 128), (512, 512, 1024)),
            ((16, 64, 128), (512, 1024, 1024)),
        ]:
            for name, length, bits, bandwidth in zip(
                    ("bf16", "fp8", "fp4"), lengths, (16, 8, 4), expected):
                t = generate_traffic(ArrayTile(16, 16, 768), mode(name, length, bits))
                for operand in ("A", "B"):
                    self.assertEqual(t["operands"][operand]["average_read_bytes_per_cycle"], bandwidth)

    def test_broadcast_is_counted_once_not_once_per_pe(self):
        t = generate_traffic(ArrayTile(16, 16, 256), mode())
        # A and B each contain 16*256 FP4 values; no further /16 reuse factor.
        self.assertEqual(t["operands"]["A"]["useful_read_bytes"], 2048)
        self.assertEqual(t["operands"]["B"]["useful_read_bytes"], 2048)
        self.assertEqual(t["peak_macs_per_cycle"], 16 * 16 * 64)

    def test_rectangular_array_has_asymmetric_operand_bandwidth(self):
        t = generate_traffic(ArrayTile(32, 8, 256), mode("bf16", 16, 16))
        self.assertEqual(t["operands"]["A"]["average_read_bytes_per_cycle"], 1024)
        self.assertEqual(t["operands"]["B"]["average_read_bytes_per_cycle"], 256)
        self.assertEqual(t["peak_macs_per_cycle"], 16 * 16 * 16)

    def test_issue_interval_changes_average_not_burst(self):
        t = generate_traffic(ArrayTile(16, 16, 256), mode(ii=2))
        self.assertEqual(t["compute_issue_cycles"], 8)
        self.assertEqual(t["operands"]["A"]["average_read_bytes_per_cycle"], 256)
        self.assertEqual(t["operands"]["A"]["peak_read_bytes_per_issue"], 512)

    def test_useful_bytes_do_not_grow_when_dot_length_increases(self):
        a = generate_traffic(ArrayTile(16, 16, 256), mode(length=64))
        b = generate_traffic(ArrayTile(16, 16, 256), mode(length=128))
        for operand in ("A", "B"):
            self.assertEqual(a["operands"][operand]["useful_read_bytes"],
                             b["operands"][operand]["useful_read_bytes"])
        self.assertEqual(a["compute_issue_cycles"], 2 * b["compute_issue_cycles"])

    def test_k_tail_keeps_true_work_and_rounds_physical_transactions(self):
        t = generate_traffic(ArrayTile(16, 16, 193), mode(length=96))
        spec, macro = operand_layout()
        e = evaluate_lane(spec, macro, t, "A", 4, 16, "slice")
        self.assertTrue(e["feasible"], e["reasons"])
        self.assertEqual(t["k_groups"], 3)
        self.assertEqual(t["useful_macs"], 16 * 16 * 193)
        # 96 + 96 + 1 useful elements => 3 + 3 + 1 128-bit words per row.
        self.assertEqual(e["read_transactions"], 16 * (3 + 3 + 1))
        self.assertEqual(e["physical_read_bytes"], 1792)
        self.assertEqual(e["useful_read_bytes"], 1544)
        self.assertAlmostEqual(e["read_payload_efficiency"], 1544 / 1792)

    def test_full_group_does_not_turn_partial_payload_into_half_access(self):
        t = generate_traffic(ArrayTile(16, 16, 256), mode("bf16", 16, 16))
        spec, macro = operand_layout()
        sliced = evaluate_lane(spec, macro, t, "A", 4, 16, "slice")
        full = evaluate_lane(spec, macro, t, "A", 4, 16, "full_group")
        self.assertTrue(sliced["feasible"], sliced["reasons"])
        self.assertTrue(full["feasible"], full["reasons"])
        self.assertEqual(sliced["read_transactions"], 512)
        self.assertEqual(full["read_transactions"], 1024)
        self.assertEqual(full["physical_read_bytes"], 16384)
        self.assertEqual(full["useful_read_bytes"], 8192)
        self.assertAlmostEqual(full["read_payload_efficiency"], 0.5)

    def test_slice_gating_reduces_dynamic_but_not_deployed_leakage(self):
        t = generate_traffic(ArrayTile(16, 16, 256), mode("bf16", 16, 16))
        spec, macro = operand_layout()
        sliced = evaluate_lane(spec, macro, t, "A", 4, 16, "slice")
        full = evaluate_lane(spec, macro, t, "A", 4, 16, "full_group")
        self.assertAlmostEqual(sliced["dynamic_power_w"] * 2, full["dynamic_power_w"])
        self.assertEqual(sliced["area_mm2"], full["area_mm2"])
        self.assertEqual(sliced["area_mm2"], macro["area_mm2"])
        self.assertEqual(sliced["leakage_w"], full["leakage_w"])
        self.assertEqual(sliced["leakage_w"], macro["leakage_w"])

    def test_e96_roundup_reports_three_of_four_active_slices(self):
        t = generate_traffic(ArrayTile(16, 16, 192), mode(length=96))
        spec, macro = operand_layout()
        e = evaluate_lane(spec, macro, t, "A", 4, 16, "slice")
        self.assertTrue(e["feasible"], e["reasons"])
        self.assertAlmostEqual(e["peak_read_lane_activity"], 0.75)
        self.assertAlmostEqual(e["read_bandwidth_utilization"], 0.75)

    def test_per_macro_capacity_matters_even_when_total_slot_capacity_fits(self):
        t = generate_traffic(ArrayTile(16, 16, 512), mode("bf16", 16, 16))
        spec, macro = operand_layout()
        self.assertEqual(t["operands"]["A"]["useful_read_bytes"], macro["slot_capacity_bytes"])
        e = evaluate_lane(spec, macro, t, "A", 4, 16, "slice")
        self.assertFalse(e["feasible"])
        self.assertGreater(e["per_macro_required_bytes_max"], macro["macro_capacity_bytes"])

    def test_low_average_cannot_hide_insufficient_per_issue_lanes(self):
        t = generate_traffic(ArrayTile(16, 16, 256), mode(length=128, ii=2))
        spec, macro = operand_layout(slices=1, width=256)
        # Average is 512 B/cycle, but one issue requires 1024 B simultaneously.
        self.assertEqual(t["operands"]["A"]["average_read_bytes_per_cycle"], 512)
        self.assertEqual(macro["read_bandwidth_bytes_per_cycle"], 512)
        e = evaluate_lane(spec, macro, t, "A", 1, 16, "slice")
        self.assertFalse(e["feasible"])

    def test_shared_macro_must_hold_both_slot_footprints(self):
        t = generate_traffic(ArrayTile(16, 16, 512), mode("bf16", 16, 16))
        spec, macro = operand_layout(layout="shared_macro", rw_ports=0,
                                     read_ports=1, write_ports=1)
        self.assertEqual(t["operands"]["A"]["useful_read_bytes"], macro["slot_capacity_bytes"])
        e = evaluate_lane(spec, macro, t, "A", 4, 16, "slice")
        self.assertEqual(e["per_macro_required_bytes_max"], macro["macro_capacity_bytes"])
        # Every active macro holds both ping-pong tiles, not just the read tile.
        self.assertFalse(e["feasible"])

    def test_resident_accumulator_writes_once_and_optional_initial_read(self):
        for initial in (False, True):
            t = generate_traffic(ArrayTile(16, 16, 256, initial_c=initial), mode())
            c = t["operands"]["C"]
            self.assertEqual(c["useful_write_bytes"], 1024)
            self.assertEqual(c["useful_read_bytes"], 1024 if initial else 0)
            self.assertEqual(c["average_write_bytes_per_cycle"], 256)
            self.assertEqual(c["peak_write_bytes_per_issue"], 1024)

    def test_resident_c_average_changes_but_writeback_burst_does_not(self):
        a = generate_traffic(ArrayTile(16, 16, 256), mode(length=64))["operands"]["C"]
        b = generate_traffic(ArrayTile(16, 16, 256), mode(length=128))["operands"]["C"]
        self.assertEqual(a["average_write_bytes_per_cycle"], 256)
        self.assertEqual(b["average_write_bytes_per_cycle"], 512)
        self.assertEqual(a["peak_write_bytes_per_issue"], b["peak_write_bytes_per_issue"])

    def test_per_update_c_traffic_does_not_scale_with_dot_length(self):
        tile = ArrayTile(16, 16, 256, accumulator_policy="l0c_rmw")
        a = generate_traffic(tile, mode(length=64))["operands"]["C"]
        b = generate_traffic(tile, mode(length=128))["operands"]["C"]
        self.assertEqual(a["useful_write_bytes"], 4096)
        self.assertEqual(b["useful_write_bytes"], 2048)
        self.assertEqual(a["useful_read_bytes"], 3072)
        self.assertEqual(b["useful_read_bytes"], 1024)
        self.assertEqual(a["peak_write_bytes_per_issue"], 1024)
        self.assertEqual(b["peak_write_bytes_per_issue"], 1024)
        self.assertEqual(a["average_write_bytes_per_cycle"], 1024)
        self.assertEqual(b["average_write_bytes_per_cycle"], 1024)

    def test_per_update_c_requires_real_read_write_ports(self):
        t = generate_traffic(ArrayTile(16, 16, 256, accumulator_policy="l0c_rmw"), mode())
        spec, shared = operand_layout(operand="C", slices=1, width=512)
        bad = evaluate_lane(spec, shared, t, "C", 1, 16, "slice")
        self.assertFalse(bad["feasible"])
        spec, separate_ports = operand_layout(operand="C", slices=1, width=512,
                                             rw_ports=0, read_ports=1, write_ports=1)
        good = evaluate_lane(spec, separate_ports, t, "C", 1, 16, "slice")
        self.assertTrue(good["feasible"], good["reasons"])

    def test_per_update_c_cannot_use_independent_pingpong_to_hide_dependency(self):
        t = generate_traffic(ArrayTile(16, 16, 256, accumulator_policy="l0c_rmw"), mode())
        spec, macro = operand_layout(operand="C", slices=1, width=512,
                                     slots=2, layout="separate_slots")
        e = evaluate_lane(spec, macro, t, "C", 1, 16, "slice")
        self.assertFalse(e["feasible"])

    def test_c_state_footprint_is_not_sum_of_all_updates(self):
        t = generate_traffic(ArrayTile(16, 16, 1024, accumulator_policy="l0c_rmw"), mode())
        spec, macro = operand_layout(operand="C", slices=1, width=512,
                                     rw_ports=0, read_ports=1, write_ports=1)
        e = evaluate_lane(spec, macro, t, "C", 1, 16, "slice")
        self.assertTrue(e["feasible"], e["reasons"])
        self.assertEqual(e["physical_tile_footprint_bytes"], 1024)
        self.assertGreater(e["physical_write_bytes"], e["physical_tile_footprint_bytes"])


if __name__ == "__main__":
    unittest.main()
