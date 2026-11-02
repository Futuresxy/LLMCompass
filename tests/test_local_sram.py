"""Mathematical checks for the array-local SRAM planner.

The fake CACTI responses are deliberately simple and are not physical estimates.
Real CACTI integration is exercised separately by the executable example.
Run: .venv/bin/python -m unittest discover -s tests -p 'test_local_sram.py'
"""

import unittest

from cost_model.cacti import MacroResult
from cost_model.local_sram import (
    BufferSpec,
    Demand,
    Organization,
    characterize,
    evaluate,
    plan_upgrade,
    scale_mode_demands,
)


class FakeCacti:
    """Return positive, deterministic macro costs with per-macro overhead."""

    def __init__(self, cycle_time_ns=0.5):
        self.cycle_time_ns = cycle_time_ns
        self.calls = []

    def simulate(self, macro):
        self.calls.append(macro)
        # Linear bit capacity plus independent macro and interface overhead.
        # More banks at fixed aggregate capacity therefore have a real cost.
        area = macro.capacity_bytes * 0.0001 + 0.1 + macro.width_bits * 0.0001
        return MacroResult(
            area_mm2=area,
            read_energy_nj=1.0,
            write_energy_nj=2.0,
            leakage_mw=4.0,
            gate_leakage_mw=1.0,
            access_time_ns=3.0,
            cycle_time_ns=self.cycle_time_ns,
            artifact_dir="unit-test-fake",
            provenance={"source": "unit-test-fake-not-hardware"},
        )


def buffer(**overrides):
    values = dict(
        name="L0A-like",
        capacity_bytes=4096,
        frequency_hz=1e9,
        slots=2,
        layout="separate_slots",
        capacity_basis="total",
        technology_nm=22,
        temperature_k=300,
    )
    values.update(overrides)
    return BufferSpec(**values)


class LocalSramTests(unittest.TestCase):
    def test_fixed_total_capacity_and_pingpong_active_bandwidth(self):
        spec = buffer()
        result = characterize(spec, Organization(banks=2, width_bits=128), FakeCacti())
        self.assertEqual(result["total_capacity_bytes"], 4096)
        self.assertEqual(result["macro_count"], 4)
        self.assertEqual(result["macro_capacity_bytes"], 1024)
        # Two active read banks, not four: the other slot is being refilled.
        self.assertEqual(result["read_bandwidth_bytes_per_cycle"], 32)
        self.assertEqual(result["write_bandwidth_bytes_per_cycle"], 32)

    def test_capacity_per_slot_is_explicit(self):
        spec = buffer(capacity_basis="per_slot")
        result = characterize(spec, Organization(banks=2, width_bits=128), FakeCacti())
        self.assertEqual(result["total_capacity_bytes"], 8192)
        self.assertEqual(result["macro_count"], 4)
        self.assertEqual(result["macro_capacity_bytes"], 2048)

    def test_shared_macro_has_no_extra_physical_capacity(self):
        spec = buffer(layout="shared_macro")
        result = characterize(spec, Organization(banks=2, width_bits=128), FakeCacti())
        self.assertEqual(result["total_capacity_bytes"], 4096)
        self.assertEqual(result["macro_count"], 2)
        self.assertEqual(result["macro_capacity_bytes"], 2048)

    def test_shared_single_rw_port_cannot_read_and_write_at_full_rate(self):
        spec = buffer(layout="shared_macro")
        result = characterize(spec, Organization(banks=1, width_bits=128), FakeCacti())
        self.assertTrue(evaluate(spec, result, Demand(16, 0))["feasible"])
        self.assertTrue(evaluate(spec, result, Demand(0, 16))["feasible"])
        simultaneous = evaluate(spec, result, Demand(16, 16))
        self.assertFalse(simultaneous["feasible"])
        self.assertTrue(simultaneous["reasons"])

    def test_shared_rw_port_allows_time_sharing_within_budget(self):
        spec = buffer(layout="shared_macro")
        result = characterize(spec, Organization(banks=1, width_bits=128), FakeCacti())
        self.assertTrue(evaluate(spec, result, Demand(8, 8))["feasible"])

    def test_dedicated_ports_allow_simultaneous_read_and_write(self):
        spec = buffer(layout="shared_macro")
        org = Organization(banks=1, width_bits=128, rw_ports=0, read_ports=1, write_ports=1)
        result = characterize(spec, org, FakeCacti())
        self.assertTrue(evaluate(spec, result, Demand(16, 16))["feasible"])

    def test_physical_pingpong_slots_allow_simultaneous_read_and_refill(self):
        spec = buffer(layout="separate_slots")
        result = characterize(spec, Organization(banks=1, width_bits=128), FakeCacti())
        self.assertTrue(evaluate(spec, result, Demand(16, 16))["feasible"])

    def test_cycle_time_limits_issue_rate_not_access_latency(self):
        spec = buffer()
        org = Organization(banks=2, width_bits=128)
        fast = characterize(spec, org, FakeCacti(cycle_time_ns=0.5))
        slow = characterize(spec, org, FakeCacti(cycle_time_ns=1.5))
        self.assertEqual(fast["issue_interval_cycles"], 1)
        self.assertEqual(slow["issue_interval_cycles"], 2)
        self.assertEqual(fast["read_bandwidth_bytes_per_cycle"], 32)
        self.assertEqual(slow["read_bandwidth_bytes_per_cycle"], 16)

    def test_power_uses_access_rate_and_counts_all_macro_leakage(self):
        spec = buffer()
        result = characterize(spec, Organization(banks=1, width_bits=128), FakeCacti())
        evaluation = evaluate(spec, result, Demand(16, 16))
        # At 1 GHz, one 16-byte read/cycle at 1 nJ and one write at 2 nJ.
        self.assertAlmostEqual(evaluation["dynamic_power_w"], 3.0)
        # Both physical slots leak, including the inactive read slot.
        self.assertAlmostEqual(result["leakage_w"], 2 * (4 + 1) / 1000)
        self.assertAlmostEqual(evaluation["total_power_w"], 3.01)

    def test_double_buffering_does_not_hide_an_inadequate_refill_window(self):
        spec = buffer()
        result = characterize(spec, Organization(banks=1, width_bits=64), FakeCacti())
        # The same 64-byte next tile has a feasible long window but not a short
        # one. Two slots alone cannot make an 8-byte/cycle write path faster.
        long_window = Demand(0, refill_bytes_per_tile=64, compute_window_cycles=20)
        short_window = Demand(0, refill_bytes_per_tile=64, compute_window_cycles=4)
        self.assertTrue(evaluate(spec, result, long_window)["feasible"])
        self.assertFalse(evaluate(spec, result, short_window)["feasible"])

    def test_refill_tile_must_fit_in_one_slot(self):
        spec = buffer()
        result = characterize(spec, Organization(banks=1, width_bits=128), FakeCacti())
        oversized = Demand(0, refill_bytes_per_tile=4096, compute_window_cycles=10000)
        self.assertFalse(evaluate(spec, result, oversized)["feasible"])
        with self.assertRaises(ValueError):
            buffer(live_tile_bytes=4096)

    def test_infeasible_service_rate_does_not_report_achieved_power(self):
        spec = buffer()
        result = characterize(spec, Organization(banks=1, width_bits=64), FakeCacti())
        evaluation = evaluate(spec, result, Demand(1000))
        self.assertFalse(evaluation["feasible"])
        self.assertIsNone(evaluation["dynamic_power_w"])
        self.assertIsNone(evaluation["total_power_w"])

    def test_unsupported_12nm_is_rejected_before_running_cacti(self):
        runner = FakeCacti()
        with self.assertRaises(ValueError):
            characterize(buffer(technology_nm=12), Organization(banks=1, width_bits=64), runner)
        self.assertEqual(runner.calls, [])

    def test_existing_headroom_preserves_hardware_and_zero_area_increment(self):
        spec = buffer()
        organizations = [Organization(banks=n, width_bits=64) for n in (1, 2, 4)]
        plan = plan_upgrade(
            spec, Demand(4), Demand(12), organizations, FakeCacti(),
            installed=organizations[1],
        )
        self.assertEqual(plan["status"], "retained")
        self.assertTrue(plan["target"]["feasible"])
        self.assertEqual(plan["delta_area_mm2"], 0)
        self.assertEqual(plan["baseline"]["area_mm2"], plan["target"]["area_mm2"])
        self.assertGreater(plan["target"]["dynamic_power_w"],
                           plan["baseline"]["dynamic_power_w"])

    def test_crossing_installed_bandwidth_triggers_upgrade(self):
        spec = buffer()
        organizations = [Organization(banks=n, width_bits=64) for n in (1, 2, 4)]
        plan = plan_upgrade(
            spec, Demand(4), Demand(24), organizations, FakeCacti(),
            installed=organizations[1],
        )
        self.assertEqual(plan["status"], "upgraded")
        self.assertTrue(plan["target"]["feasible"])
        self.assertGreater(plan["delta_area_mm2"], 0)
        self.assertEqual(plan["target"]["total_capacity_bytes"], 4096)

    def test_explicit_baseline_headroom_affects_baseline_selection(self):
        spec = buffer()
        organizations = [Organization(banks=n, width_bits=64) for n in (1, 2, 4)]
        plan = plan_upgrade(
            spec, Demand(4), Demand(12), organizations, FakeCacti(),
            baseline_headroom=3.0,
        )
        self.assertEqual(plan["status"], "retained")
        self.assertGreaterEqual(plan["baseline"]["read_bandwidth_bytes_per_cycle"], 12)
        self.assertEqual(plan["delta_area_mm2"], 0)

    def test_no_feasible_candidate_is_reported_without_capacity_growth(self):
        spec = buffer()
        organizations = [Organization(banks=n, width_bits=64) for n in (1, 2)]
        plan = plan_upgrade(spec, Demand(4), Demand(1000), organizations, FakeCacti())
        self.assertEqual(plan["status"], "infeasible")
        self.assertIsNone(plan["target"])
        self.assertEqual(plan["baseline"]["total_capacity_bytes"], 4096)

    def test_fixed_candidate_pool_minimum_area_is_monotonic(self):
        spec = buffer()
        organizations = [Organization(banks=n, width_bits=64) for n in (1, 2, 4)]
        selected = []
        for bandwidth in (4, 12, 24):
            plan = plan_upgrade(
                spec, Demand(bandwidth), Demand(bandwidth), organizations, FakeCacti()
            )
            self.assertEqual(plan["status"], "retained")
            selected.append(plan["target"]["area_mm2"])
        self.assertEqual(selected, sorted(selected))
        self.assertLess(selected[0], selected[-1])

    def test_mutually_exclusive_modes_do_not_add_read_and_write_traffic(self):
        spec = buffer(layout="shared_macro")
        org = Organization(banks=1, width_bits=64)
        modes = {"read_mode": Demand(8, 0), "write_mode": Demand(0, 8)}
        plan = plan_upgrade(spec, modes, modes, [org], FakeCacti(), installed=org)
        self.assertEqual(plan["status"], "retained")
        self.assertTrue(plan["target"]["feasible"])
        # Each mode alone fits one RW port. Their powers are alternatives too:
        # 1 W read or 2 W write, plus one macro's 5 mW leakage.
        self.assertAlmostEqual(plan["target"]["dynamic_power_w"], 2.0)
        self.assertAlmostEqual(plan["target"]["total_power_w"], 2.005)
        self.assertEqual(plan["delta_area_mm2"], 0)

    def test_one_selected_organization_must_support_every_mode(self):
        spec = buffer(layout="shared_macro")
        installed = Organization(banks=1, width_bits=64)
        read_specialist = Organization(banks=1, width_bits=64, rw_ports=0,
                                       read_ports=2, write_ports=1)
        write_specialist = Organization(banks=1, width_bits=64, rw_ports=0,
                                        read_ports=1, write_ports=2)
        common = Organization(banks=2, width_bits=64)
        baseline = {"read_mode": Demand(8, 0), "write_mode": Demand(0, 8)}
        target = {"read_mode": Demand(16, 0), "write_mode": Demand(0, 16)}
        plan = plan_upgrade(
            spec, baseline, target,
            [installed, read_specialist, write_specialist, common],
            FakeCacti(), installed=installed,
        )
        self.assertEqual(plan["status"], "upgraded")
        # Neither smaller specialist serves both modes; a mode switch must not
        # magically exchange the physical SRAM implementation.
        self.assertEqual(plan["target"]["organization"]["banks"], 2)
        self.assertTrue(all(m["feasible"] for m in
                            plan["target"]["mode_evaluations"].values()))

    def test_precision_ratios_scale_only_the_changed_mode(self):
        original = {
            "bf16": {"A": Demand(8, 2), "B": Demand(4, 1)},
            "fp8": {"A": Demand(8, 2), "B": Demand(4, 1)},
            "fp4": {"A": Demand(8, 2), "B": Demand(4, 1)},
        }
        scaled = scale_mode_demands(
            {"bf16": 1, "fp8": 2, "fp4": 4},
            {"bf16": 1, "fp8": 2, "fp4": 8},
            original,
        )
        for mode in ("bf16", "fp8"):
            self.assertEqual(scaled[mode]["A"].read_bytes_per_cycle, 8)
            self.assertEqual(scaled[mode]["A"].write_bytes_per_cycle, 2)
        self.assertEqual(scaled["fp4"]["A"].read_bytes_per_cycle, 16)
        self.assertEqual(scaled["fp4"]["A"].write_bytes_per_cycle, 4)
        self.assertEqual(scaled["fp4"]["B"].read_bytes_per_cycle, 8)
        # The utility must not mutate the baseline observations.
        self.assertEqual(original["fp4"]["A"].read_bytes_per_cycle, 8)


if __name__ == "__main__":
    unittest.main()
