"""Reference compute tests with real SCALE-Sim traces, not GPU measurements."""

from copy import deepcopy
from math import prod
import unittest

import numpy as np

from hardware_model.device import device_dict
from hardware_model.precision import KExpansionProfile
from scripts.study_precision_ratios import run_study
from software_model.matmul import BatchedMatmul, Matmul
from software_model.utils import Tensor, data_type_dict
from systolic_array_model.precision_k import simulate_gemm, token_trace


def device(ratio=(1, 2, 4)):
    result = deepcopy(device_dict["A100_80GB_fp16"])
    result.compute_module.core.systolic_array.precision_profile = KExpansionProfile(ratios=ratio)
    return result


class FormatTests(unittest.TestCase):
    def test_legacy_constructor_and_tensor_bytes(self):
        self.assertEqual(Tensor([3, 7], data_type_dict["fp16"]).nbytes, 42)
        self.assertEqual(Tensor([3, 7], data_type_dict["fp32"]).nbytes, 84)

    def test_compute_classes_are_independent_of_storage_width(self):
        expected = {"fp16": (16, 16), "bf16": (16, 16), "mxfp8": (8, 8),
                    "mxfp6": (6, 8), "mxfp4": (4, 4), "nvfp4": (4, 4), "hif4": (4, 4)}
        for name, pair in expected.items():
            t = data_type_dict[name]
            self.assertEqual((t.payload_bits, t.compute_class), pair)

    def test_real_format_metadata_and_padding(self):
        # Whole blocks: MXFP4 16+1 B/32; NVFP4 8+1 B/16 + 4 B/tensor;
        # HiF4 32+4 B/64; MXFP6 24+1 B/32.
        for name, shape, expected in [
            ("mxfp8", [2, 32], 66), ("mxfp6", [2, 32], 50),
            ("mxfp4", [2, 32], 34), ("nvfp4", [2, 32], 40),
            ("hif4", [2, 64], 72), ("mxfp4", [2, 33], 68)]:
            self.assertEqual(data_type_dict[name].storage_bytes(shape), expected)

    def test_b_groups_along_k_not_n(self):
        t = data_type_dict["mxfp4"]
        self.assertEqual(t.storage_bytes([33, 5], block_axis=-2), 170)
        self.assertNotEqual(t.storage_bytes([33, 5], block_axis=-2), t.storage_bytes([33, 5]))

    def test_encoding_names_survive_shared_performance_classes(self):
        a, b = data_type_dict["mxfp6_e2m3"], data_type_dict["mxfp6_e3m2"]
        self.assertNotEqual(a.name, b.name)
        self.assertEqual(a.compute_class, b.compute_class)
        self.assertEqual(a.storage_bytes([17, 63]), b.storage_bytes([17, 63]))

    def test_invalid_storage_and_profile_fail(self):
        for shape in ([0, 16], [1, -2], [True, 2], []):
            with self.assertRaises(ValueError):
                data_type_dict["fp16"].storage_bytes(shape)
        for ratio in ((1, 2), (1, 2, 0), (1, 2, 2.5), (True, 2, 4), (2, 4, 8)):
            with self.assertRaises(ValueError):
                KExpansionProfile(ratios=ratio)


class ScaleSimDotTests(unittest.TestCase):
    def result(self, ratio, k=4096, dtype="mxfp4", m=16, n=16, engines=1, batch=1):
        return simulate_gemm(m, n, k, data_type_dict[dtype],
                             KExpansionProfile(ratios=ratio), engine_count=engines, batch_count=batch)

    def test_requested_ratios_expand_only_k(self):
        for ratio, lengths in [((1, 2, 4), (16, 32, 64)), ((1, 2, 6), (16, 32, 96)),
                               ((1, 2, 8), (16, 32, 128)), ((1, 4, 8), (16, 64, 128))]:
            p = KExpansionProfile(ratios=ratio)
            self.assertEqual(tuple(p.dot_length(data_type_dict[d]) for d in ("bf16", "mxfp8", "mxfp4")), lengths)

    def test_pipeline_is_not_divided_by_peak_multiplier(self):
        a, b = self.result((1, 2, 4)), self.result((1, 2, 8))
        self.assertEqual(a["compute_cycles"], 94)
        self.assertEqual(b["compute_cycles"], 62)
        self.assertLess(a["compute_cycles"] / b["compute_cycles"], 2)

    def test_single_cell_inclusive_cycle_not_zero_timestamp(self):
        result = simulate_gemm(1, 1, 16, data_type_dict["fp16"], KExpansionProfile(), rows=1, cols=1)
        self.assertEqual(result["compute_cycles"], 1)
        self.assertEqual(result["compute_utilization"], 1)

    def test_k_tail_and_useful_work(self):
        result = self.result((1, 2, 6), k=193)
        self.assertEqual(result["k_groups"], 3)
        self.assertEqual(result["k_tail_valid_elements"], 1)
        self.assertEqual(result["useful_macs"], 16 * 16 * 193)
        self.assertAlmostEqual(result["k_packing_efficiency"], 193 / 288)

    def test_short_k_does_not_gain_when_group_count_unchanged(self):
        a = self.result((1, 2, 4), k=16)
        b = self.result((1, 2, 8), k=16)
        self.assertEqual(a["compute_cycles"], b["compute_cycles"])

    def test_non_multiple_k_ratio_can_lose_to_tail_rounding(self):
        six = self.result((1, 2, 6), k=97)
        eight = self.result((1, 2, 8), k=97)
        self.assertEqual(six["k_groups"], 2)
        self.assertEqual(eight["k_groups"], 1)
        self.assertEqual(six["compute_cycles"] - eight["compute_cycles"], 1)

    def test_unaffected_precision_keeps_identical_compute(self):
        for dtype in ("fp16", "bf16", "mxfp8", "mxfp6"):
            self.assertEqual(self.result((1, 2, 4), dtype=dtype)["compute_cycles"],
                             self.result((1, 2, 8), dtype=dtype)["compute_cycles"])

    def test_same_class_formats_share_schedule_not_storage(self):
        results = [self.result((1, 2, 8), dtype=d) for d in ("mxfp4", "nvfp4", "hif4")]
        self.assertEqual(len(set(r["compute_cycles"] for r in results)), 1)
        self.assertEqual(len(set(data_type_dict[d].storage_bytes([16, 4096]) for d in ("mxfp4", "nvfp4", "hif4"))), 3)

    def test_spatial_wave_pipeline_k_utilization_are_separate(self):
        r = self.result((1, 2, 6), k=193, m=17, n=19, engines=3)
        self.assertEqual((r["jobs"], r["waves"], r["compute_cycles"]), (4, 2, 66))
        self.assertEqual(r["vector_tokens"]["c"], 17 * 19)
        self.assertAlmostEqual(r["compute_utilization"], prod(r[x] for x in (
            "mn_spatial_efficiency", "wave_efficiency", "pipeline_efficiency", "k_packing_efficiency")))

    def test_batches_keep_independent_outputs(self):
        r = self.result((1, 2, 4), k=16, m=1, n=1, engines=3, batch=7)
        self.assertEqual((r["jobs"], r["waves"]), (7, 3))
        self.assertEqual(r["vector_tokens"]["c"], 7)

    def test_trace_forwarding_performs_exact_dot_gemm(self):
        # Independent arithmetic oracle: replay edge tokens through row/column
        # links, reduce actual scalar slices (including tail), compare numpy GEMM.
        m, n, k, length, rows, cols = 2, 2, 193, 96, 2, 3
        groups = (k + length - 1) // length
        ta, tb, tc = token_trace(m, n, groups, rows, cols)
        a = (np.arange(m * k).reshape(m, k) % 11) - 5
        b = (np.arange(k * n).reshape(k, n) % 7) - 3
        out = np.zeros((m, n), dtype=np.int64)
        visited = set()
        for cycle in range(ta.shape[0] + rows + cols):
            for row in range(rows):
                for col in range(cols):
                    at = ta[cycle - col, row] if 0 <= cycle - col < len(ta) else -1
                    bt = tb[cycle - row, col] if 0 <= cycle - row < len(tb) else -1
                    if at < 0 or bt < 0:
                        continue
                    am, ag = divmod(int(at), groups)
                    bg, bn = divmod(int(bt), n)
                    self.assertEqual(ag, bg)
                    self.assertNotIn((am, bn, ag), visited)
                    visited.add((am, bn, ag))
                    start, end = ag * length, min(k, (ag + 1) * length)
                    out[am, bn] += a[am, start:end] @ b[start:end, bn]
        self.assertEqual(len(visited), m * n * groups)
        np.testing.assert_array_equal(out, a @ b)
        self.assertEqual(np.count_nonzero(tc >= 0), m * n)

    def test_ws_and_unsupported_formats_fail_explicitly(self):
        with self.assertRaises(ValueError):
            simulate_gemm(16, 16, 256, data_type_dict["mxfp4"], KExpansionProfile(), dataflow="ws")
        with self.assertRaises(ValueError):
            self.result((1, 2, 4), dtype="fp32")


class OperatorIntegrationTests(unittest.TestCase):
    def test_matmul_output_and_accumulator_are_separate(self):
        dtype, output = data_type_dict["mxfp4"], data_type_dict["bf16"]
        op = Matmul(dtype, output_type=output)
        result = op(Tensor([17, 193], dtype), Tensor([193, 19], dtype))
        self.assertIs(result.data_type, output)
        self.assertIs(op.accumulator_type, data_type_dict["fp32"])
        latency = op.compile_and_simulate(device((1, 2, 6)))
        self.assertGreater(latency, 0)
        self.assertEqual(op.precision_result["dot_length"], 96)
        self.assertEqual(op.precision_result["operand_storage_bytes"]["b"], dtype.storage_bytes([193, 19], block_axis=0))
        self.assertEqual(op.precision_result["operand_storage_bytes"]["d"], 17 * 19 * 2)

    def test_decode_dispatch_uses_array_not_vector_shortcut(self):
        dtype = data_type_dict["mxfp4"]
        op = Matmul(dtype)
        op(Tensor([1, 4096], dtype), Tensor([4096, 128], dtype))
        op.compile_and_simulate(device((1, 2, 8)), "heuristic-GPU")
        self.assertEqual(op.precision_result["dot_length"], 128)
        self.assertLess(op.precision_result["mn_spatial_efficiency"], 1)

    def test_batch_dispatch_never_concatenates_independent_k(self):
        dtype = data_type_dict["mxfp6"]
        op = BatchedMatmul(dtype)
        op(Tensor([7, 1, 193], dtype), Tensor([7, 193, 1], dtype))
        op.compile_and_simulate(device(), "heuristic-GPU")
        self.assertEqual(op.precision_result["batch_count"], 7)
        self.assertEqual(op.precision_result["useful_macs"], 7 * 193)

    def test_block_formats_cannot_silently_use_legacy_backend(self):
        dtype = data_type_dict["mxfp4"]
        op = Matmul(dtype)
        op(Tensor([16, 32], dtype), Tensor([32, 16], dtype))
        with self.assertRaises(ValueError):
            op.compile_and_simulate(device_dict["A100_80GB_fp16"])

    def test_peak_is_mode_specific(self):
        cm = device((1, 4, 8)).compute_module
        self.assertEqual(cm.systolic_flops(data_type_dict["mxfp8"]) / cm.systolic_flops(data_type_dict["bf16"]), 4)
        self.assertEqual(cm.systolic_flops(data_type_dict["mxfp4"]) / cm.systolic_flops(data_type_dict["bf16"]), 8)

    def test_legacy_decode_still_executes_without_precision_profile(self):
        dtype = data_type_dict["fp16"]
        op = Matmul(dtype)
        op(Tensor([1, 64], dtype), Tensor([64, 32], dtype))
        self.assertGreater(op.compile_and_simulate(device_dict["A100_80GB_fp16"], "heuristic-GPU"), 0)
        self.assertFalse(hasattr(op, "precision_result"))

    def test_json_adapter_accepts_profile_without_changing_physical_dtype(self):
        import json
        from pathlib import Path
        from design_space_exploration.dse import template_to_system
        config = json.loads((Path(__file__).resolve().parents[1] / "configs/GA100.json").read_text())
        config["device"]["compute_chiplet"]["core"]["systolic_array"]["precision_profile"] = {
            "base_dot_length": 16, "ratios": [1, 2, 6]}
        cm = template_to_system(config).device.compute_module
        self.assertEqual(cm.core.systolic_array.precision_profile.name, "1:2:6")
        self.assertEqual(cm.total_systolic_array_flops, cm.systolic_flops(data_type_dict["fp16"]))
        from cost_model.cost_model import calc_compute_chiplet_area_mm2
        with self.assertRaisesRegex(ValueError, "supplied PPA"):
            calc_compute_chiplet_area_mm2(config)

    def test_batch_roofline_accounts_one_tensor_scale_per_operand(self):
        dtype = data_type_dict["nvfp4"]
        op = BatchedMatmul(dtype)
        op(Tensor([7, 1, 33], dtype), Tensor([7, 33, 1], dtype))
        dev = device()
        dev.io_module.bandwidth = 1
        expected = (dtype.storage_bytes([7, 1, 33])
                    + dtype.storage_bytes([7, 33, 1], block_axis=-2)
                    + dtype.storage_bytes([7, 1, 1]))
        self.assertEqual(op.roofline_model(dev), expected)


class StudyTests(unittest.TestCase):
    def config(self):
        return {"hardware": {"rows": 2, "cols": 2, "core_count": 1, "arrays_per_core": 1,
                             "frequency_hz": 1e9, "hbm_bytes_per_second": 1e15, "l2_bytes_per_cycle": 1e9},
                "profiles": [[1, 2, 4], [1, 2, 8]], "baseline": "1:2:4", "formats": ["bf16", "mxfp4"],
                "workloads": [{"name": "selection", "split": "design", "ops": [{"name": "linear", "m": 2, "n": 2, "k": 4096}]},
                              {"name": "unseen", "split": "holdout", "ops": [{"name": "linear", "m": 1, "n": 2, "k": 193}]}]}

    def test_uniform_precision_sweep_keeps_unknown_ppa_null(self):
        report = run_study(self.config())
        self.assertEqual(len(report["results"]), 8)
        self.assertTrue(all(r["tc_area_relative"] is None for r in report["results"]))
        for r in report["results"]:
            self.assertTrue(all(op["result"]["precision"] == r["format"] for op in r["operators"]))
            if r["format"] == "bf16":
                self.assertEqual(r["speedup"], 1)
        self.assertEqual({r["split"] for r in report["aggregates"]}, {"design", "holdout"})

    def test_supplied_ppa_is_used_without_inventing_chip_cost(self):
        ppa = {"scope": "one_tensor_core", "profiles": {
            "1:2:4": {"area_relative": 1, "active_power_relative": {"16": 1, "4": 1}},
            "1:2:8": {"area_relative": 1.2, "active_power_relative": {"16": 1.1, "4": 1.6}}}}
        report = run_study(self.config(), ppa)
        for r in report["results"]:
            if r["ratio"] == "1:2:8":
                self.assertEqual(r["tc_area_relative"], 1.2)
                self.assertAlmostEqual(r["throughput_per_tc_area_relative"], r["speedup"] / 1.2)
            self.assertNotIn("chip_area_relative", r)

    def test_non_gemm_time_dilutes_gain_and_memory_floor_can_erase_it(self):
        config = self.config()
        base = run_study(config)["results"][3]["speedup"]
        config["workloads"][0]["non_gemm_seconds"] = 0.1
        with_other = run_study(config)["results"][3]["speedup"]
        self.assertGreater(base, with_other)
        config["hardware"]["hbm_bytes_per_second"] = 1
        report = run_study(config)
        self.assertTrue(all(r["speedup"] == 1 for r in report["results"]))

    def test_invalid_ppa_frequency_and_empty_ops_fail(self):
        with self.assertRaises(ValueError):
            run_study(self.config(), {"scope": "one_tensor_core", "frequency_hz": 2e9})
        config = self.config()
        config["workloads"][0]["ops"] = []
        with self.assertRaises(ValueError):
            run_study(config)

    def test_existing_transformer_shapes_export_and_execute(self):
        from scripts.export_transformer_gemms import export_workload
        config = self.config()
        for phase in ("prefill", "decode"):
            workload = export_workload(512, 8, 2, 32, phase, history=128)
            self.assertEqual(len(workload["ops"]), 8)
            qk = next(op for op in workload["ops"] if op["name"] == "Q_mul_K")
            self.assertEqual(qk["batch_count"], 16)
            self.assertEqual(qk["n"], 32 if phase == "prefill" else 129)
            config["workloads"] = [workload]
            report = run_study(config)
            self.assertEqual(len(report["results"]), 4)

    def test_per_format_power_overrides_shared_class(self):
        ppa = {"scope": "one_tensor_core", "profiles": {
            "1:2:4": {"active_power_relative": {"4": 1}},
            "1:2:8": {"active_power_relative": {"4": 99},
                       "active_power_relative_by_format": {"mxfp4": 1}}}}
        report = run_study(self.config(), ppa)
        row = next(r for r in report["results"] if r["workload"] == "selection" and r["format"] == "mxfp4" and r["ratio"] == "1:2:8")
        self.assertLess(row["tc_active_energy_proxy_relative"], 1)


if __name__ == "__main__":
    unittest.main()
