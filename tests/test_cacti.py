"""CACTI adapter contract tests with synthetic fixtures and a fake executable.

These fixtures test configuration, parsing, subprocess failures, and caching;
their numerical values are not SRAM predictions. Real CACTI integration is run
separately by the model_local_sram example.
"""

import csv
from pathlib import Path
import sys
import tempfile
import textwrap
import unittest

from cost_model.cacti import CactiError, CactiRunner, MacroSpec


# A small, self-contained input dialect fixture. Deliberately use values unlike
# the requested macro so tests check that the adapter actually overrides them.
TEMPLATE = """\
-size (bytes) 32768
-block size (bytes) 64
-associativity 8
-read-write port 3
-exclusive read port 2
-exclusive write port 2
-single ended read ports 1
-UCA bank count 8
-technology (u) 0.045
-Data array cell type - "itrs-lstp"
-Data array peripheral type - "itrs-lstp"
-output/input bus width 512
-operating temperature (K) 350
-cache type "cache"
-access mode (normal, sequential, fast) - "fast"
-design objective (weight delay, dynamic power, leakage power, cycle time, area) 100:0:0:0:0
-deviate (delay, dynamic power, leakage power, cycle time, area) 20:20:20:20:20
-Optimize ED or ED^2 (ED, ED^2, NONE): "ED"
-Cache model (NUCA, UCA) - "NUCA"
-Add ECC - "true"
-Print level (DETAILED, CONCISE) - "DETAILED"
-Print input parameters - "true"
-Force cache config - "true"
-Array Power Gating - "true"
-WL Power Gating - "true"
-CL Power Gating - "true"
-Bitline floating - "true"
-Interconnect Power Gating - "true"
"""


def synthetic_row(**changes):
    row = {
        "Tech node (nm)": "22",
        "Capacity (bytes)": "4096",
        "Number of banks": "1",
        "Associativity": "1",
        "Output width (bits)": "128",
        "Area (mm2)": "0.1",
        "Dynamic read energy (nJ)": "0.015",
        "Dynamic write energy (nJ)": "0.020",
        "Standby leakage per bank(mW)": "0.32",
        "Access time (ns)": "1.2",
        "Random cycle time (ns)": "0.8",
    }
    row.update(changes)
    return row


STDOUT = """\
Synthetic fixture, not physical data.
Total leakage power of a bank (mW): 0.3
Total gate leakage power of a bank (mW): 0.02
"""


class CactiAdapterTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="llmcompass-cacti-test-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.backend = self.root / "fake-cacti"
        self.backend.mkdir()
        (self.backend / "cache.cfg").write_text(TEMPLATE)
        (self.backend / "tech_params").mkdir()
        (self.backend / "tech_params/22nm.dat").write_text("synthetic fixture only\n")
        self.cache = self.root / "cache"
        self.spec = MacroSpec(capacity_bytes=4096, width_bits=128)
        self.write_executable("pass\n")

    def runner(self, timeout_seconds=2):
        return CactiRunner(self.backend, self.cache, timeout_seconds=timeout_seconds)

    def write_executable(self, body):
        executable = self.backend / "cacti"
        executable.write_text("#!{}\n".format(sys.executable) + textwrap.dedent(body))
        executable.chmod(0o755)

    def parse_fixture(self, row=None, stdout=STDOUT, extra_rows=None):
        artifact = self.root / "parse-fixture"
        artifact.mkdir(exist_ok=True)
        row = synthetic_row() if row is None else row
        with (artifact / "macro.cfg.out").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)
            for extra in extra_rows or []:
                writer.writerow(extra)
        (artifact / "stdout.log").write_text(stdout)
        return artifact

    def valid_executable(self):
        # Keep this executable independent of the adapter's parser: it produces
        # the public CSV/stdout contract just as an external process would.
        self.write_executable("""\
            import csv
            from pathlib import Path
            import sys
            counter = Path(__file__).with_name('invocations.txt')
            counter.write_text(str(int(counter.read_text()) + 1) if counter.exists() else '1')
            row = {!r}
            with Path(sys.argv[2] + '.out').open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(row))
                writer.writeheader()
                writer.writerow(row)
            print({!r})
        """.format(synthetic_row(), STDOUT))

    def test_configuration_is_one_tagless_no_ecc_macro(self):
        configuration = self.runner()._configuration(self.spec)
        lines = set(configuration.splitlines())
        for expected in (
            '-size (bytes) 4096',
            '-block size (bytes) 16',
            '-output/input bus width 128',
            '-associativity 1',
            '-UCA bank count 1',
            '-cache type "ram"',
            '-Cache model (NUCA, UCA) - "UCA"',
            '-Add ECC - "false"',
            '-technology (u) 0.022',
            '-operating temperature (K) 300',
        ):
            self.assertIn(expected, lines)

    def test_configuration_preserves_requested_port_topology(self):
        spec = MacroSpec(4096, 256, rw_ports=0, read_ports=1, write_ports=1)
        lines = set(self.runner()._configuration(spec).splitlines())
        self.assertIn('-read-write port 0', lines)
        self.assertIn('-exclusive read port 1', lines)
        self.assertIn('-exclusive write port 1', lines)
        self.assertIn('-block size (bytes) 32', lines)
        self.assertIn('-output/input bus width 256', lines)

    def test_missing_or_duplicate_configuration_fields_fail(self):
        for template in (
            TEMPLATE.replace('-Add ECC - "true"\n', ''),
            TEMPLATE + '-Add ECC - "false"\n',
        ):
            with self.subTest(template=template):
                (self.backend / "cache.cfg").write_text(template)
                with self.assertRaises(CactiError):
                    self.runner()._configuration(self.spec)

    def test_valid_csv_retains_units_and_separates_gate_leakage(self):
        artifact = self.parse_fixture()
        result = CactiRunner._parse(self.spec, artifact, {"test_fixture": True})
        self.assertEqual(result.area_mm2, 0.1)
        self.assertEqual(result.read_energy_nj, 0.015)
        self.assertEqual(result.write_energy_nj, 0.020)
        self.assertEqual(result.leakage_mw, 0.3)
        self.assertEqual(result.gate_leakage_mw, 0.02)
        self.assertEqual(result.cycle_time_ns, 0.8)
        self.assertEqual(result.access_time_ns, 1.2)

    def test_missing_csv_is_not_success(self):
        with self.assertRaises(CactiError):
            CactiRunner._parse(self.spec, self.root, {})

    def test_missing_required_csv_field_is_rejected(self):
        row = synthetic_row()
        del row["Dynamic write energy (nJ)"]
        with self.assertRaises(CactiError):
            CactiRunner._parse(self.spec, self.parse_fixture(row), {})

    def test_multiple_rows_are_rejected_to_prevent_stale_results(self):
        artifact = self.parse_fixture(extra_rows=[synthetic_row()])
        with self.assertRaises(CactiError):
            CactiRunner._parse(self.spec, artifact, {})

    def test_bad_numeric_values_are_rejected(self):
        for value in ("not-a-number", "nan", "inf", "-1", "0"):
            with self.subTest(value=value):
                row = synthetic_row(**{"Area (mm2)": value})
                with self.assertRaises(CactiError):
                    CactiRunner._parse(self.spec, self.parse_fixture(row), {})

    def test_nonpositive_timing_and_negative_energy_are_rejected(self):
        for field, value in (
            ("Random cycle time (ns)", "0"),
            ("Access time (ns)", "-1"),
            ("Dynamic read energy (nJ)", "-0.1"),
        ):
            with self.subTest(field=field):
                row = synthetic_row(**{field: value})
                with self.assertRaises(CactiError):
                    CactiRunner._parse(self.spec, self.parse_fixture(row), {})

    def test_csv_must_describe_the_requested_macro(self):
        for field, value in (
            ("Tech node (nm)", "32"),
            ("Capacity (bytes)", "8192"),
            ("Number of banks", "2"),
            ("Output width (bits)", "256"),
        ):
            with self.subTest(field=field):
                row = synthetic_row(**{field: value})
                with self.assertRaises(CactiError):
                    CactiRunner._parse(self.spec, self.parse_fixture(row), {})

    def test_missing_or_inconsistent_leakage_is_rejected(self):
        for stdout in ("No leakage fields\n", STDOUT.replace(": 0.3", ": 0.5")):
            with self.subTest(stdout=stdout):
                with self.assertRaises(CactiError):
                    CactiRunner._parse(self.spec, self.parse_fixture(stdout=stdout), {})

    def test_exit_zero_without_results_is_rejected(self):
        self.write_executable("print('Invalid technology nodes')\n")
        with self.assertRaises(CactiError):
            self.runner().simulate(self.spec)
        self.assertFalse(list(self.cache.glob("*/result.json")))
        logs = list(self.cache.glob("*/stdout.log"))
        self.assertEqual(len(logs), 1)
        self.assertIn("Invalid technology nodes", logs[0].read_text())

    def test_nonzero_process_exit_is_rejected_and_logged(self):
        self.write_executable("import sys\nprint('synthetic failure', file=sys.stderr)\nsys.exit(3)\n")
        with self.assertRaises(CactiError):
            self.runner().simulate(self.spec)
        logs = list(self.cache.glob("*/stderr.log"))
        self.assertEqual(len(logs), 1)
        self.assertIn("synthetic failure", logs[0].read_text())
        self.assertFalse(list(self.cache.glob("*/result.json")))

    def test_timeout_is_rejected_without_caching_success(self):
        self.write_executable("import time\ntime.sleep(2)\n")
        with self.assertRaises(CactiError):
            self.runner(timeout_seconds=0.05).simulate(self.spec)
        self.assertFalse(list(self.cache.glob("*/result.json")))

    def test_successful_external_contract_is_cached(self):
        self.valid_executable()
        runner = self.runner()
        first = runner.simulate(self.spec)
        second = runner.simulate(self.spec)
        self.assertEqual(first, second)
        self.assertEqual((self.backend / "invocations.txt").read_text(), "1")
        artifact = Path(first.artifact_dir)
        self.assertTrue((artifact / "macro.cfg").is_file())
        self.assertTrue((artifact / "request.json").is_file())
        self.assertTrue((artifact / "result.json").is_file())
        self.assertIsNone(first.provenance["source_revision"])

    def test_technology_data_changes_invalidate_cached_results(self):
        self.valid_executable()
        runner = self.runner()
        first = runner.simulate(self.spec)
        (self.backend / "tech_params/22nm.dat").write_text("changed synthetic fixture\n")
        second = runner.simulate(self.spec)
        self.assertNotEqual(first.provenance["cache_key"], second.provenance["cache_key"])
        self.assertEqual((self.backend / "invocations.txt").read_text(), "2")

    def test_unsupported_technology_and_subword_bus_are_rejected(self):
        with self.assertRaises(ValueError):
            MacroSpec(4096, 128, technology_nm=12)
        for width in (8, 16, 32):
            with self.subTest(width=width):
                with self.assertRaises(ValueError):
                    MacroSpec(4096, width)


if __name__ == "__main__":
    unittest.main()
