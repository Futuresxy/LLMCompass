"""CACTI adapter for one independently interfaced, tagless SRAM macro.

CACTI supplies access energy, timing, and area. The caller is responsible for
bank-level parallelism, read/write arbitration, clocking, and workload activity.
The supported configuration dialect is HewlettPackard/cacti (CACTI 7.0).
"""

from dataclasses import asdict, dataclass
import csv
import fcntl
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
from typing import Dict


SCHEMA_VERSION = 1
SUPPORTED_TECHNOLOGY_NM = (22, 32, 45, 65, 90)


class CactiError(RuntimeError):
    """CACTI could not produce a valid result for the requested macro."""


def _integer(value, name, minimum):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError("{} must be an integer >= {}".format(name, minimum))


@dataclass(frozen=True)
class MacroSpec:
    capacity_bytes: int
    width_bits: int
    read_ports: int = 0
    write_ports: int = 0
    rw_ports: int = 1
    technology_nm: float = 22
    temperature_k: int = 300
    cell_type: str = "itrs-hp"

    def __post_init__(self):
        _integer(self.capacity_bytes, "capacity_bytes", 64)
        _integer(self.width_bits, "width_bits", 64)
        if self.capacity_bytes & (self.capacity_bytes - 1):
            raise ValueError("capacity_bytes must be a power of two for this adapter")
        if self.width_bits & (self.width_bits - 1):
            raise ValueError("width_bits must be a power of two for this adapter")
        if self.capacity_bytes < max(8, self.width_bits // 8):
            raise ValueError("An access line cannot exceed macro capacity")
        for name in ("read_ports", "write_ports", "rw_ports"):
            _integer(getattr(self, name), name, 0)
        if self.read_ports + self.write_ports + self.rw_ports == 0:
            raise ValueError("At least one SRAM port is required")
        if (isinstance(self.technology_nm, bool)
                or self.technology_nm not in SUPPORTED_TECHNOLOGY_NM):
            raise ValueError(
                "Supported CACTI technology nodes are {} nm; 12 nm needs an "
                "explicit calibrated technology model, not silent scaling".format(
                    SUPPORTED_TECHNOLOGY_NM))
        _integer(self.temperature_k, "temperature_k", 300)
        if self.temperature_k > 400 or self.temperature_k % 10:
            raise ValueError("temperature_k must be 300..400 K in increments of 10 K")
        if self.cell_type not in ("itrs-hp", "itrs-lstp", "itrs-lop"):
            raise ValueError("cell_type must be itrs-hp, itrs-lstp, or itrs-lop")


@dataclass(frozen=True)
class MacroResult:
    area_mm2: float
    read_energy_nj: float
    write_energy_nj: float
    leakage_mw: float
    gate_leakage_mw: float
    access_time_ns: float
    cycle_time_ns: float
    artifact_dir: str
    provenance: Dict


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_result(result):
    for name in ("area_mm2", "read_energy_nj", "write_energy_nj", "leakage_mw",
                 "gate_leakage_mw", "access_time_ns", "cycle_time_ns"):
        value = getattr(result, name)
        if isinstance(value, bool) or not math.isfinite(value) or value < 0:
            raise CactiError("Invalid CACTI {}: {!r}".format(name, value))
        if name in ("area_mm2", "access_time_ns", "cycle_time_ns") and value == 0:
            raise CactiError("CACTI {} must be positive".format(name))


class CactiRunner:
    def __init__(self, cacti_dir: Path, cache_dir: Path, timeout_seconds=60):
        self.cacti_dir = Path(cacti_dir).expanduser().resolve()
        self.cache_dir = Path(cache_dir).expanduser().resolve()
        self.binary = self.cacti_dir / "cacti"
        self.template = self.cacti_dir / "cache.cfg"
        if (isinstance(timeout_seconds, bool)
                or not math.isfinite(timeout_seconds) or timeout_seconds <= 0):
            raise ValueError("timeout_seconds must be finite and positive")
        self.timeout_seconds = timeout_seconds

    def _provenance(self):
        if not self.binary.is_file() or not self.template.is_file():
            raise CactiError("Build CACTI first; expected cacti and cache.cfg in {}".format(
                self.cacti_dir))
        technology_files = sorted((self.cacti_dir / "tech_params").glob("*.dat"))
        if not technology_files:
            raise CactiError("CACTI tech_params/*.dat are missing")
        revision = None
        try:
            repository = subprocess.run(
                ["git", "-C", str(self.cacti_dir), "rev-parse", "--show-toplevel"],
                capture_output=True, text=True, timeout=5, check=False)
            # An unpacked CACTI directory may live inside the caller's repository;
            # do not mislabel that parent's commit as the CACTI source revision.
            if (repository.returncode == 0
                    and Path(repository.stdout.strip()).resolve() == self.cacti_dir):
                process = subprocess.run(
                    ["git", "-C", str(self.cacti_dir), "rev-parse", "HEAD"],
                    capture_output=True, text=True, timeout=5, check=False)
                if process.returncode == 0:
                    revision = process.stdout.strip()
        except (OSError, subprocess.TimeoutExpired):
            pass
        return {
            "schema_version": SCHEMA_VERSION,
            "backend": "HewlettPackard/cacti",
            "source_revision": revision,
            "binary_sha256": _digest(self.binary),
            "template_sha256": _digest(self.template),
            "technology_sha256": {
                str(path.relative_to(self.cacti_dir)): _digest(path)
                for path in technology_files},
            "model_scope": "One independent UCA bank; tagless SRAM; ECC disabled",
            "optimization": "CACTI cycle-time objective; outer search selects among macros",
            "leakage_fields": "leakage_mw excludes gate leakage; add gate_leakage_mw once",
        }

    def _configuration(self, spec):
        config = self.template.read_text(encoding="utf-8")
        changes = {
            "-size (bytes)": str(spec.capacity_bytes),
            "-block size (bytes)": str(max(8, spec.width_bits // 8)),
            "-associativity": "1",
            "-read-write port": str(spec.rw_ports),
            "-exclusive read port": str(spec.read_ports),
            "-exclusive write port": str(spec.write_ports),
            "-single ended read ports": "0",
            "-UCA bank count": "1",
            "-technology (u)": "{:g}".format(spec.technology_nm / 1000),
            "-Data array cell type": '- "{}"'.format(spec.cell_type),
            "-Data array peripheral type": '- "{}"'.format(spec.cell_type),
            "-output/input bus width": str(spec.width_bits),
            "-operating temperature (K)": str(spec.temperature_k),
            "-cache type": '"ram"',
            "-access mode (normal, sequential, fast)": '- "normal"',
            "-design objective (weight delay, dynamic power, leakage power, cycle time, area)":
                "0:0:0:100:0",
            "-deviate (delay, dynamic power, leakage power, cycle time, area)":
                "100000:100000:100000:100000:100000",
            "-Optimize ED or ED^2 (ED, ED^2, NONE):": '"NONE"',
            "-Cache model (NUCA, UCA)": '- "UCA"',
            "-Add ECC": '- "false"',
            "-Print level (DETAILED, CONCISE)": '- "CONCISE"',
            "-Print input parameters": '- "false"',
            "-Force cache config": '- "false"',
            "-Array Power Gating": '- "false"',
            "-WL Power Gating": '- "false"',
            "-CL Power Gating": '- "false"',
            "-Bitline floating": '- "false"',
            "-Interconnect Power Gating": '- "false"',
        }
        for key, value in changes.items():
            pattern = r"^" + re.escape(key) + r"[ \t]+.*$"
            config, count = re.subn(pattern, key + " " + value, config, flags=re.MULTILINE)
            if count != 1:
                raise CactiError("Expected one active {!r} field in cache.cfg, found {}".format(
                    key, count))
        return config

    @staticmethod
    def _parse(spec, artifact, provenance):
        csv_path = artifact / "macro.cfg.out"
        if not csv_path.is_file():
            raise CactiError("CACTI returned no CSV; inspect {}".format(artifact))
        with csv_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle, skipinitialspace=True))
        if len(rows) != 1:
            raise CactiError("Expected exactly one CACTI CSV row in {}".format(csv_path))
        row = {key.strip(): value.strip() for key, value in rows[0].items()
               if key and value is not None}
        try:
            for key, expected in (
                ("Tech node (nm)", spec.technology_nm),
                ("Capacity (bytes)", spec.capacity_bytes),
                ("Number of banks", 1), ("Associativity", 1),
                ("Output width (bits)", spec.width_bits),
            ):
                if float(row[key]) != expected:
                    raise CactiError("CACTI CSV {} does not match requested macro".format(key))
            stdout = (artifact / "stdout.log").read_text(encoding="utf-8")

            def leakage(label):
                match = re.search(re.escape(label) + r"\s*:\s*(\S+)", stdout)
                if not match:
                    raise CactiError("Missing {} in CACTI stdout".format(label))
                return float(match.group(1))

            subthreshold = leakage("Total leakage power of a bank (mW)")
            gate = leakage("Total gate leakage power of a bank (mW)")
            total_leakage = float(row["Standby leakage per bank(mW)"])
            if not math.isclose(subthreshold + gate, total_leakage, rel_tol=5e-5, abs_tol=1e-9):
                raise CactiError("CACTI stdout and CSV leakage disagree")
            result = MacroResult(
                area_mm2=float(row["Area (mm2)"]),
                read_energy_nj=float(row["Dynamic read energy (nJ)"]),
                write_energy_nj=float(row["Dynamic write energy (nJ)"]),
                leakage_mw=subthreshold, gate_leakage_mw=gate,
                access_time_ns=float(row["Access time (ns)"]),
                cycle_time_ns=float(row["Random cycle time (ns)"]),
                artifact_dir=str(artifact), provenance=provenance)
        except (KeyError, TypeError, ValueError) as exc:
            raise CactiError("Invalid CACTI CSV/stdout in {}: {}".format(artifact, exc)) from exc
        _validate_result(result)
        return result

    def simulate(self, spec: MacroSpec) -> MacroResult:
        if not isinstance(spec, MacroSpec):
            raise TypeError("simulate expects a MacroSpec")
        provenance = self._provenance()
        configuration = self._configuration(spec)
        identity = {"spec": asdict(spec), "provenance": provenance,
                    "configuration_sha256": hashlib.sha256(configuration.encode()).hexdigest()}
        cache_key = hashlib.sha256(json.dumps(
            identity, sort_keys=True, allow_nan=False).encode()).hexdigest()
        provenance = dict(provenance, cache_key=cache_key,
                          configuration_sha256=identity["configuration_sha256"])
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        artifact = self.cache_dir / cache_key
        with (self.cache_dir / (cache_key + ".lock")).open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            artifact.mkdir(exist_ok=True)
            result_path = artifact / "result.json"
            if result_path.is_file():
                try:
                    cached = json.loads(result_path.read_text(encoding="utf-8"))
                    if cached["cache_key"] == cache_key:
                        result = MacroResult(**cached["result"])
                        _validate_result(result)
                        return result
                except (KeyError, TypeError, ValueError, CactiError):
                    pass
            config_path = artifact / "macro.cfg"
            config_path.write_text(configuration, encoding="utf-8")
            (artifact / "request.json").write_text(
                json.dumps(identity, indent=2, sort_keys=True, allow_nan=False) + "\n",
                encoding="utf-8")
            csv_path = artifact / "macro.cfg.out"
            if csv_path.exists():
                csv_path.unlink()  # CACTI appends; never parse a row from an earlier attempt.
            with (artifact / "stdout.log").open("w", encoding="utf-8") as stdout, \
                    (artifact / "stderr.log").open("w", encoding="utf-8") as stderr:
                try:
                    process = subprocess.run(
                        [str(self.binary), "-infile", str(config_path)],
                        cwd=str(self.cacti_dir), stdout=stdout, stderr=stderr,
                        timeout=self.timeout_seconds, check=False)
                except (OSError, subprocess.TimeoutExpired) as exc:
                    raise CactiError("CACTI execution failed; inspect {}: {}".format(
                        artifact, exc)) from exc
            if process.returncode != 0:
                raise CactiError("CACTI exited with {}; inspect {}".format(
                    process.returncode, artifact))
            # Unsupported technology can exit(0); valid, finite results are mandatory.
            result = self._parse(spec, artifact, provenance)
            temporary = artifact / "result.json.tmp"
            temporary.write_text(json.dumps(
                {"schema_version": SCHEMA_VERSION, "cache_key": cache_key,
                 "spec": asdict(spec), "result": asdict(result)},
                indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
            temporary.replace(result_path)
            return result
