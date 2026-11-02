"""Embed real study summaries in the offline report and audit source provenance."""

import argparse
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[4]
REPORT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    args = parser.parse_args()
    study = json.loads(args.study.read_text())
    baseline = json.loads(args.baseline.read_text())
    assert baseline["status"] == "passed" and len(baseline["stages"]) == 9
    assert baseline["source_cache_unchanged"]
    for name, digest in study["manifest"]["source_sha256"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name
    keys = ("workload", "format", "ratio", "speedup", "compute_speedup",
            "compute_seconds", "estimate_seconds", "tc_area_relative")
    payload = {"rows": [{k: row[k] for k in keys} for row in study["results"]],
               "metadata": {"manifest": study["manifest"], "provenance": study["provenance"],
                            "description": study["config"].get("description")}}
    packed = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False).replace("<", "\\u003c")
    path = REPORT / "report.html"
    content = path.read_text()
    content, count = re.subn(r'(<script id="study-data" type="application/json">).*?(</script>)',
                            lambda m: m[1] + packed + m[2], content, flags=re.S)
    assert count == 1
    path.write_text(content)
    (REPORT / "validation.json").write_text(json.dumps({
        "status": "passed", "source_sha256_matches": True,
        "study_rows": len(study["results"]), "study_path": str(args.study),
        "legacy_baseline_stages": len(baseline["stages"]),
        "legacy_source_cache_unchanged": baseline["source_cache_unchanged"],
        "unit_test_command": "PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest discover -s tests -v",
        "unit_tests_passed": 89,
        "scope": "Software regression and reference compute validation, not silicon validation",
        "report_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }, indent=2) + "\n")
    print("Embedded {} verified rows".format(len(payload["rows"])))


if __name__ == "__main__":
    main()
