#!/usr/bin/env python3
"""Fetch and build the pinned official CACTI version in an isolated directory."""
import argparse
from pathlib import Path
import subprocess

REVISION = "1ffd8dfb10303d306ecd8d215320aea07651e878"
URL = "https://github.com/HewlettPackard/cacti.git"
ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=ROOT / ".runtime/cacti")
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    directory = args.directory.expanduser().resolve()
    if not directory.exists():
        directory.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", URL, str(directory)], check=True)
        subprocess.run(["git", "-C", str(directory), "checkout", "--detach", REVISION], check=True)
    revision = subprocess.check_output(
        ["git", "-C", str(directory), "rev-parse", "HEAD"], text=True).strip()
    if revision != REVISION:
        parser.error("Existing CACTI revision differs; use a new --directory (no files overwritten)")
    changes = subprocess.check_output(
        ["git", "-C", str(directory), "status", "--porcelain", "--untracked-files=no"], text=True)
    if changes.strip():
        parser.error("Existing CACTI sources have tracked changes; refusing to build as pinned")
    subprocess.run(["make", "-j{}".format(args.jobs)], cwd=str(directory), check=True)
    print("CACTI ready: {} (revision {})".format(directory / "cacti", REVISION))


if __name__ == "__main__":
    main()
