#!/usr/bin/env python3
"""Export shaped GEMMs from the existing dense Transformer block into a study.

This exports operator shapes, not traces from an inference framework. Attention
remains the repository's unfused two-GEMM implementation; non-GEMM time must be
measured/supplied for an end-to-end workload estimate.
"""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from software_model.matmul import BatchedMatmul, Matmul
from software_model.transformer import TransformerBlockAutoRegressionTP, TransformerBlockInitComputationTP
from software_model.utils import Tensor, data_type_dict


def export_workload(hidden, heads, batch, sequence, phase, history=128, layers=1,
                    device_count=1, split="design", non_gemm_seconds=0):
    values = (hidden, heads, batch, sequence, history, layers, device_count)
    if any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in values):
        raise ValueError("dimensions/counts must be positive integers")
    if hidden % heads or hidden % device_count or heads % device_count:
        raise ValueError("hidden/heads and tensor-parallel dimensions must divide exactly")
    dtype = data_type_dict["fp16"]
    if phase == "prefill":
        block = TransformerBlockInitComputationTP(hidden, heads, device_count, dtype)
        block(Tensor([batch, sequence, hidden], dtype))
    elif phase == "decode":
        block = TransformerBlockAutoRegressionTP(hidden, heads, device_count, dtype)
        block(Tensor([batch, 1, hidden], dtype), history)
    else:
        raise ValueError("phase must be prefill or decode")
    ops = []
    for name, op in vars(block).items():
        if isinstance(op, (Matmul, BatchedMatmul)):
            ops.append({"name": name, "m": op.M, "n": op.N, "k": op.K,
                        "batch_count": getattr(op, "bs", 1), "count": layers})
    return {"name": "legacy_dense_{}_d{}_b{}".format(phase, hidden, batch),
            "family": "legacy_dense_transformer", "split": split,
            "non_gemm_seconds": non_gemm_seconds, "ops": ops,
            "source": {"kind": "existing LLMCompass shaped block, not a measured framework trace",
                       "hidden": hidden, "heads": heads, "batch": batch,
                       "sequence": sequence, "phase": phase, "history": history,
                       "layers": layers, "device_count": device_count},
            "scope": "Per-rank GEMMs only; unfused attention; excludes embeddings, output head, non-GEMM, communication and runtime overhead unless separately supplied"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--sequence", type=int, default=512)
    parser.add_argument("--history", type=int, default=2048)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--device-count", type=int, default=1)
    parser.add_argument("--phase", choices=("prefill", "decode"), default="prefill")
    parser.add_argument("--split", choices=("design", "holdout"), default="design")
    parser.add_argument("--non-gemm-seconds", type=float, default=0)
    parser.add_argument("--template", type=Path, default=ROOT / "configs/precision_ratio_study.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists; choose a new path")
    config = json.loads(args.template.read_text())
    config["description"] = "GEMM shapes exported from the existing LLMCompass dense block; not measured LLM performance."
    config["workloads"] = [export_workload(
        args.hidden, args.heads, args.batch, args.sequence, args.phase, args.history,
        args.layers, args.device_count, args.split, args.non_gemm_seconds)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(config, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "operators": len(config["workloads"][0]["ops"])}))


if __name__ == "__main__":
    main()
