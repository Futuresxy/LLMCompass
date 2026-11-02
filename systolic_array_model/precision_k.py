"""SCALE-Sim OS schedules with vector-dot tokens along K.

One OS cell now performs L scalar MACs per step. SCALE-Sim sees ceil(K/L)
tokens and retains its skew/fold/drain schedule. This is a compute-only model:
tokens are NOT SRAM addresses or transactions. It assumes a one-step dot unit,
FP32 resident accumulation, ready operands/scales, and no local-memory stalls.
"""

from functools import lru_cache
from hashlib import sha256
from importlib import metadata
import inspect
from math import ceil

import numpy as np
from scalesim.compute.systolic_compute_os import systolic_compute_os


def _positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(name + " must be a positive integer")


@lru_cache(maxsize=1)
def provenance():
    source = inspect.getsource(systolic_compute_os).encode()
    return {"backend": "scalesim-os-vector-dot-v1",
            "scalesim_version": metadata.version("scalesim"),
            "compute_source_sha256": sha256(source).hexdigest(),
            "validated_version": "2.0.2"}


class _ArrayConfig:
    def __init__(self, rows, cols):
        self.dims = rows, cols

    def get_array_dims(self):
        return self.dims


def token_trace(m, n, k_groups, rows, cols):
    for name, value in (("m", m), ("n", n), ("k_groups", k_groups),
                        ("rows", rows), ("cols", cols)):
        _positive_int(value, name)
    if m > rows or n > cols:
        raise ValueError("One job must fit one physical PE grid")
    # Address-like integers identify vector tokens, not scalar/SRAM addresses.
    comp = systolic_compute_os()
    comp.set_params(
        config_obj=_ArrayConfig(rows, cols),
        ifmap_op_mat=np.arange(m * k_groups, dtype=np.int64).reshape(m, k_groups),
        filter_op_mat=np.arange(k_groups * n, dtype=np.int64).reshape(k_groups, n),
        ofmap_op_mat=np.arange(m * n, dtype=np.int64).reshape(m, n))
    return comp.get_demand_matrices()


@lru_cache(maxsize=2048)
def _schedule(m, n, k_groups, rows, cols):
    a, b, c = token_trace(m, n, k_groups, rows, cols)
    if a.shape[0] != b.shape[0] or b.shape[0] != c.shape[0]:
        raise RuntimeError("SCALE-Sim returned inconsistent schedule lengths")
    outputs = int(np.count_nonzero(c >= 0))
    if outputs != m * n:
        raise RuntimeError("SCALE-Sim output token count does not match the tile")
    # Inclusive allocated clock slots, NOT the last zero-based timestamp and
    # NOT SCALE-Sim 2.0.2's inaccurate OS get_ofmap_requests() report.
    return {"cycles": int(a.shape[0]), "a_tokens": int(np.count_nonzero(a >= 0)),
            "b_tokens": int(np.count_nonzero(b >= 0)), "c_tokens": outputs,
            "last_valid_output_cycle": int(np.nonzero(c >= 0)[0].max())}


def simulate_gemm(m, n, k, data_type, profile, rows=16, cols=16,
                  engine_count=1, batch_count=1, dataflow="os"):
    for name, value in (("m", m), ("n", n), ("k", k), ("rows", rows),
                        ("cols", cols), ("engine_count", engine_count),
                        ("batch_count", batch_count)):
        _positive_int(value, name)
    if dataflow != "os":
        raise ValueError("K-compressed dot tokens are implemented for OS only; WS/IS require another mapping")
    length = profile.dot_length(data_type)
    groups = (k + length - 1) // length
    mt, nt = (m + rows - 1) // rows, (n + cols - 1) // cols
    jobs = batch_count * mt * nt
    waves = (jobs + engine_count - 1) // engine_count
    # SCALE-Sim reserves a full physical grid's skew/drain slots, including in
    # tail jobs. Use the four possible active shapes and count their traces.
    m_shapes = [(rows, m // rows), (m % rows, 1)]
    n_shapes = [(cols, n // cols), (n % cols, 1)]
    a_tokens = b_tokens = c_tokens = 0
    job_cycles = None
    for am, mc in m_shapes:
        for an, nc in n_shapes:
            if not am or not an or not mc or not nc:
                continue
            schedule = _schedule(am, an, groups, rows, cols)
            if job_cycles is not None and job_cycles != schedule["cycles"]:
                raise RuntimeError("Backend no longer reserves uniform job slots; update wave scheduling")
            job_cycles = schedule["cycles"]
            count = mc * nc * batch_count
            a_tokens += count * schedule["a_tokens"]
            b_tokens += count * schedule["b_tokens"]
            c_tokens += count * schedule["c_tokens"]
    cycles = waves * job_cycles
    useful_macs = batch_count * m * n * k
    peak = engine_count * rows * cols * length
    return {
        "precision": data_type.name, "compute_class": data_type.compute_class,
        "ratio": profile.name, "dot_length": length, "shape": [m, n, k],
        "batch_count": batch_count, "array_shape": [rows, cols],
        "engine_count": engine_count, "k_groups": groups,
        "k_tail_valid_elements": k - (groups - 1) * length,
        "jobs": jobs, "waves": waves, "cycles_per_job": job_cycles,
        "compute_cycles": cycles, "total_array_active_cycles": jobs * job_cycles,
        "useful_macs": useful_macs, "peak_macs_per_cycle": peak,
        "compute_utilization": useful_macs / (peak * cycles),
        "k_packing_efficiency": k / (groups * length),
        "mn_spatial_efficiency": m * n / (mt * rows * nt * cols),
        "wave_efficiency": jobs / (waves * engine_count),
        "pipeline_efficiency": groups / job_cycles,
        "vector_tokens": {"a": a_tokens, "b": b_tokens, "c": c_tokens},
        "provenance": dict(provenance()),
        "scope": "OS reference compute schedule; one-cycle dot/update and links; ideal operand/scale supply; no SRAM/bank stalls, conversion cost, or vendor microarchitecture claim",
    }


def analyze_matmul(m, n, k, data_type, output_type, device, batch_count=1):
    """Compute schedule plus an explicitly optimistic memory overlap envelope.

    Compulsory bytes read A/B and write D once; no initial C. Inter-tile
    duplication, local bandwidth and conversion are omitted, not fitted away.
    """
    cm = device.compute_module
    array = cm.core.systolic_array
    if array.precision_profile is None:
        raise ValueError("A precision_profile is required")
    result = simulate_gemm(
        m, n, k, data_type, array.precision_profile,
        rows=array.array_height, cols=array.array_width,
        engine_count=cm.core_count * cm.core.systolic_array_count,
        batch_count=batch_count)
    a = data_type.storage_bytes([batch_count, m, k])
    b = data_type.storage_bytes([batch_count, k, n], block_axis=-2)
    d = output_type.storage_bytes([batch_count, m, n])
    compulsory_bytes = a + b + d
    memory_seconds = max(compulsory_bytes / device.io_module.bandwidth,
                         compulsory_bytes / (cm.l2_bandwidth_per_cycle * cm.clock_freq))
    compute_seconds = result["compute_cycles"] / cm.clock_freq
    result.update(
        operand_storage_bytes={"a": a, "b": b, "d": d},
        compulsory_bytes=compulsory_bytes,
        accumulator_format="fp32", output_format=output_type.name,
        compute_seconds=compute_seconds, compulsory_memory_seconds=memory_seconds,
        latency_seconds=max(compute_seconds, memory_seconds),
        latency_model="max(OS compute schedule, compulsory HBM/L2 transfer); ideal overlap envelope, not measured kernel latency")
    return result
