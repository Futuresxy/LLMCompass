# Precision-ratio exploration

LLMCompass compares shared multi-format Tensor Core designs against inference
workloads. Precision semantics and hardware resource allocation are distinct.

## Language

**Numeric format**:
The encoding of operand values and associated scaling metadata, such as BF16,
MXFP6, NVFP4 or HiF4. Formats with equal payload width can have different storage
and numerical behavior.

**Compute class**:
One of the 16-bit, 8/6-bit or 4-bit throughput groups in the reference design.
Membership expresses a hardware design choice, not equivalence of formats.

**Precision ratio**:
The peak MAC throughput ratio of the three compute classes on the same physical
engine, normalized to a fixed 16-bit mode. It is not a model's precision mix.

**K expansion**:
Increasing the number of products reduced by each PE's dot-product update while
retaining its M/N grid. Its feasibility depends on implementation timing and
operand supply.

**Uniform-GEMM workload**:
A workload whose GEMM operators all use one selected operand format. Non-GEMM
operations and FP32 accumulation can retain their required formats.
_Avoid_: All model operations execute at 4 bit

**Compute schedule**:
The allocated array clock slots for mapped computation, under explicitly stated
operand-readiness and pipeline assumptions.

**Memory overlap envelope**:
An estimate combining a compute schedule and compulsory transfer costs with
ideal overlap. It does not establish local-memory feasibility or kernel latency.

**Deployment portfolio**:
A fixed, weighted set of model families, execution phases, shapes and precision
policies used to evaluate one hardware choice.

**Held-out workload**:
A model family or deployment condition excluded from hardware selection and used
only after the design and evaluation rules have been frozen.
