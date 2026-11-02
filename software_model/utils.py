from math import ceil, prod
from typing import List
from utils import size


class DataType:
    def __init__(self, name: str, word_size, *, payload_bits=None,
                 compute_class=None, block_size=1, metadata_bits=0,
                 tensor_metadata_bits=0) -> None:
        self.name = name
        self.word_size = word_size  # Average bytes/value; use storage_bytes for allocation.
        self.payload_bits = int(word_size * 8) if payload_bits is None else payload_bits
        self.compute_class = compute_class
        self.block_size = block_size
        self.metadata_bits = metadata_bits
        self.tensor_metadata_bits = tensor_metadata_bits

    def storage_bytes(self, shape, block_axis=-1):
        """Packed storage, padding the selected grouping axis to a full block.

        Metadata is explicit. Vendor swizzles, macro word rounding and replicas
        belong to the memory backend and are not included here.
        """
        if not shape or any(isinstance(x, bool) or not isinstance(x, int) or x < 1 for x in shape):
            raise ValueError("storage shape must contain positive integers")
        if not -len(shape) <= block_axis < len(shape):
            raise ValueError("block_axis is outside the tensor rank")
        axis = block_axis % len(shape)
        rows = prod(shape[:axis] + shape[axis + 1:])
        blocks = (shape[axis] + self.block_size - 1) // self.block_size
        row_bits = blocks * (self.block_size * self.payload_bits + self.metadata_bits)
        return rows * ceil(row_bits / 8) + ceil(self.tensor_metadata_bits / 8)

data_type_dict = {
    "int8": DataType("int8", 1),
    "fp16": DataType("fp16", 2, compute_class=16),
    "bf16": DataType("bf16", 2, compute_class=16),
    "fp32": DataType("fp32", 4),
}


def _block_format(name, bits, compute_class, block_size, metadata_bits,
                  tensor_metadata_bits=0):
    data_type_dict[name] = DataType(
        name, (bits + metadata_bits / block_size) / 8,
        payload_bits=bits, compute_class=compute_class, block_size=block_size,
        metadata_bits=metadata_bits, tensor_metadata_bits=tensor_metadata_bits)


_block_format("mxfp8", 8, 8, 32, 8)  # E4M3 payload, E8M0 block scale.
_block_format("mxfp6", 6, 8, 32, 8)  # E2M3 payload; shares the 8/6-bit compute class.
_block_format("mxfp4", 4, 4, 32, 8)  # E2M1 payload, E8M0 block scale.
_block_format("nvfp4", 4, 4, 16, 8, 32)  # E4M3 block scale + one FP32 tensor scale.
_block_format("hif4", 4, 4, 64, 32)  # HiF4 hierarchical shared metadata.
for alias, canonical in {
    "mxfp8_e4m3": "mxfp8", "mxfp8_e5m2": "mxfp8",
    "mxfp6_e2m3": "mxfp6", "mxfp6_e3m2": "mxfp6",
}.items():
    # Preserve the encoding name even when the performance/storage class agrees.
    base = data_type_dict[canonical]
    data_type_dict[alias] = DataType(
        alias, base.word_size, payload_bits=base.payload_bits,
        compute_class=base.compute_class, block_size=base.block_size,
        metadata_bits=base.metadata_bits)

class Tensor:
    def __init__(
        self, shape: List, data_type=data_type_dict["fp16"]
    ) -> None:
        self.shape = shape
        self.size = size(shape)
        self.data_type = data_type

    @property
    def nbytes(self):
        return self.data_type.storage_bytes(self.shape)
        
