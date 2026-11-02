"""Precision classes for a shared, K-expanded dot-product array.

These describe a reference design, not an inferred NVIDIA/Ascend physical array.
All modes share the same PE grid; a step performs one vector dot product per PE.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class KExpansionProfile:
    base_dot_length: int = 16
    ratios: tuple = (1, 2, 4)

    def __post_init__(self):
        if isinstance(self.base_dot_length, bool) or not isinstance(self.base_dot_length, int) or self.base_dot_length < 1:
            raise ValueError("base_dot_length must be a positive integer")
        object.__setattr__(self, "ratios", tuple(self.ratios))
        if len(self.ratios) != 3 or any(isinstance(x, bool) or not isinstance(x, int) or x < 1 for x in self.ratios):
            raise ValueError("ratios must contain three positive integers for 16, 8/6, 4 bit")
        if self.ratios[0] != 1:
            raise ValueError("ratios must be normalized to the fixed 16-bit baseline (first entry 1)")

    def dot_length(self, data_type):
        precision_class = data_type.compute_class
        if precision_class not in (16, 8, 4):
            raise ValueError("Unsupported Tensor Core format: {}".format(data_type.name))
        return self.base_dot_length * self.ratios[(16, 8, 4).index(precision_class)]

    @property
    def name(self):
        return ":".join(str(x) for x in self.ratios)

    @classmethod
    def from_dict(cls, value):
        return cls(base_dot_length=value.get("base_dot_length", 16),
                   ratios=tuple(value.get("ratios", (1, 2, 4))))
