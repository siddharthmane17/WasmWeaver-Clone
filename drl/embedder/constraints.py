import numpy as np
from typing import List
from gymnasium.spaces import Box

from core.config.config import MAX_CONSTRAINTS
from core.constraints import AbstractConstraint, FuelConstraint, ByteCodeSizeConstraint


class ConstraintsEmbedder:
    def __init__(self):
        self.constraint_dim = 5  # index, type, min, max, value
        self.total_dim = MAX_CONSTRAINTS * self.constraint_dim

    def get_space(self):
        return Box(low=0, high=1, shape=(self.total_dim,), dtype=np.float32)

    def __call__(self, constraints: List[AbstractConstraint]):
        data = np.zeros((MAX_CONSTRAINTS, self.constraint_dim), dtype=np.float32)

        for index, constraint in enumerate(constraints):
            if index >= MAX_CONSTRAINTS:
                break

            if isinstance(constraint, FuelConstraint):
                constraint_type = 0
            elif isinstance(constraint, ByteCodeSizeConstraint):
                constraint_type = 1
            else:
                constraint_type = 2  # For future-proofing

            # Normalize all values between 0 and 1 for stability
            min_val = float(constraint.min_target) / 1e6
            max_val = float(constraint.max_target) / 1e6
            val = float(constraint.resource) / 1e5

            data[index] = np.array([
                index / MAX_CONSTRAINTS,
                constraint_type,
                min_val,
                max_val,
                val
            ])

        return data.flatten()  # Final shape: (MAX_CONSTRAINTS * 5,)


if __name__ == "__main__":
    constraints = [FuelConstraint(50, 100, 100), ByteCodeSizeConstraint(0, 200, 300)]
    embedder = ConstraintsEmbedder()
    embedding = embedder(constraints)
    print(embedding)
    print(embedder.get_space().contains(embedding))
    print(embedder.get_space())
