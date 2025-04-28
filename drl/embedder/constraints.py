from typing import List
from gymnasium.spaces import Sequence, Dict, Box, Discrete

from core.config.config import MAX_CONSTRAINTS
from core.constraints import AbstractConstraint, FuelConstraint, ByteCodeSizeConstraint


class ConstraintsEmbedder:

    def get_space(self):
        return Sequence(Dict({"index":Discrete(MAX_CONSTRAINTS),
                              "type": Discrete(MAX_CONSTRAINTS),
                              "min": Box(low=0, high=1000000, shape=()),
                              "max": Box(low=0, high=1000000, shape=()),
                              "value": Box(low=0, high=100000, shape=())
                              }))

    def __call__(self, constraints: List[AbstractConstraint]):
        constraints_list = []
        for index, constraint in enumerate(constraints):
            if isinstance(constraint, FuelConstraint):
                constraint_type = 0
            elif isinstance(constraint, ByteCodeSizeConstraint):
                constraint_type = 1
            else:
                raise ValueError("Unknown constraint type")

            constraints_list.append({
                "index": int(index),
                "type": int(constraint_type),
                "min": float(constraint.min_target),
                "max": float(constraint.max_target),
                "value": float(constraint.resource)
            })

        return tuple(constraints_list)




if __name__ == "__main__":
    constraints = [FuelConstraint(50, 100, 100), ByteCodeSizeConstraint(0, 200, 300)]
    embedder = ConstraintsEmbedder()
    embedding = embedder(constraints)
    print(embedder.get_space().contains(embedding))
    print(embedder.get_space())