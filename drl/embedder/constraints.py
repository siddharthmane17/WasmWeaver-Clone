from typing import List
import numpy as np
import torch

from core.constraints import AbstractConstraint, FuelConstraint, ByteCodeSizeConstraint


class ConstraintsEmbedder:

    def embed_constraint(self, constraint: AbstractConstraint):

        a = np.clip((constraint.resource - constraint.min_target) / (constraint.max_target - constraint.min_target),0,1)
        so = max(0, (constraint.resource-constraint.max_target)/(constraint.max_target - constraint.min_target))
        su = max(0, (constraint.min_target-constraint.resource)/(constraint.max_target - constraint.min_target))
        b = constraint.is_violated() or not constraint.is_fulfilled()
        return torch.tensor([a,so, su, b])


    def __call__(self, constraints: List[AbstractConstraint]):
        combined_embedding = torch.zeros(0)
        for constraint in constraints:
            embedding = self.embed_constraint(constraint)
            combined_embedding = torch.cat((combined_embedding, embedding), dim=0)
        #Make sure, that the output is float32
        return combined_embedding.to(torch.float32)


if __name__ == "__main__":
    constraints = [FuelConstraint(50, 100, 100), ByteCodeSizeConstraint(0, 200, 300)]
    embedder = ConstraintsEmbedder()
    embedding = embedder(constraints)
    print(embedding.shape)
    print(embedding)