from typing import List

from core.value import Val


class Locals:
    """A simple local state that stores local variables."""

    def __init__(self):
        self.locals: List[Val] = []

    def add(self, local: Val):
        """Adds a local variable to the state and returns the index."""
        if not isinstance(local, Val):
            raise ValueError("Local is not of type Val")
        self.locals.append(local)
        return len(self.locals) - 1

    def __setitem__(self, key, value):
        if not isinstance(value, Val):
            raise ValueError("Local is not of type Val")
        self.locals[key] = value

    def __getitem__(self, item):
        return self.locals[item]

    def __len__(self):
        return len(self.locals)
