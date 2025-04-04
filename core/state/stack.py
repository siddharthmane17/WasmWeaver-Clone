from typing import List

from core.config.config import MAX_FUNCTION_CALL_DEPTH, MAX_STACK_SIZE
from core.state.locals import Locals
from core.value import Val


class StackFrame:
    """A simple stack frame representation."""

    def __init__(self, params: List[Val] = None, stack: List[Val] = None, name=None):
        self.locals = Locals()
        self.name = name
        self.stack: List[Val] = []
        if params:
            for param in params:
                self.locals.add(param)

        if stack:
            for val in stack:
                self.stack.append(val)

    def stack_push(self, value: Val):
        if not isinstance(value, Val):
            raise ValueError(f"Value {value} is not of type Val")
        if MAX_STACK_SIZE < len(self.stack):
            raise ValueError("Stack size limit reached")
        self.stack.append(value)

    def can_push_to_stack(self, n: int = 1):
        return MAX_STACK_SIZE >= len(self.stack) + n

    def stack_pop(self):
        return self.stack.pop()

    def stack_peek(self, n=1):
        return self.stack[-n]

    def stack_pop_n_in_order(self, n):
        return [self.stack.pop() for _ in range(n)][::-1]

    def stack_peek_n_in_order(self, n):
        return [self.stack[-i] for i in range(1, n + 1)][::-1]


class Stack:
    """A simple global stack representation."""

    def __init__(self):
        self.stack_frames: List[StackFrame] = []

    def get_current_frame(self):
        """Returns the current stack frame."""
        return self.stack_frames[-1]

    def can_add_new_stack_frame(self):
        """Checks if a new stack frame can be added or if the depth limit has been reached."""
        return len(self.stack_frames) < MAX_FUNCTION_CALL_DEPTH

    def get_last_frame(self):
        """Returns the 'parent' stack frame."""
        return self.stack_frames[-2]

    def push_frame(self, params: List[Val] = None, stack: List[Val] = None, name=None):
        """Pushes a new stack frame to the stack."""
        if not self.can_add_new_stack_frame():
            raise ValueError("Cannot add new stack frame, limit reached.")
        self.stack_frames.append(StackFrame(params, stack, name=name))

    def pop_frame(self):
        """Pops the current stack frame from the stack."""
        return self.stack_frames.pop()
