import random
from typing import Generator, Any

from config.config import MEMORY_MAX_WRITE_INDEX
from core.constraints import ByteCodeSizeConstraint, FuelConstraint, ConstraintsViolatedError
from core.converter import global_state_to_wasm_program
from core.debug.debugger import generate_trace
from core.loader import TileLoader
from core.runner import run_global_state, wat_code_to_wasm, AbstractRunResult
from core.state.stack import StackOverflowError, StackValueError
from core.state.state import GlobalState
from core.strategy import AbstractSelectionStrategy, RandomSelectionStrategy
from core.util import generate_function

random.seed(0)
tile_loader = TileLoader("core/instructions/")


class GeneratorResult:
    def __init__(self, seed: int, code_str: str, byte_code: bytearray, run_result: AbstractRunResult,
                 initial_memory: bytearray, canary_output=None):
        if canary_output is None:
            canary_output = []
        self.seed = seed
        self.code_str = code_str
        self.byte_code = byte_code
        self.initial_memory = initial_memory
        self.abstract_run_result = run_result
        self.canary_output = canary_output

    def __dict__(self):
        return {
            "seed": self.seed,
            "code_str": self.code_str,
            "byte_code": self.byte_code,
            "initial_memory": self.initial_memory,
            "abstract_run_result": self.abstract_run_result.__dict__,
            "canary_output": self.canary_output
        }

def add_line_numbers_to_code(code_str: str) -> str:
    """Adds line numbers to the code string"""
    lines = code_str.split("\n")
    for i in range(len(lines)):
        lines[i] = f"{i + 1}: {lines[i]}"
    return "\n".join(lines)

def generate_code(start_seed: int, min_byte_code_size: int = 20, max_byte_code_size: int = 512, min_fuel: int = 0,
                  max_fuel: int = 2000, verbose: bool = False, selection_strategy: AbstractSelectionStrategy = None) -> \
Generator[GeneratorResult, Any, None]:
    while True:
        start_seed = start_seed + 1
        if verbose:
            print(f"Seed: {start_seed}")
        try:
            random.seed(start_seed)
            global_state = GlobalState()
            global_state.constraints.add(ByteCodeSizeConstraint(min_byte_code_size, max_byte_code_size))
            global_state.constraints.add(FuelConstraint(min_fuel, max_fuel))
            global_state.stack.push_frame(params=None, stack=[], name="origin")
            generate_function(tile_loader, "run", [], False, global_state,
                              is_entry=True, selection_strategy=selection_strategy,fixed_output_types=[])
            code_str = global_state_to_wasm_program(global_state)
            if verbose:
                print(code_str)
                print(add_line_numbers_to_code(code_str))
            try:
                result = run_global_state(global_state)
            except Exception as e:
                print(f"Error: {e}")
                #Reset memory
                global_state.memory.memory = bytearray(global_state.memory.initial_values[:MEMORY_MAX_WRITE_INDEX])
                generate_trace(global_state, start_function="run", start_seed=start_seed)
                raise e
            byte_code = wat_code_to_wasm(code_str)
            if verbose:
                print(f"Fuel consumption: {result}")
                print(f"Byte code size: {len(byte_code)}")



            canary_output = global_state.canary_output

            yield GeneratorResult(start_seed, code_str, byte_code, result,
                                  global_state.memory.initial_values[:MEMORY_MAX_WRITE_INDEX], canary_output)
        except (ConstraintsViolatedError, StackOverflowError, StackValueError):
            if verbose:
                print("Constraints violated")
            continue

