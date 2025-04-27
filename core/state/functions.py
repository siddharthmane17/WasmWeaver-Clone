import copy
from typing import List, TYPE_CHECKING, Type, Dict

from core.formater import indent_code

if TYPE_CHECKING:
    from core.tile import AbstractTile
    from state import GlobalState

if TYPE_CHECKING:
    pass

from core.value import Val


class Block:
    """A simple block representation"""

    def __init__(self, name: str):
        self.name = name
        self.inputs: List[Type[Val]] = []
        self.outputs: List[Type[Val]] = []
        self.tiles: List["AbstractTile"] = []

    def get_byte_code_size(self):
        """Returns the byte code size of the block"""
        return sum(tile.get_byte_code_size() for tile in self.tiles)

    def get_fuel_cost(self):
        """Returns the fuel cost of the block"""
        return sum(tile.get_fuel_cost() for tile in self.tiles)

    def get_response_time(self):
        """Returns the response time of the block"""
        return sum(tile.get_response_time() for tile in self.tiles)

    def generate_code(self, current_state: "GlobalState", current_function: "Function", current_blocks: List["Block"]) -> str:
        """Generates the code of the block"""
        result_str = (f"block ${self.name}" if self.name not in ["if", "else"] else f"{self.name}")
        # Add inputs
        if self.inputs and self.name not in ["else"]:
            result_str += f" (param {' '.join(input_type.get_wasm_type() for input_type in self.inputs)})"
        # Add outputs
        if self.outputs and self.name not in ["else"]:
            result_str += f" (result {' '.join(output_type.get_wasm_type() for output_type in self.outputs)})"
        result_str += "\n"
        if self.tiles:
            result_str += indent_code(("\n").join(tile.generate_code(current_state, current_function,current_blocks=current_blocks+[self]) for tile in self.tiles))
            result_str += "\n"
        if self.name not in ["if"]:
            result_str += "end\n"
        return result_str


class Function:
    """A simple function representation"""

    def __init__(self, name, inputs: List[Type[Val]], outputs: List[Type[Val]], is_external=False):
        self.name = name
        self.inputs = inputs
        self.outputs = outputs
        self.is_external = is_external
        self.tiles: List["AbstractTile"] = []
        self.local_types: List[Type[Val]] = []
        self.blocks: List[Block] = []
        self.if_else_counter = 0
        self.checkpoints = {}
        self.current_block_depth = 0
        self.selection_strategy = None

    def create_checkpoint(self):
        checkpoint_id = 0
        while checkpoint_id in self.checkpoints:
            checkpoint_id += 1
        self.checkpoints[checkpoint_id] = {
            "name": copy.deepcopy(self.name),
            "inputs": copy.deepcopy(self.inputs),
            "outputs": copy.deepcopy(self.outputs),
            "tiles": copy.deepcopy(self.tiles),
            "local_types": copy.deepcopy(self.local_types),
            "blocks": copy.deepcopy(self.blocks)
        }
        return checkpoint_id

    def restore_checkpoint(self, checkpoint_id, delete=False):
        checkpoint = self.checkpoints[checkpoint_id]
        self.name = copy.deepcopy(checkpoint["name"])
        self.inputs = copy.deepcopy(checkpoint["inputs"])
        self.outputs = copy.deepcopy(checkpoint["outputs"])
        self.tiles = copy.deepcopy(checkpoint["tiles"])
        self.local_types = copy.deepcopy(checkpoint["local_types"])
        self.blocks = copy.deepcopy(checkpoint["blocks"])
        if delete:
            del self.checkpoints[checkpoint_id]

    def delete_checkpoint(self, checkpoint_id):
        del self.checkpoints[checkpoint_id]

    def get_byte_code_size(self):
        """Returns the byte code size of the function"""
        return sum(tile.get_byte_code_size() for tile in self.tiles)

    def get_fuel_cost(self):
        """Returns the fuel cost of the function"""
        return sum(tile.get_fuel_cost() for tile in self.tiles)

    def get_response_time(self):
        """Returns the response time of the function"""
        return sum(tile.get_response_time() for tile in self.tiles)

    def get_sig_name(self):
        return f"sig_{self.name}"

    def generate_signature(self):
        """Generates the signature of the function"""
        result_str = f"(type ${self.get_sig_name()} (func"
        if self.inputs:
            result_str += f" (param {' '.join(input_type.get_wasm_type() for input_type in self.inputs)})"
        if self.outputs:
            result_str += f" (result {' '.join(output_type.get_wasm_type() for output_type in self.outputs)})"
        result_str += "))"
        return result_str

    def generate_code(self, current_state: "GlobalState", current_function: "Function") -> str:
        """Generates the code of the function"""
        if self.is_external:
            result_str = f"    (import \"env\" \"{self.name}\" (func ${self.name} "
            if self.inputs:
                result_str += f"(param {' '.join(input_type.get_wasm_type() for input_type in self.inputs)})"
            if self.outputs:
                result_str += f"(result {' '.join(output_type.get_wasm_type() for output_type in self.outputs)})"
            result_str += "))\n"
            return result_str
        else:
            result_str = f"(func ${self.name} (export \"{self.name}\")"
            if self.inputs:
                result_str += f" (param {' '.join(input_type.get_wasm_type() for input_type in self.inputs)})"
            if self.outputs:
                result_str += f" (result {' '.join(output_type.get_wasm_type() for output_type in self.outputs)})"
            result_str += "\n"
            #Add temp local
            if self.local_types[len(self.inputs):]:
                result_str += indent_code("(local " + " ".join(
                    local_type.get_wasm_type() for local_type in self.local_types[len(self.inputs):]) + ")\n")
            result_str += indent_code("(local $temp i32)\n")
            result_str += indent_code("\n".join(tile.generate_code(current_state, current_function,[]) for tile in self.tiles))
            result_str += "\n"
            result_str += ")\n"
            return result_str


class Functions:
    """A simple function state that stores functions."""

    def __init__(self):
        self.functions: Dict[str, Function] = {}

    def set(self, value: Function):
        self.functions[value.name] = value

    def get(self, name):
        return self.functions[name]

    def __len__(self):
        return len(self.functions)
