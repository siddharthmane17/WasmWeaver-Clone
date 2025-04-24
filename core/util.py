import random
from typing import List, Type

from core.constraints import ConstraintsViolatedError
from core.loader import AbstractTileLoader
from core.state.functions import Function, Block
from core.state.stack import StackFrame
from core.state.state import GlobalState
from core.strategy import AbstractSelectionStrategy
from core.tile import AbstractTile
from core.value import Val


def stack_matches(global_state: GlobalState, expected: List[Type[Val]]):
    """Checks if the last values of the current stack frame match the expected types. This function does not modify the global state."""
    stack = global_state.stack.get_current_frame().stack_peek_n_in_order(len(expected))
    for i in range(len(stack)):
        if not isinstance(stack[i], expected[i]):
            return False
    return True


class Finish(AbstractTile):
    """Used only to stop function and block generation."""
    name = "Finish"

    def __init__(self, seed: int):
        super().__init__(seed)

    @staticmethod
    def can_be_placed(current_state: GlobalState, current_function: Function):
        #Is checked by function generation code
        raise NotImplementedError()

    def apply(self, current_state: GlobalState, current_function: Function):
        return current_state

    def generate_code(self, current_state: GlobalState, current_function: Function) -> str:
        return f""


def generate_function(tile_loader: AbstractTileLoader, name: str, input_types: List[Type[Val]],
                      is_external,
                      global_state: GlobalState, selection_strategy: AbstractSelectionStrategy = None, is_entry=False):
    """Generates a internal/external function with the given name, input types, output types in the current global state."""

    global_checkpoint_count_before = len(global_state.checkpoints)

    if not stack_matches(global_state, input_types):
        raise ValueError(f"Input types do not match stack. Expected {input_types}, got {global_state.stack}")

    if is_external:
        raise NotImplementedError

    #Check whether constraints are already violated
    if global_state.constraints.any_violated():
        raise ValueError(f"Constraints are already violated at the beginning of function generation")

    f = Function(name, inputs=input_types, outputs=[], is_external=is_external)
    f.selection_strategy = selection_strategy
    #Add function stack frame
    global_state.stack.push_frame(global_state.stack.get_current_frame().stack_pop_n_in_order(len(input_types)),
                                  name=name)
    # Add params to function local types
    for i, inp in enumerate(input_types):
        f.local_types.append(inp)

    before_generation_checkpoint = global_state.create_checkpoint()

    while True:

        #Get all placeable tiles
        placeable_tiles = tile_loader.get_placeable_tiles(global_state, current_function=f)

        #Check if function can be finished
        if (is_entry and global_state.constraints.is_finished()) or not is_entry:
            placeable_tiles.append(Finish)

        #If no placeable tiles, reset to checkpoint
        if not placeable_tiles:
            global_state.restore_checkpoint(before_generation_checkpoint)
            f = Function(name, inputs=input_types, outputs=[], is_external=is_external)
            f.selection_strategy = selection_strategy
            for i, inp in enumerate(input_types):
                f.local_types.append(inp)
            continue

        #Select tile with max weight
        tile = max(placeable_tiles, key=lambda x: x.get_weight(global_state, f, selection_strategy))(random.randint(0, 2 ** 32 - 1))

        #Apply tile to global state
        tile.apply(global_state, f)
        tile.apply_constraints(global_state, f)

        #Add tile to selected tiles
        if tile.name != "Finish":
            f.tiles.append(tile)

        #Check if constraints are violated
        if global_state.constraints.any_violated():
            raise ConstraintsViolatedError()

        #Check if constraints are finished
        if tile.name == "Finish":
            if is_entry:
                #Drop all values from stack if function is entry
                drop_tile_type = tile_loader.get_tile_type_by_name("Drop")
                for _ in range(len(global_state.stack.get_current_frame().stack)):
                    drop_tile = drop_tile_type(random.randint(0, 2 ** 32 - 1))
                    drop_tile.apply(global_state, f)
                    f.tiles.append(drop_tile)

            else:
                #Add all values from stack to output if function is not entry
                for val in global_state.stack.get_current_frame().stack_pop_n_in_order(
                        len(global_state.stack.get_current_frame().stack)):
                    global_state.stack.get_last_frame().stack_push(val)
                    f.outputs.append(type(val))
            #Check if stack is empty
            if len(global_state.stack.get_current_frame().stack) != 0:
                raise ValueError(
                    f"Local abstract stack frame is not empty at the end of function generation. Stack: {global_state.stack.get_current_frame().stack}")
            #Remove function stack frame
            global_state.stack.pop_frame()
            #Add function to global state
            global_state.functions.set(f)
            break

    #Check if function in global state
    if f not in global_state.functions.functions.values():
        raise ValueError(f"Function {f} not in global state")
    global_state.delete_checkpoint(before_generation_checkpoint)

    #Check if checkpoint count is the same as at the beginning
    if len(global_state.checkpoints) != global_checkpoint_count_before:
        raise ValueError(
            f"Checkpoint count changed during function generation. Expected {global_checkpoint_count_before}, got {len(global_state.checkpoints)}")


def compare_stack_frame_for_type_equality(current_stack_frame: StackFrame, expected_var_types: List[Type[Val]]):
    """Compares two stack frames for type equality of their tiles. Returns True if the stack frames are equal, False otherwise."""
    if expected_var_types is None:
        return True
    if len(current_stack_frame.stack) != len(expected_var_types):
        return False
    for stack_val_0, expected in zip(current_stack_frame.stack, expected_var_types):
        if type(stack_val_0) != expected:
            return False
    return True


def generate_block(tile_loader: AbstractTileLoader, global_state: GlobalState, current_function: Function,
                   input_types: List[Type[Val]], name: str,
                   depth=0, fixed_output_types: None | List[Type[Val]] = None):
    """Generates a block in the current function with the given name"""

    if not stack_matches(global_state, input_types):
        raise ValueError(f"Input types do not match stack. Expected {input_types}, got {global_state.stack}")

    #Check whether constraints are already violated
    if global_state.constraints.any_violated():
        raise ValueError(f"Constraints are already violated at the beginning of block generation")


    block = Block(name, depth=depth)
    block.inputs = input_types
    current_function.blocks.append(block)
    #Stack frame with shared locals with previous function
    global_state.stack.push_frame(stack=global_state.stack.get_current_frame().stack_pop_n_in_order(len(input_types)),
                                  name=name)
    global_state.stack.get_current_frame().locals = global_state.stack.get_last_frame().locals
    before_generation_checkpoint = global_state.create_checkpoint()
    before_generation_function_checkpoint = current_function.create_checkpoint()
    while True:
        placeable_tiles = tile_loader.get_placeable_tiles(global_state, current_function)
        #Check if can stop
        if compare_stack_frame_for_type_equality(global_state.stack.get_current_frame(), fixed_output_types):
            placeable_tiles.append(Finish)
        if not placeable_tiles:
            global_state.restore_checkpoint(before_generation_checkpoint)
            current_function.restore_checkpoint(before_generation_function_checkpoint)
            block = current_function.blocks[-1]
            raise ConstraintsViolatedError()
            #continue
        tile = max(placeable_tiles, key=lambda x: x.get_weight(global_state, current_function, current_function.selection_strategy))(random.randint(0, 2 ** 32 - 1))
        tile.apply(global_state, current_function)
        tile.apply_constraints(global_state, current_function)
        if tile.name != "Finish":
            block.tiles.append(tile)


        if global_state.constraints.any_violated():
            raise ConstraintsViolatedError()

        if tile.name == "Finish":
            global_state.delete_checkpoint(before_generation_checkpoint)
            current_function.delete_checkpoint(before_generation_function_checkpoint)
            #Add all values from stack to output
            for val in global_state.stack.get_current_frame().stack_pop_n_in_order(
                    len(global_state.stack.get_current_frame().stack)):
                global_state.stack.get_last_frame().stack_push(val)
                block.outputs.append(type(val))
            global_state.stack.pop_frame()
            break

    return block
