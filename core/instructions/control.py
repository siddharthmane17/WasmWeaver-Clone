import copy
import random
import uuid
from typing import Type

from core.config.config import MAX_FUNCTIONS_PER_MODULE, MAX_BLOCKS_PER_FUNCTION
from core.constraints import FuelConstraint, ByteCodeSizeConstraint
from core.state.functions import Function
from core.state.state import GlobalState
from core.tile import AbstractTileFactory, AbstractTile
from core.util import stack_matches, generate_function, generate_block
from core.value import I32, get_random_val, RefFunc


def generate_random_function_name(global_state: GlobalState) -> str:
    """Generates a random function name that is not already used in the global state"""
    while True:
        name = f"function_{random.randint(0, 2 ** 32 - 1)}"
        for function in global_state.functions.functions.values():
            if function.name == name:
                continue
        return name


def generate_random_block_name(function: Function) -> str:
    """Generates a random block name that is not already used in the function"""
    while True:
        name = f"block_{random.randint(0, 2 ** 32 - 1)}"
        for block in function.blocks:
            if block.name == name:
                continue
        return name


def generate_random_loop_name(function: Function) -> str:
    """Generates a random block name that is not already used in the function"""
    while True:
        name = f"loop_{random.randint(0, 2 ** 32 - 1)}"
        for block in function.blocks:
            if block.name == name:
                continue
        return name


class AbstractFunctionTileFactory(AbstractTileFactory):

    def __init__(self, seed: int, tile_loader):
        super().__init__(seed, tile_loader)

    def create_function_create_tile(self, global_state: GlobalState, is_external=False) -> Type[AbstractTile]:
        name = generate_random_function_name(global_state)
        tile_loader = self.tile_loader
        tile = type(f"CreateFunctionTile", (AbstractTile,), {})
        tile.name = f"Create and call function ({name})"
        function = None

        def can_be_placed(current_state: GlobalState, current_function: Function):
            nonlocal function
            if not current_state.stack.can_add_new_stack_frame():
                return False
            if function is None:
                return len(
                    current_state.functions) < MAX_FUNCTIONS_PER_MODULE and current_state.constraints.remaining_resources(
                    FuelConstraint) >= 10 and current_state.constraints.remaining_resources(ByteCodeSizeConstraint) >= 10

            # Same from here as in call tile function
            return self.create_function_call_tile(function).can_be_placed(current_state, current_function)

        def apply(self, current_state: GlobalState, current_function: Function):
            nonlocal function
            if function is None:
                generate_function(tile_loader,
                                  name,
                                  [type(stack_var) for stack_var in
                                   current_state.stack.get_current_frame().stack_peek_n_in_order(
                                       random.randint(0, len(current_state.stack.get_current_frame().stack)))],
                                  is_external,
                                  current_state,
                                  is_entry=False,selection_strategy=current_function.selection_strategy)

                function = current_state.functions.functions[name]
                self.get_byte_code_size = function.get_byte_code_size
                self.get_fuel_cost = function.get_fuel_cost
                self.get_response_time = function.get_response_time
            else:
                current_state.stack.push_frame(
                    params=current_state.stack.get_current_frame().stack_pop_n_in_order(len(function.inputs)),
                    name=name)
                # Check for each tile if it can be placed and apply it
                for tile in function.tiles:
                    tile.apply(current_state, function)
                    # Check if the constraints are still satisfied
                    tile.apply_constraints(current_state, function)
                    # Check if constraints are still satisfied

                # Write back values on local stack frame to parent stack frame
                for stack_val in current_state.stack.get_current_frame().stack_pop_n_in_order(
                        len(function.outputs)):
                    current_state.stack.get_last_frame().stack_push(stack_val)

                # Remove the stack frame
                current_state.stack.pop_frame()

        def generate_code(self, current_state: GlobalState, current_function: Function) -> str:
            return f"call ${name}"

        tile.apply = apply
        tile.can_be_placed = staticmethod(can_be_placed)
        tile.generate_code = generate_code
        return tile

    def create_function_call_tile(self, function: Function) -> Type[AbstractTile]:
        """Returns a tile that represents the function"""
        tile = type(f"FunctionCallTile", (AbstractTile,), {})
        tile.name = f"Call function ({function.name})"

        def function_can_be_placed(current_state: GlobalState, current_function: Function):
            """Returns if the function can be placed in the current state"""
            # Check if new stack frame can be added
            if not current_state.stack.can_add_new_stack_frame():
                return False
            backup_state = current_state.create_checkpoint()
            # Check if stack has correct inputs
            if len(current_state.stack.get_current_frame().stack) < len(function.inputs):
                current_state.restore_checkpoint(backup_state, delete=True)
                return False
            for inp, stack_val in zip(function.inputs, current_state.stack.get_current_frame().stack_peek_n_in_order(
                    len(function.inputs))):
                if not isinstance(stack_val, inp):
                    current_state.restore_checkpoint(backup_state, delete=True)
                    return False

            #Add stack frame
            current_state.stack.push_frame(
                params=current_state.stack.get_current_frame().stack_pop_n_in_order(len(function.inputs)),
                name=function.name)
            # Check for each tile if it can be placed and apply it
            for tile in function.tiles:
                if not type(tile).can_be_placed(current_state, function):
                    current_state.restore_checkpoint(backup_state, delete=True)
                    return False
                tile.apply(current_state, function)
                # Check if the constraints are still satisfied
                tile.apply_constraints(current_state, function)
                # Check if constraints are still satisfied
                if current_state.constraints.any_violated():
                    current_state.restore_checkpoint(backup_state, delete=True)
                    return False

            # Check if the stack matches the expected output
            if not stack_matches(current_state, function.outputs):
                current_state.restore_checkpoint(backup_state, delete=True)
                return False
            current_state.restore_checkpoint(backup_state, delete=True)
            return True

        def apply(self, current_state: GlobalState, current_function: Function):
            # Add stack frame
            current_state.stack.push_frame(
                params=current_state.stack.get_current_frame().stack_pop_n_in_order(len(function.inputs)),
                name=function.name)
            # Check for each tile if it can be placed and apply it
            for tile in function.tiles:
                tile.apply(current_state, function)
                # Check if the constraints are still satisfied
                tile.apply_constraints(current_state, function)
                # Check if constraints are still satisfied

            #Write back values on local stack frame to parent stack frame
            for stack_val in current_state.stack.get_current_frame().stack_pop_n_in_order(len(function.outputs)):
                current_state.stack.get_last_frame().stack_push(stack_val)

            #Remove the stack frame
            current_state.stack.pop_frame()

        def generate_code(self, current_state: GlobalState, current_function: Function) -> str:
            return f"call ${function.name}"

        tile.apply = apply
        tile.can_be_placed = staticmethod(function_can_be_placed)
        tile.get_byte_code_size = function.get_byte_code_size
        tile.get_fuel_cost = function.get_fuel_cost
        tile.get_response_time = function.get_response_time
        tile.generate_code = generate_code
        return tile


    def create_function_indirect_call_tile(self, function: Function, table_name: str, elem_index: int) -> Type[AbstractTile]:
        """Returns a tile that represents the function"""
        tile = type(f"FunctionIndirectCallTile", (AbstractTile,), {})
        tile.name = f"Indirect call function ({function.name})"

        def function_can_be_placed(current_state: GlobalState, current_function: Function):
            """Returns if the function can be placed in the current state"""
            # Check if new stack frame can be added
            if not current_state.stack.can_add_new_stack_frame():
                return False
            backup_state = current_state.create_checkpoint()
            # Check if stack has correct inputs
            if len(current_state.stack.get_current_frame().stack) < len(function.inputs):
                current_state.restore_checkpoint(backup_state, delete=True)
                return False
            for inp, stack_val in zip(function.inputs, current_state.stack.get_current_frame().stack_peek_n_in_order(
                    len(function.inputs))):
                if not isinstance(stack_val, inp):
                    current_state.restore_checkpoint(backup_state, delete=True)
                    return False

            #Add stack frame
            current_state.stack.push_frame(
                params=current_state.stack.get_current_frame().stack_pop_n_in_order(len(function.inputs)),
                name=function.name)
            # Check for each tile if it can be placed and apply it
            for tile in function.tiles:
                if not type(tile).can_be_placed(current_state, function):
                    current_state.restore_checkpoint(backup_state, delete=True)
                    return False
                tile.apply(current_state, function)
                # Check if the constraints are still satisfied
                tile.apply_constraints(current_state, function)
                # Check if constraints are still satisfied
                if current_state.constraints.any_violated():
                    current_state.restore_checkpoint(backup_state, delete=True)
                    return False

            # Check if the stack matches the expected output
            if not stack_matches(current_state, function.outputs):
                current_state.restore_checkpoint(backup_state, delete=True)
                return False
            current_state.restore_checkpoint(backup_state, delete=True)
            return True

        def apply(self, current_state: GlobalState, current_function: Function):
            # Add stack frame
            current_state.stack.push_frame(
                params=current_state.stack.get_current_frame().stack_pop_n_in_order(len(function.inputs)),
                name=function.name)
            # Check for each tile if it can be placed and apply it
            for tile in function.tiles:
                tile.apply(current_state, function)
                # Check if the constraints are still satisfied
                tile.apply_constraints(current_state, function)
                # Check if constraints are still satisfied

            #Write back values on local stack frame to parent stack frame
            for stack_val in current_state.stack.get_current_frame().stack_pop_n_in_order(len(function.outputs)):
                current_state.stack.get_last_frame().stack_push(stack_val)

            #Remove the stack frame
            current_state.stack.pop_frame()

        def generate_code(self, current_state: GlobalState, current_function: Function) -> str:
            return f"indirect_call ${table_name} ${elem_index} ${function.get_sig_name()}"

        tile.apply = apply
        tile.can_be_placed = staticmethod(function_can_be_placed)
        tile.get_byte_code_size = function.get_byte_code_size
        tile.get_fuel_cost = function.get_fuel_cost
        tile.get_response_time = function.get_response_time
        tile.generate_code = generate_code
        return tile

    def create_function_ref_to_stack_tile(self, function: Function) -> Type[AbstractTile]:
        """Returns a tile that represents the function"""
        tile = type(f"FunctionRefToStackTile", (AbstractTile,), {})
        tile.name = f"Push reference function ({function.name}) to stack"

        def function_can_be_placed(current_state: GlobalState, current_function: Function):
            """Returns if the function can be placed in the current state"""

            return current_state.stack.get_current_frame().can_push_to_stack()

        def apply(self, current_state: GlobalState, current_function: Function):
            current_state.stack.get_current_frame().stack_push(RefFunc(function))

        def generate_code(self, current_state: GlobalState, current_function: Function) -> str:
            return f"ref.func ${function.name}"


        tile.apply = apply
        tile.can_be_placed = staticmethod(function_can_be_placed)
        tile.generate_code = generate_code
        return tile

    def generate_all_placeable_tiles(self, current_state: GlobalState, current_function: Function):
        """Generates all function tiles that can be placed in the current state"""
        function_tiles = []
        start_state = copy.deepcopy(current_state)
        #Generate the tile
        for function in current_state.functions.functions.values():

            if not self.create_function_call_tile(function).can_be_placed(current_state, current_function):
                continue
            function_tiles.append(self.create_function_call_tile(function))
        #Function reference to stack
        for function in current_state.functions.functions.values():
            if not self.create_function_ref_to_stack_tile(function).can_be_placed(current_state, current_function):
                continue
            function_tiles.append(self.create_function_ref_to_stack_tile(function))
        #Function indirect call
        for table_name, table in current_state.tables.tables.items():
            for elem_index, elem in enumerate(table.elements):
                if not isinstance(elem, RefFunc) or elem.value is None:
                    continue
                function = elem.value
                if not self.create_function_indirect_call_tile(function, table_name, elem_index).can_be_placed(
                        current_state, current_function):
                    continue
                function_tiles.append(
                    self.create_function_indirect_call_tile(function, table_name, elem_index ))
        #Generate the create function tile
        create_function_tile = self.create_function_create_tile(current_state)
        if create_function_tile.can_be_placed(current_state, current_function):
            function_tiles.append(create_function_tile)
        #Restore the state
        return function_tiles


class AbstractBlockTileFactory(AbstractTileFactory):

    def __init__(self, seed: int, tile_loader):
        super().__init__(seed, tile_loader)

    def generate_all_placeable_tiles(self, global_state: GlobalState, current_function: Function):
        """Generates all possible tiles"""
        block_tiles = []
        block_tile = self.create_block_tile(global_state)
        if block_tile.can_be_placed(global_state, current_function):
            block_tiles.append(block_tile)
        return block_tiles

    def create_block_tile(self, global_state: GlobalState) -> Type[AbstractTile]:

        tile = type(f"BlockTile", (AbstractTile,), {"block": None})
        tile.name = f"A simple block"
        tile_loader = self.tile_loader

        def can_be_placed(current_state: GlobalState, current_function: Function):
            nonlocal tile
            if not current_state.stack.can_add_new_stack_frame():
                return False
            nonlocal tile
            if tile.block is None:
                if current_state.constraints.remaining_resources(
                        FuelConstraint) >= 10 and current_state.constraints.remaining_resources(
                    ByteCodeSizeConstraint) >= 10:
                    return True
                if tile.block is None and MAX_BLOCKS_PER_FUNCTION <= len(current_function.blocks):
                    return False
                return False
            #Check if block can be applied
            backup_state = current_state.create_checkpoint()
            function_backup = current_function.create_checkpoint()
            for block_tile in tile.block.tiles:
                if not type(block_tile).can_be_placed(current_state, current_function):
                    current_state.restore_checkpoint(backup_state, delete=True)
                    current_function.restore_checkpoint(function_backup, delete=True)
                    return False
                block_tile.apply(current_state, current_function)
                block_tile.apply_constraints(current_state, current_function)
                if current_state.constraints.any_violated():
                    current_state.restore_checkpoint(backup_state, delete=True)
                    current_function.restore_checkpoint(function_backup, delete=True)
                    return False

            current_state.restore_checkpoint(backup_state, delete=True)
            current_function.restore_checkpoint(function_backup, delete=True)
            return True

        def apply(self, current_state: GlobalState, current_function: Function):
            nonlocal tile, tile_loader
            if tile.block is None:
                tile.name = generate_random_block_name(current_function)
                tile.block = generate_block(tile_loader, global_state, current_function,
                                            [type(stack_var) for stack_var in
                                             current_state.stack.get_current_frame().stack_peek_n_in_order(
                                                 min(2, random.randint(0,
                                                                       len(current_state.stack.get_current_frame().stack))))],
                                            tile.name, depth=0)
                tile.generate_code = tile.block.generate_code
                tile.get_byte_code_size = tile.block.get_byte_code_size
                tile.get_fuel_cost = tile.block.get_fuel_cost
                tile.get_response_time = tile.block.get_response_time
            else:
                for block_tile in tile.block.tiles:
                    block_tile.apply(current_state, current_function)
                    block_tile.apply_constraints(current_state, current_function)

        tile.apply = apply
        tile.can_be_placed = staticmethod(can_be_placed)
        return tile


class ConditionTileFactory(AbstractTileFactory):

    def __init__(self, seed: int, tile_loader):
        super().__init__(seed, tile_loader)

    def generate_all_placeable_tiles(self, global_state: GlobalState, current_function: Function):
        """Generates all possible tiles"""
        condition_tiles = []
        condition_tile = self.create_block_tile(global_state)
        if condition_tile.can_be_placed(global_state, current_function):
            condition_tiles.append(condition_tile)
        return condition_tiles

    def create_block_tile(self, global_state: GlobalState) -> Type[AbstractTile]:

        tile = type(f"ConditionTile", (AbstractTile,), {"if_block": None, "else_block": None})
        tile.name = f"A simple condition block"
        tile_loader = self.tile_loader

        def can_be_placed(current_state: GlobalState, current_function: Function):
            nonlocal tile
            if not current_state.stack.can_add_new_stack_frame():
                return False
                # Check if stack is larger then 0 and if the top value is an i32
            if len(current_state.stack.get_current_frame().stack) < 1:
                return False

            if not isinstance(current_state.stack.get_current_frame().stack_peek(1), I32):
                return False
            #If and else blocks are generated at the same time, so it is ok to only check for the presence of the if_block
            if tile.if_block is None and MAX_BLOCKS_PER_FUNCTION <= len(current_function.blocks):
                return False
            if tile.if_block is None and current_state.constraints.remaining_resources(
                    FuelConstraint) >= 10 and current_state.constraints.remaining_resources(ByteCodeSizeConstraint) >= 10:
                return True

            if tile.if_block is None:
                return False

            #Check if block can be applied
            backup_state = current_state.create_checkpoint()
            function_backup = current_function.create_checkpoint()

            #Pop the value from the stack
            block = tile.if_block if current_state.stack.get_current_frame().stack_pop().value != 0 else tile.else_block

            for block_tile in block.tiles:
                if not type(block_tile).can_be_placed(current_state, current_function):
                    current_state.restore_checkpoint(backup_state, delete=True)
                    current_function.restore_checkpoint(function_backup, delete=True)
                    return False
                block_tile.apply(current_state, current_function)
                block_tile.apply_constraints(current_state, current_function)
                if current_state.constraints.any_violated():
                    current_state.restore_checkpoint(backup_state, delete=True)
                    current_function.restore_checkpoint(function_backup, delete=True)
                    return False

            current_state.restore_checkpoint(backup_state, delete=True)
            current_function.restore_checkpoint(function_backup, delete=True)
            return True

        def apply(self, current_state: GlobalState, current_function: Function):
            nonlocal tile, tile_loader
            should_execute_if = current_state.stack.get_current_frame().stack_pop().value != 0

            if tile.if_block is None:
                #Generate both blocks at the same time
                n_inputs = random.randint(0, min(2, len(current_state.stack.get_current_frame().stack)))
                input_types = [type(stack_var) for stack_var in
                               current_state.stack.get_current_frame().stack_peek_n_in_order(n_inputs)]
                forced_output_types = [type(get_random_val()) for _ in range(0, len(input_types))]  # Change output type of if block
                old_globals = {}
                for global_var in current_state.globals.globals:
                    old_globals[global_var.name] = copy.deepcopy(global_var)
                old_tables = copy.deepcopy(current_state.tables.tables)
                old_functions = copy.deepcopy(current_state.functions).functions
                new_tables = {}
                new_globals = {}
                new_functions = {}

                #Backup the state
                current_state_backup = current_state.create_checkpoint()
                current_function_backup = current_function.create_checkpoint()
                #print("Else block start")
                #print("Globals before else", [f"{var.name} {var.value}" for var in current_state.globals.globals])
                #Generate else block first
                tile.else_block = generate_block(tile_loader, global_state, current_function,
                                                 input_types,
                                                 "else", depth=0, fixed_output_types=forced_output_types)

                for global_var in current_state.globals.globals:
                    if global_var.name not in old_globals:
                        new_globals[global_var.name] = copy.deepcopy(global_var)
                #Reset value of new globals
                for global_var in new_globals.values():
                    global_var.value = global_var.init_value

                for f_name, function in current_state.functions.functions.items():
                    if f_name not in old_functions:
                        new_functions[f_name] = function

                for table_name, table in current_state.tables.tables.items():
                    if table_name not in old_tables:
                        new_tables[table_name] = copy.deepcopy(table)
                        #Wipe all tables
                        new_tables[table_name].wipe()

                required_local_types_else = copy.deepcopy(current_function.local_types)

                current_state.restore_checkpoint(current_state_backup)
                current_function.restore_checkpoint(current_function_backup)

                for global_var in new_globals.values():
                    current_state.globals.add(copy.deepcopy(global_var))

                for table in new_tables.values():
                    current_state.tables.set(copy.deepcopy(table))

                current_function.local_types = copy.deepcopy(required_local_types_else)
                #print(current_state.stack.get_current_frame().locals.locals)
                #print("abb", current_function.local_types)

                #Generate if block
                #print("If block start")
                tile.if_block = generate_block(tile_loader, global_state, current_function,
                                               input_types,
                                               "if", depth=0, fixed_output_types=forced_output_types)

                for global_var in current_state.globals.globals:
                    if global_var.name not in old_globals:
                        new_globals[global_var.name] = copy.deepcopy(global_var)

                for f_name, function in current_state.functions.functions.items():
                    if f_name not in old_functions:
                        new_functions[f_name] = function

                for table_name, table in current_state.tables.tables.items():
                    if table_name not in old_tables:
                        new_tables[table_name] = copy.deepcopy(table)
                        #Wipe all tables
                        new_tables[table_name].wipe()

                local_types_if = copy.deepcopy(current_function.local_types)

                #Reset all globals to their initial state
                for global_var in new_globals.values():
                    global_var.value = global_var.init_value

                #Restore the state
                current_state.restore_checkpoint(current_state_backup, delete=True)
                current_function.restore_checkpoint(current_function_backup, delete=True)

                #Set local types to function depending on the recorded locals
                current_function.local_types = []
                for i in range(max(len(required_local_types_else), len(local_types_if))):
                    if i < len(required_local_types_else):
                        current_function.local_types.append(required_local_types_else[i])

                    elif i < len(local_types_if):
                        current_function.local_types.append(local_types_if[i])

                    if i < len(local_types_if) and i < len(required_local_types_else):
                        if local_types_if[i] != required_local_types_else[i]:
                            print(local_types_if)
                            print(required_local_types_else)
                            raise ValueError("Local types do not match")

                #print(current_function.local_types)

                #Add the new globals and functions to the state
                for global_var in new_globals.values():
                    current_state.globals.add(global_var)

                for table in new_tables.values():
                    current_state.tables.set(table)

                for f_name, function in new_functions.items():
                    current_state.functions.set(function)

                tile.get_byte_code_size = lambda \
                        x: tile.if_block.get_byte_code_size() + tile.else_block.get_byte_code_size()

            block = tile.if_block if should_execute_if else tile.else_block
            #print("Globals before block", [f"{var.name} {var.value}" for var in current_state.globals.globals])
            for block_tile in block.tiles:
                #print(f"Globals before block {block_tile.name}", [f"{var.name} {var.value}" for var in current_state.globals.globals])
                block_tile.apply(current_state, current_function)
                block_tile.apply_constraints(current_state, current_function)

            tile.get_fuel_cost = lambda \
                    s: tile.if_block.get_fuel_cost() if should_execute_if else tile.else_block.get_fuel_cost()
            tile.get_response_time = lambda \
                    s: tile.if_block.get_response_time() if should_execute_if else tile.else_block.get_response_time()
            tile.generate_code = lambda se, st, f: tile.if_block.generate_code(st, f) + tile.else_block.generate_code(
                st, f)

        tile.apply = apply
        tile.can_be_placed = staticmethod(can_be_placed)
        return tile


class LoopTileFactory(AbstractTileFactory):

    def __init__(self, seed: int, tile_loader):
        super().__init__(seed, tile_loader)

    def generate_all_placeable_tiles(self, global_state: GlobalState, current_function: Function):
        """Generates all possible tiles"""
        condition_tiles = []
        condition_tile = self.create_loop_tile(global_state)
        if condition_tile.can_be_placed(global_state, current_function):
            condition_tiles.append(condition_tile)
        return condition_tiles

    def create_loop_tile(self, global_state: GlobalState) -> Type[AbstractTile]:

        tile = type(f"LoopTile", (AbstractTile,), {"inner_block": None})
        tile.name = f"A simple loop block"
        tile.loop_name = None
        tile_loader = self.tile_loader
        tile.local_name = uuid.uuid4().hex

        def can_be_placed(current_state: GlobalState, current_function: Function):
            nonlocal tile
            if not current_state.stack.can_add_new_stack_frame():
                return False
                # Check if stack is larger then 0 and if the top value is an i32
            if len(current_state.stack.get_current_frame().stack) < 1:
                return False

            if not isinstance(current_state.stack.get_current_frame().stack_peek(1), I32):
                return False
            #Get repetition count i32
            rep_count = current_state.stack.get_current_frame().stack_peek(1).value
            if rep_count <= 0 or rep_count > 10000:
                return False
            #If and else blocks are generated at the same time, so it is ok to only check for the presence of the if_block
            if tile.inner_block is None and MAX_BLOCKS_PER_FUNCTION <= len(current_function.blocks):
                return False
            if tile.inner_block is None and current_state.constraints.remaining_resources(
                    FuelConstraint) >= rep_count * 10 and current_state.constraints.remaining_resources(
                ByteCodeSizeConstraint) >= 10:
                return True

            if tile.inner_block is None:
                return False

            #Check if block can be applied
            backup_state = current_state.create_checkpoint()
            function_backup = current_function.create_checkpoint()
            for i in range(0, rep_count):
                for block_tile in tile.inner_block.tiles:
                    if not type(block_tile).can_be_placed(current_state, current_function):
                        current_state.restore_checkpoint(backup_state, delete=True)
                        current_function.restore_checkpoint(function_backup, delete=True)
                        return False
                    block_tile.apply(current_state, current_function)
                    block_tile.apply_constraints(current_state, current_function, ignore_byte_code_size=True)
                    if current_state.constraints.any_violated():
                        current_state.restore_checkpoint(backup_state, delete=True)
                        current_function.restore_checkpoint(function_backup, delete=True)
                        return False

            current_state.restore_checkpoint(backup_state, delete=True)
            current_function.restore_checkpoint(function_backup, delete=True)
            return True

        def apply(self, current_state: GlobalState, current_function: Function):
            nonlocal tile, tile_loader
            repetition_count = current_state.stack.get_current_frame().stack_pop().value

            if tile.inner_block is None:
                #Generate both blocks at the same time
                name = generate_random_loop_name(current_function)
                tile.loop_name = name
                #Backup the state
                current_state_backup = current_state.create_checkpoint()
                current_function_backup = current_function.create_checkpoint()
                #print("Else block start")
                #print("Globals before else", [f"{var.name} {var.value}" for var in current_state.globals.globals])
                #Generate else block first
                while True:
                    current_state.restore_checkpoint(current_state_backup)
                    current_function.restore_checkpoint(current_function_backup)

                    #Modify fuel constraint
                    current_state.constraints[FuelConstraint].resource = current_state.constraints[
                                                                             FuelConstraint].resource // repetition_count

                    tile.inner_block = generate_block(tile_loader, global_state, current_function,
                                                      [],
                                                      "loop $" + name, depth=0, fixed_output_types=[])

                    #Restore the fuel constraint
                    current_state.constraints[FuelConstraint].resource = current_state.constraints[
                                                                             FuelConstraint].resource * repetition_count

                    #Now check if the block can be applied n times
                    can_be_executed = True
                    for i in range(0, repetition_count):
                        for block_tile in tile.inner_block.tiles:
                            if not type(block_tile).can_be_placed(current_state, current_function):
                                can_be_executed = False
                                break
                            block_tile.apply(current_state, current_function)
                            block_tile.apply_constraints(current_state, current_function, ignore_byte_code_size=True)
                            if current_state.constraints.any_violated():
                                can_be_executed = False
                                break

                    if can_be_executed:
                        break

                current_state.delete_checkpoint(current_state_backup)
                current_function.delete_checkpoint(current_function_backup)

            else:
                for i in range(0, repetition_count):
                    for block_tile in tile.inner_block.tiles:
                        block_tile.apply(current_state, current_function)
                        block_tile.apply_constraints(current_state, current_function)

            def generate_code(self, current_state: GlobalState, current_function: Function) -> str:
                result_str = "(loop $" + tile.loop_name + "\n (param i32)"
                # Add inputs
                result_str += "\n".join(
                    loop_tile.generate_code(current_state, current_function) for loop_tile in self.inner_block.tiles)
                result_str += "\n"
                result_str += f"i32.const 1\n"
                result_str += f"i32.sub\n"
                result_str += f"local.tee $temp\n"
                result_str += f"local.get $temp\n"
                result_str += f"i32.const 0\n"
                result_str += f"i32.gt_s\n"
                result_str += f"br_if $" + tile.loop_name + "\n"
                result_str += "drop\n"
                result_str += ")"

                return result_str

            tile.get_fuel_cost = lambda \
                    s: (tile.inner_block.get_fuel_cost() + 8) * repetition_count
            tile.get_response_time = lambda \
                    s: tile.inner_block.get_response_time() * repetition_count
            tile.generate_code = generate_code

        tile.apply = apply
        tile.can_be_placed = staticmethod(can_be_placed)
        return tile

