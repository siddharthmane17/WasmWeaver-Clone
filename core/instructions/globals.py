import copy
import random
from typing import Type, List

from core.config.config import MAX_GLOBALS_PER_MODULE
from core.state.functions import Function, Block
from core.state.globals import Global
from core.state.state import GlobalState
from core.tile import AbstractTile, AbstractTileFactory
from core.value import get_random_random_val


def generate_random_global_name(global_state: GlobalState) -> str:
    while True:
        name = f"global_{random.randint(0, 2 ** 32 - 1)}"
        for global_var in global_state.globals.globals:
            if global_var.name == name:
                continue
        return name


class AbstractGlobalFactory(AbstractTileFactory):
    def __init__(self, seed: int, tile_loader):
        super().__init__(seed, tile_loader)

    def generate_all_placeable_tiles(self, global_state: GlobalState, current_function: Function, current_blocks: List[Block]) -> [Type[AbstractTile]]:
        for global_var in global_state.globals.globals:
            global_get_tile = self.create_global_get_tile(global_var.name)
            if global_get_tile.can_be_placed(global_state, current_function,current_blocks):
                yield global_get_tile
            global_set_tile = self.create_global_set_tile(global_var.name)
            if global_set_tile.can_be_placed(global_state, current_function,current_blocks):
                yield global_set_tile

        new_set_tile = self.create_global_set_tile(generate_random_global_name(global_state),
                                                   create_global=True)
        if new_set_tile.can_be_placed(global_state, current_function, current_blocks):
            yield new_set_tile

        # Add random local type
        get_tile = self.create_global_get_tile(generate_random_global_name(global_state),
                                               create_global=True)
        if get_tile.can_be_placed(global_state, current_function, current_blocks):
            yield get_tile

    def create_global_get_tile(self, global_name, create_global: bool = False):

        class GlobalGet(AbstractTile):
            name = f"Get global"

            def __init__(self, seed: int):
                nonlocal global_name, create_global
                super().__init__(seed)
                self.name = f"Get global"
                self.global_name = global_name
                self.create_local = create_global
                self.last_value = None

            @staticmethod
            def can_be_placed(current_state: GlobalState, current_function: Function, current_blocks: List[Block]):
                if not current_state.stack.get_current_frame().can_push_to_stack():
                    return False
                if create_global:
                    return MAX_GLOBALS_PER_MODULE > len(current_state.globals.globals)
                return True

            def apply(self, current_state: GlobalState, current_function, current_blocks: List[Block]):
                global_var = current_state.globals.get_global_by_name(self.global_name)
                if not create_global and global_var is None:
                    raise ValueError("Global is not set and is not marked for being created")

                if create_global and global_var is None:
                    #print(f"Creating global {self.global_name}")
                    global_var = Global(get_random_random_val(), self.global_name, random.randint(0, 10) > 2)
                    current_state.globals.add(global_var)

                #print(f"Getting global {self.global_name} value", global_var.value)
                self.last_value = copy.deepcopy(global_var.value)
                current_state.stack.get_current_frame().stack_push(global_var.value)

            def generate_code(self, current_state: GlobalState, current_function, current_blocks: List[Block]) -> str:
                return f"global.get ${self.global_name}"

            def get_byte_code_size(self):
                return 2

        return GlobalGet

    def create_global_set_tile(self, global_name, create_global: bool = False):
        """Used for creating local set tiles"""

        class GlobalSet(AbstractTile):
            name = f"Set global"

            def __init__(self, seed: int):
                nonlocal global_name, create_global
                super().__init__(seed)
                self.name = f"Set global"
                self.global_name = global_name
                self.create_local = create_global
                self.last_value = None

            @staticmethod
            def can_be_placed(current_state: GlobalState, current_function: Function, current_blocks: List[Block]):
                if len(current_state.stack.get_current_frame().stack) < 1:
                    return False
                if not create_global:
                    global_var = current_state.globals.get_global_by_name(global_name)
                    return isinstance(current_state.stack.get_current_frame().stack[-1],
                                      type(global_var.value)) and global_var.mutable
                else:
                    return MAX_GLOBALS_PER_MODULE > len(current_state.globals.globals)

            def apply(self, current_state: GlobalState, current_function: Function, current_blocks: List[Block]):
                global_var = current_state.globals.get_global_by_name(self.global_name)
                if create_global and not global_var:
                    val = current_state.stack.get_current_frame().stack_pop()
                    #print(f"Setting global {self.global_name} value to", val.value)
                    #Set initial value
                    global_var = Global(val.get_random_val(), self.global_name, True)
                    #Set current value
                    global_var.value = val
                    current_state.globals.add(global_var)

                    #Add to function local types
                else:

                    value = current_state.stack.get_current_frame().stack_pop()
                    #print(f"Setting global {self.global_name} value to", value.value)
                    current_state.globals.get_global_by_name(self.global_name).value = value
                self.last_value = copy.deepcopy(global_var.value)

            def generate_code(self, current_state: GlobalState, current_function, current_blocks: List[Block]) -> str:
                return f"global.set ${global_name}"

            def get_byte_code_size(self):
                return 2

        return GlobalSet
