from typing import Type, List

from core.config.config import MAX_LOCALS_PER_FUNCTION
from core.state.functions import Function, Block
from core.state.state import GlobalState
from core.tile import AbstractTile, AbstractTileFactory
from core.value import get_random_val


class AbstractLocalFactory(AbstractTileFactory):
    def __init__(self, seed: int, tile_loader):
        super().__init__(seed, tile_loader)

    def generate_all_placeable_tiles(self, global_state: GlobalState, current_function: Function, current_blocks: List[Block]) -> [
        Type[AbstractTile]]:
        for index, local in enumerate(global_state.stack.get_current_frame().locals.locals):
            local_get_tile = self.create_local_get_tile(index)
            if local_get_tile.can_be_placed(global_state, current_function, current_blocks):
                yield local_get_tile
            local_set_tile = self.create_local_set_tee_tile(index)
            if local_set_tile.can_be_placed(global_state, current_function, current_blocks):
                yield local_set_tile

        new_set_tile = self.create_local_set_tee_tile(len(global_state.stack.get_current_frame().locals),
                                                      create_local=True, is_tee=False)
        if new_set_tile.can_be_placed(global_state, current_function, current_blocks):
            yield new_set_tile
        new_tee_tile = self.create_local_set_tee_tile(len(global_state.stack.get_current_frame().locals),
                                                      create_local=True, is_tee=True)
        if new_tee_tile.can_be_placed(global_state, current_function, current_blocks):
            yield new_tee_tile

        # Add random local type
        get_tile = self.create_local_get_tile(len(global_state.stack.get_current_frame().locals),
                                              create_local=True)
        if get_tile.can_be_placed(global_state, current_function, current_blocks):
            yield get_tile

    def create_local_get_tile(self, index, create_local: bool = False):

        class LocalGet(AbstractTile):
            name = f"Get local"

            def __init__(self, seed: int):
                super().__init__(seed)
                self.index = index
                self.create_local = create_local

            @staticmethod
            def can_be_placed(current_state: GlobalState, current_function: Function, current_blocks: List[Block]):
                if not current_state.stack.get_current_frame().can_push_to_stack():
                    return False
                if create_local:
                    return MAX_LOCALS_PER_FUNCTION > len(current_state.stack.get_current_frame().locals)
                return True

            def apply(self, current_state: GlobalState, current_function: Function, current_blocks: List[Block]):
                if not create_local and len(current_state.stack.get_current_frame().locals) <= self.index:
                    raise ValueError("Local is not set and is not marked for being created")

                if create_local:
                    if len(current_function.local_types) > self.index:
                        self.local = current_function.local_types[self.index].get_default_value()
                        #print(f"Local type due to larger get: {current_function.local_types[self.index]}")
                    else:
                        self.local = get_random_val()
                        #print(f"Local type random get: {type(self.local)}")
                        #print(self.local.value)

                if create_local:
                    if len(current_state.stack.get_current_frame().locals) <= self.index:
                        current_state.stack.get_current_frame().locals.add(self.local)

                    if len(current_function.local_types) <= self.index:
                        current_function.local_types.append(type(self.local))
                    else:
                        if current_function.local_types[self.index] != type(self.local):
                            raise ValueError(
                                f"Local type mismatch. Expected {current_function.local_types[self.index]} got {type(self.local)}")

                current_state.stack.get_current_frame().stack_push(current_state.stack.get_current_frame().locals[index])

            def generate_code(self, current_state: GlobalState, current_function: Function, current_blocks: List[Block]) -> str:
                return f"local.get {index}"

            def get_byte_code_size(self):
                return 2

        return LocalGet

    def create_local_set_tee_tile(self, index, create_local: bool = False, is_tee: bool = False):
        """Used for creating local set tiles"""

        class LocalTeeSet(AbstractTile):
            name = f"{'Set' if not is_tee else 'Tee'} local"

            def __init__(self, seed: int):

                super().__init__(seed)
                self.index = index
                self.create_local = create_local

            @staticmethod
            def can_be_placed(current_state: GlobalState, current_function: Function, current_blocks: List[Block]):
                if len(current_state.stack.get_current_frame().stack) < 1:
                    return False
                if not create_local:
                    return isinstance(current_state.stack.get_current_frame().stack[-1], current_function.local_types[index])
                else:
                    if len(current_function.local_types) > index:
                        return isinstance(current_state.stack.get_current_frame().stack[-1], current_function.local_types[index]) and MAX_LOCALS_PER_FUNCTION > len(current_state.stack.get_current_frame().locals)
                    else:
                        return MAX_LOCALS_PER_FUNCTION > len(current_state.stack.get_current_frame().locals)

            def apply(self, current_state: GlobalState, current_function: Function, current_blocks: List[Block]):
                if create_local:
                    local = current_state.stack.get_current_frame().stack_pop() if not is_tee else \
                        current_state.stack.get_current_frame().stack[-1]
                    if len(current_state.stack.get_current_frame().locals) <= self.index:
                        current_state.stack.get_current_frame().locals.add(local)

                    #Add to function local types
                    if len(current_function.local_types) <= self.index:
                        current_function.local_types.append(type(local))
                    else:
                        if current_function.local_types[self.index] != type(local):
                            raise ValueError(
                                f"Local type mismatch. Expected {current_function.local_types[self.index]} got {type(local)}")
                else:

                    value = current_state.stack.get_current_frame().stack_pop() if not is_tee else \
                        current_state.stack.get_current_frame().stack[-1]
                    current_state.stack.get_current_frame().locals[index] = value

            def generate_code(self, current_state: GlobalState, current_function: Function, current_blocks: List[Block]) -> str:
                if not is_tee:
                    return f"local.set {index}"
                else:
                    return f"local.tee {index}"

            def get_byte_code_size(self):
                return 2

        return LocalTeeSet
