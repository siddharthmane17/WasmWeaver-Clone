import copy
import random
from typing import Type, List

from core.config.config import MAX_TABLE_SIZE, MAX_TABLES_PER_MODULE
from core.state.functions import Function, Block
from core.state.state import GlobalState
from core.state.tables import Table
from core.tile import AbstractTile, AbstractTileFactory
from core.value import I32, RefFunc


def generate_random_table_name(global_state: GlobalState) -> str:
    while True:
        name = f"tab_{random.randint(0, 2 ** 32 - 1)}"
        for table_name in global_state.tables.tables:
            if table_name == name:
                continue
        return name


class AbstractTableFactory(AbstractTileFactory):
    def __init__(self, seed: int, tile_loader):
        super().__init__(seed, tile_loader)

    def generate_all_placeable_tiles(self, global_state: GlobalState, current_function: Function, current_blocks: List[Block]) -> [
        Type[AbstractTile]]:
        for table in global_state.tables.tables.values():
            table_get_tile = self.create_table_get_tile(table.name,global_state.tables.get(table.name).index)
            if table_get_tile.can_be_placed(global_state, current_function, current_blocks):
                yield table_get_tile
            table_set_tile = self.create_table_set_tile(table.name,
                                                        global_state.tables.get(table.name).index)
            if table_set_tile.can_be_placed(global_state, current_function, current_blocks):
                yield table_set_tile

        new_set_tile = self.create_table_set_tile(generate_random_table_name(global_state),len(global_state.tables),
                                                  create_table=True)
        if new_set_tile.can_be_placed(global_state, current_function, current_blocks):
            yield new_set_tile
        #Currently disabled to not flood the stack with null ref
        new_get_tile = self.create_table_get_tile(generate_random_table_name(global_state),len(global_state.tables),
                                                    create_table=True)
        if new_get_tile.can_be_placed(global_state, current_function, current_blocks):
            yield new_get_tile

    def create_table_get_tile(self, table_name: str, table_index: int, create_table: bool = False):

        class TableGet(AbstractTile):
            name = f"Get table"
            table_size = random.randint(1, MAX_TABLE_SIZE) if create_table else 0
            index = table_index

            def __init__(self, seed: int):
                nonlocal table_name, create_table
                super().__init__(seed)
                self.name = f"Get table"
                self.create_table = create_table
                self.table_name = table_name
                self.last_value = None

            @staticmethod
            def can_be_placed(current_state: GlobalState, current_function: Function, current_blocks: List[Block]):
                if not create_table:
                    TableGet.table_size = current_state.tables.tables[table_name].size
                #Check if last value is a valid index
                if len(current_state.stack.get_current_frame().stack) < 1:
                    return False
                if not isinstance(current_state.stack.get_current_frame().stack[-1], I32):
                    return False

                if TableGet.table_size <= current_state.stack.get_current_frame().stack[-1].value or current_state.stack.get_current_frame().stack[-1].value < 0:
                    return False
                if create_table:
                    return MAX_TABLES_PER_MODULE > len(current_state.tables.tables)
                return True

            def apply(self, current_state: GlobalState, current_function: Function, current_blocks: List[Block]):
                if create_table:
                    table = Table(self.table_name, len(current_state.tables), RefFunc, self.table_size)
                    current_state.tables.set(table)
                table = current_state.tables.tables[self.table_name]
                index = current_state.stack.get_current_frame().stack_pop().value
                self.last_value = copy.deepcopy(table.elements[index])
                current_state.stack.get_current_frame().stack_push(table.elements[index])

            def generate_code(self, current_state: GlobalState, current_function: Function, current_blocks: List[Block]) -> str:
                return f"table.get ${self.table_name}"

            def get_byte_code_size(self):
                return 2

        return TableGet

    def create_table_set_tile(self, table_name: str, table_index: int, create_table: bool = False):
        """Used for creating local set tiles"""

        class TableSet(AbstractTile):
            name = f"Set table"
            table_size = random.randint(1, MAX_TABLE_SIZE) if create_table else 0
            index = table_index

            def __init__(self, seed: int):
                nonlocal table_name, create_table
                super().__init__(seed)
                self.name = f"Set table"
                self.table_name = table_name
                self.create_local = create_table
                self.last_value = None

            @staticmethod
            def can_be_placed(current_state: GlobalState, current_function: Function, current_blocks: List[Block]):
                if not create_table:
                    TableSet.table_size = current_state.tables.tables[table_name].size

                if len(current_state.stack.get_current_frame().stack) < 2:
                    return False
                if not isinstance(current_state.stack.get_current_frame().stack[-2], I32):
                    return False

                if TableSet.table_size <= current_state.stack.get_current_frame().stack[-2].value or current_state.stack.get_current_frame().stack[-2].value < 0:
                    return False
                if not isinstance(current_state.stack.get_current_frame().stack[-1], RefFunc):
                    return False
                if create_table:
                    return MAX_TABLES_PER_MODULE > len(current_state.tables.tables)
                return True

            def apply(self, current_state: GlobalState, current_function: Function, current_blocks: List[Block]):
                if create_table:
                    table = Table(table_name,len(current_state.tables), RefFunc, self.table_size)
                    current_state.tables.set(table)

                table = current_state.tables.tables[table_name]
                value = current_state.stack.get_current_frame().stack_pop()
                index = current_state.stack.get_current_frame().stack_pop().value
                table.elements[index] = value
                self.last_value = copy.deepcopy(value)

            def generate_code(self, current_state: GlobalState, current_function: Function, current_blocks: List[Block]) -> str:
                return f"table.set ${table_name}"

            def get_byte_code_size(self):
                return 2

        return TableSet
