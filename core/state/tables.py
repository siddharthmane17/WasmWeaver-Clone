from typing import Dict, List, Type

from core.value import Ref, RefFunc


class Table:
    def __init__(self, name: str, index:int, table_type: Type[RefFunc], size: int):
        self.name = name
        self.table_type = table_type
        self.elements: List[Ref] = [table_type(None)] * size
        self.size = size
        self.index = index

    def wipe(self):
        self.elements = [self.table_type(None)] * self.size

    def generate_code(self) -> str:
        return f"(table ${self.name} {self.size} funcref)"

    def __str__(self):
        return f"{self.name} {self.table_type} {self.elements}"


class Tables:
    """A simple table state that stores all tables."""

    def __init__(self):
        self.tables: Dict[str, Table] = {}

    def set(self, value: Table):
        self.tables[value.name] = value

    def get(self, name):
        return self.tables[name]

    def __len__(self):
        return len(self.tables)

    def reinit_tables(self):
        for table in self.tables.values():
            table.wipe()
