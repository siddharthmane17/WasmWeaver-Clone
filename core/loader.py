import importlib
import os
from typing import List, Type, TypeVar

from core.state.functions import Function
from core.state.state import GlobalState
from core.tile import AbstractTile, AbstractTileFactory

T = TypeVar('T')


class AbstractTileLoader:

    def get_placeable_tiles(self, state: GlobalState, current_function: Function) -> List[Type[AbstractTile]]:
        """Return a list of tiles that can be placed in the current state."""
        raise NotImplementedError

    def get_tile_type_by_name(self, name: str) -> Type[AbstractTile]:
        """Return a tile type by name."""
        raise NotImplementedError


class TileLoader(AbstractTileLoader):
    """Loads all tiles from a given directory."""

    def __init__(self, path: str):
        self.path = path
        self.tiles: List[Type[AbstractTile]] = self._load_classes(path, AbstractTile)
        self.factories: List[AbstractTileFactory] = [factory(0, self) for factory in
                                                     self._load_classes(path, AbstractTileFactory)]

    def get_placeable_tiles(self, state: GlobalState, current_function: Function) -> List[Type[AbstractTile]]:
        """Return a list of tiles that can be placed in the current state."""
        static_tiles = [tile for tile in self.tiles if tile.can_be_placed(state, current_function)]
        dynamic_tiles = []
        for factory in self.factories:
            dynamic_tiles.extend(factory.generate_all_placeable_tiles(state, current_function))
        return static_tiles + dynamic_tiles

    def get_tile_type_by_name(self, name: str) -> Type[AbstractTile]:
        """Return a tile type by name."""
        for tile in self.tiles:
            if tile.name == name:
                return tile
        raise ValueError(f"Tile with name '{name}' not found")

    def __str__(self):
        return f"TileLoader({self.path})\n" + "\n".join([f"  {tile}" for tile in self.tiles])

    def _find_python_files(self, directory):
        """Yield directory and filenames of Python files."""
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith('.py'):
                    yield root, file

    def _import_subclasses(self, root, file, subclass: Type[T]) -> List[Type[T]]:
        """Attempt to import subclasseses of AbstractTile from the given file."""
        sub_classes = []
        print("Root",root)
        print("File",file)
        module_path = os.path.splitext(os.path.join(root, file))[0]
        print(module_path)
        print(os.sep)
        relative_module = module_path.replace("/", '.')
        print(relative_module)
        try:
            module = importlib.import_module(relative_module)
            for name, obj in vars(module).items():
                if isinstance(obj, type) and issubclass(obj, subclass) and obj is not subclass:
                    sub_classes.append(obj)
        except ModuleNotFoundError as e:
            print(f"Failed to import {relative_module}: {e}")
        return sub_classes

    def _load_classes(self, directory: str, subclass: Type[T]) -> List[Type[T]]:
        """Load all classes inheriting from AbstractTile from Python files within the directory."""
        sub_classes = []
        for root, file in self._find_python_files(directory):
            cls = self._import_subclasses(root, file, subclass)
            if cls:
                sub_classes.extend(cls)
        return sub_classes
