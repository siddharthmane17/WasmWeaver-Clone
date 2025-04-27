import random
from typing import Type, TYPE_CHECKING, List
from core.state.functions import Function, Block
from core.state.state import GlobalState
if TYPE_CHECKING:
    from core.tile import AbstractTile


class AbstractSelectionStrategy:
    """
    Abstract base class for tile selection strategies.
    """
    name = "AbstractSelectionStrategy"  # This is the name of the strategy

    def get_weight(self,tile: Type["AbstractTile"], current_state: GlobalState, current_function: Function,current_blocks: List[Block]):
        """
        Returns the weight of the tile.
        """
        raise NotImplementedError

    def get_name(self):
        """
        Returns the name of the strategy.
        """
        return self.name

class RandomSelectionStrategy(AbstractSelectionStrategy):
    """
    Random selection strategy.
    """
    name = "RandomSelectionStrategy"  # This is the name of the strategy

    def get_weight(self,tile: Type["AbstractTile"], current_state: GlobalState, current_function: Function,current_blocks: List[Block]):
        """
        Returns a random weight for the tile.
        """

        #Disable all load and store instructions
        #if "Load" in tile.name or "Store" in tile.name:
        #    return -1

        #Reduce const instructions
        #if "Const" in tile.name:
        #    return random.random()*0.1

        #if "GlobalGet" in tile.name:
        #    return random.random()*0.1

        #if "LocalGet" in tile.name:
        #    return random.random()*0.1

        if tile.name == "Canary":
            return -1

        return random.random()