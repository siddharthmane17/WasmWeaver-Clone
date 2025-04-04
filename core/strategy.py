from typing import Type, TYPE_CHECKING

from core.state.functions import Function
from core.state.state import GlobalState
if TYPE_CHECKING:
    from core.tile import AbstractTile


class AbstractSelectionStrategy:
    """
    Abstract base class for tile selection strategies.
    """
    name = "AbstractSelectionStrategy"  # This is the name of the strategy

    def get_weight(self,tile: Type["AbstractTile"], current_state: GlobalState, current_function: Function):
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

    def get_weight(self,tile: Type["AbstractTile"], current_state: GlobalState, current_function: Function):
        """
        Returns a random weight for the tile.
        """
        return 1