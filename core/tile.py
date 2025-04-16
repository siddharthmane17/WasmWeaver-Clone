from typing import Type

from core.constraints import ResponseTimeConstraint, FuelConstraint, ByteCodeSizeConstraint
from core.state.functions import Function
from core.state.state import GlobalState
from core.strategy import AbstractSelectionStrategy

global_apply_callbacks = []

class ApplyMeta(type):
    def __new__(cls, name, bases, attrs):
        orig_apply = attrs.get('apply', None)

        if callable(orig_apply):
            def wrapped_apply(self, *args, **kwargs):
                for callback in global_apply_callbacks:
                    callback(self, *args, **kwargs)
                return orig_apply(self, *args, **kwargs)

            attrs['apply'] = wrapped_apply

        return super().__new__(cls, name, bases, attrs)


class AbstractTile(metaclass=ApplyMeta):
    metrics_dependent_on_input = False
    name = "AbstractTile"  # This is the name of the tile

    # This is a class variable that is used to determine if the metrics of the tile are dependent on the input.

    def __init__(self, seed: int):
        self.seed = seed
        self.response_time = 0.0001
        self.fuel_cost = 1
        self.byte_code_size = 1

    @classmethod
    def get_weight(cls,current_state: GlobalState, current_function: Function, selection_strategy: AbstractSelectionStrategy):
        return selection_strategy.get_weight(cls, current_state, current_function)

    @staticmethod
    def can_be_placed(current_state: GlobalState, current_function: Function):
        """Returns if the tile can be placed in the current state"""
        raise NotImplementedError

    def apply(self, current_state: GlobalState, current_function: Function):
        """Applies the tile to the current state"""
        raise NotImplementedError

    def apply_constraints(self, current_state: GlobalState, current_function: Function, ignore_byte_code_size=False):
        """Applies the constraints of the tile to the current state"""
        constraints = current_state.constraints
        for constraint in constraints.constraints:
            if isinstance(constraint, ByteCodeSizeConstraint) and not ignore_byte_code_size:
                constraint.update_resource(self.get_byte_code_size())
            elif isinstance(constraint, FuelConstraint):
                constraint.update_resource(self.get_fuel_cost())
            elif isinstance(constraint, ResponseTimeConstraint):
                constraint.update_resource(self.get_response_time())
        return current_state

    def generate_code(self, current_state: GlobalState, current_function: Function) -> str:
        """Returns the code that the tile represents"""
        raise NotImplementedError

    def get_byte_code_size(self):
        """Returns the byte code size of the tile"""
        return self.byte_code_size

    def get_fuel_cost(self):
        """Returns the fuel cost of the tile"""
        return self.fuel_cost

    def get_response_time(self):
        """Returns the response time of the tile"""
        return self.response_time


class AbstractTileFactory:
    """This is an abstract factory class, which allows to generate tiles dependent on the current global state. E.g. to call functions"""
    name: str = "AbstractTileFactory"

    def __init__(self, seed: int,tile_loader):
        self.seed = seed
        self.tile_loader = tile_loader

    def generate_all_placeable_tiles(self, global_state: GlobalState, current_function: Function) -> [Type[AbstractTile]]:
        """Generates all possible tiles"""
        raise NotImplementedError
