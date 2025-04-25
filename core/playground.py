import random

from core.loader import TileLoader
from core.state.functions import Function
from core.state.state import GlobalState
from core.strategy import RandomSelectionStrategy

tile_loader = TileLoader("core/instructions/")

def main():
    global_state = GlobalState()
    global_state.stack.push_frame(params=None, stack=[], name="origin")
    function = Function(name=f"run", inputs=[], outputs=[], is_external=False)
    dummy_function = Function(name="f_dummy", inputs=[], outputs=[], is_external=False)
    global_state.functions.functions["f_dummy"] = dummy_function
    function.selection_strategy = RandomSelectionStrategy()

    while True:

        print("Selected tiles:")
        print("-" * 20)
        for tile in function.tiles:
            print(tile.name)
        print("-" * 20)

        print("Current stack:")
        print("-" * 20)
        for value in global_state.stack.get_current_frame().stack:
            print(value)
        print("-" * 20)

        potential_tiles = tile_loader.get_placeable_tiles(global_state, function)
        print("Please select a tile to place:")
        for i, tile in enumerate(potential_tiles):
            print(f"{i}: {tile.name}")

        print("q: Quit")

        choice = input("Enter your choice: ")
        if choice == "q":
            break

        if not choice.isdigit():
            print("Invalid choice. Please enter a number.")
            continue

        if int(choice) >= len(potential_tiles) or int(choice) < 0:
            print("Invalid choice. Please enter a valid number.")
            continue

        for i, tile in enumerate(potential_tiles):
            if str(i) == choice:
                tile_instance = tile(random.randint(0, 2**32 - 1))
                tile_instance.apply(global_state, function)
                function.tiles.append(tile_instance)
                print(f"Placed tile: {tile.name}")






if __name__ == "__main__":
    main()