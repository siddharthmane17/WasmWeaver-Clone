from core.config.config import MEMORY_MAX_WRITE_INDEX
from core.state.functions import Function
from core.state.state import GlobalState
from core.tile import global_apply_callbacks


def generate_trace(global_state: GlobalState, start_function="run", start_seed=0):
    global_state.memory.reinit_memory()
    global_state.globals.reinit_globals()
    global_state.tables.reinit_tables()

    instruction_counter = 0
    def dummy_apply_callback(instance, current_state: GlobalState, current_function: Function):
        nonlocal instruction_counter
        print("\tStack:  ",current_state.stack.get_current_frame())
        print("\tMemory: ", '\t'.join(format(byte) for byte in current_state.memory.memory[:MEMORY_MAX_WRITE_INDEX]))
        print("\tGlobals:",current_state.globals)
        print(f"{instruction_counter}: {instance.__class__.__name__}")


        instruction_counter+=1
    global_apply_callbacks.append(dummy_apply_callback)
    print("------------TRACE START------------")

    for tile in global_state.functions.functions["run"].tiles:
        tile.apply(global_state, global_state.functions.functions["run"])

    #Remove the callback
    global_apply_callbacks.remove(dummy_apply_callback)
    print("------------TRACE END------------")