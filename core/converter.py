from core.state.state import GlobalState


def _generate_functions(global_state: GlobalState, external=False) -> str:
    """Generates a strings for all external and internal functions"""
    function_str = ""
    for function in global_state.functions.functions.values():
        if function.is_external != external:
            continue
        function_str += function.generate_code(global_state, function)

    return function_str

def _generate_ext_functions(global_state: GlobalState) -> str:
    """Generates a strings for all external functions"""
    function_str = ""
    for function in global_state.ext_functions.functions.values():
        function_str += function.generate_code()

    return function_str

def _generate_tables(global_state: GlobalState) -> str:
    """Generates a strings for all tables"""
    table_str = ""
    for table in global_state.tables.tables.values():
        table_str += table.generate_code() + "\n"

    return table_str
def _generate_globals(global_state: GlobalState) -> str:
    """Generates a strings for all global variables"""
    global_str = ""
    for global_var in global_state.globals.globals:
        global_str += global_var.generate_code() + "\n"

    return global_str


def _generate_signatures(global_state: GlobalState) -> str:
    """Generates a strings for all function signatures"""
    signature_str = ""
    for function in global_state.functions.functions.values():
        signature_str += function.generate_signature() + "\n"
    return signature_str


def global_state_to_wasm_program(global_state: GlobalState) -> str:
    """Converts a global state to a wasm program"""
    template = f"(module\n"
    #Signatures
    if _generate_signatures(global_state):
        template += f'{_generate_signatures(global_state)}'

    #Function imports
    if _generate_functions(global_state, external=True):
        template += f'{_generate_functions(global_state, external=True)}'
    #External functions
    if _generate_ext_functions(global_state):
        template += f'{_generate_ext_functions(global_state)}'
    #Memory
    template += '(import "env" "memory" (memory 1))\n'
    # Tables
    if _generate_tables(global_state):
        template += f'{_generate_tables(global_state)}'
    #Globals
    if _generate_globals(global_state):
        template += f'{_generate_globals(global_state)}'
    template += f"{_generate_functions(global_state, external=False)}"
    template += "\n)"
    return template
