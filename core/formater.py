def indent_code(code: str, indent: int=4) -> str:
    """Indents the code by the given number of spaces"""
    code_fragments = code.split("\n")
    #Only indent non-empty lines, keep empty lines as is
    indented_lines = [
        " " * indent + line if line.strip() else line for line in code_fragments
    ]
    return "\n".join(indented_lines)

def add_line_numbers_to_code(code_str: str) -> str:
    """Adds line numbers to the code string"""
    lines = code_str.split("\n")
    for i in range(len(lines)):
        lines[i] = f"{i + 1}: {lines[i]}"
    return "\n".join(lines)