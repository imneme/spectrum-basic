from .core import *  # Re-export everything

# As you move things to new files, you can explicitly list what to export:
__all__ = [
    'parse_file', 'parse_string', 'list_program',
    'minimize_variables', 'renumber', 'number_lines',
    'find_variables',
    # 'Walk', 'ASTNode', 'Statement', 'Expression',
    # etc.
]
