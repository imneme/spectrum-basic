# Spectrum BASIC Tools

A Python toolkit for parsing, transforming, and manipulating ZX Spectrum BASIC programs. This tool can help you work with both classic Spectrum BASIC and an enhanced dialect that supports modern programming constructs.

## Features

- Full parser for ZX Spectrum BASIC
- Support for an enhanced dialect with:
    - Optional line numbers
    - Labels (e.g., `@loop:`)
    - Label references in expressions and GOTOs
- Program transformations:
    - Line numbering and renumbering
    - Variable name minimization
    - Label elimination (for Spectrum compatibility)
- Detailed variable analysis
- Pretty printing with authentic Spectrum BASIC formatting

## Installation

Currently available as a single Python file. Requires Python 3.10+ and the TextX parsing library:

```bash
pip install textx
```

## Usage

### Command Line

```bash
# Show the parsed and pretty-printed program
python spectrum_basic.py program.bas --show

# Number unnumbered lines and remove labels
python spectrum_basic.py program.bas --delabel

# Minimize variable names
python spectrum_basic.py program.bas --minimize

# Combine transformations
python spectrum_basic.py program.bas --delabel --minimize

# Analyze variables
python spectrum_basic.py program.bas --find-vars
```

### As a Library

```python
from spectrum_basic import parse_file, number_lines, minimize_variables, list_program

# Parse a program
program = parse_file("my_program.bas")

# Apply transformations
number_lines(program, remove_labels=True)
minimize_variables(program)

# Output the result
list_program(program)
```

## Enhanced BASIC Features

The tool supports an enhanced dialect of BASIC that's compatible with ZX Spectrum BASIC. Additional features include:

### Labels
```basic
@loop:
FOR I = 1 TO 10
    PRINT I
NEXT I
GOTO @loop
```

Labels can be used in:

- GOTO/GOSUB statements
- Arithmetic expressions (e.g., `(@end - @start)/10`)
- Any line of code (numbered or unnumbered)

## Working with the AST

If you want to analyze or transform BASIC programs, you'll need to work with the Abstract Syntax Tree (AST) that represents the program's structure. This section provides an overview of the AST nodes and how to traverse them.

### AST Walking

The AST can be traversed using the `walk()` generator, which yields tuples of `(event, node)`. Events are:

```python
class Walk(Enum):
    ENTERING = auto()  # Entering a compound node
    VISITING = auto()  # At a leaf node or simple value
    LEAVING  = auto()  # Leaving a compound node
```

Example usage:

```python
def find_variables(program):
    """Find all variables in a program"""
    variables = set()
    for event, obj in walk(program):
        if event == Walk.VISITING and isinstance(obj, Variable):
            variables.add(obj.name)
    return sorted(variables)
```

You can control traversal by sending `Walk.SKIP` back to the generator to skip processing a node's children.  You can also just abandon the generator at any time.

### Key AST Nodes

Common patterns for matching AST nodes:

```python
# Basic nodes
Variable(name=str)          # Variable reference (e.g., "A" or "A$")
Number(value=int|float)     # Numeric literal
Label(name=str)             # Label reference (e.g., "@loop")

# Built-in commands/functions
BuiltIn(action=str,         # Command name (e.g., "PRINT", "GOTO")
        args=tuple)         # Command arguments

# Special cases
ColoredBuiltin(action=str,  # Graphics commands (PLOT, DRAW, CIRCLE)
              colors=list,  # Color parameters
              args=tuple)   # Coordinates/dimensions

# Program structure
Program(lines=list)         # Complete program
SourceLine(                 # Single line of code
    line_number=int|None,
    label=Label|None,
    statements=list)
```

(Note: Currently `Program` and `SourceLine` are textX automagic classes rather than custom AST nodes. This only matters if you want to use them in pattern matching, which is unlikely.)

Example pattern matching:

```python
match obj:
    case BuiltIn(action="GOTO", args=[target]) if isinstance(target, Number):
        # Handle simple GOTO with numeric line number
        line_num = target.value
        ...
    case Variable(name=name) if name.endswith("$"):
        # Handle string variable
        ...
```

## License

MIT License. Copyright (c) 2024 Melissa O'Neill

## Requirements

- Python 3.10 or later
- TextX parsing library