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

For developer mode, clone the repository and install the package in editable mode:

```bash
git clone https://github.com/imneme/spectrum-basic.git
cd spectrum-basic
pip install -e .
```

Install from PyPI:

```bash
pip install speccy-basic
``` 


Requires Python 3.10 or later. 

## Usage

### Command Line

The package installs a command-line tool called `speccy-basic`:

```bash
# Show the parsed and pretty-printed program
speccy-basic program.bas --show

# Number unnumbered lines and remove labels
speccy-basic program.bas --delabel

# Minimize variable names
speccy-basic program.bas --minimize

# Combine transformations
speccy-basic program.bas --delabel --minimize

# Analyze variables
speccy-basic program.bas --find-vars
```

### As a Library

```python
from spectrum_basic.core import parse_file, number_lines, minimize_variables, list_program

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

Label names are written `@identifier`. Lines are labeled by putting the label at the start of the line, followed by a colon. They can be used anywhere where you would write a line number, including:

- `GOTO`/`GOSUB` statements
- Arithmetic expressions (e.g., `(@end - @start)/10`)


## Working with the AST

If you want to analyze or transform BASIC programs, you'll need to work with the Abstract Syntax Tree (AST) that represents the program's structure. Import the AST nodes from the ast module:

```python
from spectrum_basic.ast import Variable, Number, Label, BuiltIn
```

The AST nodes have attributes that correspond to the fields of the original BASIC code. For example:

```text
>>> from spectrum_basic import *
>>> prog = parse_string('10 PRINT "Hello World!";')
>>> len(prog.lines)
1
>>> (stmt := prog.lines[0].statements[0])
PRINT "Hello World!";
>>> (arg := stmt.args[0])
"Hello World!";
>>> arg.value
"Hello World!"
>>> arg.sep
';'
```

However, for many applications where you want to traverse syntax tree, you may prefer to use the AST walking API described below.

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

You can control traversal by sending `Walk.SKIP` back to the generator to skip processing a node's children. You can also just abandon the generator at any time.

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
