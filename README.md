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

## License

MIT License. Copyright (c) 2024 Melissa O'Neill

## Requirements

- Python 3.10 or later
- TextX parsing library
