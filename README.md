# Spectrum BASIC Tools

A Python toolkit for parsing, transforming, and manipulating ZX Spectrum BASIC programs. This tool can help you work with both classic Spectrum BASIC and an enhanced dialect that supports modern programming constructs.

## Features

- Full parser for ZX Spectrum BASIC
- Support for an enhanced dialect with:
    - Optional line numbers
    - Labels (e.g., `@loop:`)
    - Label references in expressions and GOTOs
    - Additional control structures from Spectrum Next BASIC
- Program transformations:
    - Line numbering and renumbering
    - Variable name minimization
    - Label elimination (for Spectrum compatibility)
- Detailed variable analysis
- Pretty printing with authentic Spectrum BASIC formatting
- TAP file generation for loading programs on a real Spectrum
- Run a subset of BASIC programs locally for algorithm testing

## Installation

For developer mode, clone the repository and install the package in editable mode:

```bash
git clone https://github.com/imneme/spectrum-basic.git
cd spectrum-basic
pip install -e .
```

Install from PyPI:

```bash
pip install spectrum-basic
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

# Convert Spectrum Next control structures to GOTOs
speccy-basic program.bas --decontrol

# Minimize variable names
speccy-basic program.bas --minimize

# Combine transformations
speccy-basic program.bas --delabel --minimize

# Analyze variables
speccy-basic program.bas --find-vars

# Generate a TAP file
speccy-basic program.bas --tap output.tap --tap-name "My Program"

# Run a program locally
speccy-basic program.bas --run
```

### As a Library

```python
from spectrum_basic import parse_file, number_lines, minimize_variables, make_program_tap, write_tap

# Parse a program
program = parse_file("my_program.bas")

# Apply transformations
number_lines(program, remove_labels=True)
minimize_variables(program)

# Output the result
str(program)

# Program image and tape generation
binary_code = bytes(program)
tap = make_program_tap(binary_code, name="My Program", autostart=9000)
write_tap(tap, "output.tap")
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

### Spectrum Next Control Structures

The version of Spectrum BASIC used on the Spectrum Next includes a variety of additional new features. This tool can supports _some_ of those features, converting them to equivalent ZX Spectrum BASIC constructs.

The option `--decontrol` will convert these constructs to ZX Spectrum BASIC, using auto-generated labels, and adding `--delabel` will remove the labels to make the program compatible with classic ZX Spectrum BASIC.

#### Multi-line `IF` statements

On the Spectrum Next, if we omit `THEN`, we can write multi-line `IF` statements:

```basic
IF A = 1
    PRINT "One"
ELSE IF A = 2
    PRINT "Two"
ELSE
    PRINT "Not one or two"
END IF
PRINT "Done"
```

becomes :

```basic
10 IF A <> 1 THEN GOTO 40
20 PRINT "One"
30 GOTO 80
40 IF A <> 2 THEN GOTO 70
50 PRINT "Two"
60 GOTO 80
70 PRINT "Not one or two"
80 PRINT "Done"
```

(`ELSE IF` can also be written as `ELSEIF`, `ELSIF` or `ELIF`.)

#### `REPEAT` loops

Spectrum Next BASIC has a `REPEAT` loop construct (where the loop ends with `REPEAT UNTIL` and not just `UNTIL`; an infinite loop would be `REPEAT UNTIL 0`):

```basic
LET I = 0
REPEAT
    PRINT "Hello ";I
    LET I = I * 2
REPEAT UNTIL I = 16
```

Translated to Classic ZX Spectrum BASIC, this becomes:

```basic
10 LET I = 0
20 PRINT "Hello ";I
30 LET I = I * 2
40 IF I <> 16 THEN GOTO 20
```

#### Adding an exit condition with `WHILE`

`WHILE` is not a looping construct itself, it's a way to add an exit condition to a `REPEAT` loop:

```basic
LET I = 0
REPEAT
    LET I = I * 2
    WHILE I < 16
    PRINT "Hello ";I
REPEAT UNTIL 0
```

is transformed to:

```
10 LET I = 0
20 LET I = I * 2
30 IF I >= 16 THEN GOTO 60
40 PRINT "Hello ";I
50 GOTO 20
```

#### Exiting a loop with `EXIT`

Both `FOR` and `REPEAT` loops can be exited early with `EXIT`:

```basic
FOR I = 1 TO 10
    FOR J = 1 TO 10
        IF I * J > 50 THEN EXIT
        PRINT I * J
    NEXT J
NEXT I
```

becomes:

```basic
10 FOR I = 1 TO 10
20 FOR J = 1 TO 10
30 IF I * J > 50 THEN GOTO 60
40 PRINT I * J
50 NEXT J
60 NEXT I
```

To exit from multiple levels of loop, just write multiple `EXIT` statements. Changing the above code to use `EXIT:EXIT` would exit both loops.

You can also use `EXIT` like a `GOTO` statement to exit the loop and continue execution at a specific line number.  This is just translated to a `GOTO` statement when translating code to classic ZX Spectrum BASIC, but is necessary on the Spectrum Next to allow the BASIC interpreter to know that the loop is being exited.

#### Continuing a loop with `GOTO NEXT`

Many languages have a `continue` statement to skip the rest of the loop and go to the next iteration.  Spectrum Next BASIC does not have this feature, but the translation tool does support it.  You can write:

```basic
DEF FN M(x,d) = x - d*INT(x/d)
FOR I = 1 TO 10
    IF M(I,2) = 0 THEN GOTO NEXT
    PRINT I
NEXT I
```

becomes:

```basic
10 DEF FN M(x, d) = x - d * INT (x / d)
20 FOR I = 1 TO 10
30 IF M(I, 2) = 0 THEN GOTO 50
40 PRINT I
50 NEXT I
```

(You can also use `GOTO NEXT NEXT` to the continuation of an outer loop, and so on.)

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

# Built-in commands/functions (most statements)
BuiltIn(action=str,         # Command name (e.g., "PRINT", "GOTO")
        args=tuple)         # Command arguments

# Statements that don't just take expressions usually have their own AST nodes
# for example:
Let(var=Variable|ArrayRef,  # Assignment statement
    expr=Expression)        # Expression to assign

# Program structure
Program(lines=list)         # Complete program
SourceLine(                 # Single line of code
    line_number=int|None,
    label=Label|None,
    statements=list)
```

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
