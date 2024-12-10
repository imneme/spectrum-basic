#!/usr/bin/env python3
# 
# spectrum_basic.py
#
# A parser and language tool for ZX Spectrum BASIC, built on the textX
# language tool.
#
# The MIT License (MIT)
#
# Copyright (c) 2024 Melissa E. O'Neill
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#

import textx
from textx import metamodel_from_file
import functools
from os.path import dirname, join, exists, getmtime
import sys

def maybe_regenerate_ast_py():
    """Regenerate ast.py if necessary"""
    ast_py_filename = join(dirname(__file__), "ast.py")
    gen_ast_filename = join(dirname(__file__), "gen_ast.py")

    not_there = not exists(ast_py_filename)
    needs_regen = (not_there or 
                (exists(gen_ast_filename) and 
                    getmtime(gen_ast_filename) > getmtime(ast_py_filename)))

    if needs_regen:
        print("Creating ast.py" if not_there else "Updating ast.py", file=sys.stderr)
        from .gen_ast import gen_ast_py
        gen_ast_py(ast_py_filename)

maybe_regenerate_ast_py()

from .ast import *

# Find spectrum_basic.tx in the same directory as this script
META_PATH = join(dirname(__file__), "spectrum_basic.tx")

# Create meta-model
metamodel = metamodel_from_file(META_PATH, ws='\t ', ignore_case=True, classes=[Statement, Let, For, Next, If, Dim, DefFn, PrintItem, Variable, BinValue, ArrayRef, Fn, Slice, Number, String, ChanSpec, Rem, Label])

# Object processors
#
# The above code provides the core AST classes, but we map many of the
# concrete syntax elements to generic AST classes.  This is done with
# object processors, which are functions that take a concrete syntax
# element and return an AST object.  We use a few different object
# processors, depending on the kind of syntax element, but mostly
# we map things to the generic BuiltIn class.

def get_name(obj):
    """Get the name of an AST object"""
    return obj.name if hasattr(obj, "name") else obj.__class__.__name__.upper()

def make_ap_to_builtin(name=None, sep=", "):
    """Create an object processor for syntax elements that become generic BuiltIn objects, optionally specifying a name and separator"""
    def ap_to_builtin(obj):
        """Object processor for syntax elements that become generic BuiltIn objects"""
        if isinstance(obj, str):
            return BuiltIn(None, name or obj)
        builtin_name = name or get_name(obj)
        args = [getattr(obj, field) for field in obj._tx_attrs if field != 'name' or name is not None]
        while (args and args[-1] is None):
            args.pop()
        return BuiltIn(obj.parent, builtin_name, *args, sep=sep)
    return ap_to_builtin

ap_standard = make_ap_to_builtin()
ap_saveload = make_ap_to_builtin(sep=" ")

def ap_coloured(obj):
    """Object processor for PLOT/DRAW/CIRCLE commands with optional colour parameters"""
    # Circle or Draw with angle
    if hasattr(obj, "expr3") and obj.expr3 is not None:
        return ColouredBuiltin(obj.parent, get_name(obj), obj.colours, obj.expr1, obj.expr2, obj.expr3)
    else:  # Plot or Draw without angle
        return ColouredBuiltin(obj.parent, get_name(obj), obj.colours, obj.expr1, obj.expr2)

# Object processor for PRINT-like statements

def ap_print_like(obj):
    items = [PrintItem(x.item, x.sep) for x in obj.items]
    if obj.final:
        items.append(PrintItem(obj.final, None))
    return BuiltIn(obj.parent, get_name(obj), *items, sep="")

def ap_expr(ap_func):
    """Wrap an object processor to make the result an expression"""
    def new_ap_func(obj):
        newobj = ap_func(obj)
        newobj.is_expr = True
        return newobj
    return new_ap_func

def ap_binop(obj):
    """Object processor for binary operators"""
    # Need to reduce to chain of binary operations
    return functools.reduce(lambda l, r: BinaryOp(r.op, l, r.expr), obj.rest, obj.first)


# Register object processors

metamodel.register_obj_processors({
    # 0-argument commands
    "New": ap_standard,
    "Stop": ap_standard,
    "Return": ap_standard,
    "Continue": ap_standard,
    "Copy": ap_standard,
    "Cls": ap_standard,
    "Cat": ap_standard,
    # 1-argument commands
    "Goto": ap_standard,
    "Gosub": ap_standard,
    "Restore": ap_standard,
    "Pause": ap_standard,
    "Border": ap_standard,
    "Run": ap_standard,
    "Clear": ap_standard,
    "Randomize": ap_standard,
    "ColourParam": ap_standard,
    # 2-argument commands
    "Beep": ap_standard,
    "Out": ap_standard,
    "Poke": ap_standard,
    "Plot": ap_coloured,
    # 3-argument commands
    "Draw": ap_coloured,
    "Circle": ap_coloured,
    # File-related commands
    "Save": ap_saveload,
    "Load": ap_saveload,
    "Merge": ap_saveload,
    "Verify": ap_saveload,
    "SaveLine": ap_standard,
    "SaveCode": ap_standard,
    "LoadCode": ap_standard,
    "FileData": ap_standard,
    "FileScreen": ap_standard,
    "OpenHash": make_ap_to_builtin("OPEN #"),
    "CloseHash": make_ap_to_builtin("CLOSE #"),
    # PRINT-like statements
    "Print": ap_print_like,
    "Lprint": ap_print_like,
    "Input": ap_print_like,
    # 1-argument modifiers
    "Tab": ap_standard,
    "SaveLine": ap_standard,
    "InputLine": ap_standard,
    # 2-argument print-modifiers
    "At": ap_standard,
    # 0-arity functions
    "PiValue": ap_standard,
    # 1-arity functions
    "Function": ap_expr(ap_standard),
    # 2-arity functions
    "TwoArgFn": ap_expr(ap_standard),
    # Binary operators
    "OrExpr": ap_binop,
    "AndExpr": ap_binop,
    "CompareExpr": ap_binop,
    "AddExpr": ap_binop,
    "MultExpr": ap_binop,
    "PowerExpr": ap_binop,
})

def parse_file(filename):
    """Parse a BASIC program from a file"""
    return metamodel.model_from_file(filename)
    
def parse_string(program):
    """Parse a BASIC program from a string"""
    return metamodel.model_from_str(program)

# A simple walker to find the names of all the variables in a program

# def find_variables(program):
#     """Find all the variables in a program"""
#     variables = {}
#     for event, obj in walk(program):
#         if event == Walk.VISITING:
#             if isinstance(obj, Variable):
#                 lowname = obj.name.lower()
#                 if lowname not in variables:
#                     variables[lowname] = obj.name
#     return sorted(variables.values())

def find_variables(program):
    """Find all the variables in a program"""
    vars = {kind: {} for kind in ["numeric", "string", "numeric-array", "fn", "fn-info", "param", "loop-var"]}
    def used_var(kind, var, varDict=None):
        lowname = var.lower()
        if varDict is None:
            varDict = vars
        return varDict.setdefault(kind, {}).setdefault(lowname, var)
    def vars_to_lists(vars):
        def process_kind(kind):
            match kind:
                case "fn-info":         # fn-info is not a simple mapping
                    return vars[kind]
                case "param":          # leave params in order encountered
                    return vars[kind].values()
                case _:                 # sort the rest
                    return sorted(vars[kind].values())
        return {kind: process_kind(kind) for kind in vars}
    
    stack = []
    for event, obj in walk(program):
        if event == Walk.VISITING:
            if isinstance(obj, Variable):
                kind = "string" if obj.name.endswith("$") else "numeric"
                used_var(kind, obj.name)
        elif event == Walk.ENTERING:
            if isinstance(obj, (ArrayRef, Dim)):
                kind = "string" if obj.name.endswith("$") else "numeric-array"
                used_var(kind, obj.name)
            elif isinstance(obj, For):
                used_var("loop-var", obj.var.name)
            elif isinstance(obj, Fn):
                used_var("fn", obj.name)
            elif isinstance(obj, DefFn):
                name = used_var("fn", obj.name)
                # Push the names of the unbound parameters
                lparams = [p.name.lower() for p in obj.params]
                newvars = {}
                for lparam, param in zip(lparams, obj.params):
                    used_var("param", param.name)
                    used_var("param", param.name, newvars)
                stack.append((name, lparams, vars))
                vars = newvars
        elif event == Walk.LEAVING:
            if isinstance(obj, DefFn):
                # Restore and update the numeric variables
                (name, lparams, oldvars) = stack.pop()
                # Remove the params to leave the free variables
                freevars = vars["numeric"]
                for lparam in lparams:
                    freevars.pop(lparam, None)
                vars["numeric"] = freevars
                # Remember what this function used (its free variables)
                oldvars["fn-info"][name] = vars_to_lists(vars)
                # Merge the the two sets of variables
                for kind in vars:
                    oldvars[kind].update(vars[kind])
                vars = oldvars
    # Convert to sorted lists
    return vars_to_lists(vars)

def list_program(program, file=None):
    """List the program in a BASIC-like format"""
    for line in program.lines:
        spacer = "\t"
        if line.line_number and line.label:
            print(f"{line.line_number} {line.label}:", end="", file=file)
            spacer = " "
        elif line.line_number:
            print(f"{line.line_number}", end="", file=file)
        elif line.label:
            print(f"{line.label}:", end="", file=file)
            spacer = " " if len(line.label.name) < 6 else spacer
        
        if line.statements:
            print(spacer, ": ".join(str(stmt) for stmt in line.statements), file=file, sep="")
        else:
            print(file=file)

# Seems silly a function like this one isn't in the standard library

def baseN(num,b,numerals="0123456789abcdefghijklmnopqrstuvwxyz"):
    """Convert a number to a base-N numeral"""
    return numerals[0] if num == 0 else baseN(num // b, b, numerals).lstrip(numerals[0]) + numerals[num % b]

def var_generator(start='A', one_letter=True, taken_names=None):
    """Generate fresh variable names, avoiding those in taken_names"""
    if taken_names is None:
        taken_names = set()
    offset = ord(start) - ord('A')
    pos = 0
    cycles = 0
    while cycles == 0 or not one_letter:
        name = chr((pos + offset) % 26 + ord('A'))
        if cycles > 0:
            # Convert to base 36
            name += baseN(cycles-1, 36)
        if name not in taken_names:
            taken_names.add(name)
            yield name
        pos += 1
        if pos >= 26:
            pos = 0
            cycles += 1
    # If we get here, we've run out of names
    raise ValueError("Out of variable names")


def calculate_remapping(vars):
    """Create minimal remapping for each namespace"""
    namespaceInfo = {
        'numeric': {'start': 'A', 'one_letter': False},
        'string': {'start': 'A', 'one_letter': True},
        'numeric-array': {'start': 'A', 'one_letter': True},
        'fn': {'start': 'F', 'one_letter': True},
    }
    def make_remapper(info, taken_names=None, remapping=None):
        taken_names = taken_names or set()
        remapping = remapping or {}
        generator = var_generator(info['start'], info['one_letter'], taken_names)
        def remapper(var):
            lvar = var.lower()
            if lvar in remapping:
                return remapping[lvar]
            # If they're already using a single letter, don't remap if the
            # name isn't already taken
            is_string = var.endswith("$")
            lvar_no_sigil = lvar[:-1] if is_string else lvar
            if len(lvar_no_sigil) == 1 and not lvar_no_sigil in taken_names:
                taken_names.add(lvar_no_sigil)
                return remapping.setdefault(lvar, var)
            newname = next(generator)
            newname += "$" if is_string else ""
            result = remapping.setdefault(lvar, newname)
            return result
        return remapping, remapper

    remappingFor = {}
    for kind in namespaceInfo:
        info = namespaceInfo[kind]
        if kind == "numeric":
            # For numeric variables, we first remapp the loop variables,
            # since they must be single letters
            taken1 = set()
            remapping1, remap1 = make_remapper({'start': 'I', 'one_letter': True}, taken_names=taken1)
            for var in vars["loop-var"]:
                remap1(var)
            # Then we remap the rest as usual
            remapping, remap = make_remapper(info, taken_names=taken1, remapping=remapping1)
        else:
            remapping, remap = make_remapper(info)
        # We'll sort by length to put the single-letter variables first
        for var in sorted(vars[kind], key=len):
            remap(var)
        remappingFor[kind] = remapping

    # Remap the parameters of functions
    remappingFor["fn-params"] = {}
    for fn, fn_info in vars["fn-info"].items():
        # taken1 still holds all the numeric vairables, which we must avoid
        remapping, remap = make_remapper({'start': 'X', 'one_letter': True}, taken_names=taken1)
        for param in fn_info["param"]:
           remap(param)
        lfn = fn.lower()
        remappingFor["fn-params"][remappingFor["fn"][lfn].lower()] = remapping

    return remappingFor

def remap_variables(program, remapping):
    """Apply the remapping to all variables in the program"""
    stack = []
    for event, obj in walk(program):
        if event == Walk.VISITING:
            if isinstance(obj, Variable):
                lowname = obj.name.lower()
                kind = "string" if obj.name.endswith("$") else "numeric"
                if lowname in remapping[kind]:
                    obj.name = remapping[kind][lowname]
        elif event == Walk.ENTERING:
            if isinstance(obj, (ArrayRef, Dim)):
                lowname = obj.name.lower()
                kind = "string" if obj.name.endswith("$") else "numeric-array"
                if lowname in remapping[kind]:
                    obj.name = remapping[kind][lowname]
            elif isinstance(obj, Fn):
                lowname = obj.name.lower()
                if lowname in remapping["fn"]:
                    obj.name = remapping["fn"][lowname]
            elif isinstance(obj, DefFn):
                lowname = obj.name.lower()
                if lowname in remapping["fn"]:
                    remapped = remapping["fn"][lowname]
                    obj.name = remapped
                    lowname = remapped.lower()
                # Push current parameter mappings and create new ones
                param_state = {}
                for param in obj.params:
                    lparam = param.name.lower()
                    if lparam in remapping["numeric"]:
                        param_state[lparam] = remapping["numeric"][lparam]
                        # Remove from current mapping while in function
                        del remapping["numeric"][lparam]
                    # Apply parameter mapping
                    param_remap = remapping["fn-params"][lowname]
                    if lparam in param_remap:
                        mparam = param_remap[lparam]
                        param.name = mparam
                        remapping["numeric"][lparam] = mparam
                stack.append((param_state, param_remap.keys()))
        elif event == Walk.LEAVING:
            if isinstance(obj, DefFn):
                # Restore parameter mappings
                param_state, lparams = stack.pop()
                for lparam in lparams:
                    remapping["numeric"].pop(lparam, None)
                remapping["numeric"].update(param_state)

def minimize_variables(program):
    """Find all variables and remap them to minimal form"""
    vars = find_variables(program)
    remapping = calculate_remapping(vars)
    remap_variables(program, remapping)

def renumber(program, start_line=10, increment=10):
    """Renumber a BASIC program with given start line and increment"""
    # First pass: build line number mapping
    line_map = {}
    new_line = start_line
    last_line = None
    for line in program.lines:
        curr_line = line.line_number
        if curr_line is not None:
            if last_line is not None and curr_line <= last_line:
                raise ValueError(f"Huh? Line numbers should increase in order: {curr_line} after {last_line}")
            if curr_line > new_line and curr_line % 500 == 0:
                # If the original code was broken up neat sections, try
                # to preserve that
                new_line = curr_line
            line_map[curr_line] = new_line
            line.line_number = new_line
            new_line += increment

    # Check the we didn't go over 10000
    final_line = new_line - increment
    if (final_line) >= 10000:
        raise ValueError(f"Renumbering would exceed line number limit: {final_line}")

    # Second pass: update GOTO/GOSUB targets
    for event, obj in walk(program):
        if event == Walk.ENTERING:
            match obj:
                case BuiltIn(action="GOTO" | "GOSUB" | "RESTORE" | "RUN", args=[target]) if isinstance(target, Number):
                    # Simple numeric constant
                    line_num = int(target.value)
                    if line_num not in line_map:
                        raise ValueError(f"Invalid {obj.action} to non-existent line {line_num}")
                    obj.args = (line_map[line_num],)
                case BuiltIn(action="GOTO" | "GOSUB"):
                    raise ValueError(f"Cannot renumber {obj.action} with computed line number: {obj.args[0]}")
    
    return program

def number_lines(program, remove_labels=True, default_increment=10, start_line=None):
    """Number any unnumbered lines and optionally remove labels"""
    # If a start line is specified, and the first line is not numbered
    # edit the first line to use the start line
    if start_line is not None and program.lines and not program.lines[0].line_number:
        program.lines[0].line_number = start_line

    # First pass: build line number mapping for all lines
    line_map = {}  # Maps labels to line numbers
    numbered_lines = []  # List of (position, line_num, is_blank) for existing numbers
    lines_to_number = []  # List of (position, label, is_blank) for lines needing numbers

    for i, line in enumerate(program.lines):
        is_blank = not line.statements
        if line.line_number:
            if numbered_lines and line.line_number <= numbered_lines[-1][1]:
                raise ValueError(f"Line numbers must increase: {line.line_number} after {numbered_lines[-1][1]} at line {i}")
            numbered_lines.append((i, line.line_number, is_blank))
            if line.label:
                line_map[line.label.name] = line.line_number
        else:
            lines_to_number.append((i, line.label.name if line.label else None, is_blank))
    
    # Now fill in gaps with appropriate line numbers
    prev_pos, prev_num, prev_blank = -1, 0, False
    for next_pos, next_num, next_blank in numbered_lines + [(len(program.lines), 10000, False)]:
        gap_lines = [x for x in lines_to_number if prev_pos < x[0] < next_pos]
        if gap_lines:
            # Calculate how many lines we need to fit
            available_space = next_num - prev_num
            needed_spaces = sum(1 for _, _, is_blank in gap_lines if not is_blank) + 1
            increment = min(default_increment, available_space // needed_spaces)
            if increment < 1:
                raise ValueError(f"Cannot fit {len(gap_lines)} lines between {prev_num} and {next_num}")
            
            new_line = prev_num + (increment if not prev_blank else 0)
            if prev_blank:
                # We're overwriting the previous line, so remove its number
                program.lines[prev_pos].line_number = None
            for i, label, is_blank in gap_lines:
                if label:
                    line_map[label] = new_line
                if not is_blank or (label and not remove_labels):
                    program.lines[i].line_number = new_line
                    new_line += increment
        
        prev_pos, prev_num, prev_blank = next_pos, next_num, next_blank

    # Now, filter out any lines we chose not to number (blank ones)
    program.lines = [line for line in program.lines if line.line_number]

    # Second pass: update label references and optionally remove labels
    deadly_magic = False
    for event, obj in walk(program):
        if event == Walk.ENTERING:
            match obj:
                case Statement() as stmt:
                    deadly_magic = True
        elif event == Walk.LEAVING:
            match obj:
                case Statement() as stmt:
                    deadly_magic = False
        elif deadly_magic and event == Walk.VISITING:
            match obj:
                case Label(name=label):
                    # We shall perform deadly magic on this poor label.
                    # Don't try this at home kids! Experts only!  We shall
                    # wave our magic wand and turn this label into a number.
                    if label not in line_map:
                        raise ValueError(f"Reference to undefined label '{label}'")
                    obj.__class__ = Number
                    obj.__init__(obj.parent, line_map[label])
    
    if remove_labels:
        for line in program.lines:
            line.label = None
    
    return program

def main():
    import argparse
    import sys
    import json

    parser = argparse.ArgumentParser(description="Parse a ZX BASIC program")
    parser.add_argument("filename", nargs="?", help="Filename of BASIC program to parse (omit to read stdin)")
    parser.add_argument("--show", action="store_true", help="Show the parsed program")
    parser.add_argument("--number", action="store_true", help="Number any unnumbered lines")
    parser.add_argument("--delabel", action="store_true", help="Number any unnumbered lines and remove labels")
    parser.add_argument("--renumber", action="store_true", help="Renumber the program")
    parser.add_argument("--start-line", help="Starting line number for renumbering, numbering and delabeling", type=int, default=10)
    parser.add_argument("--increment", help="Increment for renumbering, numbering and delabeling", type=int, default=10)
    parser.add_argument("--minimize", action="store_true", help="Minimize(/legalize) the variable names")
    parser.add_argument("--find-vars", action="store_true", help="Find all the variables in the program and dump them as JSON")
    args = parser.parse_args()

    if not any((args.show, args.find_vars)):
        args.show = True

    if not args.filename:
        args.filename = "/dev/stdin"

    # Sanity check args for renumbering, etc
    if args.start_line < 1 or args.start_line >= 10000:
        print("Start line must be in the range 1-9999")
        sys.exit(1)
    if args.increment < 1 or args.increment > 5000:
        print("Increment should be sensible")
        sys.exit(1)

    try:
        program = parse_file(args.filename)

        if args.find_vars:
            print(json.dumps(find_variables(program), indent=4))
        if args.number or args.delabel:
            number_lines(program, remove_labels=args.delabel, start_line=args.start_line, default_increment=args.increment)
        if args.renumber:
            program = renumber(program, args.start_line, args.increment)
        if args.minimize:
            minimize_variables(program)
        if args.show:
            list_program(program)

    except textx.exceptions.TextXSyntaxError as e:
        print(f"Parse error: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
