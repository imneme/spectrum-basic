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
from os.path import dirname, join, exists, getmtime, splitext, basename
import sys

def maybe_regenerate_ast_py():
    """Regenerate ast.py if necessary"""
    ast_py_filename = join(dirname(__file__), "ast.py")
    gen_ast_filename = join(dirname(__file__), "gen_ast.py")

    not_there = not exists(ast_py_filename)
    needs_regen = (not_there or 
                (exists(gen_ast_filename) and 
                    getmtime(gen_ast_filename) > getmtime(ast_py_filename) + 1))

    if needs_regen:
        print("Creating ast.py" if not_there else "Updating ast.py", file=sys.stderr)
        from .gen_ast import gen_ast_py
        gen_ast_py(ast_py_filename)

maybe_regenerate_ast_py()

from .ast import *
from .tokenizer import *

# Find spectrum_basic.tx in the same directory as this script
META_PATH = join(dirname(__file__), "spectrum_basic.tx")

# Create meta-model
metamodel = metamodel_from_file(META_PATH, ws='\t ', ignore_case=True, classes=[JankyStatement, Statement, Let, For, Next, If, LongIf, Else, ElseIf, EndIf, Repeat, Until, While, Exit, ContinueLoop, Dim, DefFn, Data, Read, PrintItem, JankyFunctionExpr, Variable, BinValue, ArrayRef, InputExpr, Not, Neg, Fn, Slice, Number, String, ChanSpec, Rem, Label, Program, SourceLine, CommentLine])

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

def make_ap_listlike(name=None, sep=", ", minItems=0, maxItems=None):
    """Create an object processor for syntax elements that become generic BuiltIn objects with lists of arguments"""
    def ap_listlike(obj):
        """Object processor for syntax elements that become generic BuiltIn objects with lists of arguments"""
        builtin_name = name or get_name(obj)
        if minItems is not None and len(obj.items) < minItems:
            # raise ValueError(f"Too few items in {builtin_name}")
            raise textx.exceptions.TextXSyntaxError(f'Too few items in {builtin_name}', **textx.get_location(obj))
        if maxItems is not None and len(obj.items) > maxItems:
            # raise ValueError(f"Too many items in {builtin_name}")
            raise textx.exceptions.TextXSyntaxError(f'Too many items in {builtin_name}', **textx.get_location(obj))
        return BuiltIn(obj.parent, builtin_name, *obj.items, sep=sep)
    return ap_listlike

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

ap_standard_expr = ap_expr(ap_standard)

def ap_string_subscript(obj):
    """Object processor for string subscript expressions"""
    if obj.subscript is not None:
        return StringSubscript(obj.expr, obj.subscript)
    return obj.expr

def ap_drop_junk(obj):
    """Object processor to drop unneed prefix nodes"""
    if not obj.before and not obj.after:
        return obj.actual
    return obj

# Register object processors

metamodel.register_obj_processors({
    # Prefix statements
    "JankyStatement": ap_drop_junk,
    "JankyFunctionExpr": ap_drop_junk,
    # Subscript expressions
    "LitStringExpr": ap_string_subscript,
    "ParenExpr": ap_string_subscript,
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
    "List": ap_standard,
    "LList": ap_standard,
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
    # Music commands
    "Play": make_ap_listlike(minItems=1, maxItems=3),
    # PRINT-like statements
    "Print": ap_print_like,
    "Lprint": ap_print_like,
    "Input": ap_print_like,
    # 1-argument modifiers
    "Tab": ap_standard,
    "InputLine": make_ap_to_builtin("LINE"),
    # 2-argument print-modifiers
    "At": ap_standard,
    # 0-arity functions
    "CompValue": ap_standard,
    # 1-arity functions
    "Function": ap_standard_expr,
    # 2-arity functions
    "TwoArgFn": ap_standard_expr,
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

def parse_fh(fh):
    """Parse a BASIC program from a file-like object"""
    return metamodel.model_from_str(fh.read())
    
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
                    return list(vars[kind].values())
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
                lparams = [p.lower() for p in obj.params]
                newvars = {}
                for lparam, param in zip(lparams, obj.params):
                    used_var("param", param)
                    used_var("param", param, newvars)
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
        if isinstance(line, CommentLine):
            print(f"{line}", file=file)
            continue
        elif line.line_number and line.label:
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
        name = chr((pos + offset) % 26 + ord('a'))
        if cycles > 0:
            # Convert to base 36
            name += baseN(cycles-1, 36)
        if name not in taken_names:
            taken_names.add(name)
            yield name.upper()
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
        taken_names = taken_names if taken_names is not None else set()
        remapping = remapping if remapping is not None else {}
        generator = var_generator(info['start'], info['one_letter'], taken_names)
        def remapper(var):
            lvar = var.lower()
            if lvar in remapping:
                return remapping[lvar]
            # If they're already using a single letter, don't remap if the
            # name isn't already taken
            is_string = var.endswith("$")
            lvar_no_sigil = lvar[:-1] if is_string else lvar
            lvar_no_sigil = lvar_no_sigil.lower()
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
    numericRemapping = remappingFor["numeric"]
    for fn, fn_info in vars["fn-info"].items():
        free_vars = fn_info["numeric"]
        fn_taken = set(numericRemapping[v].lower() for v in free_vars)
        remapping, remap = make_remapper({'start': 'X', 'one_letter': True}, taken_names=fn_taken)
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
                for i, param in enumerate(obj.params):
                    lparam = param.lower()
                    if lparam in remapping["numeric"]:
                        param_state[lparam] = remapping["numeric"][lparam]
                        # Remove from current mapping while in function
                        del remapping["numeric"][lparam]
                    # Apply parameter mapping
                    param_remap = remapping["fn-params"][lowname]
                    if lparam in param_remap:
                        mparam = param_remap[lparam]
                        obj.params[i] = mparam
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
        if isinstance(line, CommentLine):
            continue
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
                case BuiltIn(action="GOTO" | "GOSUB" | "RESTORE" | "RUN", args=[target]):
                    if not isinstance(target, Number):
                        raise ValueError(f"Cannot renumber {obj.action} to non-numeric line number: {target}")
                    # Simple numeric constant
                    line_num = int(target.value)
                    if line_num not in line_map:
                        raise ValueError(f"Invalid {obj.action} to non-existent line {line_num}")
                    obj.args = (line_map[line_num],)
                case BuiltIn(action="SAVE", args=[filename, BuiltIn(action="LINE", args=[target])]):
                    if not isinstance(target, Number):
                        raise ValueError(f"Cannot renumber SAVE .. LINE to non-numeric line number: {target}")
                    # Simple numeric constant
                    line_num = int(target.value)
                    if line_num not in line_map:
                        raise ValueError(f"Invalid SAVE .. LINE to non-existent line {line_num}")
                    obj.args[1].args = (line_map[line_num],)
    
    return program

def number_lines(program, remove_labels=True, default_increment=10, start_line=None):
    """Number any unnumbered lines and optionally remove labels"""
    # If a start line is specified, and the first line is not numbered
    # edit the first line to use the start line
    # Find index of first non CommentLine
    first_line = next((i for i, line in enumerate(program.lines) if not isinstance(line, CommentLine)), None)
    if first_line is None:
        raise ValueError("No actual code lines in program")
    if start_line is not None and not program.lines[first_line].line_number:
        program.lines[first_line].line_number = start_line

    # First pass: build line number mapping for all lines
    line_map = {}  # Maps labels to line numbers
    numbered_lines = []  # List of (position, line_num, is_blank) for existing numbers
    lines_to_number = []  # List of (position, label, is_blank) for lines needing numbers

    for i, line in enumerate(program.lines):
        if isinstance(line, CommentLine):
            continue
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
            deferred_labels = []
            for i, label, is_blank in gap_lines:
                if not is_blank or (label and not remove_labels):
                    program.lines[i].line_number = new_line
                    if label:
                        line_map[label] = new_line
                    if deferred_labels:
                        for dlabel in deferred_labels:
                            line_map[dlabel] = new_line
                        deferred_labels = []
                    new_line += increment
                elif label:
                    deferred_labels.append(label)
            if deferred_labels:
                good_line = next_num if next_num < 10000 else new_line
                for dlabel in deferred_labels:
                    line_map[dlabel] = good_line
        
        prev_pos, prev_num, prev_blank = next_pos, next_num, next_blank

    # Now, filter out any lines we chose not to number (blank ones)
    program.lines = [line for line in program.lines if isinstance(line, CommentLine) or line.line_number]

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

# A general utility function we'll use
def split_and_insert(lst, func):
    """
    Processes each element of 'lst' with 'func'. 
    
    If func(elem) returns None, the element is added to the current run.
    If func(elem) returns a list, the current run (if any) is finalized 
    and appended to the result, then func(elem) is appended as a separate list.
    
    Example:
        >>> def f(n):
        ...     return None if n % 2 == 0 else [n // 2, (n // 2) + 1]
        >>> split_and_insert([2, 4, 6, 7, 21, 42], f)
        [[2, 4, 6], [3, 4], [10, 11], [42]]
        
    :param lst: A list of elements to be processed.
    :param func: A callable that returns either None or a list.
    :return: A list of lists (runs and inserted lists).
    """
    result = []
    current_run = []
    
    for elem in lst:
        outcome = func(elem)
        if outcome is None:
            # Keep accumulating elements until we hit a list outcome
            current_run.append(elem)
        else:
            # If there's a run in progress, finalize it
            if current_run:
                result.append(current_run)
                current_run = []
            # Then insert the outcome as its own list
            if isinstance(outcome, tuple):
                result.extend(outcome)
            else:
                result.append(outcome)
    
    # If there's a remaining run, finalize it
    if current_run:
        result.append(current_run)
    
    return result

def break_lines(program, stmt_matcher):
    """Break lines at a given statement type"""
    new_lines = []
    for line in program.lines:
        if isinstance(line, CommentLine) or len(line.statements) == 0:
            new_lines.append(line)
            continue
        first_stmt = line.statements[0]
        splits = split_and_insert(line.statements, stmt_matcher)
        line.statements = splits[0]
        new_lines.append(line)
        if len(splits) == 1:
            continue
        for stmts in splits[1:]:
            new_line = SourceLine(line.parent, None, None, stmts, [])
            if stmts[0] == first_stmt:
                new_line.line_number = line.line_number
                new_line.label = line.label
                line.line_number = None
                line.label = None
            new_lines.append(new_line)
        new_line.after = line.after
        line.after = []
    program.lines = new_lines

class PreElseStmt(Statement): pass
class PreElseIfStmt(Statement): pass
class PostUntilStmt(Statement):
    def __init__(self, exit_fixups):
        self.exit_fixups = exit_fixups
class PostNextStmt(Statement):
    def __init__(self, exit_fixups):
        self.exit_fixups = exit_fixups

def transmute_to_goto(stmt, line_expr):
    """Overwrite an existing statement with a GOTO"""
    # Deadly magic, do not try this at home
    stmt.__class__ = BuiltIn
    stmt.__init__(stmt.parent, "GOTO", line_expr)

def break_control_lines(program):
    """Break lines at multi-line control statements"""
    control_stack = []   # Pairs, [control_stmt, list_of_exits, list_of_conts]
    def exit_scan(stmt):
        match stmt:
            case Exit(exits=exits, line=None):
                try:
                    control_info = control_stack[-len(exits)]
                    control_info[1].append(stmt)
                except IndexError:
                    raise ValueError(f"{stmt} without corresponding control statement")
            case Exit(line=line):
                transmute_to_goto(stmt, line)
            case ContinueLoop(nexts=nexts):
                try:
                    control_info = control_stack[-len(nexts)]
                    control_info[2].append(stmt)
                except IndexError:
                    raise ValueError(f"{stmt} without corresponding control statement")
            case If(statements=stmts) | Else(statements=stmts):
                for stmt in stmts:
                    exit_scan(stmt)
    def breaker(stmt):
        exit_scan(stmt)
        match stmt:
            case Else():
                stmts = stmt.statements
                stmt.statements = []
                bstmts = split_and_insert(stmts, breaker)
                if len(bstmts) == 1:
                    return ([PreElseStmt()], [stmt] + bstmts[0])
                else:
                    return ([PreElseStmt()], [stmt], *bstmts)
            case ElseIf():
                return ([PreElseIfStmt()], [stmt])
            case Until():
                if not control_stack:
                    raise ValueError("Misplaced {stmt} (no active loop)")
                control_info = control_stack.pop()
                if control_info[0].__class__ is not Repeat:
                    raise ValueError(f"Mismatched {stmt} (matches {control_info[0]})")
                stmt.continues = control_info[2]
                return ([stmt], [PostUntilStmt(control_info[1])])
            case LongIf() | EndIf() | While():
                return [stmt]
            case Repeat():
                control_stack.append([stmt, [], []])
                return [stmt]
            case For():
                control_stack.append([stmt, [], []])
                return None
            case Next(var=var):
                if not control_stack:
                    raise ValueError("Misplaced {stmt} (no active loop)")
                control_info = control_stack.pop()
                if control_info[0].__class__ is not For or control_info[0].var.name != var.name:
                    raise ValueError(f"Mismatched {stmt} (matches {control_info[0]})")
                stmt.continues = control_info[2]
                if not control_info[1] and not control_info[2]:
                    return None
                return ([stmt],[PostNextStmt(control_info[1])])
            case _:
                return None

    break_lines(program, breaker)

def negate_condition(cond):
    """Negate a condition"""
    match cond:
        case Not(expr=subcond):
            return subcond
        case BinaryOp(op=op, lhs=lhs, rhs=rhs):
            match op:
                case "<":  return BinaryOp(">=", lhs, rhs)
                case "<=": return BinaryOp(">", lhs, rhs)
                case "=":  return BinaryOp("<>", lhs, rhs)
                case "<>": return BinaryOp("=", lhs, rhs)
                case ">":  return BinaryOp("<=", lhs, rhs)
                case ">=": return BinaryOp("<", lhs, rhs)
                case "AND": 
                    return BinaryOp("OR", negate_condition(lhs), 
                                          negate_condition(rhs))
                case "OR":
                    return BinaryOp("AND", negate_condition(lhs),
                                           negate_condition(rhs))
                case _:
                    return Not(None, "NOT", cond)
        case _:
            return Not(None, "NOT", cond)

def eliminate_control_lines(program):
    """Eliminate multi-line control statements"""
    break_control_lines(program)
    def new_label():
        counter = 0
        while True:
            yield f"@__control_{counter}"
            counter += 1
    def fixup_if(line, label):
        if_cond = line.statements[0].condition
        line.statements = [
            If(line, negate_condition(if_cond), [BuiltIn(None, "GOTO", label)], [])
        ]
    def fixup_goto_placeholder(line, label):
        line.statements = [BuiltIn(None, "GOTO", label)]
    def line_label(line):
        if (label := line.label) is None:
            label = Label(line, next(label_gen))
            line.label = label
        return label
    def drop_stmt(line,i):
        dropped = line.statements[0]
        line.statements = line.statements[1:]
        if not line.statements and line.line_number:
            # If the next line lacks a line number, give it ours
            if i+1 < len(program.lines) and not program.lines[i+1].line_number:
                program.lines[i+1].line_number = line.line_number
                line.line_number = None
            else:
                # No luck, we need to hang onto the line number in case someone
                # wants to GOTO it
                line.statements = [Rem(line, str(dropped))]
    label_gen = new_label()
    if_stack = []       # IFs to patch
    else_stack = []     # Lines-Before-ELSEs to patch
    repeat_stack = []   # Labels of REPEAT lines
    while_stack = []    # WHILEs to patch
    for i, line in enumerate(program.lines):
        if isinstance(line, CommentLine) or len(line.statements) == 0:
            continue
        first_stmt = line.statements[0]
        match first_stmt:
            case Repeat():
                repeat_stack.append(line_label(line))
                while_stack.append([])
                drop_stmt(line,i)
            case While():
                if not while_stack:
                    raise ValueError("Misplaced WHILE")
                while_stack[-1].append(line)
            case Until(condition=cond, continues=fixup_conts):
                if not repeat_stack:
                    raise ValueError("Misplaced UNTIL")
                label = repeat_stack[-1]
                match cond:
                    case Number(value=0):
                        fixup_goto_placeholder(line, label)
                        cont_label = label
                    case _:
                        fixup_if(line, label)
                        cont_label = line_label(line)
                for fix_cont in fixup_conts:
                    transmute_to_goto(fix_cont, cont_label)
            case Next(continues=fixup_conts):
                for fix_cont in fixup_conts:
                    transmute_to_goto(fix_cont, line_label(line))
            case PostUntilStmt(exit_fixups=fixups):
                repeat_stack.pop()
                label = line_label(line)
                for fixup in while_stack.pop():
                    fixup_if(fixup, label)
                for fix_exit in fixups:
                    transmute_to_goto(fix_exit, label)
                drop_stmt(line,i)
            case PostNextStmt(exit_fixups=fixups):
                for fix_exit in fixups:
                    transmute_to_goto(fix_exit, line_label(line))
                drop_stmt(line,i)
            case LongIf():
                if_stack.append(line)
                else_stack.append([])
            case PreElseStmt():
                # Always directly before an ELSE
                if not else_stack or isinstance(else_stack[-1], tuple):
                    raise ValueError("Misplaced ELSE")
                else_stack[-1].append(line)
                else_stack[-1] = tuple(else_stack[-1])
            case PreElseIfStmt():
                # Always directly before an ELSE IF
                if not else_stack or isinstance(else_stack[-1], tuple):
                    raise ValueError("Misplaced ELSE IF")
                else_stack[-1].append(line)
            case Else():
                fixup_if(if_stack[-1], line_label(line))
                if_stack[-1] = None
                drop_stmt(line,i)
            case ElseIf():
                fixup_if(if_stack[-1], line_label(line))
                if_stack[-1] = line
            case EndIf():
                label = line_label(line)
                if not if_stack:
                    raise ValueError("Misplaced ENDIF")
                if else_stack[-1]:
                    for fixup in else_stack[-1]:
                        fixup_goto_placeholder(fixup, label)
                if if_stack[-1]:
                    fixup_if(if_stack[-1], label)
                if_stack.pop()
                else_stack.pop()
                drop_stmt(line,i)
    if if_stack:
        raise ValueError(f"Unterminated IF ({if_stack[-1]})")
    if repeat_stack:
        raise ValueError(f"Unterminated REPEAT")


def make_header(type_code: int, name: str, length: int, param1: int, param2: int) -> bytes:
    """Create a ZX Spectrum tape header block.
    
    Args:
        type_code: 0=Program, 1=Number array, 2=Character array, 3=Code file
        name: Filename (will be padded/truncated to 10 chars)
        length: Length of the data block that follows
        param1: Parameter 1 (meaning depends on type_code)
        param2: Parameter 2 (meaning depends on type_code)
    """
    # Ensure name is exactly 10 bytes, space padded
    name_bytes = name.encode('ascii', 'replace')[:10].ljust(10, b' ')
    
    # Pack header: type (1 byte) + name (10 bytes) + 3 shorts
    return bytes([type_code]) + name_bytes + \
           length.to_bytes(2, 'little') + \
           param1.to_bytes(2, 'little') + \
           param2.to_bytes(2, 'little')

def tape_checksum(data: bytes) -> int:
    """Calculate Spectrum tape checksum (XOR of all bytes)"""
    result = 0
    for b in data:
        result ^= b
    return result

def make_tape_block(marker: int, data: bytes) -> bytes:
    """Create a tape block with marker, data, and checksum.
    Returns the block prefixed with its length (TAP format)."""
    block = bytes([marker]) + data
    block += bytes([tape_checksum(block)])
    
    # TAP format: length (2 bytes) followed by block
    return len(block).to_bytes(2, 'little') + block

def make_program_tap(progname: str, program: bytes, 
                      autostart: int = -1) -> bytes:
    """Create a TAP file containing a BASIC program.
    
    Args:
        filename: Name to give the program (max 10 chars)
        program: The tokenized BASIC program
        autostart: LINE parameter (≥32768 means no LINE given)
        vars_offset: Start of variable area relative to program start
    """
    autostart = autostart if autostart >= 0 else 32768
    # Make header block (type 0 = Program)
    proglen = len(program)
    header = make_header(0, progname, proglen, autostart, proglen)
    header_block = make_tape_block(0x00, header)
    
    # Make data block
    data_block = make_tape_block(0xff, program)
    
    return header_block + data_block

def write_tap(tap_data: bytes, filename: str):
    """Write TAP data to a file."""
    with open(filename, 'wb') as f:
        f.write(tap_data)

def main():
    import argparse
    import sys
    import json
    from .run import run_program, ZXBasicError

    parser = argparse.ArgumentParser(description="Parse a ZX BASIC program")
    # parser.add_argument("filename", help="Filename of BASIC program to parse")
    parser.add_argument("filename", help="Filename of BASIC program to parse (use - for stdin)")
    parser.add_argument("--show", action="store_true", help="Show the parsed program")
    parser.add_argument("--run", action="store_true", help="Run the program (where possible)")
    parser.add_argument("--decontrol", action="store_true", help="Simplify multi-line control statements to GOTOs")
    parser.add_argument("--number", action="store_true", help="Number any unnumbered lines")
    parser.add_argument("--delabel", action="store_true", help="Number any unnumbered lines and remove labels")
    parser.add_argument("--renumber", action="store_true", help="Renumber the program")
    parser.add_argument("--start-line", help="Starting line number for renumbering, numbering and delabeling", type=int, default=10)
    parser.add_argument("--increment", help="Increment for renumbering, numbering and delabeling", type=int, default=10)
    parser.add_argument("--minimize", action="store_true", help="Minimize(/legalize) the variable names")
    parser.add_argument("--find-vars", action="store_true", help="Find all the variables in the program and dump them as JSON")
    parser.add_argument("--tap", help="Write the program to a TAP file", metavar="FILENAME")
    # program name for tap file
    parser.add_argument("--tap-name", help="Name to give the program in the TAP file", default="SpeccyFun!")
    parser.add_argument("--tap-line", help="Run the program from a specific line", type=int, default=-1)
    args = parser.parse_args()

    if not any((args.show, args.find_vars, args.tap, args.run)):
        args.show = True

    # Currently running the code requires it have no labels, as does making a TAP file
    if args.run or args.tap:
        args.decontrol = True
        args.delabel = True

    # Sanity check args for renumbering, etc
    if args.start_line < 1 or args.start_line >= 10000:
        print("Start line must be in the range 1-9999")
        sys.exit(1)
    if args.increment < 1 or args.increment > 5000:
        print("Increment should be sensible")
        sys.exit(1)

    if args.tap == "+auto":
        filename_without_ext = splitext(args.filename)[0]
        filename_without_path = basename(filename_without_ext)
        args.tap = "taps/" + filename_without_path + ".tap"

    # Make sure the tap file ends with .tap
    if args.tap and not args.tap.endswith(".tap"):
        print("TAP filename must end with .tap")
        sys.exit(1)

    try:
        if args.filename != "-":
            program = parse_file(args.filename)
        else:
            program = parse_fh(sys.stdin)

        if not program:
            print("No program parsed", file=sys.stderr)
            sys.exit(1)

        if args.find_vars:
            print(json.dumps(find_variables(program), indent=4))
        if args.decontrol:
            eliminate_control_lines(program)
        if args.number or args.delabel:
            number_lines(program, remove_labels=args.delabel, start_line=args.start_line, default_increment=args.increment)
        if args.renumber:
            program = renumber(program, args.start_line, args.increment)
        if args.minimize:
            minimize_variables(program)
        if args.show:
            list_program(program)
        if args.tap:
            code = bytes(program)
            tap = make_program_tap(args.tap_name, code, autostart=args.tap_line)
            write_tap(tap, args.tap)
        if args.run:
            run_program(program)

    except textx.exceptions.TextXSyntaxError as e:
        # Print GCC-style error message
        message = e.message
        # Make message more user-friendly
        SWITCHAROOS = {
            "@[A-Z][A-Z0-9_]*": "Label",
            r'[\r\n]\s*': "Newline",
            r'[ \t]*': " ",
            r'\b': "",
            "Colon": "':'",
            "ColourCode": "ColourEscape",
            "New": "'NEW'",
            "Stop": "'STOP'",
            "Return": "'RETURN'",
            "Continue": "'CONTINUE'",
            "Copy": "'COPY'",
            "Cls": "'CLS'",
            "Cat": "'CAT'",
        }
        for pattern, replacement in SWITCHAROOS.items():
            message = message.replace(pattern, replacement)
        print(f"{e.filename}:{e.line}:{e.col}: {message}")
        print(f"\tstopped here: '{e.context}' <-- '*' indicates point where parse failed" if e.context else "")
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        sys.exit(1)
    except BrokenPipeError:
        sys.exit(1)
    except ZXBasicError as e:
        print(f"Spectrum Basic Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
