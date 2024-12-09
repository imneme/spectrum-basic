import textx
from textx import metamodel_from_file
import functools
from os.path import dirname, join
from enum import Enum, auto

class Walk(Enum):
    ENTERING = auto()
    VISITING = auto()
    LEAVING = auto()
    SKIP = auto()

# The ZX Spectrum BASIC Grammar is found in spectrum_basic.tx

# Operator precedence table (higher number = tighter binding)
PRECEDENCE = {
    'OR': 2,
    'AND': 3,
    '=': 5, '<': 5, '>': 5, '<=': 5, '>=': 5, '<>': 5,
    '+': 6, '-': 6,
    '*': 8, '/': 8,
    '^': 10,
}

def is_complex(expr):
    """Determine if an expression needs parentheses in function context"""
    if isinstance(expr, BinaryOp):
        return True
    # Could add other cases here
    return False

def needs_parens(expr, parent_op=None, is_rhs=False):
    """Determine if expression needs parentheses based on context"""
    if not isinstance(expr, BinaryOp):
        return False
        
    expr_prec = PRECEDENCE[expr.op]
    
    if parent_op is None:
        return False
        
    parent_prec = PRECEDENCE[parent_op]
    
    # Different cases where we need parens:
    
    # Lower precedence always needs parens
    if expr_prec < parent_prec:
        return True
        
    # Equal precedence depends on operator and position
    if expr_prec == parent_prec:
        # For subtraction and division, right side always needs parens
        if parent_op in {'-', '/'} and is_rhs:
            return True
        # For power, both sides need parens if same precedence
        if parent_op == '^':
            return True
    
    return False

# Rather than a visitor patter, we use a generator-based approach with
# a walk function that yields “visit events” for each node in the tree

def walk(obj):
    """Handles walking over the AST, but particularly non-AST nodes"""
    if obj is None:
        return
    if isinstance(obj, (list, tuple)):
        for item in obj:
            yield from walk(item)
    elif isinstance(obj, dict):
        for key, value in obj.items():
            yield from walk(value)
    elif isinstance(obj, (str, int, float)):
        yield (Walk.VISITING, obj)
    elif hasattr(obj, "walk"):
        yield from obj.walk()
    # raw AST nodes have a _tx_attrs attribute whose keys are the names of the attributes
    elif hasattr(obj, "_tx_attrs"):
        yield (Walk.VISITING, obj)
        for attr in obj._tx_attrs:
            yield from walk(getattr(obj, attr))
        yield (Walk.LEAVING, obj)
    else:
        yield (Walk.VISITING, obj)

# Classes for the BASIC language

class Statement:
    """Base class for all BASIC statements"""
    def __repr__(self):
        return str(self)
    
    def walk(self):
        """Base walk method for all statements"""
        yield (Walk.VISITING, self)

class BuiltIn(Statement):
    """Represents simple built-in commands with fixed argument patterns"""
    def __init__(self, parent, action, *args, sep=", "):
        self.parent = parent
        self.action = action
        self.args = args
        self.is_expr = False
        self.sep = sep
    
    def __str__(self):
        if not self.args:
            return self.action

        present_args = [str(arg) for arg in self.args if arg is not None]
        if self.is_expr:
            if len(present_args) == 1:
                # For single argument function-like expressions, only add parens if needed
                arg_str = present_args[0]
                if is_complex(self.args[0]):
                    return f"{self.action} ({arg_str})"
                return f"{self.action} {arg_str}"
            elif len(present_args) == 0:
                return f"{self.action}"
            else:
                return f"{self.action}({self.sep.join(present_args)})"
        else:
            return f"{self.action} {self.sep.join(present_args)}"
        
    def walk(self):
        """Walk method for built-in commands"""
        if (yield (Walk.ENTERING, self)) == Walk.SKIP: return
        yield from walk(self.args)
        yield (Walk.LEAVING, self)

class ColouredBuiltin(BuiltIn):
    """Special case for commands that can have colour parameters"""
    def __init__(self, parent, action, colours, *args):
        super().__init__(parent, action, *args)
        self.colours = colours or []
    
    def __str__(self):
        parts = [self.action]
        if self.colours:
            colour_strs = [str(c) for c in self.colours]
            parts.append(" ")
            parts.append("; ".join(colour_strs))
            parts.append(";")
        if self.args:
            if self.colours:
                parts.append(" ")
            parts.append(self.sep.join(map(str, self.args)))
        return "".join(parts)

    def walk(self):
        """Walk method for coloured built-in commands"""
        if (yield (Walk.ENTERING, self)) == Walk.SKIP: return
        yield from walk(self.colours)
        yield from walk(self.args)
        yield (Walk.LEAVING, self)

class Let(Statement):
    """Assignment statement"""
    def __init__(self, parent, var, expr):
        self.parent = parent
        self.var = var
        self.expr = expr
    
    def __str__(self):
        return f"LET {self.var} = {self.expr}"
    
    def walk(self):
        """Walk method for LET statements"""
        if (yield (Walk.ENTERING, self)) == Walk.SKIP: return
        yield from walk(self.var)
        yield from walk(self.expr)
        yield (Walk.LEAVING, self)

class For(Statement):
    """FOR loop statement"""
    def __init__(self, parent, var, start, end, step=None):
        self.parent = parent
        self.var = var
        self.start = start
        self.end = end
        self.step = step
    
    def __str__(self):
        if self.step:
            return f"FOR {self.var} = {self.expr} TO {self.end} STEP {self.step}"
        return f"FOR {self.var} = {self.start} TO {self.end}"
    
    def walk(self):
        """Walk method for FOR statements"""
        if (yield (Walk.ENTERING, self)) == Walk.SKIP: return
        yield from walk(self.var)
        yield from walk(self.start)
        yield from walk(self.end)
        if self.step:
            yield from walk(self.step)
        yield (Walk.LEAVING, self)
                                       
class Next(Statement):
    """NEXT statement"""
    def __init__(self, parent, var):
        self.parent = parent
        self.var = var
    
    def __str__(self):
        return f"NEXT {self.var}"
    
    def walk(self):
        """Walk method for NEXT statements"""
        if (yield (Walk.ENTERING, self)) == Walk.SKIP: return
        yield from walk(self.var)
        yield (Walk.LEAVING, self)

class If(Statement):
    """IF statement with statement list"""
    def __init__(self, parent, condition, statements):
        self.parent = parent
        self.condition = condition
        self.statements = statements
    
    def __str__(self):
        stmts = ": ".join(str(stmt) for stmt in self.statements)
        return f"IF {self.condition} THEN {stmts}"
    
    def walk(self):
        """Walk method for IF statements"""
        if (yield (Walk.ENTERING, self)) == Walk.SKIP: return
        yield from walk(self.condition)
        yield from walk(self.statements)
        yield (Walk.LEAVING, self)

class Dim(Statement):
    """Array dimension statement"""
    def __init__(self, parent, name, dims):
        self.parent = parent
        self.name = name
        self.dimensions = dims
    
    def __str__(self):
        dims = ", ".join(str(d) for d in self.dimensions)
        return f"DIM {self.name}({dims})"
    
    def walk(self):
        """Walk method for DIM statements"""
        if (yield (Walk.ENTERING, self)) == Walk.SKIP: return
        yield from walk(self.name)
        yield from walk(self.dimensions)
        yield (Walk.LEAVING, self)

class DefFn(Statement):
    """Function definition"""
    def __init__(self, parent, name, params, expr):
        self.parent = parent
        self.name = name
        self.params = params or []
        self.expr = expr
    
    def __str__(self):
        if self.params:
            params = ", ".join(str(p) for p in self.params)
            return f"DEF FN {self.name}({params}) = {self.expr}"
        return f"DEF FN {self.name} = {self.expr}"
    
    def walk(self):
        """Walk method for DEF FN statements"""
        if (yield (Walk.ENTERING, self)) == Walk.SKIP: return
        yield from walk(self.params)
        yield from walk(self.expr)
        yield (Walk.LEAVING, self)

class PrintItem:
    """Represents items in PRINT statement"""
    def __init__(self, value, separator=None):
        # Does not track its parent
        self.value = value
        self.separator = separator
    
    def __str__(self):
        valrepr = str(self.value) if self.value is not None else ""
        if self.separator:
            return f"{valrepr}{self.separator}"
        return valrepr
    
    def __repr__(self):
        return str(self)
    
    def walk(self):
        """Walk method for PRINT items"""
        if (yield (Walk.ENTERING, self)) == Walk.SKIP: return
        yield from walk(self.value)
        yield from walk(self.separator)
        yield (Walk.LEAVING, self)

class Rem(Statement):
    """REM statement"""
    def __init__(self, parent, comment):
        self.parent = parent
        self.comment = comment
    
    def __str__(self):
        return f"REM {self.comment}"

class Label(Statement):  # Arguably not really a statement
    """Label for GOTO/GOSUB"""
    def __init__(self, parent, name):
        self.parent = parent
        self.name = name[1:]
    
    def __str__(self):
        return f"@{self.name}"

# Expression classes

class Expression:
    """Base class for all expressions"""
    def __repr__(self):
        return str(self)
    
    def walk(self):
        """Base walk method for all expressions"""
        yield (Walk.VISITING, self)

class Variable(Expression):
    """Variable reference"""
    def __init__(self, parent, name):
        self.parent = parent
        self.name = name.replace(" ", "").replace("\t", "")
    
    def __str__(self):
        return self.name

class Number(Expression):
    """Numeric value"""
    def __init__(self, parent, value):
        self.parent = parent
        self.value = value
    
    def __str__(self):
        return str(self.value)

class String(Expression):
    """String value"""
    def __init__(self, parent, value):
        self.parent = parent
        self.value = value[1:-1]
    
    def __str__(self):
        doubled = self.value.replace('"', '""')
        return f'"{doubled}"'

class BinValue(Expression):
    """Binary value"""
    def __init__(self, parent, digits):
        self.parent = parent
        self.digits = digits
    
    def __str__(self):
        return f"BIN {self.digits}"

class ArrayRef(Expression):
    """Array reference"""
    def __init__(self, parent, name, subscripts):
        self.parent = parent
        self.name = name
        self.subscripts = subscripts
    
    def __str__(self):
        subscripts = ", ".join(str(s) for s in self.subscripts)
        return f"{self.name}({subscripts})"
    
    def walk(self):
        """Walk method for array references"""
        if (yield (Walk.ENTERING, self)) == Walk.SKIP: return
        yield from walk(self.name)
        yield from walk(self.subscripts)
        yield (Walk.LEAVING, self)

# Fn: 'FN' name=Variable '(' args*=Expression[','] ')';

class Fn(Expression):
    """FN call (to function defined with DEF FN)"""
    def __init__(self, parent, name, args):
        self.parent = parent
        self.name = name
        self.args = args
    
    def __str__(self):
        arg_strs = ", ".join(str(arg) for arg in self.args)
        return f"FN {self.name}({arg_strs})"
    
    def walk(self):
        """Walk method for FN calls"""
        if (yield (Walk.ENTERING, self)) == Walk.SKIP: return
        yield from walk(self.name)
        yield from walk(self.args)
        yield (Walk.LEAVING, self)

class Slice(Expression):
    """Array slice expression"""
    def __init__(self, parent, min=None, max=None):
        self.parent = parent
        self.min = min
        self.max = max
    
    def __str__(self):
        if self.min is None:
            return f"TO {self.max}"
        if self.max is None:
            return f"{self.min} TO"
        return f"{self.min} TO {self.max}"
    
    def walk(self):
        """Walk method for array slices"""
        if (yield (Walk.ENTERING, self)) == Walk.SKIP: return
        yield from walk(self.min)
        yield from walk(self.max)
        yield (Walk.LEAVING, self)

class BinaryOp(Expression):
    """Binary operation with smart string formatting"""
    def __init__(self, op, left, right):
        self.op = op
        self.lhs = left
        self.rhs = right
    
    def __str__(self):
        # Format left side
        lhs_str = str(self.lhs)
        if isinstance(self.lhs, BinaryOp) and needs_parens(self.lhs, self.op, False):
            lhs_str = f"({lhs_str})"
            
        # Format right side
        rhs_str = str(self.rhs)
        if isinstance(self.rhs, BinaryOp) and needs_parens(self.rhs, self.op, True):
            rhs_str = f"({rhs_str})"
            
        return f"{lhs_str} {self.op} {rhs_str}"
    
    def walk(self):
        """Walk method for binary operations"""
        if (yield (Walk.ENTERING, self)) == Walk.SKIP: return
        yield from walk(self.lhs)
        yield from walk(self.rhs)
        yield (Walk.LEAVING, self)

# Find spectrum_basic.tx in the same directory as this script
META_PATH = join(dirname(__file__), "spectrum_basic.tx")

# Create meta-model
metamodel = metamodel_from_file(META_PATH, ws='\t ', ignore_case=True, 
                                classes=[Statement, Let, For, Next, If, Dim, DefFn, PrintItem, Variable, BinValue, ArrayRef, Fn, Slice, Number, String, Rem, Label])
    
def get_name(obj):
    """Get the name of an AST object"""
    return obj.name if hasattr(obj, "name") else obj.__class__.__name__.upper()

def ap_arg0(obj):
    """Object processor for zero-argument commands"""
    # Handle object being a simple type
    if isinstance(obj, str):
        return BuiltIn(None, obj)
    return BuiltIn(obj.parent, get_name(obj))

def ap_arg1(obj):
    """Object processor for one-argument commands"""
    if obj.expr is not None:
        return BuiltIn(obj.parent, get_name(obj), obj.expr)
    else:
        return BuiltIn(obj.parent, get_name(obj))

def ap_arg2(obj):
    """Object processor for two-argument commands"""
    return BuiltIn(obj.parent, get_name(obj), obj.expr1, obj.expr2)

def ap_arg3(obj):
    """Object to turn three-argument commands into Action objects"""
    return BuiltIn(obj.parent, get_name(obj), obj.expr1, obj.expr2, obj.expr3)

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
    "New": ap_arg0,
    "Stop": ap_arg0,
    "Return": ap_arg0,
    "Continue": ap_arg0,
    "Copy": ap_arg0,
    "Cls": ap_arg0,
    "Cat": ap_arg0,
    # 1-argument commands
    "Goto": ap_arg1,
    "Gosub": ap_arg1,
    "Restore": ap_arg1,
    "Pause": ap_arg1,
    "Border": ap_arg1,
    "Run": ap_arg1,
    "Clear": ap_arg1,
    "Randomize": ap_arg1,
    "ColourParam": ap_arg1,
    # 2-argument commands
    "Beep": ap_arg2,
    "Out": ap_arg2,
    "Poke": ap_arg2,
    "Plot": ap_coloured,
    # 3-argument commands
    "Draw": ap_coloured,
    "Circle": ap_coloured,
    # PRINT-like statements
    "Print": ap_print_like,
    "Lprint": ap_print_like,
    "Input": ap_print_like,
    # 1-argument print-modifiers
    "Tab": ap_arg1,
    "Line": ap_arg1,
    # 2-argument print-modifiers
    "At": ap_arg2,
    # 0-arity functions
    "PiValue": ap_arg0,
    # 1-arity functions
    "Function": ap_expr(ap_arg1),
    # 2-arity functions
    "TwoArgFn": ap_expr(ap_arg2),
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
    print(remapping)
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

if __name__ == '__main__':
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
