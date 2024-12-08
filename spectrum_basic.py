from textx import metamodel_from_file
import functools
from os.path import dirname, join

# The ZX Spectrum BASIC Grammar is found in zxbasic.tx

def spectrum_repr(value):
    """Print a value that can be parsed on a ZX Spectrum"""
    # We mostly need to worry about strings, since they need to be 
    # double-quoted and any quotes inside the string need to be escaped
    # BASIC-style
    if isinstance(value, str):
        doubled = value.replace('"', '""')
        return f'"{doubled}"'
    return repr(value)

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


# Classes for the BASIC language

class Statement:
    """Base class for all BASIC statements"""
    def __repr__(self):
        return str(self)

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
            
        present_args = [spectrum_repr(arg) for arg in self.args if arg is not None]
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
            parts.append(self.sep.join(map(spectrum_repr, self.args)))
        return "".join(parts)

class Let(Statement):
    """Assignment statement"""
    def __init__(self, parent, var, expr):
        self.parent = parent
        self.var = var
        self.expr = expr
    
    def __str__(self):
        return f"LET {self.var} = {spectrum_repr(self.expr)}"

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
            return f"FOR {self.var} = {spectrum_repr(self.expr)} TO {spectrum_repr(self.end)} STEP {spectrum_repr(self.step)}"
        return f"FOR {self.var} = {spectrum_repr(self.start)} TO {spectrum_repr(self.end)}"
                                       
class Next(Statement):
    """NEXT statement"""
    def __init__(self, parent, var):
        self.parent = parent
        self.var = var
    
    def __str__(self):
        return f"NEXT {self.var}"

class If(Statement):
    """IF statement with statement list"""
    def __init__(self, parent, condition, statements):
        self.parent = parent
        self.condition = condition
        self.statements = statements
    
    def __str__(self):
        stmts = ": ".join(str(stmt) for stmt in self.statements)
        return f"IF {spectrum_repr(self.condition)} THEN {stmts}"

class Dim(Statement):
    """Array dimension statement"""
    def __init__(self, parent, var, dimensions):
        self.parent = parent
        self.var = var
        self.dimensions = dimensions
    
    def __str__(self):
        dims = ", ".join(str(d) for d in self.dimensions)
        return f"DIM {self.var}({dims})"

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
            return f"DEF FN {self.name}({params}) = {spectrum_repr(self.expr)}"
        return f"DEF FN {self.name} = {spectrum_repr(self.expr)}"

class PrintItem:
    """Represents items in PRINT statement"""
    def __init__(self, value, separator=None):
        # Does not track its parent
        self.value = value
        self.separator = separator
    
    def __str__(self):
        valrepr = spectrum_repr(self.value) if self.value is not None else ""
        if self.separator:
            return f"{valrepr}{self.separator}"
        return valrepr
    
    def __repr__(self):
        return str(self)
    
# Expression classes

class Expression:
    """Base class for all expressions"""
    def __repr__(self):
        return str(self)

class Variable(Expression):
    """Variable reference"""
    def __init__(self, parent, name):
        self.parent = parent
        self.name = name
    
    def __str__(self):
        return self.name

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

# Find spectrum_basic.tx in the same directory as this script
META_PATH = join(dirname(__file__), "spectrum_basic.tx")

# Create meta-model
metamodel = metamodel_from_file(META_PATH, ws='\t ', ignore_case=True, 
                                classes=[Statement, Let, For, Next, If, Dim, DefFn, PrintItem, Variable, BinValue, ArrayRef, Slice])
    
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

if __name__ == '__main__':
    import argparse
    import sys

    # Usage: python zxbasic.py [filename]
    parser = argparse.ArgumentParser(description="Parse a ZX BASIC program")
    parser.add_argument("filename", nargs="?", help="Filename of BASIC program to parse")
    args = parser.parse_args()

    try:
        # # Parse the program
        # model = metamodel.model_from_str(test_program)
        # print("Program parsed successfully!")
        program = parse_file(args.filename)

        # Basic model inspection
        for line in program.lines:
            if line.line_number:
                print(f"{line.line_number}\t", end="")
            else:
                print("\t", end="")
            print(": ".join(str(stmt) for stmt in line.statements))

    except Exception as e:
        print(f"Parse error: {e}")
        sys.exit(1)
