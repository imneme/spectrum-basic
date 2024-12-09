from enum import Enum, auto
import re

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

class ASTNode:
    """Base class for all (non-textx) AST nodes"""
    def __repr__(self):
        return str(self)
    
    def walk(self):
        """Base walk method for all expressions"""
        yield (Walk.VISITING, self)

# Rather than hand-code all the different expression classes, we
# instead generate them programmatically.  The easiest way to do
# this is with eval.
#
# We do do some classes by hand, so if you want to know what kind
# of code this function is making, look at the hand-coded classes
# first.

def gen_class(name, fields=[], keyword=None, format=None, init=None, is_leaf=False, raw_fields=None, no_parent=False, dont_code=[], xcode="", superclass=None, globals=globals(), locals=locals()):
    """Generate an AST class with given fields"""

    keyword = keyword or name.upper()
    raw_fields = raw_fields or fields
    init = init or [None] * len(fields)
    init = {name: code or raw_name for name, raw_name, code in zip(fields, raw_fields, init)}

    # Note, format of the format string doesn't use `self.` on fields,
    # we add that automagically

    # Format of lines: Nesting of the list of strings is used for indentation
    lines = [f"class {name}({superclass or "ASTNode"}):"]
    if not "__init__" in dont_code:
        # First, code for the __init__ method
        body = [] if no_parent else [f"self.parent = parent"]
        body += [f"self.{field} = {init[field]}" for field in fields]
        func = [f"def __init__(self{'' if no_parent else ', parent'}, {', '.join(raw_fields)}):", body]
        lines.append(func)
    if not "__str__" in dont_code:
        # Then, code for the __str__ method
        if format is None:   # Create with fields (without self)
            format = f"{keyword} {' '.join(['{' + f + '}' for f in fields])}"
        # Fix the format to add self. to each field
        format = re.sub(r"\b(" + "|".join(fields) + r")\b", r"self.\1", format)
        body = [f"return f\"{format}\""]
        func = [f"def __str__(self):", body]
        lines.append(func)
    if not "__repr__" in dont_code:
        # Finally, code for the walk method, two kinds of walk methods, leaf
        # and non-leaf
        if is_leaf:
            body = [f"yield (Walk.VISITING, self)"]
        else:
            body = [f"if (yield (Walk.ENTERING, self)) == Walk.SKIP: return"]
            body += [f"yield from walk(self.{field})" for field in fields]
            body.append(f"yield (Walk.LEAVING, self)")
        func = [f"def walk(self):", body]
        lines.append(func)

    if xcode:
        lines.append(xcode)
    text = []
    def flatten(lst, indent=0):
        for item in lst:
            if isinstance(item, list):
                flatten(item, indent+1)
            else:
                text.append("    " * indent + item)
    flatten(lines)
    text = "\n".join(text)
    exec(text, globals, locals)

class Statement(ASTNode):
    """Base class for all BASIC statements"""
    pass

class BuiltIn(Statement):
    """Represents simple built-in commands with fixed argument patterns"""
    def __init__(self, parent, action, *args, sep=", "):
        self.parent = parent
        self.action = action.upper()
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

def nstr(obj):
    "Like str, but returns an empty string for None"
    return str(obj) if obj is not None else ""

def speccy_quote(s):
    """Quote a string in ZX Spectrum BASIC format"""
    doubled = s.replace('"', '""')
    return f'"{doubled}"'

gen_class("Let", ["var", "expr"], format="LET {var} = {expr}", superclass="Statement")
gen_class("For", ["var", "start", "end", "step"], format="FOR {var} = {start} TO {end}{f' STEP {step}' if step else ''}", superclass="Statement")
gen_class("Next", ["var"], superclass="Statement")
gen_class("If", ["condition", "statements"], format="IF {condition} THEN {': '.join(str(stmt) for stmt in statements)}", superclass="Statement")
gen_class("Dim", ["name", "dims"], format="DIM {name}({', '.join(str(d) for d in dims)})", superclass="Statement")
gen_class("DefFn", ["name", "params", "expr"], format="DEF FN {name}({', '.join(str(p) for p in params)}) = {expr}")
gen_class("PrintItem", ["value", "sep"], format="{nstr(value)}{nstr(sep)}", no_parent=True)
gen_class("Rem", ["comment"], is_leaf=True, format="REM {comment}", superclass="Statement")
gen_class("Label", ["name"], is_leaf=True, format="@{name}", init=["name[1:]"])

# Expression classes

class Expression(ASTNode):
    pass

gen_class("Variable", ["name"], is_leaf=True, init=["name.replace(' ', '').replace('\\t', '')"], format="{name}", superclass="Expression")
gen_class("Number", ["value"], format="{value}", is_leaf=True, superclass="Expression")
gen_class("String", ["value"], format="{speccy_quote(value)}", is_leaf=True, init=["value[1:-1]"], superclass="Expression")
gen_class("BinValue", ["digits"], keyword="BIN", is_leaf=True)
gen_class("ArrayRef", ["name", "subscripts"], format="{name}({', '.join(str(s) for s in subscripts)})")
gen_class("Fn", ["name", "args"], format="FN {name}({', '.join(str(arg) for arg in args)})")
gen_class("Slice", ["min", "max"], dont_code=["__str__"], xcode="""
    def __str__(self):
        if self.min is None:
            return f"TO {self.max}"
        if self.max is None:
            return f"{self.min} TO"
        return f"{self.min} TO {self.max}"
""")
gen_class("BinaryOp", ["op", "lhs", "rhs"], no_parent=True,dont_code="__str__", xcode="""
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
""")

gen_class("ChanSpec", ["chan"], format="#{chan}")
