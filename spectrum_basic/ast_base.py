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


# Expression classes

class Expression(ASTNode):
    pass

