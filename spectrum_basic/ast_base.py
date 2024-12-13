from enum import Enum, auto
from .tokenizer import *
import re

class Walk(Enum):
    ENTERING = auto()
    VISITING = auto()
    LEAVING = auto()
    SKIP = auto()

# The ZX Spectrum BASIC Grammar is found in spectrum_basic.tx

# Operator precedence table (higher number = tighter binding)
BINARY_PRECEDENCE = {
    'OR': 2,
    'AND': 3,
    '=': 5, '<': 5, '>': 5, '<=': 5, '>=': 5, '<>': 5,
    '+': 6, '-': 6,
    '*': 8, '/': 8,
    '^': 10,
}

UNARY_PRECEDENCE = {
    '-': 9,
    'NOT': 4,
}

def precedence(expr):
    """Get the precedence of an operator"""
    if isinstance(expr, BinaryOp):
        return BINARY_PRECEDENCE[expr.op]
    if isinstance(expr, UnaryOp):
        return UNARY_PRECEDENCE[expr.op]
    return 0

def is_complex(expr):
    """Determine if an expression needs parentheses in function context"""
    if isinstance(expr, BinaryOp):
        return True
    # Could add other cases here
    return False

def needs_parens(expr, parent=None, is_rhs=False):
    """Determine if expression needs parentheses based on context"""
    if not isinstance(expr, BinaryOp) and not isinstance(expr, UnaryOp):
        return False

    if parent is None:
        return False

    expr_prec = precedence(expr)
    parent_prec = precedence(parent)
    
    # Different cases where we need parens:
    
    # Lower precedence always needs parens
    if expr_prec < parent_prec:
        return True
        
    # Equal precedence depends on operator and position
    if expr_prec == parent_prec:
        # For subtraction and division, right side always needs parens
        if parent.op in {'-', '/'} and is_rhs:
            return True
        # For power, both sides need parens if same precedence
        if parent.op == '^':
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

def sane_bytes(s):
    """Like bytes, but works on strings without needing encoding"""
    if isinstance(s, str):
        return s.encode('ascii')
    elif s is None:
        return b""
    if hasattr(s, '__iter__'):
        return bjoin(s)
    if isinstance(s, int):
        return s.to_bytes(1, 'big')
    return bytes(s)

def bjoin(items, sep=b""):
    """Join a list of byte sequences (or convertibles) with a separator"""
    return sep.join(sane_bytes(item) for item in items)

def sjoin(items, sep=""):
    """Join a list of strings (or convertibles) with a separator"""
    return sep.join(str(item) for item in items)

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

    def __bytes__(self):
        """Return the in-memory representation of the command"""
        btoken = token_to_byte(self.action)
        bsep = self.sep.strip().encode('ascii')
        present_args = [arg for arg in self.args if arg is not None]
        if self.is_expr:
            if len(self.args) == 1:
                the_arg = self.args[0]
                if is_complex(the_arg):
                    return bjoin([btoken, b'(', the_arg, b')'])
                return bjoin([btoken, the_arg])
            elif len(self.args) == 0:
                return btoken
            else:
                return bjoin([btoken, b'(', bjoin(present_args, sep=bsep), b')'])
        else:
            return bjoin([btoken, bjoin(present_args, sep=bsep)])

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

    def __bytes__(self):
        """Return the in-memory representation of the command"""
        bparts = [token_to_byte(self.action)]
        if self.colours:
            bparts.append(bjoin(self.colours, sep=b";"))
            bparts.append(b";")
        if self.args:
            bparts.append(bjoin(self.args, sep=self.sep.strip().encode('ascii')))
        return bjoin(bparts)
        

def nstr(obj, sep="", none=""):
    "Like str, but returns an empty string for None"
    if obj is None:
        return none
    return f"{obj}{sep}"

def speccy_quote(s):
    """Quote a string in ZX Spectrum BASIC format"""
    doubled = s.replace('"', '""')
    unescaped = escapes_to_unicode(doubled)
    return f'"{unescaped}"'


# Expression classes

class Expression(ASTNode):
    pass

def is_expression(obj):
    """Determine if an object is an expression"""
    return isinstance(obj, Expression) or (isinstance(obj, BuiltIn) and obj.is_expr)
