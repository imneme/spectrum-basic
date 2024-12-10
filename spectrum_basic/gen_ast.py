# Rather than hand-code all the different expression classes, we
# instead generate them programmatically.  The easiest way to do
# this is with eval.
#
# We do do some classes by hand, so if you want to know what kind
# of code this function is making, look at the hand-coded classes
# first.

import re

def gen_ast_classes(output_file):
    def gen_class(name, fields=[], keyword=None, format=None, init=None, is_leaf=False, raw_fields=None, no_parent=False, dont_code=[], xcode="", superclass=None, globals=globals(), locals=locals()):
        """Generate an AST class with given fields"""

        keyword = keyword or name.upper()
        raw_fields = raw_fields or fields
        init = init or [None] * len(fields)
        init = {name: code or raw_name for name, raw_name, code in zip(fields, raw_fields, init)}

        # Note, format of the format string doesn't use `self.` on fields,
        # we add that automagically

        # Format of lines: Nesting of the list of strings is used for indentation
        lines = [f"class {name}({superclass or "ASTNode"}):",
                 [f'"""{name} AST node"""']]
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
            body = [f'"""Return a string representation of a {name} node"""',
                    f"return f\"{format}\""]
            func = [f"def __str__(self):", body]
            lines.append(func)
        if not "walk" in dont_code:
            # Finally, code for the walk method, two kinds of walk methods, leaf
            # and non-leaf
            body = [f'"""Walk method for {name} nodes"""']
            if is_leaf:
                body += [f"yield (Walk.VISITING, self)"]
            else:
                body += [f"if (yield (Walk.ENTERING, self)) == Walk.SKIP: return"]
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
        text = "\n".join(text).strip()
        print(text, file=output_file, end="\n\n")

    gen_class("Let", ["var", "expr"], format="LET {var} = {expr}", superclass="Statement")
    gen_class("For", ["var", "start", "end", "step"], format="FOR {var} = {start} TO {end}{f' STEP {step}' if step else ''}", superclass="Statement")
    gen_class("Next", ["var"], superclass="Statement")
    gen_class("If", ["condition", "statements"], format="IF {condition} THEN {': '.join(str(stmt) for stmt in statements)}", superclass="Statement")
    gen_class("Dim", ["name", "dims"], format="DIM {name}({', '.join(str(d) for d in dims)})", superclass="Statement")
    gen_class("DefFn", ["name", "params", "expr"], format="DEF FN {name}({', '.join(str(p) for p in params)}) = {expr}")
    gen_class("PrintItem", ["value", "sep"], format="{nstr(value)}{nstr(sep)}", no_parent=True)
    gen_class("Rem", ["comment"], is_leaf=True, format="REM {comment}", superclass="Statement")
    gen_class("Label", ["name"], is_leaf=True, format="@{name}", init=["name[1:]"])

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

def gen_ast_py(outputname):
    with open(outputname, "w") as output_file:
        print(f"""#
# This file is automatically generated by gen_ast.py
#
# Do not edit by hand!
#
              
from .ast_base import *
import sys

# Re-export everything from ast_base as if it were defined here
this_module = sys.modules[__name__]
base_module = sys.modules['spectrum_basic.ast_base']
for name in dir(base_module):
    if not name.startswith('_'):  # Skip private/special attributes
        setattr(this_module, name, getattr(base_module, name))

# Automagically generated code for the AST classes
""", file=output_file)
        gen_ast_classes(output_file)
