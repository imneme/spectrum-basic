# Rather than hand-code all the different expression classes, we
# instead generate them programmatically.  The easiest way to do
# this is with eval.
#
# We do do some classes by hand, so if you want to know what kind
# of code this function is making, look at the hand-coded classes
# first.

import re
import os

def intersperse(val, sequence):
    for i, item in enumerate(sequence):
        if i != 0:
            yield val
        yield item

def gen_ast_classes(output_file):
    def gen_class(name, fields=[], keyword=None, format=None, bsep="b','", bytescode=None, init=None, is_leaf=False, raw_fields=None, no_parent=False, no_token=False, dont_code=[], xcode="", superclass=None, globals=globals(), locals=locals()):
        """Generate an AST class with given fields"""

        keyword = keyword or name.upper()
        raw_fields = raw_fields or fields
        init = init or [None] * len(fields)
        init = {name: code or raw_name for name, raw_name, code in zip(fields, raw_fields, init)}

        # Note, format of the format string doesn't use `self.` on fields,
        # we add that automagically

        # Format of lines: Nesting of the list of strings is used for indentation
        lines = [f"class {name}({superclass or 'ASTNode'}):",
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
        if not "__bytes__" in dont_code:
            # Next, code for the __bytes__ method
            btoken = f"token_to_byte({keyword!r})"
            if bytescode is None:
                bpieces = (f'self.{field}' for field in fields)
                if bsep is not None:
                    bpieces = intersperse(bsep, bpieces) 
                bytescode = f"bjoin([{', '.join(bpieces)}])"
            else:
                bytescode = re.sub(r"\b(" + "|".join(fields) + r")\b", r"self.\1", bytescode)
                bytescode = f"bjoin({bytescode})"
            body = [f'"""Return the in-memory representation of a {name} node"""']
            if no_token:
                body.append(f"return {bytescode}")
            else:
                body.append(f"return {btoken} + {bytescode}")
            func = [f"def __bytes__(self):", body]
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

    gen_class("Program", ["lines"], format="{chr(10).join(str(line) for line in lines)}",
                bytescode="[bjoin(lines)]", no_parent=True, no_token=True)

    gen_class("SourceLine", ["line_number", "label", "statements", "after"], 
              bytescode="[line_to_bytes(line_number, bjoin(statements, b':')), bjoin(after)]",
              no_token=True, dont_code=["__str__"], xcode="""
    def __str__(self):
        str_statements = ": ".join(str(stmt) for stmt in self.statements)
        after = sjoin(self.after)
        if self.line_number and self.label:
            return f"{self.line_number} {self.label}: {str_statements}{after}"
        elif self.line_number:
            return f"{self.line_number}\t{str_statements}{after}"
        elif self.label:
            return f"{self.label}:{'\t' if len(self.label.name) < 6 else ' '}{str_statements}{after}"
        return f"\t{str_statements}"
""")
    gen_class("CommentLine", ["char", "comment"], format="{char}{comment}", bytescode="[]", no_token=True, is_leaf=True)
    gen_class("JankyStatement", ["before", "actual", "after"], no_token=True, superclass="Statement",
              format="{sjoin(junk)}{nstr(actual)}{sjoin(after)}",
              bytescode="[echars_to_bytes(j) for j in before] + [actual] + [echars_to_bytes(j) for j in after]")
    gen_class("JankyFunctionExpr", ["before", "actual", "after"], no_token=True, superclass="Statement",
              format="{sjoin(junk)}{nstr(actual)}{sjoin(after)}",
              bytescode="[echars_to_bytes(j) for j in before] + [actual] + [echars_to_bytes(j) for j in after]")

    gen_class("Let", ["var", "expr"], 
              format="LET {var} = {expr}",
              bytescode="[var, b'=', expr]",
              superclass="Statement")
    gen_class("For", ["var", "start", "end", "step"], 
              format="FOR {var} = {start} TO {end}{f' STEP {step}' if step else ''}",
              bytescode = "[var, b'=', start, token_to_byte('TO'), end] + ([token_to_byte('STEP'), step] if step else [])",
              superclass="Statement")
    gen_class("Next", ["var"], superclass="Statement")
    gen_class("If", ["condition", "statements", "after"], 
              format="IF {condition} THEN {': '.join(str(stmt) for stmt in statements)}{sjoin(after)}",
              bytescode="[condition, token_to_byte('THEN'), bjoin(statements, sep=b':'), bjoin(after)]",
              superclass="Statement")
    gen_class("Dim", ["name", "dims"], 
              format="DIM {name}({', '.join(str(d) for d in dims)})",
              bytescode="[name, b'(', bjoin(dims, sep=b','), b')']",
              superclass="Statement")
    gen_class("Data", ["items"], format="DATA {', '.join(str(v) for v in items)}", bytescode="[bjoin(items,sep=b',')]", superclass="Statement")
    gen_class("Read", ["vars"], format="READ {', '.join(str(v) for v in vars)}", bytescode="[bjoin(vars, sep=b',')]", superclass="Statement")
    gen_class("DefFn", ["name", "params", "expr"], 
              format="DEF FN {name}({', '.join(str(p) for p in params)}) = {expr}", keyword="DEF FN",
              bytescode="[name, b'(', bjoin([sane_bytes(p) + bytes((14,0,0,0,0,0)) for p in params], sep=b','), b')=', expr]",
              superclass="Statement")
    gen_class("PrintItem", ["value", "sep"], format="{nstr(value)}{nstr(sep)}", no_parent=True, no_token=True, bsep=None)
    gen_class("Rem", ["comment"], is_leaf=True, format="REM {comment}", 
              bytescode="[echars_to_bytes(comment)]", superclass="Statement")
    gen_class("Label", ["name"], is_leaf=True, format="@{name}", init=["name[1:]"])

    gen_class("Variable", ["name"], is_leaf=True, init=["name.replace(' ', '').replace('\\t', '')"], format="{name}", superclass="Expression", no_token=True)
    gen_class("Number", ["value"], format="{value}", is_leaf=True, superclass="Expression", no_token=True,
              bytescode="[num_to_bytes(value)]")
    gen_class("String", ["value"], format="{speccy_quote(value)}", is_leaf=True, init=["value[1:-1]"], superclass="Expression", no_token=True,
                bytescode="[strlit_to_bytes(value)]")
    gen_class("BinValue", ["digits"], keyword="BIN", is_leaf=True)
    gen_class("ArrayRef", ["name", "subscripts"], format="{name}({', '.join(str(s) for s in subscripts)})",
              bytescode="[name, b'(', bjoin(subscripts, sep=b','), b')']", no_token=True, superclass="Expression")
    gen_class("Fn", ["name", "args"], format="FN {name}({', '.join(str(arg) for arg in args)})",
              bytescode="[name, b'(', bjoin(args, sep=b','), b')']", superclass="Expression")
    gen_class("Slice", ["min", "max"], dont_code=["__str__","__bytes__"], xcode="""
    def __str__(self):
        if self.min is None:
            return f"TO {self.max}"
        if self.max is None:
            return f"{self.min} TO"
        return f"{self.min} TO {self.max}"
    def __bytes__(self):
        bto = token_to_byte('TO')
        if self.min is None:
            return bjoin([bto, self.max])
        if self.max is None:
            return bjoin([self.min, bto])
        return bjoin([self.min, bto, self.max])
""")
    gen_class("StringSubscript", ["expr", "index"], 
              format="{expr if isinstance(expr, String) else '(' + str(expr) + ')'}({index})",
              bytescode="([expr] if isinstance(expr, String) else [b'(', expr, b')']) + [b'(', index, b')']",
              no_token=True, no_parent=True, superclass="Expression")
    
    gen_class("BinaryOp", ["op", "lhs", "rhs"], no_parent=True, dont_code=["__str__", "__bytes__"], superclass="Expression", xcode="""
    def __str__(self):
        # Format left side
        lhs_str = str(self.lhs)
        if (isinstance(self.lhs, BinaryOp) or isinstance(self.lhs, UnaryOp)) and needs_parens(self.lhs, self, False):
            lhs_str = f"({lhs_str})"
            
        # Format right side
        rhs_str = str(self.rhs)
        if isinstance(self.rhs, BinaryOp) and needs_parens(self.rhs, self, True):
            rhs_str = f"({rhs_str})"
            
        return f"{lhs_str} {self.op} {rhs_str}"
    def __bytes__(self):
        bop = token_to_byte(self.op)
        # Format left side
        blhs = bytes(self.lhs)
        if isinstance(self.lhs, BinaryOp) and needs_parens(self.lhs, self, False):
            blhs = b'(' + blhs + b')'
        
        # Format right side
        brhs = bytes(self.rhs)
        if isinstance(self.rhs, BinaryOp) and needs_parens(self.rhs, self, True):
            brhs = b'(' + brhs + b')'
        
        return blhs + bop + brhs
""")
    gen_class("UnaryOp", ["op", "expr"], dont_code=["__str__", "__bytes__"], superclass="Expression", xcode="""
    def __str__(self):
        expr_str = str(self.expr)
        if isinstance(self.expr, BinaryOp) and needs_parens(self.expr, self, False):
            expr_str = f"({expr_str})"
        # whether to add a space after the operator depends on whehter it is a symbol
        # like - or a keyword like NOT
        spacer = ' ' if self.op.isalpha() else ''
        return f"{self.op}{spacer}{expr_str}"
    def __bytes__(self):
        bop = token_to_byte(self.op)
        bexpr = bytes(self.expr)
        if isinstance(self.expr, BinaryOp) and needs_parens(self.expr, self, False):
            bexpr = b'(' + bexpr + b')'
        return bop + bexpr
""")
    
    print("""
class Not(UnaryOp):
    pass

class Neg(UnaryOp):
    pass
""", file=output_file)

    gen_class("ChanSpec", ["chan"], format="#{chan}", no_token=True,
                bytescode="[b'#', chan]")
    gen_class("Colons", ["colons"], format="{colons}", no_token=True, is_leaf=True, bytescode="[safe_bytes(colons)]", superclass="Statement")

def gen_ast_py(outputname):
    with open(outputname, "w") as output_file:
        print(f"""#
# This file is automatically generated by gen_ast.py
#
# Do not edit by hand!
#
              
# First, we have a complete textual copy of ast_base.py (which should never
# be imported directly in normal use)
              
# ----- Start of ast_base.py copy -----
""", file=output_file)
        base_path = os.path.join(os.path.dirname(__file__), "ast_base.py")
        with open(base_path) as base_file:
            print(base_file.read(), file=output_file)
        print(f"""
# ----- End of ast_base.py copy -----
              
# Automagically generated code for the AST classes
""", file=output_file)
        gen_ast_classes(output_file)
