from .ast import *
import bisect
import math
import random

# Run ZX Spectrum BASIC code

def is_stringvar(var):
    """Check if a variable is a string variable"""
    return var.endswith("$")

def find_deffns(prog):
    """Find all the DEF FN functions in the program"""
    map = {}
    for event, node in walk(prog):
        if event != Walk.ENTERING:
            continue
        if isinstance(node, DefFn):
            if node.name in map:
                raise ValueError(f"Function {node.name} already defined")
            map[node.name.lower()] = node
    return map

class ProgramData:
    """Data for a ZX Spectrum BASIC program"""
    def __init__(self, prog):
        # call static method to gather the data
        data, index = self._gather_data(prog)
        self.data = data
        self.indexForLine = index
        self.line_numbers = sorted(index.keys())
        self.index = 0
    
    @staticmethod
    def _gather_data(prog):
        data = []
        current_line = 0
        dataIndex = {0: 0} # maps line numbers to positions in the global list of data items
        for event, node in walk(prog):
            if event != Walk.ENTERING:
                continue
            match node:
                case SourceLine(line_number=ln):
                    if ln:
                        current_line = ln
                case Data(items=items):
                    dataIndex.setdefault(current_line, len(data))
                    data.extend(items)
        return data, dataIndex

    
    def restore(self, line_number=0):
        """Restore the data index to the start of a line"""
        if (index := self.indexForLine.get(line_number)) is not None:
            self.index = index
            return
        # Need to find the next line number
        line_index = bisect.bisect_left(self.line_numbers, line_number)
        if line_index == len(self.line_numbers):
            self.index = len(self.data)
            return
        next_line = self.line_numbers[line_index]
        self.index = self.indexForLine[next_line]
    
    def next(self):
        """Get the next data item"""
        if self.index >= len(self.data):
            raise ValueError("Out of DATA")
        value = self.data[self.index]
        self.index += 1
        return value

class Environment:
    """Environment for running ZX Spectrum BASIC programs"""
    def __init__(self, lines_map, functions={}, data=None):
        self.vars = {}
        self.array_vars = {}
        self.functions = functions
        self.lines_map = lines_map
        self.gosub_stack = []
        self.data = data

    def let_var(self, var, value):
        """Set a variable"""
        if not isinstance(var, str):
            raise ValueError(f"Variable name {var} is not a string")
        var = var.lower()
        self.vars.setdefault(var, {})['value'] = value

    def for_loop(self, var, line_idx, stmt_idx, start, end, step):
        """Start a FOR loop"""
        if not isinstance(var, str):
            raise ValueError(f"Variable name {var} is not a string")
        var = var.lower()
        self.vars[var] = {
            'value': start,
            'end': end,
            'step': step,
            'line_idx': line_idx,
            'stmt_idx': stmt_idx,
        }

    def get_fn(self, name):
        """Get a function"""
        if not isinstance(name, str):
            raise ValueError(f"Function name {name} is not a string")
        try:
            name = name.lower()
            return self.functions[name]
        except KeyError as e:
            raise ValueError(f"Function {name} not defined") from e

    def get_var(self, var):
        """Get a variable"""
        if not isinstance(var, str):
            raise ValueError(f"Variable name {var} is not a string")
        try:
            var = var.lower()
            return self.vars[var]['value']
        except KeyError as e:
            raise ValueError(f"Variable {var} not defined in {self.vars}") from e
    
    def save_var(self, var):
        """Save a variable (on an internal per-variable stack)"""
        if not isinstance(var, str):
            raise ValueError(f"Variable name {var} is not a string")
        var = var.lower()
        dict = self.vars.get(var)
        self.vars[var] = {"stashed": dict}

    def restore_var(self, var):
        """Restore a variable (from an internal per-variable stack)"""
        if not isinstance(var, str):
            raise ValueError(f"Variable name {var} is not a string")
        try:
            var = var.lower()
            dict = self.vars[var].pop("stashed")
            if dict is None:
                del self.vars[var]
            else:
                self.vars[var] = dict
        except KeyError as e:
            raise ValueError(f"No stashed value for variable {var}") from e
        
    def get_var_all(self, var):
        """Get all the information about a variable"""
        try:
            var = var.lower()
            return self.vars[var]
        except KeyError as e:
            raise ValueError(f"Variable {var} not defined") from e
            
    def dim(self, var, *dims):
        """Create an array"""
        pass # TODO

    def get_array(self, var, *indices):
        """Get an array element"""
        pass # TODO

    def set_array(self, var, value, *indices):
        """Set an array element"""
        pass # TODO

    def gosub_push(self, line_idx, stmt_idx):
        """Push a GOSUB return address"""
        self.gosub_stack.append((line_idx, stmt_idx))

    def gosub_pop(self):
        """Pop a GOSUB return address"""
        try:
            return self.gosub_stack.pop()
        except IndexError as e:
            raise ValueError("RETURN without GOSUB") from e

class LineMapper:
    """Map line numbers lists of statements"""
    def __init__(self, lines):
        self.lines = {}
        for i, line in enumerate(lines):
            # Only include lines that actually have a line_number
            if line.line_number:
                self.lines[line.line_number] = i
        self.line_numbers = sorted(self.lines.keys())

    def nearest_line(self, line_number):
        """Get the nearest line number"""
        if not self.line_numbers and line_number == 0:
            return 0
        i = bisect.bisect_left(self.line_numbers, line_number)
        if i == len(self.line_numbers):
            return None
        return self.line_numbers[i]
    
    def get_index(self, line_number):
        """Get the index of a line number, if not found return the index of the next line"""
        if (i := self.lines.get(line_number)) is not None:
            return i
        if line_number == 0:
            return 0
        # Not in the list, so find the the actual next line after line_number
        i = bisect.bisect_left(self.line_numbers, line_number)
        if i == len(self.line_numbers):
            return None
        return self.lines[self.line_numbers[i]]

def flattened_statements(statements):
    """Flatten a line of statements to handle IF statements"""
    for stmt in statements:
        match stmt:
            case If(condition=cond, statements=stmts, parent=parent, after=after):
                yield If(condition=cond, statements=[], parent=parent, after=None)
                yield from flattened_statements(stmts)
            case _:
                yield stmt

def run_program(prog : Program, start=0):
    """Run a ZX Spectrum BASIC program"""
    # Set up the environment
    prog_lines = [line for line in prog.lines if not isinstance(line, CommentLine)]
    if not prog_lines:
        raise ValueError("Empty program (no non-meta-comment lines)")
    lines_map = LineMapper(prog_lines)
    lines = [list(flattened_statements(line.statements)) for line in prog_lines]
    functions = find_deffns(prog)
    env = Environment(lines_map, functions, data=ProgramData(prog))
    # Run the program
    line_idx, stmt_idx = (lines_map.get_index(start), 0) if start else (0, 0)
    while line_idx is not None and line_idx < len(lines):
        stmts = lines[line_idx]
        where = run_stmts(env, stmts, line_idx, stmt_idx)
        line_idx, stmt_idx = where if where is not None else (line_idx + 1, 0)
    return env

def run_stmts(env, stmts, line_idx=0, stmt_idx=0):
    """Run a list of statements"""
    for i in range(stmt_idx, len(stmts)):
        stmt = stmts[i]
        jump = run_stmt(env, stmt, line_idx, i)
        if jump is not None:
            return jump
    return None

def run_let(env, vardest, expr):
    """Run a LET statement"""
    value = run_expr(env, expr)
    match vardest:
        case Variable(name=v):
            env.let_var(v, value)
        case ArrayRef(name=v, subscripts=subs):
            raise ValueError("Arrays not supported yet")

def run_stmt(env, stmt, line_idx, stmt_idx):
    """Run a single statement"""
    match stmt:
        case Let(var=vardest, expr=expr):
            run_let(env, vardest, expr)
        # Special case for GOSUB as it needs to push the return address
        case BuiltIn(action="GOSUB", args=args):
            if len(args) != 1:
                raise ValueError("GOSUB requires exactly one argument")
            env.gosub_push(line_idx, stmt_idx+1)
            return (env.lines_map.get_index(run_expr(env, args[0])), 0)
        case BuiltIn(action=action, args=args):
            handler = BUILTIN_MAP.get(action)
            if handler is None:
                raise ValueError(f"The {action} command is not supported")
            return handler(env, args)
        case For(var=Variable(name=v), start=start, end=end, step=step):
            env.for_loop(v, line_idx, stmt_idx+1, run_expr(env, start), run_expr(env, end), run_expr(env, step) if step is not None else 1)
        case Next(var=Variable(name=v)):
            var_info = env.get_var_all(v)
            var_info['value'] += var_info['step']
            if var_info['value'] <= var_info['end']:
                return (var_info['line_idx'], var_info['stmt_idx'])
        case If(condition=cond, statements=stmts):
            if run_expr(env, cond):
                return # Keep executing the line
            # Skip the rest of the statements, move to the next line
            return (line_idx+1, 0)
        case Read(vars=vars):
            run_read(env, vars)
        case Rem():
            pass # Comments are ignored
        case Data():
            pass # Data is handled by the ProgramData class
        case DefFn():
            pass # Functions are found by find_deffns at the start
        case _:
            raise ValueError(f"Statement {stmt} is not supported")

BINOP_MAP = {
    '+': lambda a, b: a + b,
    '-': lambda a, b: a - b,
    '*': lambda a, b: a * b,
    '/': lambda a, b: a / b,
    '^': lambda a, b: a ** b,
    '<': lambda a, b: int(a < b),
    '>': lambda a, b: int(a > b),
    '=': lambda a, b: int(a == b),
    '<>': lambda a, b: int(a != b),
    '<=': lambda a, b: int(a <= b),
    '>=': lambda a, b: int(a >= b),
    'AND': lambda a, b: int(a and b),
    'OR': lambda a, b: int(a or b),
}

UNOP_MAP = {
    '-': lambda a: -a,
    'NOT': lambda a: int(not a),
}


def run_expr(env, expr):
    """Run an expression"""
    match expr:
        case Number(value=n):
            return n
        case String(value=s):
            return s
        case Variable(name=v):
            return env.get_var(v)
        case ArrayRef(name=v, subscripts=[sub]):
            # For now, assume it's a string
            return run_slice(env, env.get_var(v), sub)
        case BuiltIn(action=action, args=args):
            (num_args, handler) = FBUILTIN_MAP.get(action)
            if num_args is not None and len(args) != num_args:
                raise ValueError(f"{action} requires {num_args} arguments")
            if handler is None:
                raise ValueError(f"The {action} function is not supported")
            return handler(env, args)
        case Fn(name=name, args=args):
            return run_fn(env, name, args)
        case BinaryOp(op=op, lhs=lhs, rhs=rhs):
            return BINOP_MAP[op](run_expr(env, lhs), run_expr(env, rhs))
        case UnaryOp(op=op, expr=expr):
            return UNOP_MAP[op](run_expr(env, expr))
        case StringSubscript(expr=expr, index=index):
            value = run_expr(env, expr)
            return run_slice(env, value, index)
        case _:
            raise ValueError(f"Expression {expr} is not supported")


def run_slice(env, value, index):
    if isinstance(index, Slice):
        min = run_expr(env, index.min) if index.min is not None else 1
        max = run_expr(env, index.max) if index.max is not None else len(value)
        return value[min-1:max]
    else:
        return value[run_expr(env, index)-1]

FBUILTIN_MAP = {
    "PI":   (0, lambda env, args: math.pi),
    "RND":  (0, lambda env, args: random.random()),
    "ABS":  (1, lambda env, args: abs(run_expr(env, args[0]))),
    "ACS":  (1, lambda env, args: math.acos(run_expr(env, args[0]))),
    "ASN":  (1, lambda env, args: math.asin(run_expr(env, args[0]))),
    "ATN":  (1, lambda env, args: math.atan(run_expr(env, args[0]))),
    "COS":  (1, lambda env, args: math.cos(run_expr(env, args[0]))),
    "EXP":  (1, lambda env, args: math.exp(run_expr(env, args[0]))),
    "INT":  (1, lambda env, args: int(math.floor(run_expr(env, args[0])))),
    "LN":   (1, lambda env, args: math.log(run_expr(env, args[0]))),
    "SGN":  (1, lambda env, args: int(math.copysign(1, run_expr(env, args[0])))),
    "SIN":  (1, lambda env, args: math.sin(run_expr(env, args[0]))),
    "SQR":  (1, lambda env, args: math.sqrt(run_expr(env, args[0]))),
    "TAN":  (1, lambda env, args: math.tan(run_expr(env, args[0]))),
    "USR":  (1, lambda env, args: 0), # TODO
    "LEN":  (1, lambda env, args: len(run_expr(env, args[0]))),
    "CODE": (1, lambda env, args: ord(run_expr(env, args[0])[0])),
    "IN":   (1, lambda env, args: 0), # TODO
    "VAL":  (1, lambda env, args: 0), # TODO
    "PEEK": (1, lambda env, args: 0), # TODO
    "CHR$": (1, lambda env, args: chr(run_expr(env, args[0]))),
    "STR$": (1, lambda env, args: str(run_expr(env, args[0]))),
    "VAL$": (1, lambda env, args: ""), # TODO
}

def run_fn(env, name, args):
    """Handle FN, run a user-defined function"""
    fn = env.get_fn(name)
    params = fn.params
    expr = fn.expr
    if len(params) != len(args):
        raise ValueError(f"Function {name} expects {len(params)} arguments")
    for param, arg in zip(params, args):
        binding = run_expr(env, arg)
        env.save_var(param)
        env.let_var(param, binding)
    result = run_expr(env, expr)
    for param in reversed(params):
        env.restore_var(param)
    return result

def run_goto(env, args):
    """Run a GOTO statement"""
    if len(args) != 1:
        raise ValueError("GOTO requires exactly one argument")
    return (env.lines_map.get_index(run_expr(env, args[0])), 0)

# Placeholder for now
def run_print(env, args):
    """Run a PRINT statement"""
    sep = None
    for printitem in args:
        printaction = printitem.value
        sep = printitem.sep
        if printaction is not None:
            match printaction:
                case BuiltIn(action="AT", args=[x, y]):
                    # Send an ANSI escape sequence to move the cursor
                    print(f"\x1b[{1+run_expr(env, x)};{1+run_expr(env, y)}H", end="")
                case _:
                    if is_expression(printaction):
                        value = run_expr(env, printaction)
                        # Floating point numbers are printed with 7 decimal places
                        if isinstance(value, float):
                            print(f"{value:.7f}", end="")
                        else:
                            print(value, end="")
                    else:
                        raise ValueError(f"Unsupported print item {printaction}")
        match sep:
            case None:
                pass
            case ",":
                print("\t", end="")
            case ";":
                pass
            case "'":
                print()
            case _:
                raise ValueError(f"Unsupported print separator {sep}")
    # After printint everything, what was the the last sep used?
    if sep is None:
        print()

def run_read(env, args):
    """Run a READ statement"""
    for arg in args:
        expr = env.data.next()
        value = run_expr(env, expr)
        match arg:
            case Variable(name=v):
                env.let_var(v, value)
            case _:
                raise ValueError(f"READ requires variable, got {arg}")

# Maps names of builtins to their corresponding functions
BUILTIN_MAP = {
    "GOTO": run_goto,
    "RETURN": lambda env, args: env.gosub_pop(),
    "STOP": lambda env, args: (float('inf'), 0),
    "PRINT": run_print,
    # CLS send ansi to clear screen and home cursor
    "CLS": lambda env, args: print("\x1b[2J\x1b[H", end=""),
}

if __name__ == "__main__":
    from .core import parse_string
    import argparse
    # Usage spectrum_basic.run.py <filename>
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", help="The file to run")
    args = parser.parse_args()
    with open(args.filename) as f:
        prog = parse_string(f.read())
    env = run_program(prog)
