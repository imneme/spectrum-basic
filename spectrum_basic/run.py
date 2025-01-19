from .ast import *
from .tokenizer import *
from .zxoutput import ZXOutputStream
from .zxinput import pause, inkey
import bisect
import math
import random

# Exception class

class ZXBasicError(Exception):
    """ZX Spectrum BASIC error"""
    def __init__(self, errcode, message, line=None, stmt=None, action=None):
        self.errcode = errcode
        # These are usually added later, not in the constructor
        self.line = line      # Line number
        self.stmt = stmt      # Statement number within line
        self.action = action  # Actual code being executed
        super().__init__(message)

    MESSAGE_MAP = {
        '0': "OK",
        '1': "NEXT without FOR",
        '2': "Variable not found",
        '3': "Subscript wrong",
        '4': "Out of memory",
        '5': "Out of screen",
        '6': "Number too big",
        '7': "RETURN without GO SUB",
        '8': "End of file",
        '9': "STOP statement",
        'A': "Invalid argument",
        'B': "Integer out of range",
        'C': "Nonsense in BASIC",
        'D': "BREAK - CONT repeats",
        'E': "Out of DATA",
        'F': "Invalid file name",
        'G': "No room for line",
        'H': "STOP in INPUT",
        'I': "FOR without NEXT",
        'J': "Invalid I/O device",
        'K': "Invalid colour",
        'L': "BREAK into program",
        'M': "RAMTOP no good",
        'N': "Statement lost",
        'O': "Invalid stream",
        'P': "FN without DEF",
        'Q': "Parameter error",
        'R': "Tape loading error",
    }

    def __str__(self):
        if self.line is None:
            return f"{self.errcode} {self.MESSAGE_MAP.get(self.errcode, 'Unknown error')}\n\t{super().__str__()}"
        return f"{self.errcode} {self.MESSAGE_MAP.get(self.errcode, 'Unknown error')}, {self.line}:{self.stmt}\n\t{super().__str__()}\n\trunning {self.action}"


# Run ZX Spectrum BASIC code

def is_stringvar(var):
    """Check if a variable is a string variable"""
    return var.endswith("$")

def force_width(str, width):
    """Force a string to a certain width"""
    if len(str) > width:
        return str[:width]
    return str.ljust(width)

class ProgramInfo:
    """Information about a ZX Spectrum BASIC program needed for running it"""
    def __init__(self, prog):
        lines = [line for line in prog.lines if not isinstance(line, CommentLine)]
        if not lines:
            raise ZXBasicError('N', "Empty program (no non-meta-comment lines)")
        self.lines_map = LineMapper(lines)
        lines = [list(self._flattened_statements(line.statements)) for line in lines]
        posForNode = {
            id(node): (line_idx, stmt_idx)
                for line_idx, line in enumerate(lines)
                    for stmt_idx, node in enumerate(line)
        }
        self.lines = lines
        self._find_info(prog, posForNode)
        self.data = ProgramData(prog)
    
    @staticmethod
    def _flattened_statements(statements):
        """Flatten a line of statements to handle IF statements"""
        for stmt in statements:
            while isinstance(stmt, JankyStatement):
                stmt = stmt.actual
            match stmt:
                case If(condition=cond, statements=stmts, parent=parent, after=after):
                    yield If(condition=cond, statements=[], parent=parent, after=None)
                    yield from ProgramInfo._flattened_statements(stmts)
                case _:
                    yield stmt

    def _find_info(self, prog, posForNode):
        """Find information about program, incuding function definitions, and NEXT statements"""
        fn_map = {}       # Maps function names to their AST DefFn nodes
        active_for = {}   # Maps loop variable names to the most recent FOR statement
        nearest_next = {} # Maps AST For nodes to the (stmt_idx, line_idx) just past the nearest NEXT
        for event, node in walk(prog):
            match event:
                case Walk.ENTERING:
                    match node:
                        case DefFn(name=name, params=params, expr=expr):
                            if name in fn_map:
                                raise ZXBasicError('C', f"Function {name} already defined")
                            fn_map[name.lower()] = node
                        case For(var=Variable(name=v), start=start, end=end, step=step):
                            active_for[v] = node
                        case Next(var=Variable(name=v)):
                            prior_for = active_for.get(v)
                            if prior_for is None:
                                pass  # Convoluted but valid BASIC: NEXT textually before FOR, perhaps a GOTO gets us here
                            elif id(prior_for) not in nearest_next:
                                line_idx, stmt_idx = posForNode[id(node)]
                                nearest_next[id(prior_for)] = (line_idx, stmt_idx+1)
                    
        self.functions = fn_map
        self.nearest_next = nearest_next

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
            raise ZXBasicError('E', "Out of DATA")
        value = self.data[self.index]
        self.index += 1
        return value

class Environment:
    """Environment for running ZX Spectrum BASIC programs"""
    def __init__(self, prog_info, data=None):
        self.vars = {}
        self.array_vars = {}
        self.prog_info = prog_info
        self.gosub_stack = []
        self.data = data
        output = ZXOutputStream()
        self.tty = output
        self.channels = {0: output, 1: output, 2: output}
        self.memory = bytearray(64*1024)

    def let_var(self, var, value):
        """Set a variable"""
        if not isinstance(var, str):
            raise ZXBasicError('C', f"Variable name {var} is not a string")
        var = var.lower()
        vardict = self.vars.setdefault(var, {})
        if is_stringvar(var) and vardict.get('str_len') is not None:
            value = force_width(value, self.vars[var]['str_len'])
        vardict['value'] = value

    def for_loop(self, var, line_idx, stmt_idx, start, end, step):
        """Start a FOR loop"""
        if not isinstance(var, str):
            raise ZXBasicError('C', f"Variable name {var} is not a string")
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
            raise ZXBasicError('C', f"Function name {name} is not a string")
        try:
            name = name.lower()
            return self.prog_info.functions[name]
        except KeyError as e:
            raise ZXBasicError('P', f"Function {name} not defined") from e

    def get_var(self, var):
        """Get a variable"""
        if not isinstance(var, str):
            raise ZXBasicError('C', f"Variable name {var} is not a string")
        try:
            var = var.lower()
            return self.vars[var]['value']
        except KeyError as e:
            raise ZXBasicError('2', f"Variable {var} not defined") from e
    
    def save_var(self, var):
        """Save a variable (on an internal per-variable stack)"""
        if not isinstance(var, str):
            raise ZXBasicError('C', f"Variable name {var} is not a string")
        var = var.lower()
        dict = self.vars.get(var)
        self.vars[var] = {"stashed": dict}

    def restore_var(self, var):
        """Restore a variable (from an internal per-variable stack)"""
        if not isinstance(var, str):
            raise ZXBasicError('C', f"Variable name {var} is not a string")
        try:
            var = var.lower()
            dict = self.vars[var].pop("stashed")
            if dict is None:
                del self.vars[var]
            else:
                self.vars[var] = dict
        except KeyError as e:
            raise ZXBasicError('2', f"Variable {var} not defined (no stashed value)") from e
        
    def get_var_all(self, var):
        """Get all the information about a variable"""
        try:
            var = var.lower()
            return self.vars[var]
        except KeyError as e:
            raise ZXBasicError('2', f"Variable {var} not defined") from e
            
    def dim(self, var, *dims):
        """Create an array"""
        is_string = is_stringvar(var)
        var = var.lower()
        if is_string:
            dims = list(dims)
            str_len = dims.pop()
            init_val = " " * str_len
            if dims == []:
                # It's a plain old string variable with fixed length, not an array
                self.let_var(var, init_val)
                self.vars[var]['str_len'] = str_len
                return
        else:
            init_val = 0
        def nest(i):
            if i == len(dims):
                return init_val
            return [nest(i+1) for _ in range(dims[i])]
        self.array_vars[var] = {
            'bounds': dims,
            'values': nest(0),
        }
        if is_string:
            self.array_vars[var]['str_len'] = str_len
    
    def get_array_all(self, var):
        """Get all the information about an array; array is allowed not to exist"""
        return self.array_vars.get(var.lower())

    def get_array(self, var, *indices):
        """Get an array element"""
        var = var.lower()
        try:
            array_dict = self.array_vars[var]
        except KeyError as e:
            raise ZXBasicError('2', f"Array {var} not defined") from e
        bounds = array_dict['bounds']
        if len(bounds) != len(indices):
            raise ZXBasicError('3', f"Wrong number of indices for array {var}, need {len(bounds)}")
        for i, (idx, bound) in enumerate(zip(indices, bounds)):
            if idx < 1 or idx > bound:
                raise ZXBasicError('3', f"Index {idx} out of bounds for array {var} at dimension {i}")
        arrayval = array_dict['values']
        for idx in indices:
            arrayval = arrayval[idx-1]
        return arrayval
        
    def set_array(self, var, value, *indices):
        """Set an array element"""
        var = var.lower()
        try:
            array_dict = self.array_vars[var]
        except KeyError as e:
            raise ZXBasicError('2', f"Array {var} not defined") from e
        bounds = array_dict['bounds']
        if len(bounds) != len(indices):
            raise ZXBasicError('3', f"Wrong number of indices for array {var}, need {len(bounds)}")
        for i, (idx, bound) in enumerate(zip(indices, bounds)):
            if idx < 1 or idx > bound:
                raise ZXBasicError('3', f"Index {idx} out of bounds for array {var} at dimension {i}")
        arrayval = array_dict['values']
        indices = list(indices)
        last_idx = indices.pop()
        for idx in indices:
            arrayval = arrayval[idx-1]
        if is_stringvar(var):
            value = force_width(value, array_dict['str_len'])
        arrayval[last_idx-1] = value

    def gosub_push(self, line_idx, stmt_idx):
        """Push a GOSUB return address"""
        self.gosub_stack.append((line_idx, stmt_idx))

    def gosub_pop(self):
        """Pop a GOSUB return address"""
        try:
            return self.gosub_stack.pop()
        except IndexError as e:
            raise ZXBasicError('7', "Spurious RETURN") from e

class LineMapper:
    """Map line numbers to line indices"""
    def __init__(self, lines):
        self.lines = {}                   # Maps line numbers to line indices
        self.rlines = [None] * len(lines) # Maps line indices to line numbers
        last_lineno = 0
        unnumbered = 0
        for i, line in enumerate(lines):
            # Only include lines that actually have a line_number
            if line.line_number:
                last_lineno = line.line_number
                self.lines[last_lineno] = i
                self.rlines[i] = last_lineno
                unnumbered = 0
            else:
                unnumbered += 1
                self.rlines[i] = f"{last_lineno}+{unnumbered}"
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


def run_program(prog : Program, start=0):
    """Run a ZX Spectrum BASIC program"""
    # Set up the environment
    prog_info = ProgramInfo(prog)
    env = Environment(prog_info, data=ProgramData(prog))
    lines = prog_info.lines
    lines_map = prog_info.lines_map
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
        try:
            jump = run_stmt(env, stmt, line_idx, i)
        except ZXBasicError as e:
            e.line = env.prog_info.lines_map.rlines[line_idx]
            e.stmt = i+1
            e.action = stmt
            raise
        except Exception as e:
            line = env.prog_info.lines_map.rlines[line_idx]
            raise ZXBasicError('C', f"Internal error: {e}", line=line, stmt=i+1, action=stmt) from e
        if jump is not None:
            return jump
    return None

def run_let(env, vardest, expr):
    """Run a LET statement"""
    run_let_val(env, vardest, run_expr(env, expr))


def run_let_val(env, vardest, value):
    """Run a LET statement (internal)"""
    match vardest:
        case Variable(name=v):
            env.let_var(v, value)
        case ArrayRef(name=v, subscripts=subs):
            # String variables can have an extra subscript for the character index
            slice = None
            subs = list(subs) # Make a copy!
            if is_stringvar(v):
                array_info = env.get_array_all(v)
                bounds = array_info['bounds'] if array_info else []
                if len(subs) == len(bounds) + 1:
                    slice = subs.pop()
                elif len(subs) != len(bounds):
                    raise ZXBasicError('3', f"Wrong number of subscripts for array {v}")
            indices = [run_expr(env, sub) for sub in subs]
            if slice is None:
                env.set_array(v, value, *indices)
                return
            if indices == []:
                # It's a plain old string variable, not an array
                old_value = env.get_var(v)
            else:
                # It's an array
                old_value = env.get_array(v, *indices)
            if not isinstance(slice, Slice):
                # Just a regular index to change one character
                index = run_expr(env, slice)
                left = index
                right = index
            else:
                left = run_expr(env, slice.min) if slice.min is not None else 1
                right = run_expr(env, slice.max) if slice.max is not None else len(old_value)
            if left < 1 or right > len(old_value):
                raise ZXBasicError('3', f"String index out of bounds for {v}")
            value = force_width(value, right - left + 1)
            value = old_value[:left-1] + value + old_value[right:]
            if indices == []:
                env.let_var(v, value)
            else:
                env.set_array(v, value, *indices)


def run_stmt(env, stmt, line_idx, stmt_idx):
    """Run a single statement"""
    match stmt:
        case Let(var=vardest, expr=expr):
            run_let(env, vardest, expr)
        # Special case for GOSUB as it needs to push the return address
        case BuiltIn(action="GOSUB", args=args):
            if len(args) != 1:
                raise ZXBasicError('A', "GOSUB requires exactly one argument")
            env.gosub_push(line_idx, stmt_idx+1)
            return (env.prog_info.lines_map.get_index(run_expr(env, args[0])), 0)
        case BuiltIn(action=action, args=args):
            handler = BUILTIN_MAP.get(action)
            if handler is None:
                raise ZXBasicError('N', f"The {action} command is not supported")
            return handler(env, args)
        case For(var=Variable(name=v), start=start, end=end, step=step):
            start = run_expr(env, start)
            end = run_expr(env, end)
            step = run_expr(env, step) if step is not None else 1
            env.for_loop(v, line_idx, stmt_idx+1, start, end, step)
            if (end < start and step >= 0) or (end > start and step < 0):
                dest = env.prog_info.nearest_next.get(id(stmt))
                if dest is None:
                    raise ZXBasicError('I', f"FOR without NEXT for {v}")
                return dest
        case Next(var=Variable(name=v)):
            var_info = env.get_var_all(v)
            var_info['value'] += var_info['step']
            end = var_info['end']
            step = var_info['step']
            if (step >= 0 and var_info['value'] > end) or (step < 0 and var_info['value'] < end):
                # Continue to the next statement
                return
            # Jump back to the FOR statement
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
        case Dim(name=name, dims=exprs):
            dims = [run_expr(env, expr) for expr in exprs]
            env.dim(name, *dims)
        case _:
            raise ZXBasicError('N', f"Statement {stmt} is not supported")

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
            bytes = echars_to_bytes(s)
            # Turn the bytes into a string, don't try this at home, kids
            # it's moderately evil to store ZX-Spectrum encoded strings 
            # in python's Unicode strings
            return ''.join(chr(b) for b in bytes)
        case Variable(name=v):
            return env.get_var(v)
        case ArrayRef(name=v, subscripts=subs):
            # # For now, assume it's a string
            # return run_slice(env, env.get_var(v), sub)
            if not is_stringvar(v):
                return env.get_array(v, *[run_expr(env, sub) for sub in subs])
            slice = None
            subs = list(subs) # Make a copy!
            array_info = env.get_array_all(v)
            bounds = array_info['bounds'] if array_info else []
            if len(subs) == len(bounds) + 1:
                slice = subs.pop()
            elif len(subs) != len(bounds):
                raise ZXBasicError('3', f"Wrong number of subscripts for array {v}")
            if subs == []:
                # It's a plain old string variable, not an array
                value = env.get_var(v)
            else:
                # It's an array
                value = env.get_array(v, *[run_expr(env, sub) for sub in subs])
            if slice:
                return run_slice(env, value, slice)
            return value

        case BuiltIn(action=action, args=args):
            (num_args, handler) = FBUILTIN_MAP.get(action, (None, None))
            if num_args is not None and len(args) != num_args:
                raise ZXBasicError('A', f"{action} requires {num_args} arguments")
            if handler is None:
                raise ZXBasicError('N', f"The {action} function is not supported")
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
            raise ZXBasicError('N', f"Expression {expr} is not supported")


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
    "VAL":  (1, lambda env, args: float(run_expr(env, args[0]))), # PARTIAL
    "PEEK": (1, lambda env, args: env.memory[int(run_expr(env, args[0]))]),
    "CHR$": (1, lambda env, args: chr(run_expr(env, args[0]))),
    "STR$": (1, lambda env, args: format_float(run_expr(env, args[0]))),
    "INKEY$": (0, lambda env, args: inkey()),
    "VAL$": (1, lambda env, args: ""), # TODO
}

def run_fn(env, name, args):
    """Handle FN, run a user-defined function"""
    fn = env.get_fn(name)
    params = fn.params
    expr = fn.expr
    if len(params) != len(args):
        raise ZXBasicError('Q', f"Function {name} expects {len(params)} arguments")
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
        raise ZXBasicError('A', "GOTO requires exactly one argument")
    return (env.prog_info.lines_map.get_index(run_expr(env, args[0])), 0)


PRINT_CODES = {
    "INK": (16, 9),
    "PAPER": (17, 9),
    "FLASH": (18, 1),
    "BRIGHT": (19, 1),
    "INVERSE": (20, 1),
    "OVER": (21, 1),
}

def run_color(env, cmd, arg, stream=None):
    """Run a colour statement"""
    stream = stream or env.tty
    value = int(run_expr(env, arg))
    code, max = PRINT_CODES[cmd]
    if value < 0 or value > max:
        raise ZXBasicError('K', f"{cmd} {value} is out of range")
    stream.write(chr(code) + chr(value))

def format_float(value):
    """Format a floating point number"""
    return f"{value:.7f}".rstrip("0").rstrip(".")

def run_print(env, args, curchannel=2, is_input=False):
    """Run a PRINT statement"""
    sep = None
    def put(*args):
        try:
            stream = env.channels[curchannel]
        except IndexError as e:
            raise ZXBasicError('J', f"Channel {curchannel} not open") from e
        for arg in args:
            stream.write(str(arg))
    
    env.tty.push_state()

    for printitem in args:
        printaction = printitem.value
        sep = printitem.sep
        if printaction is not None and is_input:
            match printaction:
                case BuiltIn(action="LINE", args=[var]):
                    do_input(env, var, curchannel)
                    continue
                case Variable(name=v):
                    do_input(env, printaction, curchannel)
                    continue
                case ArrayRef(name=v, subscripts=subs):
                    do_input(env, printaction, curchannel)
                    continue
                case InputExpr(expr=e):
                    printaction = e
        if printaction is not None:
            match printaction:
                case BuiltIn(action="AT", args=[x, y]):
                    # Send an Spectrum escape sequence to move the cursor
                    put(chr(22), chr(run_expr(env, x)), chr(run_expr(env, y)))
                case BuiltIn(action="TAB", args=[x]):
                    # Send an Spectrum escape sequence to move the cursor
                    put(chr(23), chr(run_expr(env, x)))
                case BuiltIn(action=("INK"|"PAPER"|"FLASH"|"BRIGHT"|"INVERSE"|"OVER") as action, args=[e]):
                    run_color(env, action, e, stream=env.channels[curchannel])
                case ChanSpec(chan=c):
                    curchannel = run_expr(env, c)
                case _:
                    if is_expression(printaction):
                        value = run_expr(env, printaction)
                        # Floating point numbers are printed with 7 decimal places
                        if isinstance(value, float):
                            put(format_float(value))
                        else:
                            put(value)
                    else:
                        raise ZXBasicError('N', f"Unsupported print item {printaction}")
        match sep:
            case None:
                pass
            case ",":
                put(chr(6))
            case ";":
                pass
            case "'":
                put(chr(13))
            case _:
                raise ZXBasicError('N', f"Unsupported print separator {sep}")
    # After printint everything, what was the the last sep used?
    if sep is None and not is_input:
        put(chr(13))

    env.tty.pop_state()

def do_input(env, target, curchannel):
    try:
        stream = env.channels[curchannel]
    except IndexError as e:
        raise ZXBasicError('J', f"Channel {curchannel} not open") from e
    match target:
        case Variable(name=v):
            is_string = is_stringvar(v)
        case ArrayRef(name=v):
            is_string = is_stringvar(v)
    if stream == env.tty:
        line = input("")
    else:
        line = stream.readline().strip('\n')
    if not is_string:
        line = float(line)
    run_let_val(env, target, line)
            

def run_read(env, args):
    """Run a READ statement"""
    for arg in args:
        expr = env.data.next()
        run_let(env, arg, expr)

def run_open(env, args):
    """Run an OPEN statement"""
    channel_id = int(run_expr(env, args[0]))
    file = run_expr(env, args[1])
    if (channel_id < 0 or channel_id > 15):
        raise ZXBasicError('B', f"Channel id {channel_id} is out of range")
    if channel_id in env.channels:
        # Automagically close the channel
        if env.channels[channel_id] != env.tty:
            env.channels[channel_id].close()
        del env.channels[channel_id]
    if file == "K" or file == "S":
        env.channels[channel_id] = env.tty
    elif file == "P":
        raise ZXBasicError('O', "Printer channel not supported")
    elif file.startswith("O>") or file.startswith("U>") or file.startswith("I>"):
        if file.startswith("O>"):
            mode = "w"
        elif file.startswith("U>"):
            mode = "r+"
        else:
            mode = "r"
        filename = file[2:]
        env.channels[channel_id] = open(filename, mode)
    elif file.startswith("M>"):
        raise ZXBasicError('O', "Memory channel not supported")
    elif len(file) > 2 and file[1] != ">":
        env.channels[channel_id] = open(file, "r")
    else:
        raise ZXBasicError('O', f"Unknown file type {file}")

def run_close(env, args):
    """Run a CLOSE statement"""
    channel_id = int(run_expr(env, args[0]))
    if channel_id not in env.channels:
        raise ZXBasicError('J', f"Channel {channel_id} not open")
    if env.channels[channel_id] != env.tty:
        env.channels[channel_id].close()
    del env.channels[channel_id]

def run_clear(env, args):
    """Run a CLEAR statement, zap all variables and the gosub stack"""
    env.vars.clear()
    env.array_vars.clear()
    env.gosub_stack.clear()

def run_poke(env, args):
    """Run a POKE statement"""
    address = int(run_expr(env, args[0]))
    value = int(run_expr(env, args[1]))
    if address < 0 or address > 65535:
        raise ZXBasicError('B', f"POKE address {address} out of range")
    if value < 0 or value > 255:
        raise ZXBasicError('B', f"POKE value {value} out of range")
    env.memory[address] = value

def run_load(env, args):
    """Run a LOAD statement (only load CODE supported)"""
    if len(args) != 2:
        raise ZXBasicError('N', "Only LOAD name CODE is supported")
    filename = run_expr(env, args[0])
    length = None
    match args[1]:
        case BuiltIn(action="CODE", args=[start]):
            start = int(run_expr(env, start))
        case BuiltIn(action="CODE", args=[start, length]):
            start = int(run_expr(env, start))
            length = int(run_expr(env, length))
        case BuiltIn(action="CODE", args=[]):
            raise ZXBasicError('N', "LOAD CODE without start address")
        case _:
            raise ZXBasicError('N', "LOAD only supports CODE")
    with open(filename, "rb") as f:
        data = f.read(length)
        env.memory[start:start+len(data)] = data

def run_save(env, args):
    """Run a SAVE statement (only save CODE supported)"""
    if len(args) != 2:
        raise ZXBasicError('N', "Only SAVE name CODE is supported")
    filename = run_expr(env, args[0])
    match args[1]:
        case BuiltIn(action="CODE", args=[start, length]):
            start = int(run_expr(env, start))
            length = int(run_expr(env, length))
        case _:
            raise ZXBasicError('N', "SAVE only supports CODE")
    with open(filename, "wb") as f:
        f.write(env.memory[start:start+length])

def run_pause(env, args):
    """Run a PAUSE statement"""
    delay = run_expr(env, args[0])
    if delay < 0:
        raise ZXBasicError('B', f"Negative delay {delay} in PAUSE")
    delay = None if delay == 0 else delay
    pause(delay)

# Maps names of builtins to their corresponding functions
BUILTIN_MAP = {
    "GOTO": run_goto,
    "RETURN": lambda env, args: env.gosub_pop(),
    "STOP": lambda env, args: (float('inf'), 0),
    "PRINT": run_print,    
    "INPUT": lambda env, args: run_print(env, args, is_input=True),
    "OPEN #": run_open,
    "CLOSE #": run_close,
    "CLEAR": run_clear,
    "POKE": run_poke,
    "LOAD": run_load,
    "SAVE": run_save,
    "PAUSE": run_pause,
    "RESTORE": lambda env, args: env.data.restore(run_expr(env, args[0]) if args else 0),
    "INK":  lambda env, args: run_color(env, "INK", args[0]),
    "PAPER":  lambda env, args: run_color(env, "PAPER", args[0]),
    "FLASH":  lambda env, args: run_color(env, "FLASH", args[0]),
    "BRIGHT":  lambda env, args: run_color(env, "BRIGHT", args[0]),
    "INVERSE":  lambda env, args: run_color(env, "INVERSE", args[0]),
    "OVER":  lambda env, args: run_color(env, "OVER", args[0]),
    "BORDER": lambda env, args: None, # TODO
    "PLOT": lambda env, args: None, # TODO
    "DRAW": lambda env, args: None, # TODO
    "CLS": lambda env, args: env.tty.cls(),
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
