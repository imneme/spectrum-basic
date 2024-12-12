from .ast import *
import bisect

# Run ZX Spectrum BASIC code

def is_stringvar(var):
    """Check if a variable is a string variable"""
    return var.endswith("$")

class Environment:
    """Environment for running ZX Spectrum BASIC programs"""
    def __init__(self):
        self.vars = {}
        self.array_vars = {}
        self.functions = {}

    def let(self, var, value):
        """Set a variable"""
        self.vars[var] = value
    
    def get(self, var):
        """Get a variable"""
        try:
            return self.vars.get(var)
        except KeyError:
            raise ValueError(f"Variable {var} not defined")
    
    def dim(self, var, *dims):
        """Create an array"""
        pass # TODO

    def get_array(self, var, *indices):
        """Get an array element"""
        pass # TODO

    def set_array(self, var, value, *indices):
        """Set an array element"""
        pass # TODO

class LineMapper:
    """Map line numbers lists of statements"""
    def __init__(self, prog):
        self.lines = {}
        for i, line in enumerate(prog.lines):
            self.lines[line.line_number] = i
        self.line_numbers = sorted(self.lines.keys())
    
    def get_index(self, line_number):
        """Get the index of a line number, if not found return the index of the next line"""
        if (i := self.lines.get(line_number)) is not None:
            return i
        # Not in the list, so find the the actual next line after line_number
        i = bisect.bisect_left(self.line_numbers, line_number)
        if i == len(self.line_numbers):
            return None
        return self.lines[self.line_numbers[i]]

def run_prog(prog : Program, start=0):
    """Run a ZX Spectrum BASIC program"""
    # Set up the environment
    env = Environment()
    # Run the program
    lines_map = LineMapper(prog)
    lineindex = lines_map.get_index(start)
    while lineindex is not None and lineindex < len(prog.lines):
        stmts = prog.lines[lineindex].statements
        next_line = run_stmts(stmts, env)
        lineindex = lines_map.get_index(next_line) if next_line is not None else lineindex + 1

def run_stmts(stmts, env):
    """Run a list of statements"""
    for stmt in stmts:
        jump = run_stmt(stmt, env)
        if jump is not None:
            return jump
    return None


def run_stmt(stmt, env):
    """Run a single statement"""
    match stmt:
        case Let(var=var, expr=expr):
            value = run_expr(expr, env)
            env.let(var, value)
        case BuiltIn(action=action, args=args):
            handler = BUILTIN_MAP.get(action)
            if handler is None:
                raise ValueError(f"The {action} command is not supported")
            return handler(env, args)
        case _:
            raise ValueError(f"Statement {stmt} is not supported")

def run_expr(expr, env):
    """Run an expression"""
    match expr:
        case Number(value=n):
            return n
        case Variable(var):
            return env.get(var)
        case BuiltIn(action, args):
            handler = BUILTIN_MAP.get(action)
            if handler is None:
                raise ValueError(f"The {action} function is not supported")
            return handler(env, args)
        case _:
            raise ValueError(f"Expression {expr} is not supported")

def run_goto(env, args):
    """Run a GOTO statement"""
    if len(args) != 1:
        raise ValueError("GOTO requires exactly one argument")
    return run_expr(args[0], env)

# Placeholder for now
def run_print(env, args):
    """Run a PRINT statement"""
    for printitem in args:
        printaction = printitem.value
        sep = printaction.sep
        if printaction is not None:
            match printaction:
                case BuiltIn(action="AT", args=[x, y]):
                    # Send an ANSI escape sequence to move the cursor
                    print(f"\x1b[{run_expr(x, env)};{run_expr(y, env)}H", end="")
                case _:
                    if isinstance(printaction, Expression):
                        print(run_expr(printaction, env), end="")
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
    if sep is None or sep == "'":
        print()


# Maps names of builtins to their corresponding functions
BUILTIN_MAP = {
    "GOTO": run_goto,
    "PRINT": run_print,
}

if __name__ == "__main__":
    from .core import parse_string
    prog = parse_string("""
10 PRINT 42
20 GOTO 10
""")
    run_prog(prog)
