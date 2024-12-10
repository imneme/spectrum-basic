# All 128 ZX Spectrum BASIC tokens (including the additions from the
# ZX Spectrum Next).

TOKENS = [
    "PEEK$", "REG", "DPOKE", "DPEEK", "MOD", "<<", ">>", "UNTIL",
    "ERROR", "ON", "DEF PROC", "END PROC", "PROC", "LOCAL", "DRIVER",
    "WHILE", "REPEAT", "ELSE", "REMOUNT", "BANK", "TILE", "LAYER",
    "PALETTE", "SPRITE", "PWD", "CD", "MKDIR", "RMDIR", "SPECTRUM",
    "PLAY", "RND", "INKEY$", "PI", "FN", "POINT", "SCREEN$", "ATTR",
    "AT", "TAB", "VAL$", "CODE", "VAL", "LEN", "SIN", "COS", "TAN",
    "ASN", "ACS", "ATN", "LN", "EXP", "INT", "SQR", "SGN", "ABS", "PEEK",
    "IN", "USR", "STR$", "CHR$", "NOT", "BIN", "OR", "AND", "<=", ">=",
    "<>", "LINE", "THEN", "TO", "STEP", "DEF FN", "CAT", "FORMAT",
    "MOVE", "ERASE", "OPEN #", "CLOSE #", "MERGE", "VERIFY", "BEEP",
    "CIRCLE", "INK", "PAPER", "FLASH", "BRIGHT", "INVERSE", "OVER",
    "OUT", "LPRINT", "LLIST", "STOP", "READ", "DATA", "RESTORE", "NEW",
    "BORDER", "CONTINUE", "DIM", "REM", "FOR", "GOTO", "GOSUB",
    "INPUT", "LOAD", "LIST", "LET", "PAUSE", "NEXT", "POKE", "PRINT",
    "PLOT", "RUN", "SAVE", "RANDOMIZE", "IF", "CLS", "DRAW", "CLEAR",
    "RETURN", "COPY"
]

CODE_FOR = {token: i+128+7 for i, token in enumerate(TOKENS)}

def num_to_specfloat(num):
    """Convert a Python number to ZX Spectrum's floating point format.
    Returns 5 bytes: [exponent, mantissa_bytes[0..3]]
    For numbers in the program, the sign bit is always 0 and negativity
    is handled by the '-' token in the program text.  This code is inspired
    by similar code in `zmakebas` which is inspired by what actually happens
    in the ZX Spectrum ROM."""

    # Error out if the number is negative or if it's a NaN or infinity
    if num < 0.0 or num != num or num == float('inf'):
        raise ValueError("Negative, NaN or infinity not allowed")
    
    # 16-bit integers have a special int-inside-an-invalid-float format
    if num == (inum := int(num)) and inum & 0xFFFF == inum:
        lowbyte = inum & 0xFF
        highbyte = inum >> 8
        return bytes((0, 0, lowbyte, highbyte, 0))
        
    # Handle zero specially
    if num == 0:
        return bytes([0, 0, 0, 0, 0])
    
    # Get number into binary standard form (0.5 <= num < 1.0)
    exp = 0
    while num >= 1.0:
        num /= 2.0
        exp += 1
    while num < 0.5:
        num *= 2.0
        exp -= 1
    
    # Check if exponent is in valid range
    if not (-128 <= exp <= 127):
        raise ValueError("Number out of range")
    
    # Adjust exponent (add bias of 128)
    exp = 128 + exp
    
    # Extract mantissa bits
    num *= 2.0  # Shift so 0.5 bit is in integer part
    man = 0
    for _ in range(32):
        man <<= 1
        int_part = int(num)
        man |= int_part
        num -= int_part
        num *= 2.0
    
    # Handle rounding
    if int(num) and man != 0xFFFFFFFF:
        man += 1
    
    # Clear the top bit
    man &= 0x7FFFFFFF
    
    # Return as bytes in correct order
    return bytes([exp]) + man.to_bytes(4, 'big')

def num_to_bytes(num):
    """Convert a number to a BASIC program text representation."""
    # First, the text. If we have an int, it's easy. Floats need up to 11 digits
    # of precision.
    if num == (inum := int(num)):
        text = str(inum)
    else:
        text = format(num, '.11f').rstrip('0').rstrip('.')
    textbytes = text.encode('ascii')
    binary = num_to_specfloat(num)
    return textbytes + b'\x0e' + binary

def line_to_bytes(lineno, linebytes):
    """Convert a line number and line bytes to a BASIC program line."""
    terminated_line = linebytes + b'\x0d'
    lineno_bytes = lineno.to_bytes(2, 'big')
    linelen_bytes = len(terminated_line).to_bytes(2, 'little')
    return lineno_bytes + linelen_bytes + terminated_line

def token_to_byte(token):
    """Return a byte string (just a byte) for the given token."""
    # As a consession, if we're given a single character, we'll return it as is
    if len(token) == 1:
        return token.encode('ascii')
    return bytes([CODE_FOR[token]])
