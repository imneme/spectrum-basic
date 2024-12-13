# All 128 ZX Spectrum BASIC tokens (including the additions from the
# ZX Spectrum Next).

import re

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

BLOCK_ESCAPES = ["  ", " '", "' ", "''", " .", " :", "'.", "':",
                ". ", ".'", ": ", ":'", "..", ".:", ":.", "::"]
UNICODE_BLOCKS = {
    "\u2800",   # U+2800 BRAILLE PATTERN BLANK
    "▘",        # Top left
    "▝",        # Top right
    "▀",        # Top half (both top blocks)
    "▖",        # Bottom left
    "▌",        # Left half
    "▞",        # Diagonal
    "▛",        # Three blocks (missing bottom right)
    "▗",        # Bottom right
    "▚",        # Diagonal
    "▐",        # Right half
    "▜",        # Three blocks (missing bottom left)
    "▄",        # Bottom half
    "▙",        # Three blocks (missing top right)
    "▟",        # Three blocks (missing top left)
    "█",        # Full block
}

BYTE_FOR_BLOCK_ESCAPE = {glyph.encode("ascii"): bytes([i+128]) for i, glyph in enumerate(BLOCK_ESCAPES)}
BYTE_FOR_UNICODE = {
    "©": b"\x7f",
    "↑": b"^",
    "£": b"`",
    **{uni: bytes([i+128]) for i, uni in enumerate(UNICODE_BLOCKS)},
    **{chr(i+0x24b6): bytes([i+144]) for i in range(0, 21)},  # UDGs
}
UNICODE_BLOCK_FOR_BLOCK_ESCAPE = {e: u for e, u in zip(BLOCK_ESCAPES, UNICODE_BLOCKS)}

BYTES_BLOCK_ESCAPE_RE = re.compile(rb"\\([.:' ]{2})")
BLOCK_ESCAPE_RE = re.compile(r"\\([.:' ]{2})")

def uchar_to_byte(char):
    """Convert a (potentially unicode) character to a ZX Spectrum one."""
    if (b := BYTE_FOR_UNICODE.get(char)) is not None:
        return b
    return char.encode('ascii')

BYTE_AND_MAX_FOR_COLOURCODE = {
    b'i': [b'\x10', 8], # Ink
    b'p': [b'\x11', 8], # Paper
    b'f': [b'\x12', 1], # Flash
    b'b': [b'\x13', 1], # Bright
    b'r': [b'\x14', 1], # Inverse (a.k.a. Reverse Video)
    b'o': [b'\x15', 1], # Over
}

def echars_to_bytes(s):
    """Convert escaped characters (and Unicode) to ZX Spectrum BASIC bytes."""
    # First, convert to bytes
    bstr = b''.join(uchar_to_byte(c) for c in s)
    # Then, escape any block characters
    bstr = BYTES_BLOCK_ESCAPE_RE.sub(lambda m: BYTE_FOR_BLOCK_ESCAPE[m.group(1)], bstr)
    # \* becomes 127, the copyright symbol
    bstr = bstr.replace(b'\\*', b'\x7f')
    # \{number} becomes the corresponding exactly that ASCII character
    def number_to_byte(m):
        n = int(m.group(1))
        if n > 255:
            raise ValueError(f"Character code too large, in \\{{{n}}}")
        return bstr((n,))
    bstr = re.sub(rb"\\\{(\d+)\}", number_to_byte, bstr)
    # Escapes for ink, paper, flash, bright, inverse, and over
    # - Special case, some folks use \{vi} and \{vn} for inverse video
    def colourcodes_to_bytes(m):
        codes = [m.group(1)[i:i+2] for i in range(0, len(m.group(1)), 2)]
        pieces = []
        for code in codes:
            code = b'r1' if code == b'vi' else code
            code = b'r0' if code == b'vn' else code
            c, n = bytes((code[0],)), code[1]
            try:
                b, maxval = BYTE_AND_MAX_FOR_COLOURCODE[c]
            except KeyError:
                raise ValueError(f"Invalid colour code, in \\{chr(c)}{chr(n)}")
            ni = n - ord('0')
            if ni < 0 or ni > maxval:
                raise ValueError(f"Colour code out of range, in \\{chr(c)}{chr(n)}")
            pieces.append(b + bytes([ni]))
        return b''.join(pieces)
    bstr = re.sub(rb"\\\{((?:[ipfibo]\d+)|vi|vn)+\}", colourcodes_to_bytes, bstr)
    # \a..u becomes the corresponding UDG character (either case)
    def udg_to_byte(m):
        c = m.group(1).upper()
        if c < b'A' or c > b'U':
            raise ValueError(f"Invalid UDG character, in \\{c}")
        return bytes((c[0] - 65 + 144,))
    bstr = re.sub(rb"\\([a-u])", udg_to_byte, bstr)
    # Doubled backslashes become a single backslash
    return bstr.replace(b'\\\\', b'\\')

def strlit_to_bytes(s):
    """Convert a string literal to ZX Spectrum BASIC bytes."""
    # Then double any double quotes
    bstr = echars_to_bytes(s)
    bstr = bstr.replace(b'"', b'""')
    return b'"' + bstr + b'"'

def escapes_to_unicode(s):
    """Convert ZX Spectrum block escapes to unicode."""
    return BLOCK_ESCAPE_RE.sub(lambda m: UNICODE_BLOCK_FOR_BLOCK_ESCAPE[m.group(1)], s)

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
