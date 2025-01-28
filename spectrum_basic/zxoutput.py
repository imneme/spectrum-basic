from io import TextIOBase
import sys

class ZXOutputStream(TextIOBase):
    """
    A text stream that translates ZX Spectrum control codes to ANSI terminal sequences.
    Can be used as a drop-in replacement for sys.stdout.
    """
    
    # ZX Spectrum colors mapped to ANSI 256-color codes
    COLORS = [
        # Normal colors
        {
            0: 16,    # Black (pure black from color cube)
            1: 21,    # Blue (0,0,5 in cube)
            2: 196,   # Red (5,0,0 in cube)
            3: 201,   # Magenta (5,0,5 in cube)
            4: 46,    # Green (0,5,0 in cube)
            5: 51,    # Cyan (0,5,5 in cube)
            6: 226,   # Yellow (5,5,0 in cube)
            7: 255,   # White (lightest gray from grayscale)
            8: None,  # Transparent (use terminal default)
            9: None   # Contrast (determined by other color)
        },
        # Bright colors
        {
            0: 233,   # Black (dark gray from grayscale)
            1: 27,    # Blue (0,1,5 in cube)
            2: 203,   # Red (5,1,1 in cube)
            3: 213,   # Magenta (5,2,5 in cube)
            4: 83,    # Green (1,5,1 in cube)
            5: 123,   # Cyan (2,5,5 in cube)
            6: 228,   # Yellow (5,5,2 in cube)
            7: 231,   # White (pure white)
            8: None,  # Transparent (use terminal default)
            9: None   # Contrast (determined by other color)
        }
    ]
    
    # Character mappings for ZX Spectrum graphics chars (128-143)
    BLOCK_CHARS = {
        128: "\u2800",  # Blank
        129: "▝",       # Top right
        130: "▘",       # Top left
        131: "▀",       # Top half
        132: "▗",       # Bottom right
        133: "▐",       # Right half
        134: "▚",       # Diagonal \
        135: "▜",       # Three blocks (missing bottom left)
        136: "▖",       # Bottom left
        137: "▞",       # Diagonal /
        138: "▌",       # Left half
        139: "▛",       # Three blocks (missing bottom right)
        140: "▄",       # Bottom half
        141: "▟",       # Three blocks (missing top left)
        142: "▙",       # Three blocks (missing top right)
        143: "█",       # Full block
    }
    
    # Special character mappings
    SPECIAL_CHARS = {
        0x7f: "©",
        ord("^"): "↑",
        ord("`"): "£",
    }
    
    def __init__(self, width=32, height=24):
        super().__init__()
        self.width = width
        self.height = height
        self.cursor_x = 0
        self.cursor_y = 0
        
        # State handling
        self.wants_next_char = None
        
        # Current attributes
        self.ink = 8      # Default (transparent)
        self.paper = 8    # Default (transparent)
        self.bright = 0
        self.flash = 0
        self.inverse = 0
        self.over = 0

        # Has CLS been used? If not, try to preserve the current terminal colours    
        self.no_cls = True 

        # Stack for saving and restoring attributes
        self.state_stack = []

        # Initialize terminal state
        self._update_attributes()
    
    def push_state(self):
        """Save the current terminal attributes."""
        self.state_stack.append((self.ink, self.paper, self.bright, self.flash, self.inverse, self.over))
    
    def pop_state(self):
        """Restore the terminal attributes from the last saved state."""
        self.ink, self.paper, self.bright, self.flash, self.inverse, self.over = self.state_stack.pop()
        self._update_attributes()
    
    def write(self, text):
        """Write a string to the stream."""
        if not isinstance(text, str):
            text = str(text)
            
        for char in text:
            self._handle_char(char)
        return len(text)

    def cls(self):
        """Clear the screen and fill with current PAPER color"""
        self._update_attributes()  # Ensure colors are set
        # Clear screen, print position at top left, hide cursor
        print("\033[2J\033[H\033[?25l", end='', flush=True, file=sys.__stdout__)
        self.no_cls = self.paper == 8

    def __del__(self):
        if not self.no_cls:
            # Move cursor to bottom right and reset attributes
            print("\033[999;999H\n", end='', file=sys.__stdout__)
        self._reset_attributes()
        # Reveal cursor
        print("\033[?25h", end='', flush=True, file=sys.__stdout__)

    def _handle_char(self, char):
        """Process a single character of output."""
        code = ord(char)
        
        # If we're expecting a specific next char, handle it
        if self.wants_next_char is not None:
            handler = self.wants_next_char
            self.wants_next_char = None
            handler(code)
            return
        
        # Handle control codes
        if code < 32:
            self._handle_control_code(code)
            return
            
        # Handle special characters and block graphics
        if code in self.SPECIAL_CHARS:
            char = self.SPECIAL_CHARS[code]
        elif 128 <= code <= 143:
            char = self.BLOCK_CHARS[code]
        elif 144 <= code <= 164:
            # UDG characters - mapped to circled letters
            char = chr(0x24b6 + (code - 144))
            
        # Normal character output
        if self.cursor_x >= self.width:
            self._end_line()
        self._output_char(char)
        self.cursor_x += 1
    
    def _handle_control_code(self, code):
        """Handle ZX Spectrum control codes."""
        if code == 6:  # PRINT comma
            next_tab = ((self.cursor_x // 16) + 1) * 16
            self._tab_to(next_tab)
        elif code == 12:  # DELETE
            if self.cursor_x > 0:
                self.cursor_x -= 1
                print("\033[D \033[D", end='', flush=True, file=sys.__stdout__)
        elif code == 13 or code == 10:  # ENTER
            self._end_line()
        elif code == 16:  # INK
            self.wants_next_char = lambda c: self._set_color('ink', c)
        elif code == 17:  # PAPER
            self.wants_next_char = lambda c: self._set_color('paper', c)
        elif code == 18:  # FLASH
            self.wants_next_char = lambda c: self._set_attribute('flash', c)
        elif code == 19:  # BRIGHT
            self.wants_next_char = lambda c: self._set_attribute('bright', c)
        elif code == 20:  # INVERSE
            self.wants_next_char = lambda c: self._set_attribute('inverse', c)
        elif code == 21:  # OVER
            self.wants_next_char = lambda c: self._set_attribute('over', c)
        elif code == 22:  # AT
            self.wants_next_char = self._handle_at_y
        elif code == 23:  # TAB
            self.wants_next_char = self._tab_to
    
    def _handle_at_y(self, y):
        """Handle the Y coordinate of an AT command."""
        self.cursor_y = y
        self.wants_next_char = lambda x: self._handle_at_x(x)
    
    def _handle_at_x(self, x):
        """Handle the X coordinate of an AT command."""
        self.cursor_x = x
        self._move_cursor(self.cursor_x, self.cursor_y)
    
    def _set_color(self, type_, color):
        """Set ink or paper color."""
        if type_ == 'ink':
            self.ink = color & 15  # Allow for special colors 8 and 9
        else:
            self.paper = color & 15
        self._update_attributes()
    
    def _set_attribute(self, attr, value):
        """Set a binary attribute (BRIGHT, FLASH, etc)."""
        setattr(self, attr, value & 1)  # Ensure 0 or 1
        self._update_attributes()
    
    def _update_attributes(self):
        """Update terminal attributes based on current settings."""

        fg = self.ink    # Initially in ZX Spectrum colour space
        bg = self.paper
        
        # Handle contrasting colors (9)
        if fg == 9 or bg == 9:
            # If both are 9, default to white on black
            if fg == 9 and bg == 9:
                fg = 7  # White
                bg = 0  # Black
            # If ink is 9, contrast with paper
            elif self.ink == 9:
                fg = 0 if bg >= 4 else 7  # Black for light bg, White for dark
            # If paper is 9, contrast with ink
            else:
                bg = 0 if fg >= 4 else 7  # Black for light fg, White for dark

        # Convert to ANSI color space
        colorset = self.COLORS[self.bright]
        fg = colorset[self.ink]     
        bg = colorset[self.paper]

        # Build and print the ANSI sequence
        attrs = []
        
        # Handle transparent colors (8)
        if fg is not None:
            attrs.append(f"38;5;{fg}")  # Foreground
        else:
            attrs.append("39")  # Default foreground

        if bg is not None:
            attrs.append(f"48;5;{bg}")
        else:
            attrs.append("49")

        if self.inverse:
            attrs.append("7")  # Reverse video
        if self.flash:
            attrs.append("5")  # Blink
            
        print(f"\033[{';'.join(attrs)}m", end='', flush=True, file=sys.__stdout__)
    
    def _reset_attributes(self):
        """Reset all terminal attributes to default."""
        print("\033[0m\033[K", end='', flush=True, file=sys.__stdout__)
    
    def _end_line(self):
        """Handle end of line (either ENTER or line wrap)."""
        if self.no_cls:
            self._reset_attributes()
        print(file=sys.__stdout__)
        self.cursor_x = 0
        self.cursor_y += 1
        if self.no_cls:
            self._update_attributes()  # Restore current attributes for next line
    
    def _move_cursor(self, x, y):
        """Move cursor to absolute position."""
        print(f"\033[{y+1};{x+1}H", end='', flush=True, file=sys.__stdout__)
    
    def _tab_to(self, pos):
        """Tab to specific position, handling line wrapping."""
        if self.cursor_x >= self.width:
            self._end_line()
        while self.cursor_x < pos and self.cursor_x < self.width:
            print(" ", end='', flush=True, file=sys.__stdout__)
            self.cursor_x += 1
    
    def _output_char(self, char):
        """Output a single character with current attributes."""
        # If OVER is set, we don't move the cursor back
        if self.over:
            print(f"\033[7m{char}\033[27m" if self.inverse else char, 
                  end='', flush=True, file=sys.__stdout__)
        else:
            print(char, end='', flush=True, file=sys.__stdout__)
    
    # Required methods for TextIOBase
    def readable(self): return False
    def writable(self): return True
    def seekable(self): return False

def demo():
    zx = ZXOutputStream()
    sys.stdout = zx

    # Default colors (transparent)
    print("This uses terminal defaults")

    # Contrasting colors
    zx.write(chr(16) + chr(9))  # INK 9 (contrasting)
    zx.write(chr(17) + chr(6))  # PAPER yellow
    print("Contrasting text on yellow (contrast)")

    # Black on yellow
    zx.write(chr(16) + chr(0))  # INK black
    zx.write(chr(17) + chr(6))  # PAPER yellow
    print("Black on yellow")


    # Back to defaults
    zx.write(chr(16) + chr(8))  # INK transparent
    zx.write(chr(17) + chr(8))  # PAPER transparent
    print("Back to terminal defaults")

    # Try normal colors
    zx.write(chr(17) + chr(6))  # PAPER yellow
    zx.write(chr(16) + chr(1))  # INK blue
    print("Blue on yellow")

    # Now bright
    zx.write(chr(19) + chr(1))  # BRIGHT on
    print("Bright blue on bright yellow")


if __name__ == "__main__":
    demo()
