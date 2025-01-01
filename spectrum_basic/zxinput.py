import sys
import os
import time

def generic_pause(secs):
    "Wait for a number of seconds."
    if secs:
        time.sleep(secs)

def win_pause(secs=None):
    "Wait for a number of seconds or until a key is pressed."
    start_time = time.time()
    while True:
        if msvcrt.kbhit():
            return msvcrt.getch()
        if secs is not None and (time.time() - start_time) > secs:
            return
        time.sleep(0.1)

def unix_pause(secs=None, when=None):
    "Wait for a number of seconds or until a key is pressed. Returns the key."
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    when = when or termios.TCSAFLUSH

    try:
        # Switch terminal into raw mode
        tty.setraw(fd, when=when)
        start_time = time.time()
        waittime = secs

        while True:
            # Use select() to poll stdin
            rlist, _, _ = select.select([sys.stdin], [], [], waittime)
            if rlist:
                # Read one character
                return sys.stdin.read(1)
            if secs is not None:
                now = time.time()
                if (now - start_time) >= secs:
                    return
                waittime = secs - (now - start_time)
    finally:
        # Restore original terminal settings
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

try:
    import termios
    termios.tcgetattr, termios.tcsetattr
    import tty
    import select
    def pause(secs):
        """Pause for a number of seconds or until a key is pressed."""
        if os.isatty(sys.stdin.fileno()):
            unix_pause(secs)
        else:
            generic_pause(secs)
    def inkey():
        """Return any key currently pressed."""
        if os.isatty(sys.stdin.fileno()):
            return unix_pause(0, when=termios.TCSADRAIN)
        else:
            return sys.stdin.read(1)

except (ImportError, AttributeError):
    try:
        import msvcrt
        def pause(secs):
            """Pause for a number of seconds or until a key is pressed."""
            if os.isatty(sys.stdin.fileno()):
                win_pause(secs)
            else:
                generic_pause(secs)
        def inkey():
            """Return any key currently pressed."""
            if os.isatty(sys.stdin.fileno()):
                return win_pause(0)
            else:
                return sys.stdin.read(1)
    except ImportError:
        pause = generic_pause
        def inkey():
            return sys.stdin.read(1)

if __name__ == "__main__":
    print("Doing something unimportant... Try to push a key.")
    pause(0.5)
    print("Doing something important... So press a key to continue.")
    pause(None)
    print("We made it past the gate! Onward!")
    print("Press a key to exit.")
    while (kb := inkey()) is None:
        pass
    print(f"You pressed: {kb!r}")
