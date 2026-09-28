"""Tests for the bytes spectrum_basic writes for a line, compared with what
the Spectrum's own ROM stores when the line is typed in.

Run with:  python -m unittest discover -s tests
"""
import unittest

from spectrum_basic import parse_string


def line_body(src):
    """The bytes of a one-line program's body: without the 4-byte line
    header (number and length) and without the final 0x0D."""
    data = bytes(parse_string(src))
    assert data[-1] == 0x0D
    return data[4:-1]


def hidden(n):
    """A small integer's hidden number: 0x0E, then the 5-byte integer form."""
    return bytes([0x0E, 0, 0, n & 0xFF, n >> 8, 0])


class TestBin(unittest.TestCase):
    """BIN literals are followed by their value as a hidden number, like any other number."""

    def test_bin_digits(self):
        self.assertEqual(line_body('10 LET a=BIN 101\n'),
                         b'\xf1a=\xc4101' + hidden(5))

    def test_bin_byte(self):
        self.assertEqual(line_body('10 POKE 23606,BIN 11111111\n'),
                         b'\xf423606' + hidden(23606) + b',\xc411111111' + hidden(255))

    def test_bin_zero(self):
        self.assertEqual(line_body('10 LET a=BIN 0\n'),
                         b'\xf1a=\xc40' + hidden(0))


class TestLprint(unittest.TestCase):
    """LPRINT takes the same items and separators as PRINT."""

    def test_lprint_string(self):
        self.assertEqual(line_body('10 LPRINT "p"\n'), b'\xe0"p"')

    def test_lprint_separators(self):
        self.assertEqual(line_body('10 LPRINT "a";1,x\'\n'),
                         b'\xe0"a";1' + hidden(1) + b",x'")

    def test_lprint_same_as_print(self):
        for items in ['"x"', 'a;b', 'AT 1,2;"y"', 'TAB 5;z,', '#3;"q"']:
            p = line_body(f'10 PRINT {items}\n')
            lp = line_body(f'10 LPRINT {items}\n')
            self.assertEqual(p[0], 0xF5)
            self.assertEqual(lp, bytes([0xE0]) + p[1:], items)


class TestUnaryMinus(unittest.TestCase):
    """A minus sign before a number is stored as '-' and the number's digits,
    with the hidden number positive; also where an operand follows ^ or a
    function name."""

    def test_after_power(self):
        self.assertEqual(line_body('10 LET a=2^-1\n'),
                         b'\xf1a=2' + hidden(2) + b'^-1' + hidden(1))

    def test_after_function(self):
        self.assertEqual(line_body('10 PRINT SIN -1\n'), b'\xf5\xb2-1' + hidden(1))
        self.assertEqual(line_body('10 PRINT ABS -x\n'), b'\xf5\xbd-x')

    def test_unchanged_elsewhere(self):
        self.assertEqual(line_body('10 LET a=-1\n'), b'\xf1a=-1' + hidden(1))
        self.assertEqual(line_body('10 DATA -1,2\n'), b'\xe4-1' + hidden(1) + b',2' + hidden(2))

    def test_power_binds_tighter_than_minus(self):
        # -2^2 is -(2^2), as on the Spectrum
        self.assertEqual(' '.join(str(parse_string('10 LET a=-2^2\n')).split()), '10 LET a = -2 ^ 2')


class TestValDollar(unittest.TestCase):
    """VAL$ is a function name of its own, not VAL followed by '$'."""

    def test_val_dollar_of_function(self):
        self.assertEqual(line_body('10 LET a$=VAL$ STR$ 1\n'),
                         b'\xf1a$=\xae\xc11' + hidden(1))

    def test_len_of_val_dollar(self):
        self.assertEqual(line_body('10 LET n=LEN VAL$ "x"\n'), b'\xf1n=\xb1\xae"x"')

    def test_val_still_works(self):
        self.assertEqual(line_body('10 LET n=VAL "2"\n'), b'\xf1n=\xb0"2"')


if __name__ == '__main__':
    unittest.main()
