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


if __name__ == '__main__':
    unittest.main()
