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


if __name__ == '__main__':
    unittest.main()
