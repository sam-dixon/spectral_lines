from unittest import TestCase

import spectral_lines.base as base

class TestBase(TestCase):

    def test_lines(self):
        lines = base.lines
        self.assertTrue(len(lines.keys()) == 8)