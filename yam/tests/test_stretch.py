# Copyright 2018 Tom Eulenfeld, GPLv3

import unittest


class TestCase(unittest.TestCase):

    def test_stretch(self):
        pass


def suite():
    return unittest.makeSuite(TestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
