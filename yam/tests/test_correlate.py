# Copyright 2017-2018 Tom Eulenfeld, GPLv3

import unittest


class TestCase(unittest.TestCase):

    def test_fill_array(self):
        pass

    def test_time_norm(self):
        pass

    def test_spectral_whitening(self):
        pass

    def test_phase_shift(self):
        pass

    def test_preprocess(self):
        pass

    def test_correlate_traces(self):
        pass


def suite():
    return unittest.makeSuite(TestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
