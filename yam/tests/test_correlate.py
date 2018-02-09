# Copyright 2017-2018 Tom Eulenfeld, GPLv3

import unittest
from obspy import read
import numpy as np

from yam.correlate import _fill_array, time_norm


class TestCase(unittest.TestCase):

    def test_fill_array(self):
        data = np.arange(5, dtype=float)
        self.assertIs(_fill_array(data), data)
        data2 = np.copy(data)
        mask = [True, False, False, True, False]
        data2 = _fill_array(data, mask, fill_value=0.)
        self.assertEqual(data2.data[0], 0.)
        self.assertEqual(data2.data[-2], 0.)

    def test_time_norm(self):
        stream = read().select(component='Z')
        stream.filter('highpass', freq=0.5)
        data1 = stream[0].data  # unmasked
        t = stream[0].stats.starttime
        stream.cutout(t + 18, t + 20)
        stream.merge()
        data2 = stream[0].data  # array with masked data
        mask = data2.mask

        data1_1bit = time_norm(np.copy(data1), '1bit')
        data2_1bit = time_norm(np.ma.copy(data2), '1bit')
        self.assertSetEqual(set(data1_1bit), {-1., 0., 1.})
        self.assertSetEqual(set(data2_1bit._data), {-1., 0., 1.})
        np.testing.assert_equal(data2_1bit.mask, mask)
        np.testing.assert_equal(data2_1bit._data[mask], 0.)

        data1_clip = time_norm(np.copy(data1), 'clip', clip_factor=2)
        data2_clip = time_norm(np.ma.copy(data2), 'clip', clip_factor=2)
        data1_clip0 = time_norm(np.copy(data1), 'clip', clip_factor=2,
                                clip_set_zero=True)
        data2_clip0 = time_norm(np.ma.copy(data2), 'clip', clip_factor=2,
                                clip_set_zero=True)
        clip_mask = np.abs(data1_clip) < np.abs(data1)
        with np.errstate(invalid='ignore'):
            clip_mask2 = np.abs(data2_clip) < np.abs(data2)
        np.testing.assert_equal(data2_clip.mask, mask)
        np.testing.assert_equal(data2_clip0.mask, mask)
        self.assertGreater(np.count_nonzero(clip_mask), 0)
        self.assertGreater(np.count_nonzero(clip_mask2), 0)
        np.testing.assert_equal(data1_clip0[clip_mask], 0.)
        np.testing.assert_equal(data2_clip0[clip_mask2], 0.)
        np.testing.assert_array_less(np.abs(data2_clip[clip_mask2]),
                                     np.abs(data2[clip_mask2]))

        data1_mute_envelope = time_norm(np.copy(data1), 'mute_envelope',
                                        mute_parts=4)
        data2_mute_envelope = time_norm(np.ma.copy(data2), 'mute_envelope',
                                        mute_parts=4)
        ind = np.abs(data1) > 0.5 * np.max(data1)
        np.testing.assert_equal(data1_mute_envelope[ind], 0.)
        np.testing.assert_equal(data2_mute_envelope[ind], 0.)

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
