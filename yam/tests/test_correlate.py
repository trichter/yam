# Copyright 2017-2018 Tom Eulenfeld, GPLv3

import unittest
from obspy import read
import numpy as np
from scipy.signal import periodogram
from scipy.fftpack import next_fast_len


from yam.correlate import _fill_array, spectral_whitening, time_norm


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
        stream = read().select(component='Z')
        filter_ = [0.5, 10]
        stream.filter('highpass', freq=0.5)
        times = stream[0].times()
        sr = stream[0].stats.sampling_rate
        data1 = np.copy(stream[0].data)  # unmasked
        t = stream[0].stats.starttime
        stream.cutout(t + 18, t + 20)
        stream.merge()
        data2 = stream[0].data  # array with masked data
        mask = data2.mask

        data1 = time_norm(data1, 'mute_envelope', mute_parts=4)
        data2 = time_norm(data2, 'mute_envelope', mute_parts=4)

        wdata1 = spectral_whitening(np.copy(data1))
        wdata1_f = spectral_whitening(np.copy(data1), sr=sr, filter=filter_)
        wdata1_sm = spectral_whitening(np.copy(data1), sr=sr, smooth=5)
        wdata1_sm_f = spectral_whitening(np.copy(data1), sr=sr, smooth=5,
                                         filter=filter_)
        wdata2 = spectral_whitening(np.ma.copy(data2))
        wdata2_f = spectral_whitening(np.ma.copy(data2), sr=sr, filter=filter_)
        wdata2_sm = spectral_whitening(np.ma.copy(data2), sr=sr, smooth=5)
        wdata2_sm_f = spectral_whitening(np.ma.copy(data2), sr=sr, smooth=5,
                                         filter=filter_)

        np.testing.assert_equal(wdata2.mask, mask)
        np.testing.assert_equal(wdata2_f.mask, mask)
        np.testing.assert_equal(wdata2_sm.mask, mask)
        np.testing.assert_equal(wdata2_sm_f.mask, mask)

        nfft = next_fast_len(len(data1))
        f, psd1 = periodogram(data1, sr, nfft=nfft)
        f, psd2 = periodogram(data2, sr, nfft=nfft)
        f, wpsd1 = periodogram(wdata1, sr, nfft=nfft)
        f, wpsd1_f = periodogram(wdata1_f, sr, nfft=nfft)
        f, wpsd1_sm = periodogram(wdata1_sm, sr, nfft=nfft)
        f, wpsd1_sm_f = periodogram(wdata1_sm_f, sr, nfft=nfft)
        f, wpsd2 = periodogram(wdata2, sr, nfft=nfft)
        f, wpsd2_f = periodogram(wdata2_f, sr, nfft=nfft)
        f, wpsd2_sm = periodogram(wdata2_sm, sr, nfft=nfft)
        f, wpsd2_sm_f = periodogram(wdata2_sm_f, sr, nfft=nfft)

        ind = f < 10
        psd1 /= np.max(psd1[ind])
        wpsd1 /= np.max(wpsd1[ind])
        wpsd1_f /= np.max(wpsd1_f[ind])
        wpsd1_sm /= np.max(wpsd1_sm[ind])
        wpsd1_sm_f /= np.max(wpsd1_sm_f[ind])
        psd2 /= np.max(psd2[ind])
        wpsd2 /= np.max(wpsd2[ind])
        wpsd2_f /= np.max(wpsd2_f[ind])
        wpsd2_sm /= np.max(wpsd2_sm[ind])
        wpsd2_sm_f /= np.max(wpsd2_sm_f[ind])

        np.testing.assert_allclose(wpsd1[1:-1], 1)
        np.testing.assert_array_less(wpsd1_f[1:] / wpsd1[1:], 1.01)

#        import matplotlib.pyplot as plt
#        ax1 = plt.subplot(221)
#        plt.title('contiguous data')
#        plt.plot(times, data1 / np.max(data1) / 5, label='data')
#        plt.plot(times, wdata1, label='whitened')
#        plt.plot(times, wdata1_f, label='+filtered')
#        plt.plot(times, wdata1_sm, label='+smoothed')
#        plt.plot(times, wdata1_sm_f, label='+both')
#        plt.xlabel('time (s)')
#        plt.legend()
#
#        plt.subplot(222, sharex=ax1, sharey=ax1)
#        plt.title('gappy data')
#        plt.plot(times, data2 / np.max(data2) / 5, label='data')
#        plt.plot(times, wdata2, label='whitened')
#        plt.plot(times, wdata2_f, label='+filtered')
#        plt.plot(times, wdata2_sm, label='+smoothed')
#        plt.plot(times, wdata2_sm_f, label='+bothed')
#        plt.xlabel('time (s)')
#        plt.legend()
#
#        ax3 = plt.subplot(223)
#        plt.plot(f, psd1, label='data')
#        plt.plot(f, wpsd1, label='whitened')
#        plt.plot(f, wpsd1_f, label='+filtered')
#        plt.plot(f, wpsd1_sm, label='+smoothed')
#        plt.plot(f, wpsd1_sm_f, label='+both')
#        plt.xlabel('frequency (Hz)')
#        plt.legend()
#
#        plt.subplot(224, sharex=ax3, sharey=ax3)
#        plt.plot(f, psd2, label='data')
#        plt.plot(f, wpsd2, label='whitened')
#        plt.plot(f, wpsd2_f, label='+filtered')
#        plt.plot(f, wpsd2_sm, label='+smoothed')
#        plt.plot(f, wpsd2_sm_f, label='+both')
#        plt.xlabel('frequency (Hz)')
#        plt.ylim(0, 1.2)
#        plt.legend()
#        plt.show()


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
