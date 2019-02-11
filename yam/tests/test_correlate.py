# Copyright 2017-2018 Tom Eulenfeld, GPLv3
import unittest
import sys

import numpy as np
from obspy import read, read_inventory, UTCDateTime as UTC
from obspy.signal.cross_correlation import correlate, xcorr_max
from scipy.signal import periodogram
from scipy.fftpack import next_fast_len

from yam.correlate import (_fill_array, _downsample_and_shift,
                           _make_same_length,
                           correlate as yam_correlate,
                           correlate_traces, preprocess, spectral_whitening,
                           time_norm)


class TestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        args = sys.argv[1:]
        cls.njobs = int(args[args.index('-n') + 1]) if '-n'  in args else None

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
        tr1 = stream[0].copy()  # unmasked
        t = stream[0].stats.starttime
        stream.cutout(t + 18, t + 20)
        stream.merge()
        tr2 = stream[0]  # array with masked data
        mask = tr2.data.mask
        tr1_1bit = time_norm(tr1.copy(), '1bit')
        #tr2_1bit = time_norm(np.ma.copy(data2), '1bit', sr)
        tr2_1bit = time_norm(tr2.copy(), '1bit')
        self.assertSetEqual(set(tr1_1bit.data), {-1., 0., 1.})
        self.assertSetEqual(set(tr2_1bit.data._data), {-1., 0., 1.})
        np.testing.assert_equal(tr2_1bit.data.mask, mask)
        np.testing.assert_equal(tr2_1bit.data._data[mask], 0.)

        tr1_clip = time_norm(tr1.copy(), 'clip', clip_value=2)
        tr2_clip = time_norm(tr2.copy(), 'clip', clip_value=2)
        tr1_clip0 = time_norm(tr1.copy(), 'clip', clip_value=2,
                              clip_mode='zero')
        tr2_clip0 = time_norm(tr2.copy(), 'clip', clip_value=2,
                              clip_mode='zero')
        tr2b_clip0 = time_norm(tr2.copy(), 'clip', clip_value=2,
                               clip_mode='mask')
        clip_mask = np.abs(tr1_clip.data) < np.abs(tr1.data)
        with np.errstate(invalid='ignore'):
            clip_mask2 = np.abs(tr2_clip.data) < np.abs(tr2.data)
        np.testing.assert_equal(tr2_clip.data.mask, mask)
        np.testing.assert_equal(tr2_clip0.data.mask, mask)
        self.assertGreater(np.count_nonzero(clip_mask), 0)
        self.assertGreater(np.count_nonzero(clip_mask2), 0)
        np.testing.assert_equal(tr1_clip0.data[clip_mask], 0.)
        np.testing.assert_equal(tr2_clip0.data[clip_mask2], 0.)
        np.testing.assert_array_less(np.abs(tr2_clip.data[clip_mask2]),
                                     np.abs(tr2.data[clip_mask2]))
        self.assertGreater(np.count_nonzero(tr2b_clip0.data.mask),
                           np.count_nonzero(mask))

        tr1_me = time_norm(tr1.copy(), 'mute_envelope', mute_parts=4)
        tr2_me = time_norm(tr2.copy(), 'mute_envelope', mute_parts=4)
        ind = np.abs(tr1.data) > 0.5 * np.max(tr1.data)
        np.testing.assert_equal(tr1_me.data[ind], 0.)
        np.testing.assert_equal(tr2_me.data[ind], 0.)


    def test_spectral_whitening(self):
        stream = read().select(component='Z')
        filter_ = [0.5, 10]
        stream.filter('highpass', freq=0.5)
        tr1 = stream[0].copy()  # unmasked
        t = tr1.stats.starttime
        stream.cutout(t + 18, t + 20)
        stream.merge()
        tr2 = stream[0]  # trace with masked data
        mask = tr2.data.mask

        time_norm(tr1, 'mute_envelope', mute_parts=4)
        time_norm(tr2, 'mute_envelope', mute_parts=4)

        trw1 = spectral_whitening(tr1.copy())
        trw1_f = spectral_whitening(tr1.copy(), filter=filter_)
        trw1_sm = spectral_whitening(tr1.copy(), smooth=5)
        trw1_sm_f = spectral_whitening(tr1.copy(), smooth=5, filter=filter_)
        trw2 = spectral_whitening(tr2.copy())
        trw2_f = spectral_whitening(tr2.copy(), filter=filter_)
        trw2_sm = spectral_whitening(tr2.copy(), smooth=5)
        trw2_sm_f = spectral_whitening(tr2.copy(), smooth=5, filter=filter_)

        np.testing.assert_equal(trw2.data.mask, mask)
        np.testing.assert_equal(trw2_f.data.mask, mask)
        np.testing.assert_equal(trw2_sm.data.mask, mask)
        np.testing.assert_equal(trw2_sm_f.data.mask, mask)

        nfft = next_fast_len(len(tr1.data))
        sr = stream[0].stats.sampling_rate
        f, psd1 = periodogram(tr1.data, sr, nfft=nfft)
        f, psd2 = periodogram(tr2.data, sr, nfft=nfft)
        f, wpsd1 = periodogram(trw1.data, sr, nfft=nfft)
        f, wpsd1_f = periodogram(trw1_f.data, sr, nfft=nfft)
        f, wpsd1_sm = periodogram(trw1_sm.data, sr, nfft=nfft)
        f, wpsd1_sm_f = periodogram(trw1_sm_f.data, sr, nfft=nfft)
        f, wpsd2 = periodogram(trw2.data, sr, nfft=nfft)
        f, wpsd2_f = periodogram(trw2_f.data, sr, nfft=nfft)
        f, wpsd2_sm = periodogram(trw2_sm.data, sr, nfft=nfft)
        f, wpsd2_sm_f = periodogram(trw2_sm_f.data, sr, nfft=nfft)

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
#        times = tr1.times()
#        plt.plot(times, 20 * tr1.data, label='data')
#        plt.plot(times, trw1.data, label='whitened')
#        plt.plot(times, trw1_f.data, label='+filtered')
#        plt.plot(times, trw1_sm.data, label='+smoothed')
#        plt.plot(times, trw1_sm_f.data, label='+both')
#        plt.xlabel('time (s)')
#        plt.legend()
#
#        plt.subplot(222, sharex=ax1, sharey=ax1)
#        plt.title('gappy data')
#        plt.plot(times, tr2.data / np.max(tr2.data) / 5, label='data')
#        plt.plot(times, trw2.data, label='whitened')
#        plt.plot(times, trw2_f.data, label='+filtered')
#        plt.plot(times, trw2_sm.data, label='+smoothed')
#        plt.plot(times, trw2_sm_f.data, label='+bothed')
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

    def test_shift(self):
        tr = read()[0]
        dt = tr.stats.delta
        t = tr.stats.starttime = UTC('2018-01-01T00:00:10.000000Z')
        tr2 = tr.copy()
        _downsample_and_shift(tr2)
        self.assertEqual(tr2, tr)

        tr2 = tr.copy()
        tr2.stats.starttime = t + 0.1 * dt
        _downsample_and_shift(tr2)
        self.assertEqual(tr2.stats.starttime, t)

        tr2 = tr.copy()
        tr2.stats.starttime = t - 0.1 * dt
        _downsample_and_shift(tr2)
        self.assertEqual(tr2.stats.starttime, t)

        tr2 = tr.copy()
        tr2.stats.starttime = t - 0.49 * dt
        _downsample_and_shift(tr2)
        self.assertEqual(tr2.stats.starttime, t)

        tr2 = tr.copy()
        tr2.stats.starttime = t - 0.0001 * dt
        _downsample_and_shift(tr2)
        self.assertEqual(tr2.stats.starttime, t)

        # shift cumulatively by +1 sample
        tr2 = tr.copy()
        tr2.stats.starttime += 0.3 * dt
        _downsample_and_shift(tr2)
        tr2.stats.starttime += 0.3 * dt
        _downsample_and_shift(tr2)
        tr2.stats.starttime += 0.4 * dt
        _downsample_and_shift(tr2)
        self.assertEqual(tr2.stats.starttime, t)
        np.testing.assert_allclose(tr2.data[201:-200], tr.data[200:-201],
                                   rtol=1e-2, atol=1)
        cc = correlate(tr2.data, tr.data, 1000)
        shift, cc_max = xcorr_max(cc)
        self.assertEqual(shift, 1)
        self.assertGreater(cc_max, 0.995)

        # shift cumulatively by -1 sample
        tr2 = tr.copy()
        tr2.stats.starttime -= 0.3 * dt
        _downsample_and_shift(tr2)
        tr2.stats.starttime -= 0.3 * dt
        _downsample_and_shift(tr2)
        tr2.stats.starttime -= 0.4 * dt
        _downsample_and_shift(tr2)
        self.assertEqual(tr2.stats.starttime, t)
        np.testing.assert_allclose(tr2.data[200:-201], tr.data[201:-200],
                                   rtol=1e-2, atol=2)
        cc = correlate(tr2.data, tr.data, 1000)
        shift, cc_max = xcorr_max(cc)
        self.assertEqual(shift, -1)
        self.assertGreater(cc_max, 0.995)

    def test_downsample_and_shift(self):
        tr = read()[0]
        t = tr.stats.starttime = UTC('2018-01-01T00:00:10.000000Z')
        # decimate
        tr2 = _downsample_and_shift(tr.copy(), 50.)
        self.assertEqual(tr2.stats.sampling_rate, 50)
        # interpolate
        tr2 = _downsample_and_shift(tr.copy(), 40.)
        self.assertEqual(tr2.stats.sampling_rate, 40)
        # decimate and time shift
        tr2 = tr.copy()
        tr2.stats.starttime += 0.002
        tr2 = _downsample_and_shift(tr2, 50.)
        self.assertEqual(tr2.stats.sampling_rate, 50)
        self.assertEqual(tr2.stats.starttime, t)
        tr2 = tr.copy()
        tr2.stats.starttime -= 0.002
        tr2 = _downsample_and_shift(tr2, 50.)
        self.assertEqual(tr2.stats.sampling_rate, 50)
        self.assertEqual(tr2.stats.starttime, t)
        # interpolate and time shift
        tr2 = tr.copy()
        tr2.stats.starttime += 0.002
        tr2 = _downsample_and_shift(tr2, 40.)
        self.assertEqual(tr2.stats.sampling_rate, 40)
        self.assertEqual(tr2.stats.starttime - tr2.stats.delta, t)
        tr2 = tr.copy()
        tr2.stats.starttime -= 0.002
        tr2 = _downsample_and_shift(tr2, 40.)
        self.assertEqual(tr2.stats.sampling_rate, 40)
        self.assertEqual(tr2.stats.starttime, t)

    def test_preprocess(self):
        stream = read()
        day = UTC('2018-01-02')
        for tr in stream:
            tr.stats.starttime = day
        tr = stream[1]
        tr.id = 'GR.FUR..BH' + tr.stats.channel[-1]
        tr.stats.sampling_rate = 80.
        tr = stream[2]
        tr.id = 'GR.WET..BH' + tr.stats.channel[-1]
        tr.stats.sampling_rate = 50.
        stream = stream.cutout(day + 0.01, day + 10)
        stream = stream.cutout(day + 14, day + 16.05)
        norm = ('clip', 'spectral_whitening', 'mute_envelope', '1bit')
        # see https://docs.scipy.org/doc/numpy-1.13.0/release.html#
        # assigning-to-slices-views-of-maskedarray
        ignore_msg = r'setting an item on a masked array which has a shared'
        with np.warnings.catch_warnings():
            np.warnings.filterwarnings('ignore', ignore_msg)
            preprocess(stream, day=day, inventory=read_inventory(),
                       remove_response=True,
                       filter=None,
                       normalization=norm,
                       time_norm_options=None,
                       spectral_whitening_options=None,
                       decimate=5)
        for tr in stream:
            self.assertEqual(tr.stats.sampling_rate, 10)
        for tr in stream:
            self.assertEqual(set(tr.data._data), {-1, 0, 1})
            mask = np.ma.getmask(tr.data)
            np.testing.assert_equal(tr.data[mask]._data, 0)
            self.assertGreater(np.count_nonzero(mask), 0)
        self.assertEqual(len(stream), 3)

    def test_correlate_traces(self):
        stream = read().sort()
        tr = correlate_traces(stream[0], stream[1])
        shift, corr = xcorr_max(tr.data)
        self.assertLess(abs(shift / len(tr)), 0.01)
        self.assertGreater(abs(corr), 0.2)
        self.assertEqual(tr.id, 'RJOB.EHE.RJOB.EHN')

    def test_correlate(self):
        stream = read()
        stream2 = stream.copy()
        stream3 = stream.copy()
        for tr in stream2:
            tr.id = 'GR.FUR..BH' + tr.stats.channel[-1]
            tr.stats.sampling_rate = 80.
        for tr in stream3:
            tr.id = 'GR.WET..BH' + tr.stats.channel[-1]
            tr.stats.sampling_rate = 50.
        stream = stream + stream2 + stream3
        day = UTC('2018-01-02')
        for tr in stream:
            tr.stats.starttime = day
        # create some gaps
        stream = stream.cutout(day + 0.01, day + 10)
        stream = stream.cutout(day + 14, day + 16.05)

        # prepare mock objects for call to yam_correlate
        def data(starttime, endtime, **kwargs):
            return stream.select(**kwargs).slice(starttime, endtime)
        io = {'data': data, 'data_format': None, 'inventory': read_inventory()}
        res = yam_correlate(io, day, 'outkey', keep_correlations=True)
        self.assertEqual(len(res['corr']), 6)
        # by default only 'ZZ' combinations
        for tr in res['corr']:
            self.assertEqual(tr.stats.station[-1], 'Z')
            self.assertEqual(tr.stats.channel[-1], 'Z')
            if len(set(tr.id.split('.'))) == 2:  # autocorr
                np.testing.assert_allclose(xcorr_max(tr.data), (0, 1.))

        res = yam_correlate(
                  io, day, 'outkey',
                  station_combinations=('GR.FUR-GR.WET', 'RJOB-RJOB'),
                  component_combinations=('ZZ', 'NE', 'NR'),
                  keep_correlations=True,
                  stack='1d', njobs=self.njobs)
        self.assertEqual(len(res['corr']), 7)
        self.assertEqual(len(res['stack']), 7)
        ids = ['RJOB.EHE.RJOB.EHN', 'RJOB.EHZ.RJOB.EHZ',
               'FUR.BHE.WET.BHN', 'FUR.BHN.WET.BHE',
               'FUR.BHR.WET.BHN', 'FUR.BHN.WET.BHR',
               'FUR.BHZ.WET.BHZ']
        for tr in res['corr']:
            self.assertIn(tr.id, ids)
            if len(set(tr.id.split('.'))) == 2:  # autocorr
                np.testing.assert_allclose(xcorr_max(tr.data), (0, 1.))

        res = yam_correlate(
                  io, day, 'outkey', only_auto_correlation=True,
                  station_combinations=('GR.FUR-GR.WET', 'RJOB-RJOB'),
                  component_combinations=['ZN', 'RT'], njobs=self.njobs,
                  keep_correlations=True,
                  remove_response=True)
        self.assertEqual(len(res['corr']), 1)
        tr = res['corr'][0]
        self.assertEqual(tr.stats.station[-1], 'N')
        self.assertEqual(tr.stats.channel[-1], 'Z')

        stream.traces = [tr for tr in stream if tr.stats.channel[-1] != 'N']
        res = yam_correlate(
                  io, day, 'outkey',
                  station_combinations=('GR.FUR-GR.WET', 'RJOB-RJOB'),
                  component_combinations=('NT', 'NR'), discard=0.0,
                  keep_correlations=True)
        self.assertEqual(res, None)


    def _test_make_same_length(self):
        tr1 = read()[0]
        tr1.stats.sampling_rate = 1
        utc = tr1.stats.starttime
        tr2 = tr1.copy()
        tr2.stats.starttime -= 0.1
        for dt in np.linspace(0, 1, 101):
            tr1a = tr1.slice(utc + dt, utc + 3.7 + dt)
            tr2a = tr2.slice(utc + dt, utc + 3.7 + dt)
            _make_same_length(tr1a, tr2a)
            self.assertEqual(len(tr1a), len(tr2a))
            self.asertLess(tr1a.stats.starttime - tr2a.stats.starttime, 0.5)


def suite():
    return unittest.makeSuite(TestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
