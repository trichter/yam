# Copyright 2018 Tom Eulenfeld, GPLv3

import os.path
import tempfile
import unittest

import numpy as np
from obspy import Stream, Trace, UTCDateTime as UTC

from yam.stretch import stretch
from yam.commands import write_dict, read_dicts


class TestCase(unittest.TestCase):

    def test_stretch(self):
        h = {'sampling_rate': 100}
        # TODO: allow to call stretch without these headers
        h['network1'] = h['network2'] = 'NET'
        h['station1'] = h['station2'] = h['network'] = h['location'] = 'STA'
        h['location1'] = h['location2'] = ''
        h['channel1'] = h['channel2'] = h['location'] = h['channel'] = 'HHZ'
        h['dist'] = h['azi'] = h['baz'] = 0
        vel_changes = [0, 1, -1]
        traces = []
        dt = 24 * 3600
        t0 = UTC()
        for i, v in enumerate(vel_changes):
            mul = 1 + v / 100
            # there is a small difference, because the routines from MIIC
            # use the following approximation for stretching:
            # mul = np.exp(v / 100)
            t = np.linspace(-10 * mul, 10 * mul, 10001)
            data = np.cos(2 * np.pi * t)
            h['starttime'] = t0 + i * dt
            tr = Trace(data, header=h)
            traces.append(tr)
        d = stretch(Stream(traces), reftr=traces[0], str_range=1.1,
                    nstr=2201,
                    time_windows=[[1], 4], sides='both')
        expect = np.array(vel_changes)[:, np.newaxis]
        np.testing.assert_allclose(d['velchange_vs_time'], expect, atol=0.008)
        # routine from miic uses approximation exp(dv/v) = 1 + dv/v
        corrected = -np.log(d['velchange_vs_time'] / -100 + 1) * 100
        np.testing.assert_allclose(corrected, expect, rtol=1e-4)
        # test writing and reading
        with tempfile.TemporaryDirectory(prefix='yam_') as tmpdir:
            fname = os.path.join(tmpdir, 'stretch.h5')
            d['attrs']['key'] = 'test'
            write_dict(d, fname)
            d2 = read_dicts(fname)[0]
            for key in d:
                if key == 'sim_mat':
                    np.testing.assert_allclose(d2[key], d[key], rtol=1e-3)
                elif isinstance(d2[key], np.ndarray):
                    np.testing.assert_equal(d2[key], d[key])
                else:
                    self.assertEqual(d2[key], d[key])
            d2['attrs']['key'] = 'test2'
            write_dict(d2, fname)


def suite():
    return unittest.makeSuite(TestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
