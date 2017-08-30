# Copyright 2017 Tom Eulenfeld, GPLv3

import unittest
from pkg_resources import load_entry_point
from contextlib import redirect_stderr, redirect_stdout
import glob
import io
import logging
import os
import shutil
import sys
import time
import tempfile
import tqdm

import matplotlib
matplotlib.use('Agg')


def _replace_in_file(fname_src, fname_dest, str_src, str_dest):
    with open(fname_src) as f:
        text = f.read()
    text = text.replace(str_src, str_dest)
    with open(fname_dest, 'w') as f:
        f.write(text)


class TestCase(unittest.TestCase):

    def setUp(self):
        args = sys.argv[1:]
        self.verbose = '-v' in args
        self.permanent_tempdir = '-p' in args
        self.njobs = args[args.index('-n') + 1] if '-n'  in args else None
        if self.permanent_tempdir:
            tempdir = os.path.join(tempfile.gettempdir(), 'yam_test')
            if os.path.exists(tempdir) and '-d' in args:
                shutil.rmtree(tempdir)
            if not os.path.exists(tempdir):
                os.mkdir(tempdir)
        else:
            tempdir = tempfile.mkdtemp(prefix='yam_test_')
        self.cwd = os.getcwd()
        os.chdir(tempdir)
        self.tempdir = tempdir
        # for coverage put .coveragerc config file into tempdir
        # and append correct data_file parameter to config file
        covfn = os.path.join(self.cwd, '.coverage')
        if not os.path.exists('.coveragerc') and os.path.exists(covfn + 'rc'):
            _replace_in_file(covfn + 'rc', '.coveragerc', '[run]',
                             '[run]\ndata_file = ' + covfn)

        self.plotdir = os.path.join(self.tempdir, 'plots')
        self.script = load_entry_point('yam', 'console_scripts', 'yam')
        total = 71 - 2 * self.permanent_tempdir
        self.pbar = tqdm.tqdm(total=total, desc='CLI tests passed')

    def out(self, cmd, text=None):
        """Test if text is in output of command"""
        # for TRAVIS use maximal two cores
        if (self.njobs and '--njobs' not in cmd and
                cmd.split()[0] in ('correlate', 'stretch')):
            cmd = cmd + ' --njobs ' + self.njobs
        # disabling the logger is necessary, because the logging
        # configuration cannot be changed easily on subsequent calls
        # of yam in this test suite
        if self.verbose and cmd.split()[0] in ('correlate', 'stack', 'stretch'):
            if '-v' not in cmd:
                cmd = cmd + ' -vvv'
            logging.getLogger('yam').disabled = False
        elif self.verbose:
            logging.getLogger('yam').disabled = True
        # catching all output, print only if tests are run with -v
        try:
            with io.StringIO() as f:
                with redirect_stdout(f), redirect_stderr(f):
                    try:
                        self.script(cmd.split())
                    except SystemExit:
                        pass
                output = f.getvalue()
            if self.verbose:
                tqdm.tqdm.write(output)
        finally:
            self.pbar.update(1)
        if text is not None:
            self.assertIn(text, output)
        return output

    def checkplot(self, bname):
        """Test if plot file exists"""
        fname = os.path.join(self.plotdir, bname)
        self.assertTrue(os.path.exists(fname), msg='%s missing' % fname)

    def test_cli(self):
        # create tutorial
        self.out('-h', 'print information about')
        self.out('--version', '.')
        self.out('bla', 'invalid choice')
        if not self.permanent_tempdir:
            self.out('info', "No such file or directory: 'conf.json'")
        self.out('create', '')
        if not self.permanent_tempdir:
            self.out('info', 'Not found')
        self.out('create --tutorial')
        self.out('create --tutorial')

        # check basics
        _replace_in_file('conf.json', 'conf2.json', '"io"', '"io",')
        self.out('-c conf2.json info', 'parsing the conf')
        self.out('info', '3 stations')
        self.out('info stations', 'CX.PATCX..BHZ')
        self.out('info data', 'example_data/CX.PATCX')
        self.out('print stations', 'CX.PATCX..BHZ')
        pr = self.out('print data CX.PATCX..BHZ 2010-02-03', '2010-02-03')
        self.out('print data CX.PATCX..BHZ 2010-034', '2010-02-03')
        self.out('print prepdata CX.PATCX..BHZ 2010-02-03 1', '864001 samples')
        self.out('print data', 'seedid')
        self.out('print prepdata CX.PATCX..BHZ 2010-02-03', 'corrid')
        # check remove_response on one day
        _replace_in_file(
            'conf.json', 'conf2.json', '{"clip_factor": 2},',
            '{"clip_factor":2, "clip_set_zero":true},"remove_response":true,')
        self.out('-c conf2.json print prepdata CX.PATCX..BHZ 2010-02-03 1a')

        try:
            self.out('plot stations')
        except ImportError:
            pass
        else:
            self.checkplot('stations.png')
        self.out('plot data CX.PATCX..BHZ 2010-02-03')
        self.checkplot('data_CX.PATCX..BHZ_2010-02-03.png')
        self.out('plot prepdata CX.PATCX..BHZ 2010-02-03 1')
        self.checkplot('prepdata_CX.PATCX..BHZ_2010-02-03_c1.png')

        # check data_plugin
        data_plugin_file = """
from obspy import read
EXPR = ("example_data/{network}.{station}.{location}.{channel}__"
        "{t.year}{t.month:02d}{t.day:02d}*.mseed")
# def get_data(starttime, endtime, network, station, location, channel):
def get_data(starttime, endtime, **smeta):
     fname = EXPR.format(t=starttime, **smeta)
     stream = read(fname, 'MSEED')
     # load also previous file, because startime in day file is after 0:00
     fname = EXPR.format(t=starttime-5, **smeta)
     stream += read(fname, 'MSEED')
     return stream
        """
        with open('data.py', 'w') as f:
            f.write(data_plugin_file)
        _replace_in_file('conf.json', 'conf2.json', '"data_plugin": null',
                         '"data_plugin": "data : get_data"')
        self.out('-c conf2.json info', 'data : get_data')
        self.out('-c conf2.json info data', 'data : get_data')
        pr2 = self.out('-c conf2.json print data CX.PATCX..BHZ 2010-02-03')
        self.assertEqual(pr, pr2)

        # check correlation
        t1 = time.time()
        self.out('correlate 1')
        t2 = time.time()
        self.out('correlate 1 -v')
        t3 = time.time()
        if not self.permanent_tempdir:
            self.assertLess(t3 - t2, 0.5 * (t2 - t1))
        self.out('correlate auto')
        self.out('correlate 1a')
        self.out('info', 'c1_s1d: 7 combs')
        self.out('info', 'c1a_s1d: 3 combs')
        self.out('info', 'cauto: 2 combs')
        cauto_info = self.out('info cauto', 'CX.PATCX/.BHZ-.BHZ/2010-02-03')
        self.out('info c1_s1d/CX.PB06-CX.PB06/.BHZ-.BHZ', 'CX.PB06-CX.PB06')
        self.out('print cauto', '1201 samples')
        self.out('print c1_s1d/CX.PB06-CX.PB06/.BHZ-.BHZ', '11 Trace')
        # check if correlation without parallel processing gives the same
        # result keys
        with self.assertWarnsRegex(UserWarning, 'only top level keys'):
            self.out('remove cauto/CX.PATCX-CX.PATCX')
        self.out('remove cauto')
        self.out('correlate auto --njobs 1')
        cauto_info_seq = self.out('info cauto')
        self.assertEqual(cauto_info, cauto_info_seq)

        # check plots of correlation
        po = """--plot-options {"trim":[0,10],"figsize":[10,10]}"""
        self.out('plot c1_s1d --plottype vs_dist')
        self.checkplot('corr_vs_dist_c1_s1d_ZZ.png')
        self.out('plot c1_s1d/CX.PATCX-CX.PB01 --plottype wiggle')
        self.out('plot cauto --plottype wiggle %s' % po)
        bname = 'corr_vs_time_wiggle_c1_s1d_CX.PATCX-CX.PB01_'
        self.checkplot(bname + '.BHZ-.BHZ.png')
        self.checkplot(bname + '.BHN-.BHZ.png')
        bname = 'corr_vs_time_wiggle_cauto_CX.PATCX-CX.PATCX_'
        self.checkplot(bname + '.BHZ-.BHZ.png')
        self.checkplot(bname + '.BHN-.BHZ.png')
        self.out('plot c1_s1d/CX.PATCX-CX.PB01')
        self.out('plot cauto %s' % po)
        bname = 'corr_vs_time_c1_s1d_CX.PATCX-CX.PB01_'
        self.checkplot(bname + '.BHZ-.BHZ.png')
        self.checkplot(bname + '.BHN-.BHZ.png')
        bname = 'corr_vs_time_cauto_CX.PATCX-CX.PATCX_'
        self.checkplot(bname + '.BHZ-.BHZ.png')
        self.checkplot(bname + '.BHN-.BHZ.png')

        # check stacking
        self.out('stack c1_s1d 2d')
        self.out('info',  'c1_s1d_s2d: 7 combs')
        self.out('stack c1_s1d 2dm1d')
        self.out('info',  'c1_s1d_s2dm1d: 7 combs')
        self.out('stack c1_s1d 1')
        self.out('stack cauto 2')
        self.out('info',  'cauto_s2')
        self.out('remove c1_s1d_s2dm1d c1_s1d_s1')

        # check stretching
        po = """--plot-options {"show_line":true}"""
        self.out('stretch c1_s1d/CX.PATCX-CX.PATCX 1')
        self.out('stretch c1_s1d 1')
        self.out('stretch cauto/CX.PATCX-CX.PATCX/.BHZ-.BHZ 2')
        self.out('stretch cauto 2')
        self.out('stretch cauto 2 --njobs 1')
        self.out('stretch cauto/CX.PATCX-CX.PATCX/.BHZ-.BHZ 2')
        self.out('plot c1_s1d_t1/CX.PATCX-CX.PB01 %s' % po)
        self.out('plot cauto_t2 --plottype wiggle', 'not supported')
        self.out('plot cauto_t2')
        globexpr = os.path.join(self.plotdir, 'sim_mat_cauto_t2*.png')
        self.assertEqual(len(glob.glob(globexpr)), 6)
        globexpr = os.path.join(self.plotdir, 'sim_mat_c1_s1d_t1*.png')
        if not self.permanent_tempdir:
            self.assertEqual(len(glob.glob(globexpr)), 6)
        self.out('plot c1_s1d_t1')
        self.assertEqual(len(glob.glob(globexpr)), 21)
        self.out('info', 'c1_s1d_t1: 7 combs')
        self.out('info cauto_t2', 'CX.PATCX-CX.PATCX/.BHZ-.BHZ')
        info_t = self.out('print cauto_t2', 'lag_time_windows')
        self.assertGreater(len(info_t.splitlines()), 100)  # lots of lines

        # check export
        fname = 'dayplot.mseed'
        self.out('export data %s CX.PATCX..BHZ 2010-02-03' % fname)
        self.assertTrue(os.path.exists(fname), msg='%s missing' % fname)
        fname = 'dayplot.mseed'
        self.out('export data %s CX.PATCX..BHZ 2010-02-03' % fname)
        self.assertTrue(os.path.exists(fname), msg='%s missing' % fname)
        fname = 'some_auto_corrs.h5'
        self.out('export cauto/CX.PATCX-CX.PATCX %s --format H5' % fname)
        self.assertTrue(os.path.exists(fname), msg='%s missing' % fname)

        # check load (IPython mocked)
        sys.modules['IPython'] = unittest.mock.MagicMock()
        self.out('load cauto', 'Good Bye')
        self.out('load c1_s1d', 'Good Bye')
        self.out('load c1_s1d_t1/CX.PATCX-CX.PB01', 'Good Bye')

    def tearDown(self):
        os.chdir(self.cwd)
        if not self.permanent_tempdir:
            shutil.rmtree(self.tempdir)


def suite():
    return unittest.makeSuite(TestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
