# Copyright 2017 Tom Eulenfeld, GPLv3
"""Some utility functions"""

from importlib import import_module
import logging
import numbers
import os
from pkg_resources import resource_filename
import shutil
import sys
import tempfile

import numpy as np
import scipy.signal
import tqdm


log = logging.getLogger('yam.util')


class YamError(Exception):
    pass


class ParseError(YamError):
    pass


class ConfigError(YamError):
    pass


def _analyze_key(key):
    if '/' in key:
        key = key.split('/', 1)[0]
    return ''.join([k[0] for k in key.split('_')])


def _corr_id(trace):
    st = trace.stats
    id_ = (st.network1, st.station1, st.location1, st.channel1,
           st.network2, st.station2, st.location2, st.channel2)
    return '%s.%s.%s.%s-%s.%s.%s.%s' % id_


def _filter(stream, filter):
    if filter[0] is None:
        stream.filter("lowpass", freq=filter[1])
    elif filter[1] is None:
        stream.filter("highpass", freq=filter[0])
    else:
        stream.filter("bandpass", freqmin=filter[0], freqmax=filter[1])


def _load_func(modulename, funcname):
    """Load and return function from Python module"""
    sys.path.append(os.path.curdir)
    module = import_module(modulename)
    sys.path.pop(-1)
    func = getattr(module, funcname)
    return func


def _seedid2meta(seedid):
    net, sta, loc, cha = seedid.split('.')
    smeta = {'network': net, 'station': sta, 'location': loc, 'channel': cha}
    return smeta


def _time2sec(time):
    if not isinstance(time, numbers.Number):
        time, unit = float(time[:-1]), time[-1]
        assert unit in 'dh'
        time *= 24 * 3600 if unit == 'd' else 3600
    return time


def create_config(conf='conf.json', tutorial=False):
    shutil.copyfile(resource_filename('yam', 'conf_example.json'), conf)
    temp_dir = os.path.join(tempfile.gettempdir(), 'yam_example_data')
    template = os.path.join(temp_dir, 'example_data')
    station_template = os.path.join(temp_dir, 'example_inventory')
    try:
        num_files = (len([name for name in os.listdir(template)]),
                     len([name for name in os.listdir(station_template)]))
    except FileNotFoundError:
        num_files = (0, 0)
    if tutorial and num_files != (55, 3):
        print('Download example data from Geofon')
        from obspy import UTCDateTime as UTC
        from obspy.clients.fdsn.mass_downloader import (
            GlobalDomain, Restrictions, MassDownloader)
        domain = GlobalDomain()
        restrictions = Restrictions(
            starttime=UTC('2010-02-01'), endtime=UTC('2010-02-15'),
            network='CX', station='PATCX', location=None,
            channel_priorities=["BH[ZN]"], chunklength_in_sec=86400,
            reject_channels_with_gaps=False, minimum_length=0.5)
        mdl = MassDownloader(providers=['GFZ'])
        mdl.download(domain, restrictions, template, station_template)
        mdl = MassDownloader(providers=['GFZ'])
        restrictions.station = 'PB06'
        restrictions.endtime = UTC('2010-02-12')
        mdl.download(domain, restrictions, template, station_template)
        restrictions.station = 'PB01'
        restrictions.endtime = UTC('2010-02-05 08:00:00')
        restrictions.channel_priorities = ["BHZ"]
        mdl.download(domain, restrictions, template, station_template)
        restrictions.starttime = UTC('2010-02-08 00:00:00')
        restrictions.endtime = UTC('2010-02-09 23:55:00')
        restrictions.channel_priorities = ["BHZ"]
        mdl.download(domain, restrictions, template, station_template)
    if tutorial:
        dest_dir = os.path.dirname(conf)
        dest_dir_data = os.path.join(dest_dir, 'example_data')
        dest_dir_inv = os.path.join(dest_dir, 'example_inventory')
        if not os.path.exists(dest_dir_data):
            shutil.copytree(template, dest_dir_data)
        if not os.path.exists(dest_dir_inv):
            shutil.copytree(station_template, dest_dir_inv)


def get_data_window(stream, start=None, end=None, relative='middle'):
    """
    Return array with data in time window (start, end) around relative.

    'time' can stand for UTCDateTime, list of UTCDateTimes, header entry out of
    ('ponset', 'sonset', 'startime', 'endtime') or 'middle'
    :param stream: Stream object with data
    :param start, end: time or float (seconds) relative to param=relative
    :param relative: time, is needed if start or end in seconds (float)
    :return: np.array of shape (N_stream, N_data)
    """
    for tr in stream:
        if relative == 'middle':
            t1 = tr.stats.starttime
            t2 = tr.stats.endtime
            reft = t1 + 0.5 * (t2 - t1)
        else:
            reft = tr.stats[relative]
        if start is not None:
            start = reft + start
        if end is not None:
            end = reft + end
        tr = tr.slice(start, end)
    if len(stream) == 0:
        raise ValueError('Stream has length 0')
    samp = [tr.stats.sampling_rate for tr in stream]
    npts = [len(tr) for tr in stream]
    if min(samp) != max(samp):
        for tr in stream:
            tr.decimate(int(tr.stats.sampling_rate) // min(samp))
        log.warning('Downsampling stream because of differing sampling rate.')
    if min(npts) != max(npts):
        log.warning('Traces in stream have different NPTS. '
                    'Difference: %d samples' % (max(npts) - min(npts)))
    data = np.zeros((len(stream), max(npts)))
    for i, trace in enumerate(stream):
        data[i, :len(trace.data)] = trace.data
    return data


def smooth(x, window_len=None, window='flat', method='zeros'):
    """Smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.

    :param x: the input signal
    :param window_len: the dimension of the smoothing window; should be an
        odd integer
    :param window: the type of window from 'flat', 'hanning', 'hamming',
        'bartlett', 'blackman'
        flat window will produce a moving average smoothing.
    :param method: handling of border effects\n
        'zeros': zero padding on both ends (len(smooth(x)) = len(x))\n
        'reflect': pad reflected signal on both ends (same)\n
        'clip': pad signal on both ends with the last valid value (same)\n
        None: no handling of border effects
        (len(smooth(x)) = len(x) - len(window_len) + 1)

    See also:
    www.scipy.org/Cookbook/SignalSmooth
    """
    if window_len is None:
        return x
    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")
    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")
    if window_len < 3:
        return x
    if window not in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is one of 'flat', 'hanning', 'hamming',"
                         "'bartlett', 'blackman'")
    if method == 'zeros':
        s = np.r_[np.zeros((window_len - 1) // 2), x,
                  np.zeros(window_len // 2)]
    elif method == 'reflect':
        s = np.r_[x[(window_len - 1) // 2:0:-1], x,
                  x[-1:-(window_len + 1) // 2:-1]]
    elif method == 'clip':
        s = np.r_[x[0] * np.ones((window_len - 1) // 2), x,
                  x[-1] * np.ones(window_len // 2)]
    else:
        s = x
    if window == 'flat':
        w = np.ones(window_len, 'd')
    else:
        w = getattr(np, window)(window_len)
    return scipy.signal.fftconvolve(w / w.sum(), s, mode='valid')


class IterTime():
    def __init__(self, startdate, enddate, dt=24 * 3600):
        self.startdate = startdate
        self.enddate = enddate
        self.dt = dt

    def __len__(self):
        return int((self.enddate - self.startdate) / self.dt)

    def __iter__(self):
        t = self.startdate
        while t <= self.enddate:
            yield t
            t += self.dt


# https://stackoverflow.com/a/38739634
# not working yet for parallel processing
class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super(self.__class__, self).__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.tqdm.write(msg)
            self.flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            self.handleError(record)


LOGLEVELS = {0: 'CRITICAL', 1: 'WARNING', 2: 'INFO', 3: 'DEBUG'}


LOGGING_DEFAULT_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'capture_warnings': True,
    'formatters': {
        'file': {
            'format': ('%(asctime)s %(module)-6s%(process)-6d%(levelname)-8s'
                       '%(message)s')
        },
        'console': {
            'format': '%(levelname)-8s%(message)s'
        },
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'console',
            'level': None,
        },
        'console_tqdm': {
            'class': 'yam.util.TqdmLoggingHandler',
            'formatter': 'console',
            'level': None,
        },
        'file': {
            'class': 'logging.FileHandler',
            'formatter': 'file',
            'level': None,
            'filename': None,
        },
    },
    'loggers': {
        'yam': {
            'handlers': ['console_tqdm', 'file'],
            'level': 'DEBUG',
            'propagate': False,
        },
        'py.warnings': {
            'handlers': ['console_tqdm', 'file'],
            'level': 'DEBUG',
            'propagate': False,
        }

    }
}
