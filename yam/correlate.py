# Copyright 2017-2018 Tom Eulenfeld, GPLv3
"""Preprocessing and correlation"""
from functools import partial
import itertools
import logging
import multiprocessing

import numpy as np
from numpy.fft import rfft, irfft, rfftfreq
import obspy
from obspy.core import Stream
from obspy.geodetics import gps2dist_azimuth
from obspy.signal.cross_correlation import correlate as obscorr
from scipy.fftpack import fft, ifft, fftshift, ifftshift, next_fast_len
from scipy.signal import freqz, iirfilter, hilbert

from yam.util import _filter, IterTime, smooth as smooth_func, _time2sec
import yam.stack


log = logging.getLogger('yam.correlate')


def start_parallel_jobs_inner_loop(tasks, do_work, njobs=1):
    if njobs == 1:
        results = [do_work(task) for task in tasks]
    else:
        pool = multiprocessing.Pool(njobs)
        results = pool.map(do_work, tasks)
        pool.close()
        pool.join()
    return results


def _fill_array(data, mask=None, fill_value=None):
    """
    Mask numpy array and/or fill array value without demasking.

    Additionally set fill_value to value.
    If data is not a MaskedArray and mask is None returns silently data.

    :param mask: apply mask to array
    :param fill_value: fill value
    """
    if mask is not None and mask is not False:
        data = np.ma.MaskedArray(data, mask=mask, copy=False)
    if np.ma.is_masked(data) and fill_value is not None:
        data._data[data.mask] = fill_value
        np.ma.set_fill_value(data, fill_value)
#    elif not np.ma.is_masked(data):
#        data = np.ma.filled(data)
    return data


def time_norm(tr, method,
              clip_factor=None, clip_set_zero=None,
              clip_value=2, clip_std=True, clip_mode='clip',
              mute_parts=48, mute_factor=2, plugin=None,
              plugin_options={}):
    """
    Calculate normalized data, see e.g. Bensen et al. (2007)

    :param tr: Trace to manipulate
    :param str method:
        1bit: reduce data to +1 if >0 and -1 if <0\n
        clip: clip data to value or multiple of root mean square (rms)\n
        mute_envelope: calculate envelope and set data to zero where envelope
        is larger than specified
        plugin: use own function
    :param mask_zeros: mask values that are set to zero, they will stay zero
        in the further processing
    :param float clip_value: value for clipping or list of lower and upper
        value
    :param bool clip_std: Multiply clip_value with rms of data
    :param bool clip_mode:
        'clip': clip data
        'zero': set clipped data to zero
        'mask': set clipped data to zero and mask it
    :param int mute_parts: mean of the envelope is calculated by dividing the
        envelope into several parts, the mean calculated in each part and
        the median of this averages defines the mean envelope
    :param float mute_factor: mean of envelope multiplied by this
        factor defines the level for muting
    :param str plugin: function in the form module:func
    :param dict plugin_options: kwargs passed to plugin

    :return: normalized data
    """
    data = tr.data
    data = _fill_array(data, fill_value=0)
    mask = np.ma.getmask(data)
    if method == '1bit':
        np.sign(data, out=data)
    elif method == 'clip':
        if clip_factor is not None:
            from warnings import warn
            msg = 'clip_factor is deprecated, use clip_value instead'
            warn(msg, DeprecationWarning)
            clip_value = clip_factor
        if clip_set_zero is not None:
            from warnings import warn
            msg = 'clip_set_zero is deprecated, use clip_mode instead'
            warn(msg, DeprecationWarning)
            clip_mode = 'zero' if clip_set_zero else 'clip'
        from collections import Iterable
        if not isinstance(clip_value, Iterable):
            clip_value = [-clip_value, clip_value]
        if clip_std:
            std = np.std(data)
            clip_value = [clip_value[0] * std, clip_value[1] * std]
        if clip_mode == 'clip':
            np.clip(data, *clip_value, out=data)
        else:
            cmask = np.logical_or(data < clip_value[0], data > clip_value[1])
            if clip_mode == 'mask':
                mask = np.logical_or(np.ma.getmaskarray(data), cmask)
            elif clip_mode == 'zero':
                data[cmask] = 0
            else:
                raise ValueError('clip_mode must be one of clip, zeros, mask')
    elif method == 'mute_envelope':
        N = next_fast_len(len(data))
        envelope = np.abs(hilbert(data, N))[:len(data)]
        levels = [np.mean(d) for d in np.array_split(envelope, mute_parts)]
        level = mute_factor * np.median(levels)
        data[envelope > level] = 0
    elif method == 'plugin':
        from yam.util import _load_func
        modulename, funcname = plugin.split(':')
        func = _load_func(modulename.strip(), funcname.strip())
        func(tr, **plugin_options)
        data = tr.data
    else:
        msg = 'The method passed to time_norm is not known: %s.' % method
        raise ValueError(msg)
    tr.data = _fill_array(data, mask=mask, fill_value=0)
    return tr


# http://azitech.wordpress.com/
# 2011/03/15/designing-a-butterworth-low-pass-filter-with-scipy/
def _filter_resp(freqmin, freqmax, corners=2, zerophase=False, sr=None,
                 N=None, whole=False):
    """
    Complex frequency response of Butterworth-Bandpass Filter.

    :param freqmin: Pass band low corner frequency.
    :param freqmax: Pass band high corner frequency.
    :param corners: Filter corners
    :param zerophase: If True, apply filter once forwards and once backwards.
        This results in twice the number of corners but zero phase shift in
        the resulting filtered trace.
    :param sr: Sampling rate in Hz.
    :param N,whole: passed to scipy.signal.freqz

    :return: frequencies and complex response
    """
    df = sr
    fe = 0.5 * df
    low = freqmin / fe
    high = freqmax / fe
    # raise for some bad scenarios
    if high > 1:
        high = 1.0
        msg = "Selected high corner frequency is above Nyquist. " + \
              "Setting Nyquist as high corner."
        log.warning(msg)
    if low > 1:
        msg = "Selected low corner frequency is above Nyquist."
        raise ValueError(msg)
    [b, a] = iirfilter(corners, [low, high], btype='band',
                       ftype='butter', output='ba')
    freqs, values = freqz(b, a, N, whole=whole)
    if zerophase:
        values *= np.conjugate(values)
    return freqs, values


def spectral_whitening(tr, smooth=None, filter=None,
                       waterlevel=1e-8, mask_again=True):
    """
    Apply spectral whitening to data

    Data is divided by its smoothed (Default: None) amplitude spectrum.

    :param tr: trace to manipulate
    :param smooth: length of smoothing window in Hz
        (default None -> no smoothing)
    :param filter: filter spectrum with bandpass after whitening
        (tuple with min and max frequency)
    :param waterlevel: waterlevel relative to mean of spectrum
    :param mask_again: weather to mask array after this operation again and
        set the corresponding data to 0

    :return: whitened data
    """
    sr = tr.stats.sampling_rate
    data = tr.data
    data = _fill_array(data, fill_value=0)
    mask = np.ma.getmask(data)
    nfft = next_fast_len(len(data))
    spec = fft(data, nfft)
    spec_ampl = np.abs(spec)
    spec_ampl /= np.max(spec_ampl)
    if smooth:
        smooth = int(smooth * nfft / sr)
        spec_ampl = ifftshift(smooth_func(fftshift(spec_ampl), smooth))
    # save guard against division by 0
    spec_ampl[spec_ampl < waterlevel] = waterlevel
    spec /= spec_ampl
    if filter is not None:
        spec *= _filter_resp(*filter, sr=sr, N=len(spec), whole=True)[1]
    ret = np.real(ifft(spec, nfft)[:len(data)])
    if mask_again:
        ret = _fill_array(ret, mask=mask, fill_value=0)
    tr.data = ret
    return tr


def __get_stations(inventory):
    channels = inventory.get_contents()['channels']
    stations = sorted({ch[:-1] + '?': ch[-1] for ch in channels})
    return stations


def _iter_station_meta(inventory, components):
    """
    Return iterator yielding metadata per station and day.

    :param inventory: |Inventory| object with station and channel information
    :param components: components to yield
    """
    stations = __get_stations(inventory)
    for seedid in stations:
        for comp in components:
            net, sta, loc, cha = seedid.split('.')
            cha = cha[:2] + comp
            meta = {'network': net, 'station': sta, 'location': loc,
                    'channel': cha}
            yield meta


def get_data(smeta, data, data_format, day, overlap=0, edge=0,
             trim_and_merge=False):
    """Return data of one day

    :param smeta: dictionary with station metadata
    :param data: string with expression of data day files or
        function that returns the data (aka get_waveforms)
    :param data_format: format of data
    :param day: day as |UTC| object
    :param overlap: overlap to next day in seconds
    :param edge: additional time span requested from day before and after
        in seconds
    :param trim_and_merge: weather data is trimmed to day boundaries and merged
    """
    next_day = day + 24 * 3600
    if not isinstance(data, str):
        try:
            stream = data(starttime=day - edge,
                          endtime=next_day + overlap + edge, **smeta)
        except Exception as ex:
            log.debug('no data for %s %s: %s', day, smeta, str(ex))
            return
    else:
        fname = data.format(t=day, **smeta)
        try:
            stream = obspy.read(fname, data_format)
        except:
            return
        t1 = stream[0].stats.starttime
        t2 = stream[-1].stats.endtime
        if t1 - day < 60:
            fname = data.format(t=day - 1, **smeta)
            try:
                stream += obspy.read(fname, data_format, starttime=day - edge)
            except:
                pass
        if next_day - t2 < 60:
            endtime = next_day + overlap + edge
            fname = data.format(t=next_day, **smeta)
            try:
                stream += obspy.read(fname, data_format, endtime=endtime)
            except:
                pass
    if trim_and_merge:
        stream.merge(method=1, interpolation_samples=10)
        stream.trim(day, next_day + overlap)
    return stream


def _shift(trace, shift):
    """Shift trace by given time and correct starttime => interpolation"""
    msg = ('interpolate trace %s with starttime %s to shift by %.6fs '
           '(Fourier method)')
    log.debug(msg, trace.id, trace.stats.starttime, shift)
    nfft = next_fast_len(len(trace))
    spec = rfft(trace.data, nfft)
    freq = rfftfreq(nfft, trace.stats.delta)
    spec *= np.exp(-2j * np.pi * freq * shift)
    trace.data = irfft(spec, nfft)[:len(trace)]
    trace.stats.starttime -= shift
    return trace


def _downsample_and_shift(trace, target_sr=None, tolerance_shift=None,
                          **interpolate_options):
    """Downsample and align samples at "good" times by shifting"""
    sr = trace.stats.sampling_rate
    if target_sr is None:
        target_sr = sr
    dt = 1 / target_sr
    shift = (1e-6 * trace.stats.starttime.microsecond) % dt
    if shift > 0.5 * dt:
        shift = shift - dt
    if tolerance_shift is None:
        tolerance_shift = np.finfo(float).eps
    must_shift = abs(shift) > tolerance_shift
    if not must_shift:  # anyway correct starttime
        trace.stats.starttime -= shift
    if sr % target_sr == 0:
        if sr != target_sr:
            trace.decimate(int(sr // target_sr))
        if must_shift:
            _shift(trace, shift)
    else:
        # anti-aliasing filter
        if sr / target_sr > 16:
            msg = ('Automatic filter design is unstable for decimation'
                   ' factors above 16. '
                   'Manual decimation is necessary.')
            raise ArithmeticError(msg)
        trace.filter('lowpass_cheby_2', freq=0.5 * target_sr,
                     maxorder=12)
        if must_shift:
            starttime = trace.stats.starttime - shift
            if starttime < trace.stats.starttime:
                starttime += dt
            msg = ('interpolate trace %s with starttime %s to downsample and '
                   'shift by %.6fs (Stream.interpolate() method)')
            log.debug(msg, trace.id, trace.stats.starttime, shift)
        else:
            starttime = None
        trace.interpolate(target_sr, starttime=starttime,
                          **interpolate_options)
    return trace


def _prep1(target_sr, tolerance_shift, interpolate_options,
           remove_response, inventory, remove_response_options, demean, filter,
           tr):
    """Helper function for parallel preprocessing"""
    tr.data = tr.data.astype('float64')
    _downsample_and_shift(tr, target_sr=target_sr,
                          tolerance_shift=tolerance_shift,
                          interpolate_options=interpolate_options)
    if remove_response:
        tr.remove_response(inventory, **remove_response_options)
    if demean:
         tr.detrend('demean')
    if filter is not None:
        _filter(tr, filter)
    return tr


def _prep2(normalization, time_norm_options, spectral_whitening_options,
           decimate,
           tr):
    """Helper function for parallel preprocessing"""
    tr.data = _fill_array(tr.data, fill_value=0)
    for norm in normalization:
        if norm == 'spectral_whitening':
            spectral_whitening(tr, **spectral_whitening_options)
        else:
            time_norm(tr, norm, **time_norm_options)
    if decimate:
        mask = np.ma.getmask(tr.data)
        tr.decimate(decimate, no_filter=True)
        if mask is not np.ma.nomask:
            tr.data = np.ma.MaskedArray(tr.data, mask[::decimate],
                                        fill_value=0)
    return tr


def preprocess(stream, day=None, inventory=None,
               overlap=0,
               remove_response=False,
               remove_response_options=None,
               demean=True,
               filter=None,
               normalization=(),
               time_norm_options=None,
               spectral_whitening_options=None,
               downsample=None, tolerance_shift=None, interpolate_options=None,
               decimate=None,
               njobs=1):
    """
    Preprocess stream of 1 day

    :param stream: |Stream| object
    :param day: |UTC| object of day (for trimming)
    :param inventory: |Inventory| object (for response removal)
    :param bool remove_response: remove response
    :param filter: min and max frequency of bandpass filter
    :param normalizaton: ordered list of normalizations to apply,
        ``'sprectal_whitening'`` for `spectral_whitening` and/or
        one or several of the time normalizations listed in `time_norm`
    :param downsample: downsample before preprocessing,
        target sampling rate
    :param tolerance_shift: Samples are aligned at "good" times for the target
        sampling rate. Specify tolerance in seconds. (default: no tolerance)
    :param decimate: decimate further by given factor after preprocessing
        (see Trace.decimate)
    :param njobs: number of parallel workers
    :param \*_options: dictionary of options passed to the corresponding
        functions
    """
    if time_norm_options is None:
        time_norm_options = {}
    if spectral_whitening_options is None:
        spectral_whitening_options = {}
    spectral_whitening_options.setdefault('filter', filter)
    if remove_response_options is None:
        remove_response_options = {}
    if interpolate_options is None:
        interpolate_options = {}
    if isinstance(normalization, str):
        normalization = [normalization]
    stream.merge(1, interpolation_samples=10)
    stream.traces = stream.split().traces
    # discard traces with less than 10 samples
    stream.traces = [tr for tr in stream if len(tr) >= 10]
    if downsample is None:
        downsample = min(tr.stats.sampling_rate for tr in stream)
    # call _prep1 on all traces, merge stream and call _prep2 on all traces
    do_work = partial(_prep1, downsample, tolerance_shift, interpolate_options,
                      remove_response, inventory, remove_response_options,
                      demean, filter)
    stream.traces = start_parallel_jobs_inner_loop(stream, do_work, njobs)
    len1 = len(stream)
    stream.merge()
    if len(stream) < len1:
        log.debug('detected gaps in data')
    do_work = partial(_prep2, normalization, time_norm_options,
                      spectral_whitening_options, decimate)
    stream.traces = start_parallel_jobs_inner_loop(stream, do_work, njobs)
    if day is not None:
        next_day = day + 24 * 3600
        stream.trim(day, next_day + overlap)
    stream.sort()
    assert len({tr.stats.sampling_rate for tr in stream}) == 1
    return stream


def correlate_traces(tr1, tr2, maxshift=3600, demean=True):
    """
    Return trace of cross-correlation of two input traces

    :param tr1,tr2: two |Trace| objects
    :param maxsift: maximal shift in correlation in seconds
    """
    n1, s1, l1, c1 = tr1.id.split('.')
    n2, s2, l2, c2 = tr2.id.split('.')
    sr = tr1.stats.sampling_rate
    xdata = obscorr(tr1.data, tr2.data, int(round(maxshift * sr)),
                    demean=demean)
    header = {'network': s1, 'station': c1, 'location': s2, 'channel': c2,
              'network1': n1, 'station1': s1, 'location1': l1, 'channel1': c1,
              'network2': n2, 'station2': s2, 'location2': l2, 'channel2': c2,
              'starttime': tr1.stats.starttime,
              'sampling_rate': sr,
              }
    return obspy.Trace(data=xdata, header=header)


def _make_same_length(tr1, tr2):
    """Guarantee that tr1 and tr2 have the same length.

    Even if tr1 and tr2 have the same sampling rate and are trimmed with
    the same times, they could differ in length up to one sample.
    This is handled here.
    """
    dlen = len(tr2) - len(tr1)
    dt = tr1.stats.delta
    if dlen == -1:
        tr1, tr2 = tr2, tr1
    if abs(dlen) == 1:
        # tr2 is too long
        if tr1.stats.starttime - tr2.stats.starttime > dt / 2:
            tr2.data = tr2.data[1:]
            tr2.stats.starttime += dt
        else:
            tr2.data = tr2.data[:-1]
    elif abs(dlen) > 1:
        msg = 'This should not happen ;), traces have different length'
        raise ValueError(msg)


def _slide_and_correlate_traces(day, next_day, length, overlap, discard,
                                max_lag, outkey, demean_window,
                                task):
    """Helper function for parallel correlating"""
    tr1, tr2, dist, azi, baz = task
    sr = tr1.stats.sampling_rate
    sr2 = tr2.stats.sampling_rate
    if sr != sr2:
        msg = 'Traces have different sampling rate (%s != %s)' % (sr, sr2)
        raise ValueError(msg)
    xstream = obspy.Stream()
    for t1 in IterTime(day, next_day - length + overlap, dt=length - overlap):
        sub = obspy.Stream([tr1, tr2]).slice(t1, t1 + length)
        if len(sub) < 2:
            continue
        st = [tr.stats.starttime for tr in sub]
        et = [tr.stats.endtime for tr in sub]
        if max(st) > min(et):  # this should not happen
            continue
        sub.trim(max(st), min(et))
        _make_same_length(sub[0], sub[1])
        avail = min((tr.data.count() if hasattr(tr.data, 'count')
                     else len(tr)) / sr / length for tr in sub)
        if discard is not None and avail < discard:
            msg = ('discard trace combination %s-%s for time %s '
                   '(availability %.1f%% < %.1f%% desired)')
            log.debug(msg, sub[0].id, sub[1].id, str(max(st))[:19],
                      100 * avail, 100 * discard)
            continue
        for tr in sub:
            _fill_array(tr.data, fill_value=0)
            tr.data = np.ma.getdata(tr.data)
        xtr = correlate_traces(sub[0], sub[1], max_lag, demean=demean_window)
        xtr.stats.starttime = t1
        xtr.stats.key = outkey
        xtr.stats.dist = dist
        xtr.stats.azi = azi
        xtr.stats.baz = baz
        xtr.stats.avail = avail
        xstream += xtr
    return xstream


def _midtime(stats):
    return stats.starttime + 0.5 * (stats.endtime - stats.starttime)


def correlate(io, day, outkey,
              edge=60,
              length=3600, overlap=1800,
              demean_window=True,
              discard=None,
              only_auto_correlation=False,
              station_combinations=None,
              component_combinations=('ZZ',),
              max_lag=100,
              keep_correlations=False,
              stack='1d',
              njobs=1,
              **preprocessing_kwargs):
    """
    Correlate data of one day

    :param io: io config dictionary
    :param day: |UTC| object with day
    :param outkey: the output key for the HDF5 index
    :param edge: additional time span requested from day before and after
        in seconds
    :param length: length of correlation in seconds (string possible)
    :param overlap: length of overlap in seconds (string possible)
    :param demean_window: demean each window individually before correlating
    :param discard: discard correlations with less data coverage
        (float from interval [0, 1])
    :param only_auto_correlations: Only correlate stations with itself
        (different components possible)
    :param station_combinations: specify station combinations
        (e.g. ``'CX.PATCX-CX.PB01``, network code can be
        omitted, e.g. ``'PATCX-PB01'``, default: all)
    :param component_combinations: component combinations to calculate,
        tuple of strings with length two, e.g. ``('ZZ', 'ZN', 'RR')``,
        if ``'R'`` or ``'T'`` is specified, components will be rotated after
        preprocessing, default: all component combinations (not rotated)
    :param max_lag: max time lag in correlations in seconds
    :param keep_correlatons: write correlations into HDF5 file (dafault: False)
    :param stack: stack correlations and write stacks into HDF5 file
        (default: ``'1d'``, must be smaller than one day or one day)

        .. note::

            If you want to stack larger time spans
            use the separate stack command on correlations or stacked
            correlations.

    :param njobs: number of jobs used. Some tasks will run parallel
        (preprocessing and correlation).
    :param \*\*preprocessing_kwargs: all other kwargs are passed to
        `preprocess`

    """
    inventory = io['inventory']
    length = _time2sec(length)
    overlap = _time2sec(overlap)
    if not keep_correlations and stack is None:
        msg = ('keep_correlation is False and stack is None -> correlations '
               ' would not be saved')
        raise ValueError(msg)
    components = set(''.join(component_combinations))
    if 'R' in components or 'T' in components:
        load_components = components - {'R', 'T'} | {'N', 'E'}
    else:
        load_components = components
    if station_combinations is not None:
        load_stations = set(sta for comb in station_combinations
                            for sta in comb.split('-'))
    else:
        load_stations = None
    # load data
    stream = obspy.Stream()
    for smeta in _iter_station_meta(inventory, load_components):
        if (load_stations is not None and smeta['station'] not in load_stations
                and '.'.join((smeta['network'], smeta['station']))
                not in load_stations):
            continue
        stream2 = get_data(smeta, io['data'], io['data_format'], day,
                           overlap=overlap, edge=edge)
        if stream2:
            stream += stream2
    if len(stream) == 0:
        log.warning('empty stream for day %s', str(day)[:10])
        return
    preprocess(stream, day, inventory, overlap=overlap, njobs=njobs,
               **preprocessing_kwargs)
    # collect trace pairs for correlation
    next_day = day + 24 * 3600
    stations = sorted({tr.id[:-1] for tr in stream})
    tasks = []
    for station1, station2 in itertools.combinations_with_replacement(
            stations, 2):
        if only_auto_correlation and station1 != station2:
            continue
        if station_combinations and not any(set(station_comb.split('-')) == (
                {station1.rsplit('.', 2)[0], station2.rsplit('.', 2)[0]}
                if '.' in (station_comb) else
                {station1.split('.')[1], station2.split('.')[1]})
                for station_comb in station_combinations):
            continue
        stream1 = Stream([tr for tr in stream if tr.id[:-1] == station1])
        stream2 = Stream([tr for tr in stream if tr.id[:-1] == station2])
        datetime1 = _midtime(stream1[0].stats)
        datetime2 = _midtime(stream2[0].stats)
        msg = 'Cannot get coordinates for channel %s datetime %s'
        try:
            c1 = inventory.get_coordinates(stream1[0].id, datetime=datetime1)
        except Exception as ex:
            raise RuntimeError(msg % (stream1[0].id, datetime1)) from ex
        try:
            c2 = inventory.get_coordinates(stream2[0].id, datetime=datetime2)
        except Exception as ex:
            raise RuntimeError(msg % (stream2[0].id, datetime2)) from ex
        args = (c1['latitude'], c1['longitude'],
                c2['latitude'], c2['longitude'])
        dist, azi, baz = gps2dist_azimuth(*args)
        if ('R' in components or 'T' in components) and station1 != station2:
            stream1 = stream1.copy()
            stream1b = stream1.copy().rotate('NE->RT', azi)
            stream1.extend(stream1b.select(component='R'))
            stream1.extend(stream1b.select(component='T'))
            stream2 = stream2.copy()
            stream2b = stream2.copy().rotate('NE->RT', azi)
            stream2.extend(stream2b.select(component='R'))
            stream2.extend(stream2b.select(component='T'))
        it_ = (itertools.product(stream1, stream2) if station1 != station2 else
               itertools.combinations_with_replacement(stream1, 2))
        for tr1, tr2 in it_:
            comps = tr1.stats.channel[-1] + tr2.stats.channel[-1]
            if component_combinations and (
                    comps not in component_combinations and
                    comps[::-1] not in component_combinations):
                continue
            tasks.append((tr1, tr2, dist, azi, baz))
    # start correlation
    do_work = partial(_slide_and_correlate_traces, day, next_day, length,
                      overlap, discard, max_lag, outkey,
                      demean_window)
    streams = start_parallel_jobs_inner_loop(tasks, do_work, njobs)
    xstream = Stream()
    xstream.traces = [tr for s_ in streams for tr in s_]
    if len(xstream) > 0:
        res = {}
        if keep_correlations:
            res['corr'] = xstream
        if stack:
            res['stack'] = yam.stack.stack(xstream, stack)
        return res
