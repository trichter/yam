# Copyright 2017-2023, Tom Eulenfeld, MIT license
"""
Stretch correlations

The results are returned in a dictionary with the following entries:

:times: strings of starttimes of the traces (1D array, length ``N1``)
:velchange_values: velocity changes (%) corresponding to the used stretching
    factors (assuming a homogeneous velocity change, 1D array, length ``N2``)
:tw: used lag time window
:sim_mat: similarity matrices (2D array, dimension ``(N1, N2)``)
:velchange_vs_time: velocity changes (%) as a function of time
    (value of highest correlation/similarity for each time, length ``N1``)
:corr_vs_time: correlation values as a function of time
    (value of highest correlation/similarity for each time, length ``N1``)
:attrs: dictionary with metadata
    (e.g. network, station, channel information of both stations,
    inter-station distance and parameters passed to the stretching function)

|
"""

import logging
import functools
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
from warnings import warn

from yam.util import _corr_id, _trim, _trim_time_period
import yam.stack

log = logging.getLogger('yam.stretch')


def _intersect_sorted(l1, l2):
    i = 0
    res = []
    for el in l1:
        while l2[i] < el:
            i += 1
        if l2[i] == el:
            res.append(el)
            i += 1
    return res


def _index_sorted(l1, l2):
    index = np.empty(len(l1), dtype=bool)
    i = 0
    for j, el in enumerate(l1):
        if len(l2) <= i:
            index[j:] = False
            break
        if l2[i] == el:
            index[j] = True
            i += 1
        else:
            index[j] = False
    return index


def _update_result(res):
    sim_mat = res['sim_mat']
    res['corr_vs_time'] = np.max(sim_mat, axis=1)
    argmax = np.argmax(sim_mat, axis=1)
    res['velchange_vs_time'] = res['velchange_values'][argmax]


def join_dicts(dicts):
    """Join list of dictionaries with stretching results"""
    # TODO: write test for this function
    if len(dicts) == 0:
        return
    elif len(dicts) == 1:
        return dicts[0]
    dicts = sorted(dicts, key=lambda d: d['times'][0])
    dim1 = sum(len(d['times']) for d in dicts)
    d = dicts[0]
    dim2 = len(d['velchange_values'])
    dtype = d['sim_mat'].dtype
    res = {'sim_mat': np.empty((dim1, dim2), dtype=dtype),
           'velchange_values': d['velchange_values'],
           'times': np.empty(dim1, dtype=d['times'].dtype),
           'velchange_vs_time': np.empty(dim1, dtype=dtype),
           'corr_vs_time': np.empty(dim1, dtype=dtype),
           'lag_time_windows': d['lag_time_windows'],
           'attrs': d['attrs']}
    res['attrs']['endtime'] = dicts[-1]['attrs']['endtime']
    i = 0
    for d in dicts:
        j = i + len(d['times'])
        res['sim_mat'][i:j, :] = d['sim_mat']
        res['times'][i:j] = d['times']
        res['velchange_vs_time'][i:j] = d['velchange_vs_time']
        res['corr_vs_time'][i:j] = d['corr_vs_time']
        i = j
    return res


def average_dicts(dicts):
    """Average list of dictionaries with stretching results"""
    # TODO: write test for this function
    if len(dicts) == 0:
        return
    elif len(dicts) == 1:
        return dicts[0]
    times = functools.reduce(_intersect_sorted, [d['times'] for d in dicts])
    d = dicts[0]
    sim_mat = np.mean(
        [d['sim_mat'][_index_sorted(d['times'], times), :] for d in dicts],
        axis=0)
    res = {'sim_mat': sim_mat,
           'velchange_values': d['velchange_values'],
           'times': np.array(times),
           'tw': d['tw'],
           'attrs': d['attrs']}
    res['attrs']['channel1'] = '???'
    res['attrs']['channel2'] = '???'
    _update_result(res)
    return res


def _stretch_helper(data, refdata, mask, stretch_factor):
    N1, N2 = data.shape
    N3 = len(stretch_factor)
    assert N2 == len(refdata)
    # create a spline object for the reference trace
    # and evaluate the spline object at stretched times
    time_index_max = (N2 - 1) / 2
    time_idx = np.linspace(-time_index_max, time_index_max, N2)
    ref_tr_spline = InterpolatedUnivariateSpline(time_idx, refdata)
    ref_stretch = np.empty((N3, N2))
    for i in range(N3):
        ref_stretch[i, :] = ref_tr_spline(time_idx * stretch_factor[i])
    # correlate, normalize and return
    cc = (data * mask) @ np.transpose(ref_stretch * mask)
    sq1 = np.sum((data * mask) ** 2, axis=1)
    sq2 = np.sum((ref_stretch * mask) ** 2, axis=1)
    norm = (sq1[:, np.newaxis] * sq2) ** 0.5
    sim_mat = cc / norm
    sim_mat[np.isnan(sim_mat)] = 0
    assert sim_mat.shape == (N1, N3)
    return sim_mat


def stretch(stream, max_stretch, num_stretch, tw, tw_relative=None,
            reftr=None, sides='both', max_lag=None, time_period=None
            ):
    """
    Stretch traces in stream and return dictionary with results

    See e.g. Richter et al. (2015) for a description of the procedure.

    :param stream: |Stream| object with correlations
    :param float max_stretch: stretching range in percent
    :param int num_stretch: number of values in stretching vector
    :param tw: definition of the time window in the correlation --
        tuple of length 2 with start and end time in seconds (positive)
    :param tw_relative: time windows can be defined relative to a
        velocity, default None or 0 -- time windows relative to zero lag time,
        otherwise velocity is given in km/s
    :param reftr: reference trace, by default the stack of stream is used
        as reference
    :param sides: one of left, right, both
    :param max_lag: max lag time in seconds, stream is trimmed to
        ``(-max_lag, max_lag)`` before stretching
    :param time_period: use correlations only from this time span
        (tuple of dates)
    """
    if len(stream) <= 1:
        log.warning('For stretch the stream must have a minimum length of 2')
        return
    ids = {_corr_id(tr) for tr in stream}
    if None in ids:
        ids.discard(None)
        stream.traces = [tr for tr in stream if _corr_id(tr) is not None]
    if len(ids) != 1:
        warn('Different ids in stream: %s' % ids)
    _trim_time_period(stream, time_period)
    stream.sort()
    tr0 = stream[0]
    if max_lag is not None:
        for tr in stream:
            _trim(tr, (-max_lag, max_lag))
    rel = 0 if tw_relative is None else tr0.stats.dist / 1000 / tw_relative
    twabs = rel + np.array(tw)
    starttime = tr0.stats.starttime
    mid = starttime + (tr0.stats.endtime - starttime) / 2
    times = tr0.times(reftime=mid)
    data = np.array([tr.data for tr in stream])
    data[np.isnan(data)] = 0  # bug fix
    data[np.isinf(data)] = 0
    if reftr is None:
        reftr = yam.stack.stack(stream)[0]
    stretch_vector_perc = np.linspace(-max_stretch, max_stretch, num_stretch)
    stretch_factor = 1 + stretch_vector_perc / 100
    # MIIC approximation:
    #stretch_factor = np.exp(stretch_vector_percent / 100)

    mask1 = np.logical_and(times >= twabs[0], times <= twabs[1])
    mask2 = np.logical_and(times <= -twabs[0], times >= -twabs[1])
    if sides == 'left':
        mask = mask1
    elif sides == 'right':
        mask = mask2
    elif sides == 'both':
        mask = np.logical_or(mask1, mask2)
    else:
        raise ValueError('sides = %s not a valid option' % sides)
    sim_mat = _stretch_helper(data, reftr.data, mask, stretch_factor)
    times = np.array([str(tr.stats.starttime)[:19] for tr in stream],
                     dtype='S19')
    result = {'sim_mat': sim_mat,
              'velchange_values': stretch_vector_perc,
              'times': times,
              'tw': twabs,
              'attrs': {'num_stretch': num_stretch,
                        'max_stretch': max_stretch,
                        'sides': sides,
                        'starttime': stream[0].stats.starttime,
                        'endtime': stream[-1].stats.starttime
                        }
              }
    _update_result(result)
    for k in ('network1', 'network2', 'station1', 'station2',
              'location1', 'location2', 'channel1', 'channel2',
              'sampling_rate', 'dist', 'azi', 'baz'):
        if k in tr0.stats:
            result['attrs'][k] = tr0.stats[k]
    return result
