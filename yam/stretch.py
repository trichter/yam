# Copyright 2017, Tom Eulenfeld, GPLv3

import logging
import numpy as np
from warnings import warn

from yam._from_miic import time_windows_creation, time_stretch_estimate
from yam.util import _trim, _corr_id
import yam.stack

log = logging.getLogger('yam.stretch')


def stretch(stream, reftr=None, stretch=None, str_range=10, nstr=100,
            time_windows=None,
            time_windows_relative=None, sides='right',
            max_lag=None, time_period=None
            ):
    """
    time_windows:
        ((5, 10, 15), 5) -- 3 time windows, start in sec, length in sec

    """
    ids = {_corr_id(tr) for tr in stream}
    if len(ids) != 1:
        warn('Different ids in stream: %s' % ids)
    stream.sort()
    sr = stream[0].stats.sampling_rate
    rel = 0.
    if time_windows_relative is not None:
        rel = np.round(stream[0].stats.dist / 1000 / time_windows_relative)
    if time_windows is not None and isinstance(time_windows[1], (float, int)):
        args = ((rel + np.array(time_windows[0])) * int(sr),
                time_windows[1] * int(sr))
        tw_mat = time_windows_creation(*args)
    else:
        raise ValueError('Wrong format for time_window')
    if max_lag is not None:
        for tr in stream:
            _trim(tr, (-max_lag, max_lag))
    data = np.array([tr.data for tr in stream])
    if np.min(tw_mat) < 0 or data.shape[1] < np.max(tw_mat):
        msg = ('Defined time window outside of data. '
               'shape, mintw index, maxtw index: %s, %s, %s')
        raise ValueError(msg % (data.shape[1], np.min(tw_mat), np.max(tw_mat)))
    data[np.isnan(data)] = 0  # bug fix
    data[np.isinf(data)] = 0
    if reftr is None:
        reftr = yam.stack.stack(stream)[0]

    if reftr != 'alternative':
        if hasattr(reftr, 'stats'):
            assert reftr.stats.sampling_rate == sr
            ref_data = reftr.data
            #ref_data = _stream2matrix(obspy.Stream([reftr]))[0, :]
        else:
            ref_data = reftr
#        log.debug('calculate correlations and time shifts...')
        # convert str_range from % to decimal
        tse = time_stretch_estimate(
            data, ref_trc=ref_data, tw=tw_mat, stretch_range=str_range / 100,
            stretch_steps=nstr, sides=sides)
#    else:
#        assert len(tw_mat) == len(stretch)
#        tses = []
#        log.debug('calculate correlations and time shifts...')
#        for i in range(len(tw_mat)):
#            tw = tw_mat[i:i + 1]
#            st = stretch[i]
#            sim_mat = time_stretch_apply(data, st, single_sided=False)
#            ref_data = np.mean(sim_mat, axis=0)
#            tse = time_stretch_estimate(data, ref_trc=ref_data, tw=tw,
#                                 stretch_range=str_range, stretch_steps=nstr,
#                                 sides=sides)
#            tses.append(tse)
#        for i in ('corr', 'stretch'):
#            tse[i] = np.hstack([t[i] for t in tses])
#        i = 'sim_mat'
#        tse[i] = np.vstack([t[i] for t in tses])
    times = np.array([str(tr.stats.starttime)[:19] for tr in stream],
                     dtype='S19')
    ltw1 = rel + np.array(time_windows[0])
    # convert streching to velocity change
    # -> minus at several places
    result = {'sim_mat': tse['sim_mat'][:, ::-1, :],
              'velchange_values': -tse['second_axis'][::-1] * 100,
              'times': times,
              'velchange_vs_time': -tse['value'] * 100,
              'corr_vs_time': tse['corr'],
              'lag_time_windows': np.transpose([ltw1, ltw1 + time_windows[1]]),
              'attrs': {'nstr': nstr,
                        'str_range': str_range,
                        'sides': sides,
                        'starttime': stream[0].stats.starttime,
                        'endtime': stream[-1].stats.starttime
                        }
              }
    for k in ('network1', 'network2', 'station1', 'station2',
              'location1', 'location2', 'channel1', 'channel2',
              'sampling_rate', 'dist', 'azi', 'baz'):
        result['attrs'][k] = stream[0].stats[k]
    return result
