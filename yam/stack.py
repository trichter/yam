# Copyright 2017-2018 Tom Eulenfeld, GPLv3
"""Stack correlations"""
import numpy as np
import obspy
from obspy import UTCDateTime as UTC

from yam.util import _corr_id, _time2sec, IterTime


def stack(stream, length=None, move=None):
    """
    Stack traces in stream by correlation id

    :param stream: |Stream| object with correlations
    :param length: time span of one trace in the stack in seconds
        (alternatively a string consisting of a number and a unit
        -- ``'d'`` for days and ``'h'`` for hours -- can be specified,
        i.e. ``'3d'`` stacks together all traces inside a three days time
        window, default: None, which stacks together all traces)
    :param move: define a moving stack, float or string,
        default: None -- no moving stack,
        if specified move usually is smaller than length to get an overlap
        in the stacked traces
    :return: |Stream| object with stacked correlations
    """
    stream.sort()
    stream_stack = obspy.Stream()
    ids = {_corr_id(tr) for tr in stream}
    ids.discard(None)
    for id_ in ids:
        traces = [tr for tr in stream if _corr_id(tr) == id_]
        if length is None:
            data = np.mean([tr.data for tr in traces], dtype=float, axis=0)
            tr_stack = obspy.Trace(data, header=traces[0].stats)
            tr_stack.stats.key = tr_stack.stats.key + '_s'
            if 'num' in traces[0].stats:
                tr_stack.stats.num = sum(tr.stats.num for tr in traces)
            else:
                tr_stack.stats.num = len(traces)
            stream_stack.append(tr_stack)
        else:
            t1 = traces[0].stats.starttime
            lensec = _time2sec(length)
            movesec = _time2sec(move) if move else lensec
            if (lensec % (24 * 3600) == 0 or
                    isinstance(length, str) and 'd' in length):
                t1 = UTC(t1.year, t1.month, t1.day)
            elif (lensec % 3600 == 0 or
                    isinstance(length, str) and 'm' in length):
                t1 = UTC(t1.year, t1.month, t1.day, t1.hour)
            t2 = max(t1, traces[-1].stats.endtime - lensec)
            for t in IterTime(t1, t2, dt=movesec):
                sel = [tr for tr in traces
                       if -0.1 <= tr.stats.starttime - t <= lensec + 0.1]
                if len(sel) == 0:
                    continue
                data = np.mean([tr.data for tr in sel], dtype=float, axis=0)
                tr_stack = obspy.Trace(data, header=sel[0].stats)
                key_add = '_s%s' % length + (move is not None) * ('m%s' % move)
                tr_stack.stats.key = tr_stack.stats.key + key_add
                tr_stack.stats.starttime = t
                if 'num' in traces[0].stats:
                    tr_stack.stats.num = sum(tr.stats.num for tr in sel)
                else:
                    tr_stack.stats.num = len(sel)
                stream_stack.append(tr_stack)
    return stream_stack
