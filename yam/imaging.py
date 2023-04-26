# -*- coding: utf-8 -*-
# Copyright 2017-2023 Tom Eulenfeld, MIT license
"""
Plotting functions

Common arguments in plotting functions are:

:stream: |Stream| object with correlations
:fname: file name for the plot output
:ext: file name extension (e.g. ``'png'``, ``'pdf'``)
:figsize: figure size (tuple of inches)
:dpi: resolution of image file (not available for station plot)
:xlim: limits of x axis (tuple of lag times or tuple of UTC strings)
:ylim: limits of y axis (tuple of UTC strings or tuple of percentages)
:\*_kw: dictionary of arguments passed to calls of matplotlib methods
    (e.g. ``plot_kw`` for arguments passed to |Axes.plot|, etc). Some of these
    dictionaries might be set to ``None`` to suppress the corresponding feature
    (e.g. set ``stack_plot_kw=None`` to not plot the stack of all traces in
    ``plot_corr_vs_time()``).

|
"""

from copy import copy
from collections import OrderedDict
import os.path
from warnings import warn

import matplotlib.pyplot as plt
import numpy as np
from obspy import UTCDateTime as UTC

from yam.util import _corr_id, _trim, _trim_time_period
import yam.stack
import yam.stretch


def _get_times_no_data(x):
    """Return times without data"""
    dx = np.median(np.diff(x))
    try:
        index = np.argwhere(np.diff(x) > dx)[:, 0]
    except IndexError:
        return {}
    tl = OrderedDict()
    for i in index:
        tl[i] = int(round((x[i + 1] - x[i]) / dx)) - 1
    return tl


def _add_value(x, tl, value=None, masked=False, single_value=True):
    """Add values for missing data

    :param x: The array where to add the data
    :param tl: output of `_get_times_no_data`, dictionary with times
    :param value: fill value, if None assume a time array
    :param masked: mask filled values
    :param single_value: True -- add a single value for plotting,
        False -- fill gaps fully
    """
    dx = np.median(np.diff(x))
    keys = sorted(tl.keys())
    xs = np.split(x, [i + 1 for i in keys], axis=-1)
    dtype = xs[0].dtype
    xf = [xs[0]]
    for i, x in enumerate(xs[1:]):
        if single_value:
            shape = (1,)
        else:
            shape = (tl[keys[i]],)
        if len(x.shape) > 1:
            shape = (x.shape[0],) + shape
        if value is None:
            # shape not sorted out, but doesnt matter
            # can only be used with single_value = True!!
            # if this should be supported it needs a proper test!
            if not single_value:
                raise NotImplementedError
            xfn = xf[-1][-1] + dx
        else:
            xfn = np.ones(shape, dtype=dtype) * value
        if masked:
            xfn = np.ma.masked_all_like(xfn)
        xf.append(xfn)
        xf.append(x)
    x2 = np.ma.hstack(xf)
    return x2


def _align_values_for_pcolormesh(x):
    x = list(x)
    dx = np.median(np.diff(x))
    x.append(x[-1] + dx)
    x = [xx - 0.5 * dx for xx in x]
    return x


def plot_stations(inventory, fname, ext='png', projection='local', **kwargs):
    """
    Plot station map

    :param inventory: |Inventory| object with coordinates
    :param projection,\*\*kwargs: passed to |Inventory.plot| method
    """
    inventory.plot(projection, outfile=fname + '.' + ext, **kwargs)


def plot_data(data, fname, ext='png', show=False,
              type='dayplot', **kwargs):
    """
    Plot data (typically one day)

    :param data: |Stream| object holding the data
    :param type,\*\*kwargs: passed to |Stream.plot| method
    """
    label = os.path.basename(fname)
    data.plot(type=type, outfile=fname + '.' + ext, title=label, **kwargs)
    if show:
        data.plot(type=type, title=label, show=True, **kwargs)


def plot_corr_vs_dist(
        stream, fname=None, figsize=(10, 5), ext='png', dpi=None,
        components='ZZ', scale=1, dist_unit='km',
        xlim=None, ylim=None, time_period=None, plot_kw={}):
    """
    Plot stacked correlations versus inter-station distance

    .. image:: _static/corr_vs_dist.png
       :width: 30%

    This plot can be created from the command line with ``--plottype vs_dist``.

    :param components: component combination to plot
    :param scale: scale wiggles (default 1)
    :param dist_unit: one of ``('km', 'm', 'deg')``
    :time_period: use correlations only from this time span (tuple of dates)
    """
    plot_kw = plot_kw.copy()
    plot_kw.setdefault('color', 'black')
    # scale relative to axis
    traces = [tr for tr in stream if
              _corr_id(tr).split('-')[0][-1] + _corr_id(tr)[-1] == components]
    stream.traces = traces
    _trim_time_period(stream, time_period)
    stack = yam.stack.stack(stream)
    if len(stack) == 0:
        msg = 'Not plotting anything. No traces in stack with components %s'
        warn(msg % components)
        return
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    dist_scale = (1 if dist_unit == 'm' else 1000 * 111.2 if dist_unit == 'deg'
                  else 1000)
    max_dist = max(tr.stats.dist / dist_scale for tr in stack)
    for tr in stack:
        lag_times = _trim(tr, xlim)
        scaled_data = tr.stats.dist / dist_scale + tr.data * max_dist * scale
        ax.plot(lag_times, scaled_data, **plot_kw)
    if fname is not None:
        fname = '%s_%s' % (fname, components)
        label = os.path.basename(fname)
    else:
        label = components
    ax.set_ylim(ylim)
    ax.annotate(label, (0, 1), (10, 10), 'axes fraction', 'offset points',
                annotation_clip=False, va='bottom')
    ax.set_ylabel('distance (%s)' % dist_unit)
    ax.set_xlabel('time (s)')
    if fname is not None:
        fig.savefig(fname + '.' + ext, dpi=dpi)


def plot_corr_vs_time_wiggle(
        stream, fname=None, figsize=(10, 5), ext='png', dpi=None,
        xlim=None, ylim=None, scale=20,
        plot_kw={}):
    """
    Plot correlation wiggles versus time

    .. image:: _static/corr_vs_time_wiggle.png
       :width: 30%
    .. image:: _static/corr_vs_time_wiggle2.png
       :width: 40%

    This plot can be created from the command line with ``--plottype wiggle``.

    :param scale: scale of wiggles (default 20)
    """
    plot_kw = plot_kw.copy()
    plot_kw.setdefault('lw', 0.5)
    plot_kw.setdefault('color', 'black')
    # scale relative to neighboring wiggles
    ids = {_corr_id(tr) for tr in stream}
    if len(ids) != 1:
        warn('Different ids in stream: %s' % ids)
    stream.sort(['starttime'])
    _trim_time_period(stream, ylim)
    times = [tr.stats.starttime.matplotlib_date for tr in stream]
    dt = np.median(np.diff(times))
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    for tr in stream:
        lag_times = _trim(tr, xlim)
        scaled_data = tr.stats.starttime.matplotlib_date + tr.data * dt * scale
        ax.plot(lag_times, scaled_data, **plot_kw)
    label = '' if fname is None else os.path.basename(fname)
    ax.annotate(label, (0, 1), (10, 10), 'axes fraction', 'offset points',
                annotation_clip=False, va='bottom')
    ax.set_ylabel('date')
    ax.set_xlabel('time (s)')
    ax.yaxis_date()
    if fname is not None:
        fig.savefig(fname + '.' + ext, dpi=dpi)


def plot_corr_vs_time(
        stream, fname=None, figsize=(10, 5), ext='png', dpi=None,
        xlim=None, ylim=None, vmax=None, cmap='RdBu_r', stack_plot_kw={}):
    """
    Plot correlations versus time

    .. image:: _static/corr_vs_time.png
       :width: 30%
    .. image:: _static/corr_vs_time_zoom.png
       :width: 30%

    Default correlation plot.

    :param vmax: maximum value in colormap
    :param cmap: used colormap
    """
    ids = {_corr_id(tr) for tr in stream}
    srs = {tr.stats.sampling_rate for tr in stream}
    lens = {len(tr) for tr in stream}
    if len(ids) != 1:
        warn('Different ids in stream: %s' % ids)
    # These checks should be done when saving the correlations to hdf5.
    # On the other hand side, these problems will not occur often.
    if len(srs) != 1:
        sr = np.median([tr.stats.sampling_rate for tr in stream])
        msg = 'Different sampling rates in stream: %s -> set %s Hz (%s)'
        warn(msg % (srs, sr, stream[0].id))
        for tr in stream:
            tr.stats.sampling_rate = sr
    if len(lens) != 1:
        msg = ('Different lengths of traces in stream: %s (%s)'
               '-> Plese use xlim parameter to trim traces')
        warn(msg % (lens, stream[0].id))
    stream.sort(['starttime'])
    _trim_time_period(stream, ylim)
    for tr in stream:
        lag_times = _trim(tr, xlim)
    times = [tr.stats.starttime.matplotlib_date for tr in stream]
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0.15, 0.1, 0.75, 0.75])
    cax = fig.add_axes([0.91, 0.375, 0.008, 0.25])
    data = np.array([tr.data for tr in stream])
    if vmax is None:
        vmax = min(0.8 * np.max(data), 0.1)
    no_data = _get_times_no_data(times)
    times = _add_value(times, no_data)
    times2 = _align_values_for_pcolormesh(times)
    lag_times2 = _align_values_for_pcolormesh(lag_times)
    data2 = _add_value(np.transpose(data), no_data, value=0, masked=True)
    data2 = np.transpose(data2)
    mesh = ax.pcolormesh(lag_times2, times2, data2, cmap=cmap,
                         vmin=-vmax, vmax=vmax)
    fig.colorbar(mesh, cax)
    ax.set_ylabel('date')
    ax.set_xlabel('time (s)')
    ax.yaxis_date()
    ax_ano = ax
    if stack_plot_kw is not None:
        stack_plot_kw = stack_plot_kw.copy()
        stack_plot_kw.setdefault('color', 'black')
        stack_plot_kw.setdefault('lw', 1)
        ax2 = fig.add_axes([0.15, 0.85, 0.75, 0.05], sharex=ax)
        stack = yam.stack.stack(stream)
        ax2.plot(lag_times, stack[0].data, **stack_plot_kw)
        ax2.set_ylim(-vmax, vmax)
        ax2.set_ylabel('stack')
        plt.setp(ax2.get_xticklabels(), visible=False)
        ax_ano = ax2
    ax.set_xlim(lag_times[0], lag_times[-1])
    label = '' if fname is None else os.path.basename(fname)
    ax_ano.annotate(label, (0, 1), (10, 10), 'axes fraction', 'offset points',
                    annotation_clip=False, va='bottom')
    if fname is not None:
        fig.savefig(fname + '.' + ext, dpi=dpi)


def plot_sim_mat(res, fname=None, figsize=(10, 5), ext='png', dpi=None,
                 xlim=None, ylim=None, vmax=None, cmap='hot_r',
                 show_line=False, line_plot_kw={}):
    """
    Plot similarity matrices

    .. image:: _static/sim_mat.png
       :width: 30%

    Default plot for stretching results.

    :param res: dictionary with stretching results
    :param vmax: maximum value in colormap
    :param cmap: used colormap
    :param show_line: show line connecting best correlations for each time
    """
    tw = res['tw']
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    data = np.transpose(res['sim_mat'])
    x = [UTC(t) for t in res['times']]
    no_data = _get_times_no_data(x)
    x = _add_value(x, no_data)
    x = [t.datetime for t in x]
    x2 = _align_values_for_pcolormesh(x)
    y2 = _align_values_for_pcolormesh(copy(res['velchange_values']))
    if vmax is None:
        vmax = np.max(data)
    data = _add_value(data, no_data, value=0, masked=True)
    mesh = ax.pcolormesh(x2, y2, data, cmap=cmap, vmin=0, vmax=vmax)
    if show_line:
        line_plot_kw = line_plot_kw.copy()
        line_plot_kw.setdefault('color', 'b')
        line_plot_kw.setdefault('lw', 2)
        s = res['velchange_vs_time']
        s = _add_value(s, no_data, value=0, masked=True)
        ax.plot(x, s, **line_plot_kw)
    ax.set_xlabel('date')
    ax.set_ylabel('velocity change (%)')
    fig.autofmt_xdate()
    fig.colorbar(mesh, shrink=0.5)
    # set label and  limits
    label_tw = 'tw_{:05.1f}s-{:05.1f}s'.format(*tw)
    if fname is None:
        label = label_tw
    else:
        label = os.path.basename(fname) + '_' + label_tw
    ax.annotate(label, (0, 1), (10, 10), 'axes fraction', 'offset points',
                annotation_clip=False, va='bottom')
    if ylim:
        if isinstance(ylim, (float, int)):
            ylim = (-ylim, ylim)
        ax.set_ylim(ylim)
    t0, t1 = (None, None) if xlim is None else xlim
    t0 = x2[0] if t0 is None else UTC(t0).matplotlib_date
    t1 = x2[-1] if t1 is None else UTC(t1).matplotlib_date
    ax.set_xlim(t0, t1)
    if fname is not None:
        fig.savefig(fname + '.' + ext, dpi=dpi)
    return fig


def plot_velocity_change(
        results, fname=None, figsize=(10, 5), ext='png', dpi=None,
        xlim=None, ylim=None, plot_kw={}, joint_plot_kw={}, legend_kw={}):
    """
    Plot velocity change over time

    .. image:: _static/velocity_change.png
       :width: 30%

    Plot velocity change over time estimated from different component/station
    combinations and joint estimate.
    This plot can be created from the command line with
    ``--plottype velocity``.

    :param results: list of dictionaries with stretching results
    """
    if len(results) == 0:
        return
    tw = results[0]['tw']
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    label_stations = None
    if 'group' in results[0]:
        labels = [res['group'].replace('/', '_') for res in results]
        if len(labels) == 1:
            labels = [labels[0].rsplit('_', 1)[-1]]
            if '_' in labels[0]:
                label_stations = labels[0].rsplit('_', 2)[-2]
        else:
            if labels[0].count('_') >= 2:
                label_parts = [l.split('_')[-2:] for l in labels]
                lp1, lp2 = zip(*label_parts)
                if len(set(lp1)) == 1:
                    label_stations = lp1[0]
                    labels = lp2
                else:
                    labels = ['_'.join(lp) for lp in label_parts]
        assert len(labels) == len(results)
    else:
        labels = [None] * len(results)
    if plot_kw is not None:
        plot_kw = plot_kw.copy()
        plot_kw.setdefault('marker', '.')
        plot_kw.setdefault('ls', '')
        for res, label in zip(results, labels):
            x = [UTC(t).datetime for t in res['times']]
            y = res['velchange_vs_time']
            ax.plot(x, y, label=label, **plot_kw)
    if len(results) > 1 and joint_plot_kw is not None:
        joint_plot_kw = joint_plot_kw.copy()
        joint_plot_kw.setdefault('color', 'k')
        res = yam.stretch.average_dicts(results)
        x = [UTC(t).datetime for t in res['times']]
        y = res['velchange_vs_time']
        ax.plot(x, y, label='joint', **joint_plot_kw)
    ax.set_xlabel('date')
    ax.set_ylabel('velocity change (%)')
    if legend_kw is not None:
        ax.legend(**legend_kw)
    # set label and  limits
    label_tw = 'tw_{:05.1f}s-{:05.1f}s'.format(*tw)
    if fname is None:
        label = label_tw
    else:
        label = os.path.basename(fname) + '_' + label_tw
    if label_stations and label_stations not in label:
        label = label + '_' + label_stations
    ax.annotate(label, (0, 1), (10, 10), 'axes fraction', 'offset points',
                annotation_clip=False, va='bottom')
    if ylim:
        if isinstance(ylim, (float, int)):
            ylim = (-ylim, ylim)
        ax.set_ylim(ylim)
    t0, t1 = (None, None) if xlim is None else xlim
    t0 = x[0] if t0 is None else UTC(t0).matplotlib_date
    t1 = x[-1] if t1 is None else UTC(t1).matplotlib_date
    ax.set_xlim(t0, t1)
    fig.autofmt_xdate()
    if fname is not None:
        fig.savefig(fname + '.' + ext, dpi=dpi)
    return fig
