# -*- coding: utf-8 -*-
# Copyright 2017 Tom Eulenfeld, GPLv3

from copy import copy
from collections import OrderedDict
import os.path
from warnings import warn

from matplotlib.dates import date2num
import matplotlib.pyplot as plt
import numpy as np
import obspy

from yam.util import _corr_id
import yam.stack


def _get_times_no_data(x):
    dx = np.median(np.diff(x))
    try:
        index = np.argwhere(np.diff(x) > dx)[:, 0]
    except IndexError:
        return {}
    tl = OrderedDict()
    for i in index:
        tl[i] = int(round((x[i + 1] - x[i]) / dx))
    return tl


def _add_value(x, tl, value=None, masked=False):
    dx = np.median(np.diff(x))
    where = [i + 1 for i in tl.keys()]
    xs = np.split(x, where, axis=-1)
    xf = [xs[0]]
    for x in xs[1:]:
        shape = (1,)
        if len(x.shape) > 1:
            shape = (x.shape[0],) + shape
        if value is None:
            # shape not sorted out, but doesnt matter
            xfn = xf[-1][-1] + dx
        else:
            xfn = np.ones(shape) * value
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


def plot_stations(inventory, fname, ext='.png', projection='local', **kwargs):
    inventory.plot(projection, outfile=fname + ext, **kwargs)


def plot_data(data, fname, ext='.png', show=False,
              type='dayplot', **kwargs):
    label = os.path.basename(fname)
    data.plot(type=type, outfile=fname + ext, title=label)
    if show:
        data.plot(type=type, title=label, show=True)


def plot_corr_vs_dist(
        stream, fname, figsize=(10, 5), ext='.png',
        components='ZZ', line_style='k', scale=1, dist_unit='km',
        trim=None):
    # scale relative to axis
    traces = [tr for tr in stream if
              _corr_id(tr).split('-')[0][-1] + _corr_id(tr)[-1] == components]
    stack = yam.stack.stack(obspy.Stream(traces))
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
        mid = tr.stats.starttime + (tr.stats.endtime - tr.stats.starttime) / 2
        if trim:
            tr.trim(mid - trim[0], mid + trim[1])
        lag_time = tr.times(reftime=mid)
        scaled_data = tr.stats.dist / dist_scale + tr.data * max_dist * scale
        ax.plot(lag_time, scaled_data, line_style)
    fname = '%s_%s' % (fname, components)
    label = os.path.basename(fname)
    ax.annotate(label, (0, 1), (10, 10), 'axes fraction', 'offset points',
                annotation_clip=False, va='bottom')
    ax.set_ylabel('distance (%s)' % dist_unit)
    ax.set_xlabel('time (s)')
    fig.savefig(fname + ext)


def plot_corr_vs_time_wiggle(
        stream, fname, figsize=(10, 5), ext='.png',
        line_style='k', scale=20, line_width=0.5,
        trim=None):
    # scale relative to neighboring wiggles
    ids = {_corr_id(tr) for tr in stream}
    if len(ids) != 1:
        warn('Different ids in stream: %s' % ids)
    stream.sort(['starttime'])
    times = [date2num(tr.stats.starttime.datetime) for tr in stream]
    dt = np.median(np.diff(times))
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    for tr in stream:
        t3 = tr.stats.starttime
        mid = t3 + (tr.stats.endtime - t3) / 2
        if trim:
            tr.trim(mid - trim[0], mid + trim[1])
        lag_time = tr.times(reftime=mid)
        scaled_data = date2num(t3.datetime) + tr.data * dt * scale
        ax.plot(lag_time, scaled_data, line_style, lw=line_width)
    label = os.path.basename(fname)
    ax.annotate(label, (0, 1), (10, 10), 'axes fraction', 'offset points',
                annotation_clip=False, va='bottom')
    ax.set_ylabel('date')
    ax.set_xlabel('time (s)')
    ax.yaxis_date()
    fig.savefig(fname + ext)


def plot_corr_vs_time(
        stream, fname, figsize=(10, 5), ext='.png',
        vmax=None, cmap='RdBu_r', trim=None,
        show_stack=True, line_style='k', line_width=1):
    ids = {_corr_id(tr) for tr in stream}
    if len(ids) != 1:
        warn('Different ids in stream: %s' % ids)
    stream.sort(['starttime'])
    times = [date2num(tr.stats.starttime.datetime) for tr in stream]
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0.15, 0.1, 0.75, 0.75])
    cax = fig.add_axes([0.91, 0.375, 0.008, 0.25])
    for tr in stream:
        st = tr.stats.starttime
        mid = st + (tr.stats.endtime - st) / 2
        if trim:
            tr.trim(mid - trim[0], mid + trim[1])
    lag_times = tr.times(reftime=mid)
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
    if show_stack:
        ax2 = fig.add_axes([0.15, 0.85, 0.75, 0.05], sharex=ax)
        stack = yam.stack.stack(stream)
        ax2.plot(lag_times, stack[0].data, line_style, lw=line_width)
        ax2.set_ylim(-vmax, vmax)
        ax2.set_ylabel('stack')
        plt.setp(ax2.get_xticklabels(), visible=False)
        ax_ano = ax2
    ax.set_xlim(lag_times[0], lag_times[-1])
    label = os.path.basename(fname)
    ax_ano.annotate(label, (0, 1), (10, 10), 'axes fraction', 'offset points',
                    annotation_clip=False, va='bottom')
    fig.savefig(fname + ext)


def plot_sim_mat(res, bname=None, figsize=(10, 5), ext='.png',
                 vmax=None, cmap='hot_r',
                 show_line=False, line_style='b', line_width=2,
                 time_window=None):
    labelexpr = '{}_tw{:02d}_{:05.1f}s-{:05.1f}s'
    figs = []
    for itw, tw in enumerate(res['lag_time_windows']):
        if time_window is not None and itw != time_window:
            continue
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        np.transpose(res['sim_mat'][:, :, itw])
        data = np.transpose(res['sim_mat'][:, :, itw])
        x = copy(res['times'])
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
            s = res['velchange_vs_time'][:, itw]
            s = _add_value(s, no_data, value=0, masked=True)
            ax.plot(x, s, line_style, lw=line_width)
        ax.set_xlabel('date')
        ax.set_ylabel('velocity change (%)')
        fig.autofmt_xdate()
        fig.colorbar(mesh, shrink=0.5)
        bname2 = 'sim_mat' if bname is None else bname
        fname = labelexpr.format(bname2, itw, *tw)
        label = os.path.basename(fname)
        ax.annotate(label, (0, 1), (10, 10), 'axes fraction', 'offset points',
                    annotation_clip=False, va='bottom')
        if bname is not None:
            fig.savefig(fname + ext)
        figs.append(fig)
    if tw is None:
        return figs
    elif len(figs) == 1:
        return figs[0]


