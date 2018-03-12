# Copyright 2017-2018 Tom Eulenfeld, GPLv3
"""Reading and writing correlations and stretching results"""

import h5py

import obspyh5
import yam
import os.path
import obspy

from yam.util import _analyze_key, _get_fname


INDEX = ('{key}/{network1}.{station1}-{network2}.{station2}/'
         '{location1}.{channel1}-{location2}.{channel2}/'
         '{starttime.datetime:%Y-%m-%dT%H:%M}')
INDEX_STRETCH = ('{key}/{network1}.{station1}-{network2}.{station2}/'
                 '{location1}.{channel1}-{location2}.{channel2}')
obspyh5.set_index(INDEX)


def write_dict(dict_, fname, mode='a', libver='latest', dtype='float16'):
    """
    Write similarity matrix into HDF5 file

    :param dict_: Dictionary with stretching results
        (output from `~yam.stretch.stretch()`)
    :param fname: file name
    :param mode: file mode (default ``'a'`` -- write into file)
    :param libver: use latest version of HDF5 file format
    :param dtype: type for similarity matrix
    """
    with h5py.File(fname, mode=mode, libver=libver) as f:
        f.attrs['file_format_stretch'] = 'yam'
        f.attrs['version_stretch'] = yam.__version__
        if 'index_stretch' not in f.attrs:
            index = f.attrs['index_stretch'] = INDEX_STRETCH
        else:
            index = f.attrs['index_stretch']
        attrs = dict_['attrs']
        index = index.format(**attrs)
        group = f.require_group(index)
        for key, val in attrs.items():
            if 'time' in key:
                val = str(val)
            group.attrs[key] = val
        group.create_dataset('sim_mat', data=dict_['sim_mat'], dtype=dtype)
        for key, val in dict_.items():
            if key not in ('attrs', 'sim_mat'):
                group.create_dataset(key, data=val)


def _get_existent(fname, root, level):
    """
    Return existing keys at level in HDF5 file
    """
    if not os.path.exists(fname):
        return []
    done = []

    def visit(group, level):
        if level == 0:
            done.append(group.name)
            return
        elif level == 1:
            done.extend([group.name + '/' + subg for subg in group])
            return
        for n in group:
            visit(group[n], level - 1)
    with h5py.File(fname, 'r') as f:
        try:
            g = f[root]
        except KeyError:
            return []
        if root == '/':
            level_reached = 0
        else:
            level_reached = g.name.count('/')
        visit(g, level - level_reached)
    return sorted(done)


def _read_dict(group, readonly=None):
    """Read a single stretching dictionary from group"""
    res = {'attrs': {}}
    for key, val in group.attrs.items():
        res['attrs'][key] = val
        if key in ('starttime', 'endtime'):
            res['attrs'][key] = obspy.UTCDateTime(val)
    for key, val in group.items():
        if key != 'attrs' and (readonly is None or key in readonly):
            res[key] = val[:]
#    if 'times' in res:
#        res['times'] = [UTC(t) for t in res['times']]
    res['group'] = group.name
    return res


def _iter_dicts(fname, groupname='/', level=3, readonly=None):
    """Iterator yielding stretching dictionaries"""
    tasks = _get_existent(fname, groupname, level)
    with h5py.File(fname, 'r') as f:
        for task in tasks:
            yield task, _read_dict(f[task], readonly=readonly)


def read_dicts(fname, groupname='/', level=3, readonly=None):
    """
    Read dictionaries with stretching results

    :param fname: file name
    :param groupname: specify group to read
    :param level: level in index where the data was written, defaults to 3 and
        should not be changed
    :return: list of dictionaries with stretching results
    """
    return [obj[1] for obj in _iter_dicts(fname, groupname, level, readonly)]


def _iter_streams(fname, groupname='/', level=3):
    """Iterator yielding correlation streams"""
    tasks = _get_existent(fname, groupname, level)
    for task in tasks:
        stream = obspy.read(fname, 'H5', group=task)
        yield task, stream


def _iter_h5(io, key, level=3):
    """Iterator yielding streams or stretching results, depending on key"""
    is_stretch = 't' in _analyze_key(key)
    fname = _get_fname(io, key)
    iter_ = _iter_dicts if is_stretch else _iter_streams
    for obj in iter_(fname, key, level=level):
        yield obj


def _write_corr(result, io, dtype=None):
    """Write result from yam.correlate.correlate"""
    if result is not None:
        for key in result:
            result[key].write(io[key], 'H5', dtype=dtype, mode='a')
