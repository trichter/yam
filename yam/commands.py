# Copyright 2017-2018 Tom Eulenfeld, GPLv3
"""Commands used by the CLI interface"""
import functools
import glob
import logging
import multiprocessing
import os
import sys
import shutil
import textwrap

import h5py
import obspy
from obspy.core import UTCDateTime as UTC
import obspyh5
import tqdm

import yam
from yam.correlate import correlate
import yam.stack
import yam.stretch
from yam.util import _analyze_key, _filter, IterTime, ParseError


log = logging.getLogger('yam.commands')

INDEX = ('{key}/{network1}.{station1}-{network2}.{station2}/'
         '{location1}.{channel1}-{location2}.{channel2}/'
         '{starttime.datetime:%Y-%m-%dT%H:%M}')
INDEX_STRETCH = ('{key}/{network1}.{station1}-{network2}.{station2}/'
                 '{location1}.{channel1}-{location2}.{channel2}')
obspyh5.set_index(INDEX)


def _write_stream(queue):
    while True:
        args = queue.get()
        if len(args) == 2:
            stream, fname = args
            stream.write(fname, 'H5', mode='a')
        else:
            break


def write_dict(dict_, fname, mode='a', libver='latest'):
    """
    Write similarity matrix into HDF5 file

    :param dict_: Dictionary with stretching results
        (output from `~yam.stretch.stretch()`)
    :param fname: file name
    :param mode: file mode (default ``'a'`` -- write into file)
    :param libver: use latest version of HDF5 file format
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
        for key, val in dict_.items():
            if key not in ('attrs', ):
                group.create_dataset(key, data=val)


def _write_stretch(queue):
    while True:
        args = queue.get()
        if len(args) == 2:
            stretchres, fname = args
            write_dict(stretchres, fname)
        else:
            break


def start_parallel_jobs(tasks, do_work, write, njobs=1):
    queue = multiprocessing.Queue()
    proc = multiprocessing.Process(target=write, args=(queue, ))
    proc.start()
    do_work.func.q = queue
    try:
        if njobs == 1:
            log.info('do work sequentially')
            for task in tqdm.tqdm(tasks, total=len(tasks)):
                do_work(task)
        else:
            pool = multiprocessing.Pool(njobs)
            log.info('do work parallel (%d cores)', pool._processes)
            for _ in tqdm.tqdm(pool.imap_unordered(do_work, tasks),
                               total=len(tasks)):
                pass
            pool.close()
            pool.join()
    except KeyboardInterrupt:
        log.warning('Keyboard interrupt')
        sys.exit()
    finally:
        queue.put('finished')
        proc.join()


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


def _todo_tasks(tasks, done_tasks):
    if len(tasks) == 0:
        log.warning('no tasks found -> nothing to do')
    new_tasks = [t for t in tasks if t not in done_tasks]
    if len(new_tasks) < len(tasks):
        msg = '%d of %d tasks already processed -> skip these tasks'
        log.info(msg, len(tasks) - len(new_tasks), len(tasks))
    return sorted(new_tasks)


def start_correlate(io,
                    filter_inventory=None,
                    startdate='1990-01-01', enddate='2020-01-01',
                    njobs=None,
                    parallel_inner_loop=False,
                    keep_correlations=False,
                    stack='1d',
                    **kwargs):
    """
    Start correlation

    :param io: |io|
    :param filter_inventory: filter inventory with its select method,
        specified dict is passed to |Inventory.select|
    :param str startdate,enddate: start and end date as strings
    :param njobs: number of cores to use for computation, days are computed
        parallel, this might consume much memory, default: None -- use all
        available cores
    :param parallel_inner_loop: Run inner loops parallel instead of outer loop
        (preproccessing of different stations and correlation of different
        pairs versus processing of different days).
        Useful for a datset with many stations.
    :param keep_correlations,stack,\*\*kwargs: all other kwargs are passed to
        `~yam.correlate.correlate()` function
    """
    if filter_inventory:
        log.debug('filter inventory')
        io['inventory'] = io['inventory'].select(**filter_inventory)
    log.info('start preprocessing and correlation')
    tasks = list(IterTime(UTC(startdate), UTC(enddate)))
    done_tasks = None
    if stack is not None:
        key2 = kwargs['outkey'] + '_s' + stack
        done_tasks = [UTC(t[-16:-6]) for t in
                      _get_existent(io['stack'], key2, 4)]
    if keep_correlations:
        key2 = kwargs['outkey']
        done_tasks2 = [UTC(t[-16:-6]) for t in
                       _get_existent(io['corr'], key2, 4)]
        if done_tasks is None:
            done_tasks = done_tasks2
        else:
            done_tasks = [t for t in done_tasks if t in done_tasks2]
    tasks = _todo_tasks(tasks, done_tasks)
    kwargs.update({'keep_correlations': keep_correlations, 'stack': stack})
    if parallel_inner_loop:
        kwargs['njobs'] = njobs
        njobs = 1
    do_work = functools.partial(correlate, io, **kwargs)
    start_parallel_jobs(tasks, do_work, _write_stream, njobs=njobs)
    log.info('finished preprocessing and correlation')


def start_stack(io, key, outkey, subkey='', **kwargs):
    """
    Start stacking

    :param io: |io|
    :param key:  key to load correlations from
    :param outkey: key to write stacked correlations to
    :param subkey: only use a part of the correlations
    :param \*\*kwargs: all other kwargs are passed to
        `yam.stack.stack()` function
    """
    fname = io['stack'] if 's' in _analyze_key(key) else io['corr']
    tasks = _get_existent(fname, key + subkey, 3)
    done_tasks = [t.replace(outkey, key) for t in
                  _get_existent(io['stack'], outkey + subkey, 3)]
    tasks = _todo_tasks(tasks, done_tasks)
    for task in tqdm.tqdm(tasks, total=len(tasks)):
        stream = obspy.read(fname, 'H5', group=task)
        stack_stream = yam.stack.stack(stream, **kwargs)
        for tr in stack_stream:
            tr.stats.key = outkey
        stack_stream.write(io['stack'], 'H5', mode='a')


def stretch_wrapper(groupname, fname, fname_stretch, outkey, filter=None,
                    **kwargs):
    """
    Wrapper around `~yam.stretch.stretch()`

    :param groupname: group to load the correlations from
    :param fname: file to load correlations from
    :param fname_stretch: file for writing results
    :param outkey: key to write stacked correlations to
    :param filter: filter correlations before stretching
        (bandpass, tuple with min and max frequency)
    :param \*\*kwargs: all other kwargs are passed to
        `~yam.stretch.stretch()` function
    """
    stream = obspy.read(fname, 'H5', group=groupname)
    if filter:
        _filter(stream, filter)
    stretchres = yam.stretch.stretch(stream, **kwargs)
    if stretchres is not None:
        stretchres['attrs']['key'] = outkey
        stretch_wrapper.q.put((stretchres, fname_stretch))


def start_stretch(io, key, subkey='', njobs=None, **kwargs):
    """
    Start stretching

    :param io: |io|
    :param key:  key to load correlations from
    :param subkey: only use a part of the correlations
    :param njobs: number of cores to use for computation,
        might consume much memory, default: None -- use all available cores
    :param \*\*kwargs: all other kwargs are passed to
        `stretch_wrapper()` function
    """
    fname = _get_fname(io, key)
    outkey = kwargs['outkey']
    tasks = _get_existent(fname, key + subkey, 3)
    done_tasks = [t.replace(outkey, key) for t in
                  _get_existent(io['stretch'], outkey + subkey, 3)]
    tasks = _todo_tasks(tasks, done_tasks)
    do_work = functools.partial(stretch_wrapper, fname=fname,
                                fname_stretch=io['stretch'], **kwargs)
    start_parallel_jobs(tasks, do_work, _write_stretch, njobs=njobs)


def _read_dict(group):
    """Read a single stretching dictionary from group"""
    res = {'attrs': {}}
    for key, val in group.attrs.items():
        res['attrs'][key] = val
    for key, val in group.items():
        if key not in ('attrs'):
            res[key] = val[:]
    res['times'] = [UTC(t) for t in res['times']]
    res['group'] = group.name
    return res


def _iter_dicts(fname, groupname='/', level=3):
    """Iterator yielding stretching dictionaries"""
    tasks = _get_existent(fname, groupname, level)
    with h5py.File(fname, 'r') as f:
        for task in tasks:
            yield task, _read_dict(f[task])


def read_dicts(fname, groupname='/', level=3):
    """
    Read dictionaries with stretching results

    :param fname: file name
    :param groupname: specify group to read
    :param level: level in index where the data was written, defaults to 3 and
        should not be changed
    :return: list of dictionaries with stretching results
    """
    return [obj[1] for obj in _iter_dicts(fname, groupname, level)]


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


def _start_ipy(obj):
    from IPython import start_ipython
    print('Contents loaded into obj variable.')
    start_ipython(argv=[], user_ns={'obj': obj}, display_banner=False)
    print('Good Bye')


def _get_fname(io, key):
    fname = (io['stretch'] if 't' in _analyze_key(key)
             else io['stack'] if 's' in _analyze_key(key)
             else io['corr'])
    return fname


def _get_print2():
    num, _ = shutil.get_terminal_size()
    if num == 0:
        num = 80
    wrap = textwrap.TextWrapper(width=num - 1, initial_indent='    ',
                                subsequent_indent='    ')

    def print2(text):
        print(wrap.fill(text))
    return print2


def _get_data_glob(data):
    """
    Construct a glob expression from the data expression
    """
    from obspy import UTCDateTime as UTC
    kw = dict(network='*', station='*', location='*', channel='*',
              t=UTC('2211-11-11 11:11:11'))
    dataglob = data.format(**kw)
    dataglob = dataglob.replace('22', '*').replace('11', '*')
    return dataglob


def _get_data_files(data):
    """
    Construct a glob expression from the data expression and return file names
    """
    return glob.glob(_get_data_glob(data))


def _print_info_helper(key, io):
    print2 = _get_print2()
    is_stretch = key == 'tstretch'
    fname = _get_fname(io, key)
    keys = _get_existent(fname, '/', 1)  # 1, 3, 4
    if len(keys) == 0:
        print2('None')
    for key in sorted(keys):
        keys2 = _get_existent(fname, key, 3)
        subkey = key.split('/')[-1]
        if is_stretch:
            o = '%s: %d combs' % (subkey, len(keys2))
        else:
            keys3 = _get_existent(fname, key, 4)
            o = ('%s: %d combs, %d corrs' %
                 (subkey, len(keys2), len(keys3)))
        print2(o)


def info(io, key=None, subkey='', config=None, **unused_kwargs):
    """
    Print information about yam project

    :param io: |io|
    :param key: key to print infos about
        (key inside HDF5 file, or one of data, stations,
        default: None -- print overview)
    :param subkey: only print part of the HDF5 file
    :param config: list of configuration dictionaries
    """
    print2 = _get_print2()
    data_plugin = io.get('data_plugin')
    if key is None:
        print('Stations:')
        inventory = io['inventory']
        if inventory is None:
            print2('Not found')
        else:
            stations = inventory.get_contents()['stations']
            channels = inventory.get_contents()['channels']
            print2(' '.join(st.strip().split()[0] for st in stations))
            print2('%d stations, %d channels' % (len(stations), len(channels)))
        if data_plugin:
            print('Data plugin:')
            print2('%s' % data_plugin)
        else:
            print('Raw data (expression for day files):')
            print2(io['data'])
            print2('%d files found' % len(_get_data_files(io['data'])))
        print('Config ids:')

        def get_keys(d):
            if d is None or len(d) == 0:
                return 'None'
            else:
                return ', '.join(sorted(d.keys()))
        print2('c Corr: ' + get_keys(config[0]))
        print2('s Stack: ' + get_keys(config[1]))
        print2('t Stretch: ' + get_keys(config[2]))
        print('Correlations (channel combinations, correlations calculated):')
        _print_info_helper('corr', io)
        print('Stacks:')
        _print_info_helper('stack', io)
        print('Stretching matrices:')
        _print_info_helper('tstretch', io)
    elif key == 'stations':
        print(io['inventory'])
    elif key == 'data':
        if data_plugin:
            print('Data plugin:')
            print2('%s' % data_plugin)
        else:
            print('Raw data (expression for day files):')
            print2(io['data'])
            fnames = _get_data_files(io['data'])
            print2('%d files found' % len(fnames))
            for fname in sorted(fnames):
                print2(fname)
    else:
        is_stretch = 't' in _analyze_key(key)
        fname = _get_fname(io, key)
        level = 3 if is_stretch else 4
        for line in _get_existent(fname, key + subkey, level):
            print2(line)


def _load_data(seedid, day, data, data_format, key='data',
               **prep_kw):
    """Load preprocessed or raw data"""

    from obspy import UTCDateTime as UTC
    from yam.util import _seedid2meta
    from yam.correlate import get_data, preprocess
    smeta = _seedid2meta(seedid)
    day = UTC(day)
    if key == 'data':
        obj = get_data(smeta, data, data_format, day,
                       overlap=0, edge=0, trim_and_merge=True)
        return obj
    stream = get_data(smeta, data, data_format, day,
                      overlap=0, edge=60, trim_and_merge=False)
    preprocess(stream, day, **prep_kw)
    return stream


def load(io, key, seedid=None, day=None, do='return', prep_kw={},
         fname=None, format=None):
    """
    Load object and do something with it

    :param io: io
    :param key: key of object to load
        (key inside HDF5 file, or one of data, prepdata, stations)
    :param seedid: seed id of a  channel (for data or prepdata)
    :param day: |UTC| object with day (for data or prepdata)
    :param do: specifies what to do with the object, default is ``'return'``
        which simply returns the object, other possible values are
        ``'print'`` -- print object (used by print command),
        ``'load'`` -- load object in IPython session (used by load command),
        ``'export'`` -- export correlations to different file format
        (used by export command)
    :param dict prep_kw: options passed to preprocess (for prepdata only)
    :param fname: file name (for export command)
    :param format: target format (for export command)
    """
    if key == 'stations':
        obj = io['inventory']
    elif key in ('data', 'prepdata'):
        if seedid is None or day is None:
            msg = 'seedid and day must be given for data or prepdata'
            raise ParseError(msg)
        if key == 'prepdata':
            prep_keys = ('remove_response', 'remove_response_options',
                         'filter', 'normalization',
                         'time_norm_options', 'spectral_whitening_options',
                         'downsample')
            prep_kw = {k: prep_kw.get(k) for k in prep_keys}
        obj = _load_data(seedid, day, io['data'], io.get('data_format'),
                         key, inventory=io['inventory'], **prep_kw)
    else:
        is_stretch = 't' in _analyze_key(key)
        fname_in = _get_fname(io, key)
        if is_stretch:
            obj = read_dicts(fname_in, key)
            if do == 'print':
                obj = '\n\n'.join(str(o) for o in obj)
        else:
            obj = obspy.read(fname_in, 'H5', group=key, headonly=do == 'print')
            if do == 'print':
                obj = obj.__str__(extended=True)
    if do == 'print':
        print(obj)
    elif do == 'load':
        _start_ipy(obj)
    elif do == 'return':
        return obj
    elif do == 'export':
        print(obj)
        obspyh5.set_index()
        obj.write(fname, format)
        obspyh5.set_index(INDEX)
    else:
        raise


def plot(io, key, plottype=None, seedid=None, day=None, prep_kw={},
         corrid=None, show=False,
         **kwargs):
    """
    Plot everything

    :param io: |io|
    :param key: key of objects to plot, or one of stations, data, prepdata
    :param plottype: plot type to use
        (non default values are ``'vs_dist'`` and ``'wiggle'`` for
        correlation plots)
    :param seedid: seed id of a  channel (for data or prepdata)
    :param day: |UTC| object with day (for data or prepdata)
    :param dict prep_kw: options passed to preprocess (for prepdata only)
    :param corrid: correlation configuration (for prepdata only)
    :param show: show interactive plot
    :param \*\*kwargs: all other kwargs are passed to
        the corresponding plot function in `~yam.imaging` module
    """
    import yam.imaging
    path = io['plot']
    if not os.path.exists(path):
        os.mkdir(path)
    if key in ('stations', 'data', 'prepdata'):
        pt = key
    else:
        is_corr = 't' not in _analyze_key(key)
        if is_corr and plottype == 'vs_dist':
            pt = 'corr_vs_dist'
        elif is_corr and plottype == 'wiggle':
            pt = 'corr_vs_time_wiggle'
        elif is_corr and plottype is None:
            pt = 'corr_vs_time'
        elif not is_corr and plottype is None:
            pt = 'sim_mat'
        else:
            raise ParseError('Combination of key and plottype not supported')

    kw = kwargs.get('plot_%s_options' % pt, {})
    kw.update(kwargs.get('plot_options', {}))
    bname = os.path.join(path, pt)
    if key == 'stations':
        yam.imaging.plot_stations(io['inventory'], bname, **kw)
    elif key in ('data', 'prepdata'):
        data = load(io, key, do='return', seedid=seedid, day=day,
                    prep_kw=prep_kw)
        fname = bname + '_%s_%s' % (seedid, day)
        if key == 'prepdata':
            fname = fname + '_c' + corrid
        yam.imaging.plot_data(data, fname, show=show, **kw)
    else:
        plot_ = getattr(yam.imaging, 'plot_' + pt)
        if pt == 'corr_vs_dist':
            fname2 = _get_fname(io, key)
            stream = obspy.read(fname2, 'H5', group=key)
            fname = bname + '_' + key.replace('/', '_')
            plot_(stream, fname, **kw)
        else:
            for task, res in _iter_h5(io, key):
                fname = bname + task.replace('/', '_')
                plot_(res, fname, **kw)
    if show:
        from matplotlib.pyplot import show
        show()


def remove(io, keys):
    """
    Remove one or several keys from HDF5 file

    :param io: |io|
    :param keys: list of keys to remove
    """
    for key in keys:
        if '/' in key and key.split('/', 1) != '':
            from warnings import warn
            warn('It is highly encouraged to delete only top level keys')
        fname = _get_fname(io, key)
        with h5py.File(fname, 'a') as f:
            del f[key]
