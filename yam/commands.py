# Copyright 2017 Tom Eulenfeld, GPLv3

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

from yam.correlate import correlate
import yam.stack
import yam.stretch
from yam.util import _analyze_key, _filter, IterTime, ParseError


log = logging.getLogger('yam.commands')

# group will start with waveforms (default of obspyh5)
INDEX = ('{key}/{network1}.{station1}-{network2}.{station2}/'
         '{location1}.{channel1}-{location2}.{channel2}/'
         '{starttime.datetime:%Y-%m-%dT%H:%M}')
INDEX_STRETCH = ('stretch/{key}/{network1}.{station1}-{network2}.{station2}/'
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


def _write_stretch(queue):
    while True:
        args = queue.get()
        if len(args) == 2:
            stretchres, fname = args
            attrs = stretchres['attrs']
            with h5py.File(fname, 'a') as f:
                index = INDEX_STRETCH.format(**attrs)
                group = f.require_group(index)
                for key, val in attrs.items():
                    group.attrs[key] = val
                for key, val in stretchres.items():
                    if key not in ('attrs', ):
                        group.create_dataset(key, data=val)
        else:
            break


def start_parallel_jobs(tasks, do_work, write, njobs=None):
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
    return existing keys at level in HDF5 file
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
        level_reached = g.name.count('/')
        visit(g, level - level_reached)
    return done


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
                    njobs=None, **kwargs):
    if filter_inventory:
        log.debug('filter inventory')
        io['inventory'] = io['inventory'].select(**filter_inventory)
    log.info('start preprocessing and correlation')
    tasks = list(IterTime(UTC(startdate), UTC(enddate)))
    # check for existing days
    # Not perfect yet:
    # If a day exits for one combination and not for another station
    # combination, it will be marked as done
    done_tasks = None
    if kwargs.get('stack') is not None:
        key2 = 'waveforms/' + kwargs['outkey'] + '_s' + kwargs['stack']
        done_tasks = [UTC(t[-16:-6]) for t in
                      _get_existent(io['stack'], key2, 5)]
    if kwargs.get('keep_correlations', False):
        key2 = 'waveforms/' + kwargs['outkey']
        done_tasks2 = [UTC(t[-16:-6]) for t in
                       _get_existent(io['corr'], key2, 5)]
        if done_tasks is None:
            done_tasks = done_tasks2
        else:
            done_tasks = [t for t in done_tasks if t in done_tasks2]
    tasks = _todo_tasks(tasks, done_tasks)
    do_work = functools.partial(correlate, io, **kwargs)
    start_parallel_jobs(tasks, do_work, _write_stream, njobs=njobs)
    log.info('finished preprocessing and correlation')


def start_stack(io, key, outkey, subkey='', **kwargs):
    fname = io['stack'] if 's' in _analyze_key(key) else io['corr']
    rootkey1 = 'waveforms/' + key
    rootkey2 = 'waveforms/' + outkey
    tasks = _get_existent(fname, rootkey1 + subkey, 4)
    done_tasks = [t.replace(rootkey2, rootkey1) for t in
                  _get_existent(io['stack'], rootkey2 + subkey, 4)]
    tasks = _todo_tasks(tasks, done_tasks)
    for task in tqdm.tqdm(tasks, total=len(tasks)):
        stream = obspy.read(fname, 'H5', group=task)
        stack_stream = yam.stack.stack(stream, **kwargs)
        for tr in stack_stream:
            tr.stats.key = outkey
        stack_stream.write(io['stack'], 'H5', mode='a')


def stretch_wrapper(groupname, fname, fname_stretch, outkey, filter=None,
                    **kwargs):
    stream = obspy.read(fname, 'H5', group=groupname)
    if filter:
        _filter(stream, filter)
    reftr = yam.stack.stack(stream)[0]
    stretchres = yam.stretch.stretch(stream, reftr, **kwargs)
    stretchres['attrs']['key'] = outkey
    stretch_wrapper.q.put((stretchres, fname_stretch))


def start_stretch(io, key, subkey='', njobs=None, **kwargs):
    fname = _get_fname(io, key)
    rootkey1 = 'waveforms/' + key
    rootkey2 = 'stretch/' + kwargs['outkey']
    tasks = _get_existent(fname, rootkey1 + subkey, 4)
    done_tasks = [t.replace(rootkey2, rootkey1) for t in
                  _get_existent(io['stretch'], rootkey2 + subkey, 4)]
    tasks = _todo_tasks(tasks, done_tasks)
    do_work = functools.partial(stretch_wrapper, fname=fname,
                                fname_stretch=io['stretch'], **kwargs)
    start_parallel_jobs(tasks, do_work, _write_stretch, njobs=njobs)


def read_stretch(fname, groupname):
    res = {'attrs': {}}
    with h5py.File(fname) as f:
        group = f[groupname]
        for key, val in group.attrs.items():
            res['attrs'][key] = val
        for key, val in group.items():
            if key not in ('attrs'):
                res[key] = val[:]
        res['times'] = [UTC(t) for t in res['times']]
    return res


def _print_info_helper(key, io):
    print2 = _get_print2()
    is_stretch = key == 'tstretch'
    fname = _get_fname(io, key)
    rootkey = _get_rootkey(key)
    keys = _get_existent(fname, rootkey, 2)  # 2, 4, 5
    if len(keys) == 0:
        print2('None')
    for key in sorted(keys):
        keys2 = _get_existent(fname, key, 4)
        subkey = key.split('/')[-1]
        if is_stretch:
            o = '%s: %d combs' % (subkey, len(keys2))
        else:
            keys3 = _get_existent(fname, key, 5)
            o = ('%s: %d combs, %d corrs' %
                 (subkey, len(keys2), len(keys3)))
        print2(o)


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


def _get_rootkey(key):
    return 'stretch/' if 't' in _analyze_key(key) else 'waveforms/'


def _iterh5(key, io):
    is_stretch = 't' in _analyze_key(key)
    fname = _get_fname(io, key)
    rootkey = _get_rootkey(key)
    tasks = _get_existent(fname, rootkey + key, 4)
    for task in tasks:
        if is_stretch:
            res = read_stretch(fname, task)
            yield task, res
        else:
            stream = obspy.read(fname, 'H5', group=task)
            yield task, stream


def _load_obj(what, io, key):
    is_stretch = 't' in _analyze_key(key)
    fname = _get_fname(io, key)
    rootkey = _get_rootkey(key)
    level = 4 if is_stretch else 5
    keys2 = _get_existent(fname, rootkey + key, level)
    if what == 'info':
        return '\n'.join(keys2)
    if is_stretch:
        res = {}
        out = []
        for k in keys2:
            s = read_stretch(fname, k)
            res[k] = s
            out.append('%s\n%s' % (k, s))
        if what == 'print':
            return '\n\n'.join(out)
    else:
        if what == 'print':
            res = obspy.read(fname, 'H5', headonly=True, group=rootkey + key)
            res.sort()
            return res.__str__(extended=True)
        else:
            res = obspy.read(fname, 'H5', group=rootkey + key)
    assert what == 'load'
    return res


def _get_print2():
    num, _ = shutil.get_terminal_size()
    if num == 0:
        num = 80
    wrap = textwrap.TextWrapper(width=num - 1, initial_indent='    ',
                                subsequent_indent='    ')

    def print2(text):
        print(wrap.fill(text))
    return print2


def _get_data_files(data):
    from obspy import UTCDateTime as UTC
    kw = dict(network='*', station='*', location='*', channel='*',
              t=UTC('2211-11-11 11:11:11'))
    dataglob = data.format(**kw)
    dataglob = dataglob.replace('22', '*').replace('11', '*')
    fnames = glob.glob(dataglob)
    return fnames


def info(io, key=None, subkey='', config=None, **unused_kwargs):
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
        obj = _load_obj('info', io, key + subkey)
        for line in sorted(obj.splitlines()):
            print2(line)


def _load_data(seedid, day, data, data_format, key='data',
               **prep_kw):
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
        obj = _load_obj('print' if do == 'print' else 'load', io, key)
    if do == 'print':
        print(obj)
    elif do == 'load':
        _start_ipy(obj)
    elif do == 'return':
        return obj
    elif do == 'export':
        obspyh5.set_index()
        obj.write(fname, format)
        obspyh5.set_index(INDEX)
    else:
        raise


def plot(io, key, plottype=None, seedid=None, day=None, prep_kw={},
         corrid=None, show=False,
         **kwargs):
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
            rootkey = _get_rootkey(key)
            stream = obspy.read(fname2, 'H5', group=rootkey + key)
            fname = bname + '_' + key.replace('/', '_')
            plot_(stream, fname, **kw)
        else:
            for task, res in _iterh5(key, io):
                fname = bname + '_' + task.split('/', 2)[-1].replace('/', '_')
                plot_(res, fname, **kw)
    if show:
        from matplotlib.pyplot import show
        show()


def remove(io, keys):
    for key in keys:
        subkey = ''
        if '/' in key:
            key, subkey = key.split('/', 1)
        if subkey != '':
            from warnings import warn
            warn('It is highly encouraged to delete only top level keys')
        fname = _get_fname(io, key)
        rootkey = _get_rootkey(key)
        with h5py.File(fname, 'a') as f:
            del f[rootkey + '/' + key + '/' + subkey]
