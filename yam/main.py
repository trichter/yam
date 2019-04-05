# Copyright 2017-2018 Tom Eulenfeld, GPLv3
"""
Command line interface and main entry point
"""

import argparse
from argparse import SUPPRESS
from copy import deepcopy
import glob
import json
import logging
import logging.config
import sys
import time

import obspy

import yam.commands
from yam.util import (_load_func, create_config,
                      LOGLEVELS, LOGGING_DEFAULT_CONFIG, ParseError,
                      ConfigError)


log = logging.getLogger('yam')
log.addHandler(logging.NullHandler())


class ConfigJSONDecoder(json.JSONDecoder):
    def decode(self, s):
        """Decode JSON config with comments stripped"""
        s = '\n'.join(l.split('#', 1)[0] for l in s.split('\n'))
        return super(ConfigJSONDecoder, self).decode(s)


def configure_logging(loggingc, verbose=0, loglevel=3, logfile=None):
    if loggingc is None:
        loggingc = deepcopy(LOGGING_DEFAULT_CONFIG)
        if verbose > 3:
            verbose = 3
        loggingc['handlers']['console']['level'] = LOGLEVELS[verbose]
        loggingc['handlers']['console_tqdm']['level'] = LOGLEVELS[verbose]
        if logfile is None or loglevel == 0:
            del loggingc['handlers']['file']
            loggingc['loggers']['yam']['handlers'] = ['console_tqdm']
            loggingc['loggers']['py.warnings']['handlers'] = ['console_tqdm']
        else:
            loggingc['handlers']['file']['level'] = LOGLEVELS[loglevel]
            loggingc['handlers']['file']['filename'] = logfile
    logging.config.dictConfig(loggingc)
    logging.captureWarnings(loggingc.get('capture_warnings', False))


def __get_station(seedid):
    """Station name from seed id"""
    st = seedid.rsplit('.', 2)[0]
    if st.startswith('.'):
        st = st[1:]
    return st


def load_inventory(inventory):
    try:
        if not isinstance(inventory, obspy.Inventory):
            if isinstance(inventory, str):
                format_ = None
            else:
                inventory, format_ = inventory
            expr = inventory
            inventory = None
            for fname in glob.glob(expr):
                inv2 = obspy.read_inventory(fname, format_)
                if inventory is None:
                    inventory = inv2
                else:
                    inventory += inv2
            channels = inventory.get_contents()['channels']
            stations = list(set(__get_station(ch) for ch in channels))
            log.info('read inventory with %d stations', len(stations))
    except:
        log.exception('cannot read stations')
        return
    return inventory


def _get_kwargs(kwargs, id_):
    kw = kwargs[id_]
    if 'based_on' in kw:
        kw2 = kwargs[kw.pop('based_on')]
        kw2.update(kw)
        kw = kw2
    return kw


def run(command, conf=None, tutorial=False, less_data=False, pdb=False,
        **args):
    """Main entry point for a direct call from Python

    Example usage:

    >>> from yam import run
    >>> run(conf='conf.json')

    :param command: if ``'create'`` the example configuration is created,
       optionally the tutorial data files are downloaded

    For all other commands this function loads the configuration
    and construct the arguments which are passed to `run2()`

    All args correspond to the respective command line and
    configuration options.
    See the example configuration file for help and possible arguments.
    Options in args can overwrite the configuration from the file.
    E.g. ``run(conf='conf.json', bla='bla')`` will set bla configuration
    value to ``'bla'``.
    """
    if pdb:
        import traceback, pdb

        def info(type, value, tb):
            traceback.print_exception(type, value, tb)
            print
            # ...then start the debugger in post-mortem mode.
            pdb.pm()

        sys.excepthook = info
    if conf in ('None', 'none', 'null', ''):
        conf = None
    # Copy example files if create_config or tutorial
    if command == 'create':
        if conf is None:
            conf = 'conf.json'
        create_config(conf, tutorial=tutorial, less_data=less_data)
        return
    # Parse config file
    if conf:
        try:
            with open(conf) as f:
                conf = json.load(f, cls=ConfigJSONDecoder)
        except ValueError as ex:
            msg = 'Error while parsing the configuration: %s' % ex
            raise ConfigError(msg)
        except IOError as ex:
            raise ConfigError(ex)
        # Populate args with conf, but prefer args
        conf.update(args)
        args = conf
    run2(command, **args)


def run2(command, io,
         logging=None, verbose=0, loglevel=3, logfile=None,
         key=None, keys=None, corrid=None, stackid=None, stretchid=None,
         correlate=None, stack=None, stretch=None,
         **args):
    """
    Second main function for unpacking arguments

    Initialize logging, load inventory if necessary, load options from
    configuration dictionary into args
    (for correlate, stack and stretch commands) and run the corresponding
    command in `~yam.commands` module. If ``"based_on"`` key is set the
    configuration dictionary will be preloaded with the specified
    configuration.

    :param command: specified subcommand, will call one of
        `~yam.commands.start_correlate()`,
        `~yam.commands.start_stack()`,
        `~yam.commands.start_stretch()`,
        `~yam.commands.info()`,
        `~yam.commands.load()`,
        `~yam.commands.plot()`,
        `~yam.commands.remove()`
    :param logging,verbose,loglevel,logfile: logging configuration
    :param key: the key to work with
    :param keys: keys to remove (only remove command)
    :param correlate,stack,stretch: corresponding configuration dictionaries
    :param \*id: the configuration id to load from the config dictionaries
    :param \*\*args: all other arguments are passed to next called function
    """
    time_start = time.time()
    # Configure logging
    if command in ('correlate', 'stack', 'stretch'):
        configure_logging(loggingc=logging, verbose=verbose,
                          loglevel=loglevel, logfile=logfile)
    log.info('Yam version %s', yam.__version__)
    if key is not None and '/' in key:
        key, subkey = key.split('/', 1)
        subkey = '/' + subkey
    else:
        subkey = ''
    # load data plugin, discard unnecessary io kwargs
    data_plugin = io.get('data_plugin')
    if command == 'correlate' or (
            command in ('print', 'load', 'plot') and
            key in ('data', 'prepdata')):
        if data_plugin is not None:
            modulename, funcname = data_plugin.split(':')
            io['data'] = _load_func(modulename.strip(), funcname.strip())
    # pop plotting options
    if command != 'plot':
        for k in list(args.keys()):
            if k.startswith('plot_'):
                args.pop(k)
    # Start main routine
    if command == 'correlate':
        kw = _get_kwargs(correlate, corrid)
        kw['outkey'] = 'c' + corrid
        args.update(kw)
        io['inventory'] = load_inventory(kw.get('inventory') or
                                         io.get('inventory'))
        yam.commands.start_correlate(io, **args)
    elif command == 'stack':
        if stack is not None and stackid in stack:
            kw = _get_kwargs(stack, stackid)
        else:
            kw = {}
            if 'm' in stackid:
                kw['length'], kw['move'] = stackid.split('m')
            elif stackid in ('', 'all', 'None', 'none', 'null'):
                stackid = ''
                kw['length'] = None
            else:
                kw['length'] = stackid
        kw['outkey'] = key + '_s' + stackid
        args.update(kw)
        yam.commands.start_stack(io, key, subkey=subkey, **args)
    elif command == 'stretch':
        kw = _get_kwargs(stretch, stretchid)
        kw['outkey'] = key + '_t' + stretchid
        args.update(kw)
        yam.commands.start_stretch(io, key, subkey=subkey, **args)
    elif command == 'remove':
        yam.commands.remove(io, keys)
    elif command in ('info', 'print', 'load', 'plot', 'export'):
        if key == 'stations' or command == 'info' and key is None:
            io['inventory'] = load_inventory(io['inventory'])
        if key == 'prepdata':
            if corrid is None:
                msg = 'seed id, day and corrid need to be set for prepdata'
                raise ParseError(msg)
            kw = _get_kwargs(correlate, corrid)
            io['inventory'] = load_inventory(kw.get('inventory') or
                                             io['inventory'])
            args['prep_kw'] = kw
        if command == 'info':
            yam.commands.info(io, key=key, subkey=subkey,
                              config=(correlate, stack, stretch), **args)
        elif command == 'print':
            yam.commands.load(io, key=key + subkey, do='print', **args)
        elif command == 'load':
            yam.commands.load(io, key=key + subkey, do='load', **args)
        elif command == 'export':
            yam.commands.load(io, key=key + subkey, do='export', **args)
        else:
            yam.commands.plot(io, key + subkey, corrid=corrid, **args)
    elif command == 'scan':
        data_glob = yam.commands._get_data_glob(io['data'])
        print('Suggested call to obspy-scan:')
        if io.get('data_format') is not None:
            print('obspy-scan -f %s %s' % (io['data_format'], data_glob))
        else:
            print('obspy-scan %s' % data_glob)
        print('This will probably not work if you use the data_plugin '
              'configuration. '
              'For more options check obspy-scan -h.')
    else:
        raise ValueError('Unknown command')
    time_end = time.time()
    log.debug('used time: %.1fs', time_end - time_start)


def run_cmdline(args=None):
    """Main entry point from the command line"""
    # Define command line arguments
    from yam import __version__
    msg = ('Yam: Yet another monitoring tool using cross-correlations of '
           'ambient noise')
    epilog = 'To get help on a subcommand run: yam command -h'
    p = argparse.ArgumentParser(description=msg, epilog=epilog)
    version = '%(prog)s ' + __version__
    p.add_argument('--version', action='version', version=version)
    msg = 'configuration file to load (default: conf.json)'
    p.add_argument('-c', '--conf', default='conf.json', help=msg)
    msg = 'if an exception occurs start the debugger'
    p.add_argument('--pdb', action='store_true', help=msg)

    sub = p.add_subparsers(title='commands', dest='command')
    sub.required = True
    msg = 'create config file in current directory'
    p_create = sub.add_parser('create', help=msg)
    msg = 'preprocess and correlate data'
    p_correlate = sub.add_parser('correlate', help=msg)
    msg = 'stack correlations or stacked correlations'
    p_stack = sub.add_parser('stack', help=msg)
    msg = 'stretch correlations or stacked correlations'
    p_stretch = sub.add_parser('stretch', help=msg)
    msg = 'remove keys from HDF5 files'
    p_remove = sub.add_parser('remove', help=msg)
    msg = 'print information about project or objects'
    p_info = sub.add_parser('info', help=msg)
    msg = 'print objects'
    p_print = sub.add_parser('print', help=msg)
    msg = 'print suggested call for obspy-scan scipt (data availability)'
    p_scan = sub.add_parser('scan', help=msg)
    msg = 'plot objects'
    p_plot = sub.add_parser('plot', help=msg)
    msg = 'load objects into IPython session'
    p_load = sub.add_parser('load', help=msg)
    msg = ('export corrrelations or stacks to another file format '
           'for processing with other programs, '
           'preprocessed data can also be saved')
    p_export = sub.add_parser('export', help=msg)

    msg = 'additionaly create example files for tutorial'
    p_create.add_argument('--tutorial', help=msg, action='store_true')
    p_create.add_argument('--less-data', help=SUPPRESS, action='store_true')

    msg = 'use the data defined by this key (processing chain)'
    p_stack.add_argument('key', help=msg)
    p_stretch.add_argument('key', help=msg)
    msg = 'keys of data to be removed'
    p_remove.add_argument('keys', help=msg, nargs='+')
    msg = 'configuration key for correlation'
    p_correlate.add_argument('corrid', help=msg)
    msg = 'configuration key or explicit configuration for stacking'
    p_stack.add_argument('stackid', help=msg)
    msg = 'configuration key for stretching'
    p_stretch.add_argument('stretchid', help=msg)
    msg = 'key of reference trace (stack)'
    p_stretch.add_argument('--reftrid', help=msg)

    msg1 = 'more detailed information about '
    msg2 = ('stations, data or specific processing key '
            '(e.g. stations, data, '
            'c1_s1d, c1_s1d_t1/CX.PB06-CX.PB06/.BHZ-.BHZ)')
    p_info.add_argument('key', help=msg1 + msg2, nargs='?')
    msg1 = 'print contents of '
    msg2 = ('stations, data, preprocessed data or specific processing key '
            '(e.g. stations, data, prepdata, '
            'c1_s1d, c1_s1d_t1/CX.PB06-CX.PB06/.BHZ-.BHZ)')
    p_print.add_argument('key', help=msg1 + msg2)
    msg1 = 'load IPython session with contents of '
    p_load.add_argument('key', help=msg1 + msg2)
    msg1 = 'plot '
    p_plot.add_argument('key', help=msg1 + msg2)
    msg1 = 'export '
    msg2 = msg2.replace('stations, ', '').replace('_t1', '')
    p_export.add_argument('key', help=msg1 + msg2)
    p_export.add_argument('fname', help='filename')

    for p2 in (p_print, p_load, p_plot, p_export):
        msg = 'seed id (only for data or prepdata, e.g. CX.PATCX..BHZ)'
        p2.add_argument('seedid', help=msg, nargs='?', default=SUPPRESS)
        msg = 'day (only for data or prepdata, e.g. 2010-02-03, 2010-035)'
        p2.add_argument('day', help=msg, nargs='?', default=SUPPRESS)
        msg = 'configuration key for correlation (only for prepdata)'
        p2.add_argument('corrid', help=msg, nargs='?', default=SUPPRESS)
    for p2 in (p_correlate, p_stack, p_stretch):
        msg = 'Set chattiness on command line. Up to 3 -v flags are possible'
        p2.add_argument('-v', '--verbose', help=msg, action='count',
                        default=SUPPRESS)
    for p2 in (p_correlate, p_stack, p_stretch):
        msg = ('Number of cores to use (default: all), '
               'only applies to some commands')
        p2.add_argument('-n', '--njobs', default=None, type=int, help=msg)
    msg = ('Run inner loops parallel instead of outer loop '
           '(preproccessing of different stations and correlation of different '
           'pairs versus processing of different days). '
           'Useful for a datset with many stations.')
    p_correlate.add_argument('--parallel-inner-loop', action='store_true',
                             help=msg)
    msg = 'type of plot (a default is chosen for the given key)'
    choices = ('vs_dist', 'wiggle')
    p_plot.add_argument('--plottype', help=msg, choices=choices)
    msg = 'specify some plot options (dictionary in JSON format)'
    p_plot.add_argument('--plot-options', default=SUPPRESS, type=json.loads,
                        help=msg)
    p_plot.add_argument('--show', action='store_true', default=SUPPRESS,
                        help='show figures')
    p_plot.add_argument('--no-show', dest='show',
                        action='store_false', default=SUPPRESS,
                        help='do not show figures')
    msg = 'format supported by ObsPy (default: auto-detected by extension)'
    p_export.add_argument('-f', '--format', help=msg)

    # Get command line arguments and start run function
    args = vars(p.parse_args(args))
    try:
        run(**args)
    except ParseError as ex:
        p.error(ex)
    except ConfigError as ex:
        print(ex, file=sys.stderr)
        sys.exit(1)
