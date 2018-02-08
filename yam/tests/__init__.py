"""
Tests for the rf package.

yam-runtests [-h] [-v] [-p] [-d] [-f] [-n num]

-h    short help
-v    be verbose
-p    use permanent tempdir
-d    empty permanent tempdir at start
--full   use full tutorial dataset (tests take longer)
-n num   maximal number of cores to use (default: all)
"""

from pkg_resources import resource_filename
import sys
import unittest


def run():
    if '-h' in sys.argv[1:]:
        print(__doc__)
        return
    loader = unittest.TestLoader()
    test_dir = resource_filename('yam', 'tests')
    suite = loader.discover(test_dir)
    runner = unittest.runner.TextTestRunner()
    ret = not runner.run(suite).wasSuccessful()
    sys.exit(ret)
