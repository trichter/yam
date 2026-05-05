"""
Tests for the yam package.

yam-runtests [-h] [-v] [-p] [-d] [--full] [-n num]

-h    short help
-v    be verbose
-p    use permanent tempdir
-d    empty permanent tempdir at start
--full   use full tutorial dataset (tests take longer)
-n num   maximal number of cores to use (default: all)
"""

from importlib import resources
import sys
import unittest


def run():
    if '-h' in sys.argv[1:]:
        print(__doc__)
        return
    loader = unittest.TestLoader()
    test_pkg = resources.files('yam') / 'tests'
    with resources.as_file(test_pkg) as test_dir:
        suite = loader.discover(str(test_dir))
    runner = unittest.runner.TextTestRunner()
    ret = not runner.run(suite).wasSuccessful()
    sys.exit(ret)
