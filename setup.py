# Copyright 2017-2018 Tom Eulenfeld, GPLv3
import os.path
import re

from setuptools import find_packages, setup


def find_version(*paths):
    fname = os.path.join(os.path.dirname(__file__), *paths)
    with open(fname) as fp:
        code = fp.read()
    match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", code, re.M)
    if match:
        return match.group(1)
    raise RuntimeError("Unable to find version string.")


VERSION = find_version('yam', '__init__.py')
DESCRIPTION = (
    'Yet another monitoring tool using correlations of '
    'ambient noise (seismology)')
LONG_DESCRIPTION = (
    'Please look at the project site for tutorials and information.')

ENTRY_POINTS = {
    'console_scripts': ['yam-runtests = yam.tests:run',
                        'yam = yam.main:run_cmdline']}

REQUIRES = ['h5py', 'matplotlib', 'numpy', 'obspy>=1.1', 'obspyh5>=0.3',
            'scipy>=0.18', 'setuptools', 'tqdm']

CLASSIFIERS = [
    'Environment :: Console',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
    'Operating System :: OS Independent',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Topic :: Scientific/Engineering :: Physics'
    ]


setup(name='yam',
      version=VERSION,
      description=DESCRIPTION,
      long_description=LONG_DESCRIPTION,
      url='https://github.com/trichter/yam',
      author='Tom Eulenfeld',
      author_email='tom.eulenfeld@gmail.de',
      license='GPLv3',
      packages=find_packages(),
      package_dir={'yam': 'yam'},
      install_requires=REQUIRES,
      entry_points=ENTRY_POINTS,
      include_package_data=True,
      zip_safe=False,
      classifiers=CLASSIFIERS
      )
