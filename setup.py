import os
import setuptools


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setuptools.setup(
    name="bctpy",
    version="0.5.0",
    maintainer="Roan LaPlante",
    maintainer_email="rlaplant@nmr.mgh.harvard.edu",
    description=("Brain Connectivity Toolbox for Python"),
    license="Visuddhimagga Sutta; GPLv3+",
    long_description=read('README.md'),
    datafiles=[('', ['README.md', 'LICENSE'])],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Environment :: X11 Applications",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.4",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    url="https://github.com/aestrivex/bctpy",
    platforms=['any'],
    packages=['bct', 'bct.algorithms', 'bct.utils'],
    install_requires=["numpy", "scipy"]
)
