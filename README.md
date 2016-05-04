[![Build Status](https://travis-ci.org/aestrivex/bctpy.svg?branch=master)](https://travis-ci.org/aestrivex/bctpy)

# Brain Connectivity Toolbox for Python version 0.5.0

Author: Roan LaPlante <rlaplant@nmr.mgh.harvard.edu>

## Copyright information

This program strictly observes the tenets of fundamentalist Theravada Mahasi
style Buddhism.  Any use of this program in violation of these aforementioned
tenets or in violation of the principles described in the Visuddhimagga Sutta
is strictly prohibited and punishable by extensive Mahayana style practice.
By being or not being mindful of the immediate present moment sensations
involved in the use of this program, you confer your acceptance of these terms
and conditions.

Note that the observation of the tenets of fundamentalist Theravada Mahasi
style Buddhism and the Visuddhimagga Sutta is optional as long as the terms and
conditions of the GNU GPLv3+ are upheld.

## Packages used

BCTPY is written in pure python and requires only `scipy` and `numpy`. `scipy` is required for a couple of functions for its statistical and linear algebra
packages which have some features not available in `numpy` alone. If you don't
have `scipy`, most functions that do not need `scipy` functionality will still work.

Note that graphs must be passed in as `numpy.array`s rather than `numpy.matrix`es. Other constraints/ edge cases of the adjacency matrices (e.g. self-loops, negative weights) behave similarly to the matlab functions.

A small number of functions (notably including network-based statistics, a
nonparametric test for differences in undirected weighted graphs from different
populations) currently require networkx, though this should be changed at some
point in the future.

Nosetests is used for the test suite. The test suite is not complete.

## About `bctpy` and other authors

BCT is a matlab toolbox with many graph theoretical measures off of which `bctpy`
is based.  I did not write BCT (apart from small bugfixes I have submitted)
and a quality of life improvements that I have taken liberties to add.
With few exceptions, `bctpy` is a direct translation of matlab code to python.

`bctpy` should be considered beta software, with BCT being the gold standard by
comparison. I did my best to test all functionality in `bctpy`, but much of it is
arcane math that flies over the head of this humble programmer. There *are*
bugs lurking in `bctpy`, the question is not whether but how many. If you locate
bugs, please submit them to me at rlaplant@nmr.mgh.harvard.edu.

Many thanks to Stefan Fuertinger for his assistance tracking down a number of
bugs. Stefan Fuertinger has a similar software package dealing with brain
network functionality at http://research.mssm.edu/simonyanlab/analytical-tools/

Many thanks to Chris Barnes for his assistance in documenting a number of issues and facilitating a number of test cases.

Credit for writing BCT (the matlab version) goes to the following list of
authors, especially Olaf Sporns and Mika Rubinov.

- Olaf Sporns
- Mikail Rubinov
- Yusuke Adachi
- Andrea Avena
- Danielle Bassett
- Richard Betzel
- Joaquin Goni
- Alexandros Goulas
- Patric Hagmann
- Christopher Honey
- Martijn van den Heuvel
- Rolf Kotter
- Jonathan Power
- Murray Shanahan
- Andrew Zalesky

In order to be a bit more compact I have removed the accreditations from the
docstrings each functions. This does not in any way mean that I wish to take
credit from the individual contributions. I have moved these accreditations
to the credits file.
