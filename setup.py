import os
import setuptools

def read(fname):
	return open(os.path.join(os.path.dirname(__file__), fname)).read()

setuptools.setup(
	name="bctpy",
	version="0.2",
	maintainer="Roan LaPlante",
	maintainer_email="rlaplant@nmr.mgh.harvard.edu",
	description=("Brain Connectivity Toolbox for Python"),
	license="Visuddhimagga Sutta; GPLv3+",
	long_description=read('README'),
	datafiles=[('',['README','LICENSE'])],
	classifiers=[
		"Development Status :: 3 - Alpha",
		"Environment :: X11 Applications",
		"Intended Audience :: Science/Research",
		"License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
		"Natural Language :: English",
		"Programming Language :: Python :: 2.7",
		"Topic :: Scientific/Engineering :: Information Analysis",
	],
	url="https://github.com/aestrivex/bctpy",
	platforms=['any'],
	packages=['bct'],
	requires=["numpy","scipy"]
)
