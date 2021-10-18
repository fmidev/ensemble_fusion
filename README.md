Ensemble data fusion system FUSE, public distribution

This is a full-featured source code of the FUSE ensemble data fusion system FUSE
Version 1.1 , cross-platform implementation, 10.10.2021

Get the source code

The FUSE system
$ git clone https://github.com/fmidev/ensemble_fusion.git

Supplementary tools from the SILAM model environment
$ git clone https://github.com/fmidev/SILAM_python_3_toolbox.git

A full-fetched 2-days test case
$ git clone https://github.com/fmidev/ensemble_fusion_test.git

FUSE does not require installation, it is a Python-3.8.1 source code, which requires
the Python interpretator and the following libraries:

The first run of the system will require numpy.f2py module installed with appropriate 
FORTRAN compiler in the system: some modules of FUSE are written in dual Python-FORTRAN
manner, where the FORTRAN subroutines need to be compiled into dll and linked to the
Python code. This operation is automatic providing that the f2py is installed and 
configured.

Python-3 packages needed for FUSE to work:
numpy
numpy.f2py
scipy
sklearn
math
pickle
psutil
zipfile
netCDF4
matplotlib
copy, datetime, io, os, shutil, glob

The starting program is run_ensemble.py

Configuration of the run is recorded in the ini file, see example in FUSE_template.ini

A working test set is in FUSE_test.zip. Requires setting up the local path in 
the ini file

