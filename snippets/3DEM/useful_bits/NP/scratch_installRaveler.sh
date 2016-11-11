============
Installation
============


Prerequisites and general notes
-------------------------------
NOTE: We can't provide installation support at this time.  We haven't tested any variations on the installation procedures below.

Raveler is developed and used under Fedora 16 - 20.  While the list of required libraries is long, they should be easy to install.  Wh
enever possible, we choose libraries that are available in the Fedora repository using "yum" or in Python's PyPI index using "easy_ins
tall".  Other libraries are included with Raveler's download.  We have installed Raveler on dozens of computers to date, and we are as
 interested as anyone in minimizing installation effort!

That being said, Raveler's installation process is still manually intensive.  Improvements are scheduled, but they are in the future.

You will likely need administrator privileges to install Raveler.

NOTE: the "superpixel split" tool requires scikit-image, which has more stringent prerequisites.  See below for details.

Fedora packages
---------------

This list is a superset of the packages required to run Raveler.  We have, alas, not removed packages from this list when they are no longer needed for Raveler.  Some of these packages are already available in most typical installations of Fedora (e.g., Python).

We do not list individual version numbers.  If you need them, the versions that are part of the Fedora repository are known to work.  The sole exception is Python; Raveler currently requires Python 2.7.

* python
* python and tkinter
* numpy and its prerequisites
* tk, tcl, tk-devel, tcl-devel
* vtk, vtk-devel
* tkinter, python-imaging, python-imaging-tk, and (maybe?) python-imaging-devel
* hdf5, hdf5-devel
* freeglut

For the superpixel split tool:
* scikit-image and its prerequisites (note: this requires the Fedora 17 repository; you may be able to install it by hand for older versions, but we have not tried it)

Python packages
---------------

* PyOpenGL


Raveler installation
--------------------

* download and extract the tarball
* move the raveler directory to a reasonable location; we use /usr/local, and the rest of these instructions will assume that location
** mv raveler/ /usr/local
* compile libstack and related:
** cd /usr/local/raveler/src
** make

* install Togl; for Fedora:
** mv /usr/local/raveler/Togl /usr/lib64/tcl8.5  (or whatever your version of tcl is)
* edit startup script
** edit /usr/local/raveler/raveler-proof
** LD_LIBRARY_PATH should include /usr/local/raveler/bin
* optional: move startup script to central location, e.g., /usr/local/bin
* startup script should have executable permissions






conda create --name raveler numpy scipy pillow

tkinter x
tk ok
tcl x
tk-devel x
tcl-devel x
vtk ok
vtk-devel x
python-imaging x
python-imaging-tk x
python-imaging-devel x
hdf5 ok
hdf5-devel x
freeglut ok

conda install tk
conda install vtk
conda install hdf5
conda install pyopengl #(gives freeglut)
conda install -c flyem libdvid-cpp
conda install requests

conda create -n libdvid-cpp -c flyem libdvid-cpp
conda create -n CHOOSE_ENV_NAME -c flyem dvid-viewer


TCLLIBPATH=/vols/Data/km/michielk/workspace/raveler/Togl2.0


set auto_path [linsert $auto_path 0 /home/mystuff/someextension/unix]
