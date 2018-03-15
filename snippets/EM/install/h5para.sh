

# source activate scikit-image-devel_0.13
# conda install python=2.7
# conda install -c spectralDNS hdf5-parallel=1.8.14
# conda install -c spectralDNS h5py-parallel=2.6.0
# conda install --channel mpi4py mpich mpi4py
# pip install git+git://github.com/scikit-image/scikit-image@master
# pip install nibabel

# conda install -c spectraldns h5py-parallel


export PATH=${DATA}/anaconda2/bin:$PATH
export CONDA_PATH="$(conda info --root)"
conda create -n h5para python=2.7
source activate h5para

conda install h5py-parallel=2.6.0 -c spectralDNS
conda install hdf5-parallel=1.8.14 -c spectralDNS
conda install --channel mpi4py mpich mpi4py
pip install nibabel


# jalapeno test failed
export CONDA_PATH="$(conda info --root)"
conda create -n h5para python=2.7
source activate h5para
conda install h5py-parallel=2.6.0 -c spectralDNS

import os
import h5py
from mpi4py import MPI
datadir = '/data/ndcn-fmrib-water-brain/ndcn0180/EM/Myrf_01/SET-B/B-NT-S9-2a'
h5path = os.path.join(datadir, 'B-NT-S9-2a.h5')
# h5file = h5py.File(h5path, 'r')
h5file = h5py.File(h5path, 'r', driver='mpio', comm=MPI.COMM_WORLD)
h5file.close()


hdf5-parallel/1.8.14_mvapich2
