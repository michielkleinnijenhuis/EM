export PATH=${DATA}/anaconda2/bin:$PATH
export CONDA_PATH="$(conda info --root)"
# export scriptdir="${HOME}/workspace/EM"
# export PYTHONPATH=$scriptdir
# export PYTHONPATH=$PYTHONPATH:$HOME/workspace/pyDM3reader
# imagej=/system/software/linux-x86_64/fiji/20140602/ImageJ-linux64  # 20170515
# ilastik=$HOME/workspace/ilastik-1.2.2post1-Linux/run_ilastik.sh

# module load mpi4py/1.3.1
# module load hdf5-parallel/1.8.14_mvapich2_gcc
# module load python/2.7__gcc-4.8

# module load hdf5-parallel/1.8.14_mvapich2_gcc
# module load mpi4py

# module load mpich2/1.5.3__gcc

# source deactivate
# conda remove --name parallel --all

module load hdf5-parallel/1.8.17_mvapich2_gcc
h5pcc -showconfig
alias mpicc=/system/software/arcus-b/lib/mpi/mvapich2/2.1.0/gcc-4.9.2/bin/mpicc
export PATH=${DATA}/anaconda2/bin:$PATH
export CONDA_PATH="$(conda info --root)"
conda create --name parallel python=2.7

source activate parallel

# rsync -Pazv /Users/michielk/Downloads/mpi4py-2.0.0.tar.gz ndcn0180@arcus-b.arc.ox.ac.uk:/home/ndcn-fmrib-water-brain/ndcn0180/workspace
# cd ~/workspace/mpi4py-2.0.0
# cd ~/workspace
# git clone https://github.com/mpi4py/mpi4py.git
cd ~/workspace/mpi4py
python setup.py build
python setup.py build --mpicc=/system/software/arcus-b/lib/mpi/mvapich2/2.1.0/gcc-4.9.2/bin/mpicc
python setup.py install --prefix=$CONDA_PREFIX

# conda install -c spectraldns hdf5-parallel
# h5pcc -showconfig

# cd ~/workspace
# git clone https://github.com/h5py/h5py.git
cd ~/workspace/h5py
export CC=mpicc
python setup.py configure --mpi --hdf5=/system/software/arcus-b/lib/hdf5/1.8.17/mvapich2-2.1.0__gcc-4.9.2
python setup.py build
python setup.py install --prefix=$CONDA_PREFIX
# pip install . --prefix=$CONDA_PREFIX




# export CC=mpicc
# export HDF5_DIR=/system/software/arcus-b/lib/hdf5/1.8.17/mvapich2-2.1.0__gcc-4.9.2
# export HDF5_MPI="ON"
# pip install --no-binary=h5py h5py

conda install unittest2

cd $DATA
testfile='test_mpi4py.py'
echo "import os" > $testfile
echo "import h5py" >> $testfile
echo "from mpi4py import MPI" >> $testfile
echo "datadir = '/data/ndcn-fmrib-water-brain/ndcn0180/EM/Myrf_01/SET-B/B-NT-S9-2a'" >> $testfile
echo "h5path = os.path.join(datadir, 'B-NT-S9-2a.h5')" >> $testfile
echo "h5file = h5py.File(h5path, 'r', driver='mpio', comm=MPI.COMM_WORLD)" >> $testfile
echo "h5file.close()" >> $testfile

python test.py


python

import os
import h5py
from mpi4py import MPI
datadir = '/data/ndcn-fmrib-water-brain/ndcn0180/EM/Myrf_01/SET-B/B-NT-S9-2a'
h5path = os.path.join(datadir, 'B-NT-S9-2a.h5')
# h5file = h5py.File(h5path, 'r')
h5file = h5py.File(h5path, 'r', driver='mpio', comm=MPI.COMM_WORLD)
h5file.close()


import os
import h5py
from mpi4py import MPI
datadir = "/Users/michielk/oxdata/P01/EM/Myrf_01/SET-B/B-NT-S10-2f_ROI_00"
h5path = os.path.join(datadir, 'B-NT-S10-2f_ROI_00ds7_labels.h5')
# h5file = h5py.File(h5path, 'r')
h5file = h5py.File(h5path, 'r', driver='mpio', comm=MPI.COMM_WORLD)
h5file_in.close()

# python setup.py build --configure
# python setup.py install
