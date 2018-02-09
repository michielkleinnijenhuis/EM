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
export PATH=${DATA}/anaconda2/bin:$PATH
export CONDA_PATH="$(conda info --root)"
conda create --name parallel python=2.7
source activate parallel
# alias mpicc=/system/software/arcus-b/lib/mpi/mvapich2/2.1.0/gcc-4.9.2/bin/mpicc

# cd ~/workspace
# git clone https://github.com/mpi4py/mpi4py.git
cd ~/workspace/mpi4py
# rsync -Pazv /Users/michielk/Downloads/mpi4py-2.0.0.tar.gz ndcn0180@arcus-b.arc.ox.ac.uk:/home/ndcn-fmrib-water-brain/ndcn0180/workspace
# cd ~/workspace/mpi4py-2.0.0
python setup.py build # --mpicc=/system/software/arcus-b/lib/mpi/mvapich2/2.1.0/gcc-4.9.2/bin/mpicc
python setup.py install --prefix=/data/ndcn-fmrib-water-brain/ndcn0180/anaconda2/envs/parallel

cd ~/workspace
git clone https://github.com/h5py/h5py.git
cd ~/workspace/h5py
export CC=mpicc
python setup.py configure --mpi --hdf5=/system/software/arcus-b/lib/hdf5/1.8.17/mvapich2-2.1.0__gcc-4.9.2
python setup.py build
python setup.py install --prefix=/data/ndcn-fmrib-water-brain/ndcn0180/anaconda2/envs/parallel






# export CC=mpicc
# export HDF5_DIR=/system/software/arcus-b/lib/hdf5/1.8.17/mvapich2-2.1.0__gcc-4.9.2
# export HDF5_MPI="ON"
# pip install --no-binary=h5py h5py



python

import h5py
from mpi4py import MPI
h5path_file = '/data/ndcn-fmrib-water-brain/ndcn0180/EM/Myrf_01/SET-B/B-NT-S9-2a/B-NT-S9-2a.h5'
h5file = h5py.File(h5path_file, 'r', driver='mpio', comm=MPI.COMM_WORLD)
h5file.close()




python setup.py build --configure
python setup.py install
