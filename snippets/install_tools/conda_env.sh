###=========================================================================###
### NEW BUILDS on arcus-b and jalapeno
###=========================================================================###

# jalapeno paths
export DATA=/vols/Data/km/michielk
export CC=/vols/Data/km/michielk/anaconda2/envs/parallel/bin/mpicc
export HDF5_PREFIX=/vols/Data/km/michielk/workspace/hdf5

# arcus paths
export CC=/data/ndcn-fmrib-water-brain/ndcn0180/anaconda2/envs/parallel/bin/mpicc
export HDF5_PREFIX=/data/ndcn-fmrib-water-brain/ndcn0180/workspace/hdf5

# create conda env
PATH=${DATA}/anaconda2/bin:$PATH
CONDA_PATH="$( conda info --root )"
conda remove --name parallel --all
conda info --envs
conda create --name parallel python=2.7 mpi4py unittest2 numpy scipy scikit-image scikit-learn
source activate parallel

# parallel hdf5 build
./configure --enable-parallel --prefix=$HDF5_PREFIX
make
make check
make install
PATH=$HDF5_PREFIX/bin:$PATH

# parallel h5py
cd ~/workspace
git clone https://github.com/h5py/h5py.git
cd h5py
python setup.py configure --mpi --hdf5=$HDF5_PREFIX
python setup.py build
python setup.py install --prefix=$CONDA_PREFIX

# install the wmem package
cd ~/workspace/EM
pip install -e .
