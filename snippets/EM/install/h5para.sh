

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
conda create -n h5para

conda install h5py-parallel=2.6.0 -c spectralDNS
conda install hdf5-parallel=1.8.14 -c spectralDNS
conda install --channel mpi4py mpich mpi4py
pip install nibabel
