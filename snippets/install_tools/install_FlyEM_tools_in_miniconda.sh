# Install miniconda to the prefix of your choice, e.g. /my/miniconda

# LINUX:
wget https://repo.continuum.io/miniconda/Miniconda-latest-Linux-x86_64.sh
bash Miniconda-latest-Linux-x86_64.sh

# MAC:
wget https://repo.continuum.io/miniconda/Miniconda-latest-MacOSX-x86_64.sh
bash Miniconda-latest-MacOSX-x86_64.sh

# Activate conda
CONDA_ROOT=`conda info --root`
source ${CONDA_ROOT}/bin/activate root


conda create -n ilastik-devel -c ilastik ilastik-everything-but-tracking
source activate ilastik-devel
cd ~/workspace/h5py
python setup.py install --record files.txt
cat files.txt | xargs rm -rf
${CONDA_ROOT}/envs/ilastik-devel/run_ilastik.sh --debug
python setup.py install

conda create -n neuroproof-devel2 -c flyem neuroproof
source activate neuroproof-devel
DYLD_FALLBACK_LIBRARY_PATH=/Users/michielk/workspace/miniconda/envs/neuroproof-devel/lib
cd /Users/michielk/workspace/miniconda/envs/neuroproof/  # this only seems to work when in the neuroproof directory
neuroproof_stack_viewer




