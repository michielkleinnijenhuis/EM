########################
### DOJO in anaconda ###
########################
cd ~/workspace
git clone https://github.com/Rhoana/dojo.git
brew install hdf5 libjpeg libpng libtiff webp
conda create --name dojo pip tornado
source activate dojo
ARCHFLAGS=-Wno-error=unused-command-line-argument-hard-error-in-future pip install h5py
ARCHFLAGS=-Wno-error=unused-command-line-argument-hard-error-in-future pip install PIL --allow-unverified PIL --allow-external PIL
ARCHFLAGS=-Wno-error=unused-command-line-argument-hard-error-in-future pip install mahotas
ARCHFLAGS=-Wno-error=unused-command-line-argument-hard-error-in-future pip install cython
ARCHFLAGS=-Wno-error=unused-command-line-argument-hard-error-in-future pip install imread
conda install scipy
ARCHFLAGS=-Wno-error=unused-command-line-argument-hard-error-in-future pip install scikit-image
ARCHFLAGS=-Wno-error=unused-command-line-argument-hard-error-in-future pip install lxml

source activate dojo
cd ~/workspace/dojo
./dojo.py


##############################
### Neuroproof in anaconda ###
##############################
conda create -n neuroproof -c flyem neuroproof
#DYLD_FALLBACK_LIBRARY_PATH=/Users/michielk/anaconda/envs/neuroproof/lib

source activate neuroproof
cd /Users/michielk/anaconda/envs/neuroproof/  # neuroproof only seems to work when in the neuroproof directory!!
neuroproof_stack_viewer


###########################
### Ilastik in anaconda ###
###########################
chmod +x COSCE1262MACOS.bin
./COSCE1262MACOS.bin
CPLEX_ROOT_DIR=/Users/michielk/Applications/IBM/ILOG/CPLEX_Studio_Community1262 conda create -n throw-away -c ilastik cplex-shared
conda remove --all -n throw-away
conda create -n ilastik-devel -c ilastik ilastik-everything

source activate ilastik-devel
CONDA_ROOT=`conda info --root`
${CONDA_ROOT}/envs/ilastik-devel/run_ilastik.sh --debug


########################
### Gala in anaconda ###
########################
conda create --name gala numpy scipy pillow networkx hdf5 h5py cython scikit-learn matplotlib scikit-image progressbar
pip install viridis
# pip install gala
pip install git+git://github.com/janelia-flyem/gala@master

source activate gala
gala-segmentation-pipeline


##########################
### Rhoana in anaconda ###
##########################
conda create --name rhoana numpy scipy h5py opencv
source activate rhoana
pip install mahotas
conda install cython
pip install git+git://github.com/Rhoana/pymaxflow@master
pip install git+git://github.com/Rhoana/fast64counter@master
# (rhoana)ws126:neuroproof michielk$ pip install git+git://github.com/Rhoana/fast64counter@master
# Collecting git+git://github.com/Rhoana/fast64counter@master
#   Cloning git://github.com/Rhoana/fast64counter (to master) to /var/folders/1g/05trnwf963g8rbf8gbtbnc000000gq/T/pip-4aEKnO-build
# Installing collected packages: UNKNOWN
#   Running setup.py install for UNKNOWN
# Successfully installed UNKNOWN-0.0.0






cd ~/workspace/h5py
python setup.py install --record files.txt && cat files.txt | xargs rm -rf
python setup.py install


### development version 0.12-dev0 of scikit-image to circumvent segfault when running slic with enforce_connectivity=True ###
# local
conda create --name scikit-image-devel_p35 python=3.5
source activate scikit-image-devel_p35
conda install numpy
pip install git+git://github.com/scikit-image/scikit-image@master

# ARC
module load python/3.4__gcc-4.8
conda create --name scikit-image-devel_p34 python=3.4 numpy scipy h5py matplotlib six networkx pillow dask
source activate scikit-image-devel_p34
pip install git+git://github.com/scikit-image/scikit-image@master
# NOTE: start from scratch or unload module first before running slic
