###==================###
### Anaconda2 on ARC ###
###==================###
wget https://3230d63b5fc54e62148e-c95ac804525aac4b6dba79b00b39d1d3.ssl.cf1.rackcdn.com/Anaconda2-2.5.0-Linux-x86_64.sh
# chosen prefix: /data/ndcn-fmrib-water-brain/ndcn0180/anaconda2

###========================###
### Neuroproof in anaconda ### old, 
###========================###
# create
conda create -n neuroproof -c flyem neuroproof
source activate neuroproof
cd /Users/michielk/anaconda/envs/neuroproof/  # locally, neuroproof only seems to work when in the neuroproof directory (otherwise it can find opencv and hdf5 libs)!!
# run
source activate neuroproof
neuroproof_stack_viewer

###========================###
### Neuroproof in anaconda ### testing the segfault bug #4: https://github.com/janelia-flyem/NeuroProof/issues/4
###========================###
conda create -n neuroproof-test -c flyem neuroproof
source activate neuroproof-test
PREFIX=$(conda info --root)/envs/neuroproof-test
[ `uname` == 'Darwin' ] && 
export DYLD_FALLBACK_LIBRARY_PATH=${PREFIX}/lib || 
export LD_LIBRARY_PATH=${PREFIX}/lib

module load cmake/2.8.12
cd ~/workspace
git clone https://github.com/janelia-flyem/neuroproof
cd neuroproof
./configure-for-conda.sh ${PREFIX}
cd build
make -j4
make install
make test

# ARC:
#CMakeLists.txt
    set(Boost_NO_BOOST_CMAKE ON)
#configure_for_conda.sh
    cat $HOME/workspace/neuroproof/debug.sh | bash -x -e -s - --configure-only
#build.sh
        -DCMAKE_OSX_DEPLOYMENT_TARGET:STRING="" \
        -DCMAKE_OSX_SYSROOT:STRING=/ \
        -DCMAKE_BUILD_TYPE=Debug \


########################
### Gala in anaconda ###
########################
# create
conda create --name gala numpy scipy pillow networkx hdf5 h5py cython scikit-learn matplotlib scikit-image progressbar
source activate gala
pip install viridis
pip install git+git://github.com/janelia-flyem/gala@master
# run
source activate gala
gala-segmentation-pipeline


###########################
### Ilastik in anaconda ### local
###########################
# create
chmod +x COSCE1262MACOS.bin
./COSCE1262MACOS.bin
CPLEX_ROOT_DIR=/Users/michielk/Applications/IBM/ILOG/CPLEX_Studio_Community1262
conda create -n throw-away -c ilastik cplex-shared
conda remove --all -n throw-away
conda create -n ilastik-devel -c ilastik ilastik-everything
# run
source activate ilastik-devel
CONDA_ROOT=`conda info --root`
${CONDA_ROOT}/envs/ilastik-devel/run_ilastik.sh --debug

# 20160713 update
CPLEX_ROOT_DIR=/Users/michielk/Applications/IBM/ILOG/CPLEX_Studio_Community1262
conda create -n throw-away -c ilastik cplex-shared
conda remove --all -n throw-away
conda create -n ilastik-devel_20160714 -c ilastik ilastik-everything
source activate ilastik-devel_20160714
CONDA_ROOT=`conda info --root`
${CONDA_ROOT}/envs/ilastik-devel/run_ilastik.sh --debug

# version info
# ./ilastik-1.1.7-OSX.app/Contents/ilastik-release/bin/python -c "import ilastik; print ilastik.__version__"
CONDA_ROOT=`conda info --root`
${CONDA_ROOT}/envs/ilastik-devel/bin/python -c "import ilastik; print ilastik.__version__"


###########################
### Ilastik in anaconda ### ARC
###########################
# create
conda create -n ilastik-devel -c ilastik ilastik-everything-but-tracking
# run
source activate ilastik-devel
CONDA_ROOT=`conda info --root`
${CONDA_ROOT}/envs/ilastik-devel/run_ilastik.sh --debug


##########################
### Rhoana in anaconda ### conda env remove -n rhoana
##########################
conda create --name rhoana numpy scipy h5py opencv
source activate rhoana
pip install mahotas
conda install cython=0.19.1
pip install git+git://github.com/Rhoana/pymaxflow@master  # pip install git+https://github.com/Rhoana/pymaxflow.git#egg=pymaxflow
pip install git+git://github.com/Rhoana/fast64counter@master
# (rhoana)ws126:neuroproof michielk$ pip install git+git://github.com/Rhoana/fast64counter@master
# Collecting git+git://github.com/Rhoana/fast64counter@master
#   Cloning git://github.com/Rhoana/fast64counter (to master) to /var/folders/1g/05trnwf963g8rbf8gbtbnc000000gq/T/pip-4aEKnO-build
# Installing collected packages: UNKNOWN
#   Running setup.py install for UNKNOWN
# Successfully installed UNKNOWN-0.0.0


RDIR=~/anaconda/envs/rhoana/lib/python2.7/site-packages
cd $RDIR
git clone https://github.com/Rhoana/rhoana.git
cd $RDIR/rhoana
###========================================###
### adapt Makefile to find OPENCV and HDF5 ###
###========================================###
atom $RDIR/rhoana/ClassifyMembranes/Makefile
# CFLAGS=-I$(OPENCV)/include -I$(HDF5_DIR)/include -g -O3
# CXXFLAGS=-I$(OPENCV)/include -I$(HDF5_DIR)/include -g -O3
# LDFLAGS=-L$(OPENCV)/lib -lopencv_highgui -lopencv_imgproc -lopencv_core -L$(HDF5_DIR)/lib -lhdf5_cpp -lhdf5
CFLAGS=-I/Users/michielk/anaconda/envs/rhoana/include/opencv2 -I/Users/michielk/anaconda/envs/rhoana/include -g -O3
CXXFLAGS=-I/Users/michielk/anaconda/envs/rhoana/include/opencv2 -I/Users/michielk/anaconda/envs/rhoana/include -g -O3
LDFLAGS=-L/Users/michielk/anaconda/envs/rhoana/lib -lopencv_highgui -lopencv_imgproc -lopencv_core -lhdf5_cpp -lhdf5

cd $RDIR/rhoana/ClassifyMembranes; make  # throws cv namespace errors
cd $RDIR/rhoana/Relabeling; python setup.py install
cd $RDIR/rhoana/Renderer; python setup.py install  # for Windows :(




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
#./dojo.py /PATH/TO/MOJO/FOLDER/WITH/IDS/AND/IMAGES/SUBFOLDERS




cd ~/workspace/h5py
python setup.py install --record files.txt && cat files.txt | xargs rm -rf
python setup.py install

###============================================###
### scikit-image development version 0.12-dev0 ###  to circumvent segfault when running slic with enforce_connectivity=True
###============================================###
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
