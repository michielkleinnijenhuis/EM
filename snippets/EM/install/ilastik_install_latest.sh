###########################
### Ilastik in anaconda ###
###########################
# create
## ARC / jalapeno
conda remove --name ilastik-latest --all
# conda create -n ilastik-latest -c ilastik ilastik-everything-but-tracking
conda create -n ilastik-latest ilastik-dependencies-no-solvers -c ilastik-forge -c conda-forge




## local
conda create -n ilastik-latest -c ilastik ilastik-everything-no-solvers
export CPLEX_ROOT_DIR=/Users/michielk/Applications/IBM/ILOG/CPLEX_Studio_Community1262
conda install -n ilastik-latest -c ilastik multi-hypotheses-tracking-with-cplex

## getting the source
CONDA_ROOT=`conda info --root`
DEV_PREFIX=${CONDA_ROOT}/envs/ilastik-latest
conda remove -n ilastik-latest ilastik-meta

# Re-install ilastik-meta.pth
cat > ${DEV_PREFIX}/lib/python2.7/site-packages/ilastik-meta.pth << EOF
../../../ilastik-meta/lazyflow
../../../ilastik-meta/volumina
../../../ilastik-meta/ilastik
EOF

# Option 1: clone a fresh copy of ilastik-meta
git clone http://github.com/ilastik/ilastik-meta ${DEV_PREFIX}/ilastik-meta
cd ${DEV_PREFIX}/ilastik-meta
git submodule update --init --recursive
git submodule foreach "git checkout master"

# Option 2: Symlink to a pre-existing working copy, if you have one.
# cd ${DEV_PREFIX} && ln -s /path/to/ilastik-meta


# run
source activate ilastik-latest
CONDA_ROOT=`conda info --root`
${CONDA_ROOT}/envs/ilastik-latest/run_ilastik.sh --debug

source activate ilastik-devel
CONDA_ROOT=`conda info --root`
${CONDA_ROOT}/envs/ilastik-devel/run_ilastik.sh --debug
