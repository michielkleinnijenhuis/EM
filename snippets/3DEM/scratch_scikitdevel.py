module load python/3.4__gcc-4.8
conda create --name scikit-image-devel_0.13 python=3.4
source activate scikit-image-devel_0.13
conda install numpy
# conda install mpi4py
conda install --channel mpi4py mpich mpi4py  # https://groups.google.com/forum/#!topic/mpi4py/ULMq-bC1oQA
pip install git+git://github.com/scikit-image/scikit-image@master


module load mpi4py/1.3.1
module load python/2.7__gcc-4.8
conda create --name scikit-image-devel_0.13_p2 python=2.7
source activate scikit-image-devel_0.13_p2
conda install numpy
# conda install mpi4py
conda install --channel mpi4py mpich mpi4py  # https://groups.google.com/forum/#!topic/mpi4py/ULMq-bC1oQA
pip install git+git://github.com/scikit-image/scikit-image@master


conda env remove -n scikit-image-devel_0.13_p2



# remdir="ndcn0180@arcus-b.arc.ox.ac.uk:/data/ndcn-fmrib-water-brain/ndcn0180/EM/M3/M3_S1_GNU/reg_d4_test"
# rsync -avz $remdir/0009.tif ~

remdir="ndcn0180@arcus-b.arc.ox.ac.uk:/data/ndcn-fmrib-water-brain/ndcn0180/EM/M3/M3_S1_GNU"
rsync -avz $remdir/reg_d4/0230.tif ~

remdir="ndcn0180@arcus-b.arc.ox.ac.uk:/data/ndcn-fmrib-water-brain/ndcn0180/EM/M3/M3_S1_GNU_old"
rsync -avz $remdir/reg0250/0000.tif ~
rsync -avz $remdir/reg0250/0250.tif ~
rsync -avz $remdir/reg0250/0459.tif ~

remdir="ndcn0180@arcus-b.arc.ox.ac.uk:/data/ndcn-fmrib-water-brain/ndcn0180/EM/M3/M3_S1_GNU_old"
rsync -avz $remdir/M3_S1_GNU_ds4.h5 ~
