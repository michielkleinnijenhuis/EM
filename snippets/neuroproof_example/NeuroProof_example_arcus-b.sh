# ARCUS
source ~/.bashrc
cd $DATA/EM/NeuroProof
#git clone https://github.com/janelia-flyem/neuroproof_examples.git
#mv neuroprooof_examples examples
scriptdir=/home/ndcn-fmrib-water-brain/ndcn0180/workspace/EM
datadir=$DATA/EM/NeuroProof/examples
cd $datadir

module load mpi4py/1.3.1
module load hdf5-parallel/1.8.14_mvapich2_intel
#module load python/2.7

###############################################################################
# neuroproof graph

qsubfile=$datadir/EM_lc.sh
echo '#!/bin/bash' > $qsubfile
echo "#SBATCH --nodes=1" >> $qsubfile
echo "#SBATCH --ntasks-per-node=1" >> $qsubfile
echo "#SBATCH --time=00:10:00" >> $qsubfile
echo "#SBATCH --job-name=EM_lc" >> $qsubfile
echo "export PATH=$DATA/anaconda2/bin:\$PATH" >> $qsubfile
echo "source activate neuroproof" >> $qsubfile
echo "neuroproof_graph_learn \
$datadir/training_sample2/oversegmented_stack_labels.h5 \
$datadir/training_sample2/boundary_prediction.h5 \
$datadir/training_sample2/groundtruth.h5 \
--classifier-name $datadir/training_sample2/classifier_str2.xml \
--strategy-type 2 --num-iterations 5 --use_mito 0" >> $qsubfile
sbatch -p devel $qsubfile


qsubfile=$datadir/EM_lc.sh
echo '#!/bin/bash' > $qsubfile
echo "#SBATCH --nodes=1" >> $qsubfile
echo "#SBATCH --ntasks-per-node=1" >> $qsubfile
echo "#SBATCH --time=00:10:00" >> $qsubfile
echo "#SBATCH --job-name=EM_lc" >> $qsubfile
echo "export PATH=$DATA/anaconda2/bin:\$PATH" >> $qsubfile
echo "source activate neuroproof" >> $qsubfile
echo "neuroproof_graph_predict \
$datadir/validation_sample/supervoxels.h5 \
$datadir/validation_sample/boundary_prediction.h5 \
$datadir/training_sample2/classifier_str1.xml \
--output-file $datadir/validation_sample/segmentation.h5 \
--graph-file $datadir/validation_sample/graph.json" >> $qsubfile
sbatch -p devel $qsubfile





# LOCAL
cd ~/oxdata/P01/EM
git clone https://github.com/janelia-flyem/neuroproof_examples.git
datadir=~/oxdata/P01/EM/neuroproof_examples

gunzip $datadir/*/*.gz
source activate neuroproof
cd /Users/michielk/anaconda/envs/neuroproof

neuroproof_graph_learn \
$datadir/training_sample2/oversegmented_stack_labels.h5 \
$datadir/training_sample2/boundary_prediction.h5 \
$datadir/training_sample2/groundtruth.h5 \
--classifier-name $datadir/training_sample2/classifier_str2.xml \
--strategy-type 2 --num-iterations 5 --use_mito 0

# debugging with gdb
gdb ../bin/neuroproof_graph_learn
set args /Users/michielk/oxdata/P01/EM/neuroproof_examples/training_sample2/oversegmented_stack_labels.h5 /Users/michielk/oxdata/P01/EM/neuroproof_examples/training_sample2/boundary_prediction.h5 /Users/michielk/oxdata/P01/EM/neuroproof_examples/training_sample2/groundtruth.h5 --classifier-name /Users/michielk/oxdata/P01/EM/neuroproof_examples/training_sample2/classifier_str2.xml --strategy-type 2 --num-iterations 5 --use_mito 0
#set args /data/ndcn-fmrib-water-brain/ndcn0180/EM/NeuroProof/examples/training_sample2/oversegmented_stack_labels.h5 /data/ndcn-fmrib-water-brain/ndcn0180/EM/NeuroProof/examples/training_sample2/boundary_prediction.h5 /data/ndcn-fmrib-water-brain/ndcn0180/EM/NeuroProof/examples/training_sample2/groundtruth.h5 --classifier-name /data/ndcn-fmrib-water-brain/ndcn0180/EM/NeuroProof/examples/training_sample2/classifier_str2.xml --strategy-type 2 --num-iterations 5 --use_mito 0
break RagUtils.cpp:54
run
