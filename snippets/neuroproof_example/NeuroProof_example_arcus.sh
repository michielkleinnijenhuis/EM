# ARCUS
scriptdir=/home/ndcn-fmrib-water-brain/ndcn0180/workspace/EM
datadir=$DATA/EM/NeuroProof/examples


###############################################################################
module load python/2.7 mpi4py/1.3.1 
module load hdf5-parallel/1.8.14_mvapich2

qsubfile=$datadir/submit.sh
echo '#!/bin/bash' > $qsubfile
echo "#PBS -l nodes=1:ppn=1" >> $qsubfile
echo "#PBS -l walltime=00:10:00" >> $qsubfile
echo "#PBS -N em_s2s" >> $qsubfile
echo "#PBS -V" >> $qsubfile
echo "cd \$PBS_O_WORKDIR" >> $qsubfile
echo "python $scriptdir/EM_series2stack.py \
$datadir/training_sample1/grayscale_maps \
$datadir/training_sample1/grayscale_maps.h5 \
-f 'stack' -m -o -r '*.png' -n 5" >> $qsubfile
echo "python $scriptdir/EM_series2stack.py \
$datadir/training_sample2/grayscale_maps \
$datadir/training_sample2/grayscale_maps.h5 \
-f 'stack' -m -o -r '*.png' -n 5" >> $qsubfile
echo "python $scriptdir/EM_series2stack.py \
$datadir/validation_sample/grayscale_maps \
$datadir/validation_sample/grayscale_maps.h5 \
-f 'stack' -m -o -r '*.png' -n 5" >> $qsubfile
qsub -q develq $qsubfile

rsync -avz ndcn0180@arcus.oerc.ox.ac.uk:/data/ndcn-fmrib-water-brain/ndcn0180/EM/NeuroProof/examples/training_sample1/grayscale_maps.h5 /Users/michielk/


###############################################################################

module unload python/2.7 mpi4py/1.3.1 
module unload hdf5-parallel/1.8.14_mvapich2

mv $datadir/training_sample2/boundary_prediction.h5 $datadir/training_sample2/boundary_prediction.h5.orig

echo "export PATH=/home/ndcn-fmrib-water-brain/ndcn0180/miniconda/bin:\$PATH" > $datadir/test.sh
echo "source activate ilastik-devel" >> $datadir/test.sh
echo "CONDA_ROOT=`conda info --root`" >> $datadir/test.sh
echo "${CONDA_ROOT}/envs/ilastik-devel/run_ilastik.sh --headless \
--preconvert_stacks \
--project=$datadir/training_sample1/results/boundary_classifier_ilastik.ilp \
--output_axis_order=xyzc \
--output_format=hdf5 \
--output_filename_format=$datadir/training_sample1/boundary_prediction.h5 \
--output_internal_path=/volume/predictions \
$datadir/training_sample1/grayscale_maps.h5/stack" >> $datadir/test.sh
echo "${CONDA_ROOT}/envs/ilastik-devel/run_ilastik.sh --headless \
--preconvert_stacks \
--project=$datadir/training_sample1/results/boundary_classifier_ilastik.ilp \
--output_axis_order=xyzc \
--output_format=hdf5 \
--output_filename_format=$datadir/training_sample2/boundary_prediction.h5 \
--output_internal_path=/volume/predictions \
$datadir/training_sample2/grayscale_maps.h5/stack" >> $datadir/test.sh
echo "${CONDA_ROOT}/envs/ilastik-devel/run_ilastik.sh --headless \
--preconvert_stacks \
--project=$datadir/training_sample1/results/boundary_classifier_ilastik.ilp \
--output_axis_order=xyzc \
--output_format=hdf5 \
--output_filename_format=$datadir/validation_sample/boundary_prediction.h5 \
--output_internal_path=/volume/predictions \
$datadir/validation_sample/grayscale_maps.h5/stack" >> $datadir/test.sh
chmod u+x $datadir/test.sh
echo '#!/bin/bash' > $datadir/submit.sh
echo "#PBS -l nodes=1:ppn=1" >> $datadir/submit.sh
echo "#PBS -l walltime=00:10:00" >> $datadir/submit.sh
echo "#PBS -N em_lc" >> $datadir/submit.sh
echo "#PBS -V" >> $datadir/submit.sh
echo "cd \$PBS_O_WORKDIR" >> $datadir/submit.sh
echo "$datadir/test.sh" >> $datadir/submit.sh
qsub -q develq $datadir/submit.sh


###############################################################################

module load python/2.7 mpi4py/1.3.1 
module load hdf5-parallel/1.8.14_mvapich2

cp $datadir/training_sample2/grayscale_maps.h5 $datadir/training_sample2/supervoxels.h5
cp $datadir/validation_sample/grayscale_maps.h5 $datadir/validation_sample/supervoxels.h5

echo '#!/bin/bash' > $datadir/submit.sh
echo "#PBS -l nodes=1:ppn=1" >> $datadir/submit.sh
echo "#PBS -l walltime=00:10:00" >> $datadir/submit.sh
echo "#PBS -N em_lc" >> $datadir/submit.sh
echo "#PBS -V" >> $datadir/submit.sh
echo "cd \$PBS_O_WORKDIR" >> $datadir/submit.sh
echo "/home/ndcn-fmrib-water-brain/ndcn0180/.local/bin/gala-segmentation-pipeline \
--image-stack $datadir/training_sample2/grayscale_maps.h5 \
--ilp-file $datadir/training_sample1/results/boundary_classifier_ilastik.ilp \
--disable-gen-pixel --pixelprob-file $datadir/training_sample2/boundary_prediction.h5 \
--enable-gen-supervoxels \
--disable-gen-agglomeration \
--seed-size 5 \
--enable-raveler-output \
--enable-h5-output $datadir/training_sample2 \
--segmentation-thresholds 0.0" >> $datadir/submit.sh
echo "/home/ndcn-fmrib-water-brain/ndcn0180/.local/bin/gala-segmentation-pipeline \
--image-stack $datadir/validation_sample/grayscale_maps.h5 \
--ilp-file $datadir/training_sample1/results/boundary_classifier_ilastik.ilp \
--disable-gen-pixel --pixelprob-file $datadir/validation_sample/boundary_prediction.h5 \
--enable-gen-supervoxels \
--disable-gen-agglomeration \
--seed-size 5 \
--enable-raveler-output \
--enable-h5-output $datadir/validation_sample \
--segmentation-thresholds 0.0" >> $datadir/submit.sh
qsub -q develq $datadir/submit.sh


###############################################################################

module unload python/2.7 mpi4py/1.3.1 
module unload hdf5-parallel/1.8.14_mvapich2

echo "export PATH=/home/ndcn-fmrib-water-brain/ndcn0180/miniconda/bin:\$PATH" > $datadir/test.sh
echo "source activate neuroproof-devel" >> $datadir/test.sh
echo "neuroproof_graph_learn \
$datadir/training_sample2/oversegmented_stack_labels.h5 \
$datadir/training_sample2/boundary_prediction.h5 \
$datadir/training_sample2/groundtruth.h5 \
--classifier-name $datadir/training_sample2/classifier_str1.xml \
--strategy-type 1" >> $datadir/test.sh
chmod u+x $datadir/test.sh
echo '#!/bin/bash' > $datadir/submit.sh
echo "#PBS -l nodes=1:ppn=1" >> $datadir/submit.sh
echo "#PBS -l walltime=00:10:00" >> $datadir/submit.sh
echo "#PBS -N em_lc" >> $datadir/submit.sh
echo "#PBS -V" >> $datadir/submit.sh
echo "cd \$PBS_O_WORKDIR" >> $datadir/submit.sh
echo "$datadir/test.sh" >> $datadir/submit.sh
qsub -q develq $datadir/submit.sh


echo "export PATH=/home/ndcn-fmrib-water-brain/ndcn0180/miniconda/bin:\$PATH" > $datadir/test.sh
echo "source activate neuroproof-devel" >> $datadir/test.sh
echo "neuroproof_graph_predict \
$datadir/validation_sample/supervoxels.h5 \
$datadir/validation_sample/boundary_prediction.h5 \
$datadir/training_sample2/classifier_str1.xml \
--output-file $datadir/validation_sample/segmentation.h5 \
--graph-file $datadir/validation_sample/graph.json" >> $datadir/test.sh
chmod u+x $datadir/test.sh
echo '#!/bin/bash' > $datadir/submit.sh
echo "#PBS -l nodes=1:ppn=1" >> $datadir/submit.sh
echo "#PBS -l walltime=00:10:00" >> $datadir/submit.sh
echo "#PBS -N em_lc" >> $datadir/submit.sh
echo "#PBS -V" >> $datadir/submit.sh
echo "cd \$PBS_O_WORKDIR" >> $datadir/submit.sh
echo "$datadir/test.sh" >> $datadir/submit.sh
qsub -q develq $datadir/submit.sh






python /Users/michielk/workspace/EMseg/EM_slicvoxels0.py \
-i $datadir/training_sample2/grayscale_maps.h5 \
-o $datadir/training_sample2/slicvoxels.h5 \
-f 'stack' -g 'stack' -s 500

source activate neuroproof-devel
neuroproof_graph_learn \
$datadir/training_sample2/slicvoxels.h5 \
$datadir/training_sample2/boundary_prediction.h5 \
$datadir/training_sample2/groundtruth.h5 \
--classifier-name $datadir/training_sample2/classifier_str1_slic.xml \
--strategy-type 1



# STAGE3

python /Users/michielk/workspace/EMseg/EM_slicvoxels0.py \
-i $datadir/validation_sample/grayscale_maps.h5 \
-o $datadir/validation_sample/slicvoxels.h5 \
-f 'stack' -g 'stack' -s 500

source activate neuroproof-devel
neuroproof_graph_predict \
$datadir/validation_sample/slicvoxels.h5 \
$datadir/validation_sample/boundary_prediction.h5 \
$datadir/training_sample2/classifier_str1_slic.xml \
--output-file $datadir/validation_sample/segmentation_slic.h5 \
--graph-file $datadir/validation_sample/graph_slic.json


source activate neuroproof-devel
neuroproof_stack_viewer
