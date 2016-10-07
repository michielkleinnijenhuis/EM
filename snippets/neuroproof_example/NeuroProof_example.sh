




datadir=/home/ndcn-fmrib-water-brain/ndcn0180/workspace/testNeuroProof
datadir=/vols/Data/km/michielk/workspace/NeuroProof/examples/training_sample2
mkdir -p $datadir && cd $datadir

echo "export PATH=/home/ndcn-fmrib-water-brain/ndcn0180/miniconda/bin:\$PATH" > $datadir/test.sh
# echo "export PATH=/vols/Data/km/michielk/workspace/miniconda/bin:\$PATH" > $datadir/test.sh
echo "source activate neuroproof-devel" >> $datadir/test.sh
echo "neuroproof_graph_learn -h > $datadir/helpmessage_\$1.txt" >> $datadir/test.sh
chmod u+x $datadir/test.sh
./test.sh login

echo '#!/bin/bash' > $datadir/submit.sh
echo "#PBS -l nodes=1:ppn=1" >> $datadir/submit.sh
echo "#PBS -l walltime=00:05:00" >> $datadir/submit.sh
echo "#PBS -N em_lc" >> $datadir/submit.sh
echo "#PBS -V" >> $datadir/submit.sh
echo "cd \$PBS_O_WORKDIR" >> $datadir/submit.sh
echo "$datadir/test.sh arcus" >> $datadir/submit.sh
qsub -q develq $datadir/submit.sh

fsl_sub -q veryshort.q $datadir/test.sh jalapeno


# JALAPENO
echo "export PATH=/vols/Data/km/michielk/workspace/miniconda/bin:\$PATH" > $datadir/test.sh
echo "source activate neuroproof-devel" >> $datadir/test.sh
echo "neuroproof_graph_learn \
$datadir/oversegmented_stack_labels.h5 \
$datadir/boundary_prediction.h5 \
$datadir/groundtruth.h5 \
--strategy-type 1" >> $datadir/test.sh
chmod u+x $datadir/test.sh
fsl_sub -q long.q $datadir/test.sh







# LOCAL
datadir=/Users/michielk/workspace/FlyEM/NeuroProof/examples



# STAGE1
python /Users/michielk/workspace/EM/convert/EM_series2stack.py \
$datadir/training_sample1/grayscale_maps \
$datadir/training_sample1/grayscale_maps.h5 \
-f 'stack' -o -r '*.png'

#cd ~/workspace/h5py
#python setup.py install --record files.txt
#cat files.txt | xargs rm -rf
source activate ilastik-devel
CONDA_ROOT=`conda info --root`
${CONDA_ROOT}/envs/ilastik-devel/run_ilastik.sh --headless \
--preconvert_stacks \
--project=$datadir/training_sample1/results/boundary_classifier_ilastik.ilp \
--output_axis_order=zyxc \
--output_format=hdf5 \
--output_filename_format=$datadir/training_sample1/boundary_prediction.h5 \
--output_internal_path=/volume/predictions \
"$datadir/training_sample1/grayscale_maps.h5/stack"
#python setup.py install



# STAGE2
python /Users/michielk/workspace/EM/convert/EM_series2stack.py \
$datadir/training_sample2/grayscale_maps \
$datadir/training_sample2/grayscale_maps.h5 \
-f 'stack' -o -r '*.png'

#cd ~/workspace/h5py
#python setup.py install --record files.txt
#cat files.txt | xargs rm -rf
source activate ilastik-devel
CONDA_ROOT=`conda info --root`
${CONDA_ROOT}/envs/ilastik-devel/run_ilastik.sh --headless \
--preconvert_stacks \
--project=$datadir/training_sample1/results/boundary_classifier_ilastik.ilp \
--output_axis_order=zyxc \
--output_format=hdf5 \
--output_filename_format=$datadir/training_sample2/boundary_prediction.h5 \
--output_internal_path=/volume/predictions \
"$datadir/training_sample2/grayscale_maps.h5/stack"
#python setup.py install

#cp $datadir/training_sample2/grayscale_maps.h5 $datadir/training_sample2/supervoxels.h5
source activate gala_20160715
gala-segmentation-pipeline \
--image-stack $datadir/training_sample2/grayscale_maps.h5 \
--ilp-file $datadir/training_sample1/results/boundary_classifier_ilastik.ilp \
--disable-gen-pixel --pixelprob-file $datadir/training_sample2/boundary_prediction.h5 \
--enable-gen-supervoxels \
--disable-gen-agglomeration \
--seed-size 5 \
--enable-raveler-output \
--enable-h5-output $datadir/training_sample2 \
--segmentation-thresholds 0.0
# $datadir/training_sample2/supervoxels.h5 has to exist ?!

gunzip $datadir/training_sample2/groundtruth.h5.gz
source activate neuroproof-20160912
neuroproof_graph_learn \
$datadir/training_sample2/supervoxels.h5 \
$datadir/training_sample2/boundary_prediction.h5 \
$datadir/training_sample2/groundtruth.h5 \
--classifier-name $datadir/training_sample2/classifier_str1.xml \
--strategy-type 1

python /Users/michielk/workspace/EM/supervoxels/EM_slicvoxels.py \
$datadir/training_sample2/grayscale_maps.h5 \
$datadir/training_sample2/slicvoxels.h5 \
-f 'stack' -g 'stack' -s 500 -e 0.05 0.0073 0.0073

source activate neuroproof-20160912
neuroproof_graph_learn \
$datadir/training_sample2/slicvoxels.h5 \
$datadir/training_sample2/boundary_prediction.h5 \
$datadir/training_sample2/groundtruth.h5 \
--classifier-name $datadir/training_sample2/classifier_str1_slic.xml \
--strategy-type 1



# STAGE3
python /Users/michielk/workspace/EM/convert/EM_series2stack.py \
$datadir/validation_sample/grayscale_maps \
$datadir/validation_sample/grayscale_maps.h5 \
-f 'stack' -o -r '*.png'

#cd ~/workspace/h5py
#python setup.py install --record files.txt
#cat files.txt | xargs rm -rf
source activate ilastik-devel
CONDA_ROOT=`conda info --root`
${CONDA_ROOT}/envs/ilastik-devel/run_ilastik.sh --headless \
--preconvert_stacks \
--project=$datadir/training_sample1/results/boundary_classifier_ilastik.ilp \
--output_axis_order=xyzc \
--output_format=hdf5 \
--output_filename_format=$datadir/validation_sample/boundary_prediction.h5 \
--output_internal_path=/volume/predictions \
"$datadir/validation_sample/grayscale_maps.h5/stack"
#python setup.py install

#cp $datadir/validation_sample/grayscale_maps.h5 $datadir/validation_sample/supervoxels.h5
source activate gala_20160715
gala-segmentation-pipeline \
--image-stack $datadir/validation_sample/grayscale_maps.h5 \
--ilp-file $datadir/training_sample1/results/boundary_classifier_ilastik.ilp \
--disable-gen-pixel --pixelprob-file $datadir/validation_sample/boundary_prediction.h5 \
--enable-gen-supervoxels \
--disable-gen-agglomeration \
--seed-size 5 \
--enable-raveler-output \
--enable-h5-output $datadir/validation_sample \
--segmentation-thresholds 0.0
# $datadir/validation_sample/supervoxels.h5 has to exist ?!

source activate neuroproof-20160912
neuroproof_graph_predict \
$datadir/validation_sample/supervoxels.h5 \
$datadir/validation_sample/boundary_prediction.h5 \
$datadir/training_sample2/classifier_str1.xml \
--output-file $datadir/validation_sample/segmentation.h5 \
--graph-file $datadir/validation_sample/graph.json

python /Users/michielk/workspace/EM/supervoxels/EM_slicvoxels.py \
$datadir/validation_sample/grayscale_maps.h5 \
$datadir/validation_sample/slicvoxels.h5 \
-f 'stack' -g 'stack' -s 500 -e 0.05 0.0073 0.0073

source activate neuroproof-20160912
neuroproof_graph_predict \
$datadir/validation_sample/slicvoxels.h5 \
$datadir/validation_sample/boundary_prediction.h5 \
$datadir/training_sample2/classifier_str1_slic.xml \
--output-file $datadir/validation_sample/segmentation_slic.h5 \
--graph-file $datadir/validation_sample/graph_slic.json


source activate neuroproof-20160912
neuroproof_stack_viewer
