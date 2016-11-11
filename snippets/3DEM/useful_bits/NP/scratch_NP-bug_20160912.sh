conda create -n neuroproof-20160912 -c flyem neuroproof
source activate neuroproof-20160912
cd /Users/michielk/anaconda/envs/neuroproof-20160912/  # locally, neuroproof only seems to work when in the neuroproof directory (otherwise it can find opencv and hdf5 libs)!!
# run
source activate neuroproof-20160912
neuroproof_stack_viewer





conda env remove -n neuroproof-20160912
conda create -n neuroproof-20160912 -c flyem neuroproof
source activate neuroproof-20160912

PREFIX=$(conda info --root)/envs/neuroproof-20160912
#export LD_LIBRARY_PATH=${PREFIX}/lib
export DYLD_FALLBACK_LIBRARY_PATH=${PREFIX}/lib

conda remove neuroproof

cd ~/workspace/neuroproof
./configure-for-conda.sh ${PREFIX}

cd build
#ccmake -DCMAKE_BUILD_TYPE=Debug ..
#ccmake -G"Eclipse CDT4 - Unix Makefiles" -DCMAKE_BUILD_TYPE=Debug ..
make -j8
make install



cd ~/workspace/neuroproof/build
make -j8
make install

PREFIX=$(conda info --root)/envs/neuroproof-20160912
export DYLD_FALLBACK_LIBRARY_PATH=${PREFIX}/lib
datadir=/Users/michielk/workspace/FlyEM/NeuroProof/examples
cd /Users/michielk/anaconda/envs/neuroproof-20160912/
source activate neuroproof-20160912
neuroproof_graph_learn

neuroproof_graph_learn \
$datadir/training_sample2/supervoxels.h5 \
$datadir/training_sample2/boundary_prediction.h5 \
$datadir/training_sample2/groundtruth.h5 \
--classifier-name $datadir/training_sample2/classifier_str2.xml \
--strategy-type 2

neuroproof_graph_learn \
$datadir/training_sample2/supervoxels.h5 \
$datadir/training_sample2/boundary_prediction.h5 \
$datadir/training_sample2/groundtruth.h5 \
--classifier-name $datadir/training_sample2/classifier_str3.xml \
--strategy-type 3

make -j8
make install
neuroproof_graph_learn \
$datadir/training_sample2/supervoxels.h5 \
$datadir/training_sample2/boundary_prediction.h5 \
$datadir/training_sample2/groundtruth.h5 \
--classifier-name $datadir/training_sample2/classifier_str2_niter2.xml \
--strategy-type 2 --num-iterations 2 --use_mito 0


export DYLD_FALLBACK_LIBRARY_PATH=/Users/michielk/anaconda/envs/neuroproof-20160912/lib
mkdir -p /Users/michielk/anaconda/envs/neuroproof-20160912/examples/training_sample2/
cp $datadir/training_sample2/supervoxels.h5 /Users/michielk/anaconda/envs/neuroproof-20160912/examples/training_sample2/
cp $datadir/training_sample2/boundary_prediction.h5 /Users/michielk/anaconda/envs/neuroproof-20160912/examples/training_sample2/
cp $datadir/training_sample2/groundtruth.h5 /Users/michielk/anaconda/envs/neuroproof-20160912/examples/training_sample2/


source activate gala_20160715
gala-segmentation-pipeline \
--image-stack $datadir/training_sample2/grayscale_maps.h5 \
--ilp-file $datadir/training_sample1/results/boundary_classifier_ilastik.ilp \
--disable-gen-pixel --pixelprob-file $datadir/training_sample2/boundary_prediction.h5 \
--enable-gen-supervoxels \
--disable-gen-agglomeration \
--seed-size 50 \
--enable-raveler-output \
--enable-h5-output $datadir/training_sample2 \
--segmentation-thresholds 0.0


# FAIL
neuroproof_graph_learn \
$datadir/training_sample2/oversegmented_stack_labels.h5 \
$datadir/training_sample2/boundary_prediction.h5 \
$datadir/training_sample2/groundtruth.h5 \
--classifier-name $datadir/training_sample2/classifier_str2_niter2.xml \
--strategy-type 2 --num-iterations 2 --use_mito 0

# FAIL
neuroproof_graph_learn \
$datadir/validation_sample/oversegmented_stack_labels.h5 \
$datadir/validation_sample/boundary_prediction.h5 \
$datadir/validation_sample/groundtruth.h5 \
--classifier-name $datadir/validation_sample/classifier_str2_niter2.xml \
--strategy-type 2 --num-iterations 2 --use_mito 0




neuroproof_create_spgraph \
$datadir/training_sample2/oversegmented_stack_labels.h5 \
$datadir/training_sample2/boundary_prediction.h5

neuroproof_graph_analyze





neuroproof_graph_learn \
$datadir/training_sample2/oversegmented_stack_labels.h5 \
$datadir/training_sample2/boundary_prediction.h5 \
$datadir/training_sample2/groundtruth.h5 \
--classifier-name $datadir/training_sample2/classifier_osl_str2.xml \
--strategy-type 2

neuroproof_graph_predict \
$datadir/training_sample2/oversegmented_stack_labels.h5 \
$datadir/training_sample2/boundary_prediction.h5 \
$datadir/training_sample2/classifier_osl_str2.xml \
--output-file $datadir/training_sample2/classifier_osl_str2_predict.h5

neuroproof_graph_predict \
$datadir/validation_sample/oversegmented_stack_labels.h5 \
$datadir/validation_sample/boundary_prediction.h5 \
$datadir/training_sample2/classifier_osl_str2.xml \
--output-file $datadir/validation_sample/classifier_osl_str2_predict.h5
# the originals of validation_sample are transposed in stack_viewer!
