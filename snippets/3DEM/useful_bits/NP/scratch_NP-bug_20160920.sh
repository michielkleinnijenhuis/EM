datadir=/Users/michielk/workspace/FlyEM/NeuroProof/examples

### greyscale_maps to .h5
for dataset in 'training_sample1' 'training_sample2' 'validation_sample'; do
python /Users/michielk/workspace/EM/convert/EM_series2stack.py \
$datadir/$dataset/grayscale_maps \
$datadir/$datset/grayscale_maps.h5 \
-f 'stack' -o -r '*.png'
done

### Ilastikboundary prediction from an zyx!! greyscale_maps.h5 (i.e. output is xyzc)
source activate ilastik-devel
CONDA_ROOT=`conda info --root`
for dataset in 'training_sample1' 'training_sample2' 'validation_sample'; do
${CONDA_ROOT}/envs/ilastik-devel/run_ilastik.sh --headless \
--preconvert_stacks \
--project=$datadir/training_sample1/results/boundary_classifier_ilastik.ilp \
--output_axis_order=zyxc \
--output_format=hdf5 \
--output_filename_format=$datadir/$dataset/boundary_prediction.h5 \
--output_internal_path=/volume/predictions \
"$datadir/$dataset/grayscale_maps.h5/stack"
done

### gala supervoxels
source activate gala_20160715
for dataset in 'training_sample1' 'training_sample2' 'validation_sample'; do
gala-segmentation-pipeline \
--image-stack $datadir/$dataset/grayscale_maps.h5 \
--ilp-file $datadir/training_sample1/results/boundary_classifier_ilastik.ilp \
--disable-gen-pixel \
--pixelprob-file $datadir/$dataset/boundary_prediction.h5 \
--enable-gen-supervoxels \
--disable-gen-agglomeration \
--seed-size 5 \
--enable-raveler-output \
--enable-h5-output $datadir/$dataset \
--segmentation-thresholds 0.0
done

### slicvoxels
s=1000; c=0.02;
s=5000; c=0.02;
for dataset in 'training_sample1' 'training_sample2' 'validation_sample'; do
python /Users/michielk/workspace/EM/supervoxels/EM_slicvoxels.py \
$datadir/$dataset/grayscale_maps.h5 \
$datadir/$dataset/slicvoxels_s${s}_c${c}.h5 \
-f 'stack' -g 'stack' -s $s -c $c -o 1 -e 1 1 1 -u
done

### agglomeration classifier training (works only for one iter!!!)
source activate neuroproof-20160912
dataset=training_sample2
strtype=2
iter=2
for datavol in 'oversegmented_stack_labels' 'supervoxels' 'slicvoxels'; do
neuroproof_graph_learn \
$datadir/$dataset/$datavol.h5 \
$datadir/$dataset/boundary_prediction.h5 \
$datadir/$dataset/groundtruth.h5 \
--classifier-name $datadir/training_sample2/classifier_${datavol}_str${strtype}_iter${iter}.xml \
--strategy-type $strtype --num-iterations $iter --use_mito 0
done


### agglomeration
source activate neuroproof-20160912
strtype=2
iter=1
for dataset in 'training_sample2' 'validation_sample'; do
for datavol in 'oversegmented_stack_labels' 'supervoxels' 'slicvoxels'; do
neuroproof_graph_predict \
$datadir/$dataset/$datavol.h5 \
$datadir/$dataset/boundary_prediction.h5 \
$datadir/training_sample2/classifier_${datavol}_str${strtype}_iter${iter}.xml \
--output-file $datadir/$dataset/classifier_${datavol}_str${strtype}_iter${iter}_segmentation.h5
done
done




dataset=training_sample1
dataset=training_sample2
dataset=validation_sample
datavol=oversegmented_stack_labels
datavol=supervoxels
datavol=slicvoxels

for dataset in 'training_sample1' 'training_sample2' 'validation_sample'; do
python $HOME/workspace/EM/convert/EM_stack2stack.py \
${datadir}/${dataset}/${datavol}.h5 ${datadir}/${dataset}/${datavol}_int16.h5 -d int16
done
