### LOCAL ###
scriptdir="$HOME/workspace/EM"
datadir="$HOME/oxdata/P01/EM/M3/M3_S1_GNU/pipeline_test"
pixprob_trainingset="pixprob_training"
dataset="pixprob_training"
dataset="segclass_training"

python $scriptdir/EM_stack2stack.py \
"$datadir/${dataset}.h5" \
"$datadir/${dataset}.nii.gz" \
-e 0.05 0.0073 0.0073 \
-d 'int32' -f 'stack' -g 'stack'

### STAGE0: ilastik interactive pixprob classifier
source activate ilastik-devel
CONDA_ROOT=`conda info --root`
${CONDA_ROOT}/envs/ilastik-devel/run_ilastik.sh

### STAGE1: boundary prediction (actually exportable from Ilastik for this dataset)
source activate ilastik-devel
CONDA_ROOT=`conda info --root`
${CONDA_ROOT}/envs/ilastik-devel/run_ilastik.sh --headless \
--preconvert_stacks \
--project="$datadir/${pixprob_trainingset}.ilp" \
--output_axis_order=xyzc \
--output_format=hdf5 \
--output_filename_format="$datadir/${dataset}_probs.h5" \
--output_internal_path=/volume/predictions \
"$datadir/${dataset}.h5/stack"
python $scriptdir/EM_stack2stack.py \
"$datadir/${dataset}_probs.h5" \
"$datadir/${dataset}_probs.nii.gz" \
-e 0.05 0.0073 0.0073 \
-d 'float' -f 'volume/predictions'

# STAGE2: supervoxel generation
### watershed supervoxels
source activate gala
gala-segmentation-pipeline \
--image-stack "$datadir/${pixprob_trainingset}.h5" \
--ilp-file "$datadir/${pixprob_trainingset}.ilp" \
--disable-gen-pixel --pixelprob-file "$datadir/${dataset}_probs.h5" \
--enable-gen-supervoxels \
--disable-gen-agglomeration \
--seed-size 3 \
--enable-raveler-output \
--enable-h5-output "$datadir" \
--segmentation-thresholds 0.0
python $scriptdir/EM_stack2stack.py \
"$datadir/supervoxels.h5" \
"$datadir/supervoxels.nii.gz" \
-e 0.05 0.0073 0.0073 \
-d 'int32' -f 'stack' -g 'stack' -i 'xyz' -l 'zyx'

### slicvoxels
python $scriptdir/EM_slicvoxels.py \
-i "$datadir/${dataset}.h5" \
-o "$datadir/${dataset}_slicvoxels.h5" \
-f 'stack' -g 'stack' -s 500
python $scriptdir/EM_stack2stack.py \
"$datadir/${dataset}_slicvoxels.h5" \
"$datadir/${dataset}_slicvoxels.nii.gz" \
-e 0.05 0.0073 0.0073 \
-d 'int32' -f 'stack' -g 'stack'

python $scriptdir/EM_slicvoxels.py \
-i "$datadir/${dataset}_probs.h5" \
-o "$datadir/${dataset}_probs_slicvoxels.h5" \
-f 'volume/predictions' -g 'stack' -s 2000
python $scriptdir/EM_stack2stack.py \
"$datadir/${dataset}_probs_slicvoxels.h5" \
"$datadir/${dataset}_probs_slicvoxels.nii.gz" \
-e 0.05 0.0073 0.0073 \
-d 'int32' -f 'stack' -g 'stack'

### slic to segmentation scratch
python ${scriptdir}/EM_slicsegmentation.py \
"$datadir/${dataset}_probs.h5/volume/predictions" \
"$datadir/${dataset}_probs_slicvoxels.h5/stack" \
"$datadir/${dataset}_probs_slicsegmentation.h5/stack" \
-i 0 -t 0.8 -p 100 \
-b 40 250 250  # -b 100 500 500
python $scriptdir/EM_stack2stack.py \
"$datadir/${dataset}_probs_slicsegmentation.h5" \
"$datadir/${dataset}_probs_slicsegmentation.nii.gz" \
-e 0.05 0.0073 0.0073 \
-d 'int32' -f 'stack' -g 'stack'



### watershedding the intracellular layer (on EED-processed probabilities)
import os
import numpy as np
import h5py
import nibabel as nib
from skimage.morphology import watershed, remove_small_objects, dilation
from skimage.segmentation import random_walker, relabel_sequential
from scipy.ndimage.measurements import label
from scipy.ndimage import distance_transform_edt

datadir = "/Users/michielk/oxdata/P01/EM/M3/M3_S1_GNU/pipeline_test"
pixprob_trainingset = "pixprob_training"
dataset = "pixprob_training"
dataset = "segclass_training"

def loadh5(datadir, dname, fieldname='stack'):
    f = h5py.File(os.path.join(datadir, dname), 'r')
    if len(f[fieldname].shape) == 3:
        stack = f[fieldname][:,:,:]
    if len(f[fieldname].shape) == 4:
        stack = f[fieldname][:,:,:,:]
    f.close()
    return stack

def writeh5(stack, datadir, fp_out, fieldname='stack', dtype='int32'):
    g = h5py.File(os.path.join(datadir, fp_out), 'w')
    dset = g.create_dataset(fieldname, stack.shape, dtype=dtype)
    g[fieldname][:,:,:] = stack
    g.close()


### load the probabilities
prob_myel = loadh5(datadir, dataset + '_probs0_eed2.h5')
prob_axon = loadh5(datadir, dataset + '_probs1_eed2.h5')
prob_mito = loadh5(datadir, dataset + '_probs2_eed2.h5')
prob_memb = loadh5(datadir, dataset + '_probs3_eed2.h5')

myelin = prob_myel > 0.2
writeh5(myelin, datadir, dataset + '_probs_hardsegmentation_myelin.h5')
mito = prob_mito > 0.5
writeh5(mito, datadir, dataset + '_probs_hardsegmentation_mito.h5')

### get the axon seeds
seed_val = 0.6
seed_size = 100
seeds_axon = label(prob_axon>=seed_val)[0]
seeds_axon = remove_small_objects(seeds_axon, seed_size)
writeh5(seeds_axon, datadir, dataset + '_probs_wsseeds.h5')

### watershed on the myelinated axons (for getting the myelinated cores)
ws = watershed(prob_myel, seeds_axon, mask=~myelin)
ws, _, _ = relabel_sequential(ws)
writeh5(ws, datadir, dataset + '_probs_wssegmentation.h5')

### watershed on the myelin to separate individual sheaths
ws = loadh5(datadir, dataset + '_probs_wssegmentation.h5')
# merge the segments of oversegmented myelinated axons
if dataset == 'pixprob_training':
    MA_segments = [[13],[3,29],[8,24,33,34,53],[25],
                   [44],[9],[14,15],[22],[41]]  # [56]
elif dataset == 'segclass_training':
    MA_segments = [[1,4],[2],[3],[39,40],[44]]
elif dataset == 'm000_cutout01':
    MA_segments = [[54],[26],[14,46],[7],[34],[32],[15,16],[31],
                   [33,54],[9,10],[11],[19,37],[23],[21],[17,25],[47]]

MA = np.zeros_like(myelin, dtype='uint32')
for i, MA_segment in enumerate(MA_segments):
    for snum in MA_segment:
        MA[ws==snum] = i + 1

writeh5(MA, datadir, dataset + '_probs_wsseeds_myelin.h5')
# watershed on the distance transform
distance = distance_transform_edt(MA==0)
distance = distance_transform_edt(MA==0, sampling=[0.05,0.0073,0.0073])
distance[~myelin] = 0
writeh5(distance, datadir, dataset + '_probs_wsdistance_myelin.h5', dtype='float')
# labels = seeds[indices[0],indices[1],indices[2]]  # similar to watershed, but also gets non-connected components # labels[~myelin] = 0
ws_myel = watershed(distance, dilation(MA), mask=myelin)
writeh5(ws_myel, datadir, dataset + '_probs_wssegmentation_myelin.h5')

### watershed the unmyelinated axons
# prob = -prob_axon
prob = prob_myel + np.abs(1-prob_axon) - prob_mito + prob_memb
writeh5(prob, datadir, dataset + '_probs_prob+0i1-2+3.h5', dtype='float')
seeds_axon = loadh5(datadir, dataset + '_probs_wsseeds.h5')
ws = watershed(prob, seeds_axon, mask=~np.logical_or(ws_myel,MA))
ws, _, _ = relabel_sequential(ws)
writeh5(ws, datadir, dataset + '_probs_wssegmentation_axon.h5')


python $scriptdir/EM_stack2stack.py \
"$datadir/${dataset}_probs_prob+0i1-2+3.h5" \
"$datadir/${dataset}_probs_prob+0i1-2+3.nii.gz" \
-e 0.05 0.0073 0.0073 \
-d 'float' -f 'stack'

python $scriptdir/EM_stack2stack.py \
"$datadir/${dataset}_probs_hardsegmentation_myelin.h5" \
"$datadir/${dataset}_probs_hardsegmentation_myelin.nii.gz" \
-e 0.05 0.0073 0.0073 \
-d 'int32' -f 'stack'
python $scriptdir/EM_stack2stack.py \
"$datadir/${dataset}_probs_hardsegmentation_mito.h5" \
"$datadir/${dataset}_probs_hardsegmentation_mito.nii.gz" \
-e 0.05 0.0073 0.0073 \
-d 'int32' -f 'stack'

python $scriptdir/EM_stack2stack.py \
"$datadir/${dataset}_probs_wsseeds.h5" \
"$datadir/${dataset}_probs_wsseeds.nii.gz" \
-e 0.05 0.0073 0.0073 \
-d 'int32' -f 'stack'
python $scriptdir/EM_stack2stack.py \
"$datadir/${dataset}_probs_wssegmentation.h5" \
"$datadir/${dataset}_probs_wssegmentation.nii.gz" \
-e 0.05 0.0073 0.0073 \
-d 'int32' -f 'stack'
python $scriptdir/EM_stack2stack.py \
"$datadir/${dataset}_probs_wsseeds_myelin.h5" \
"$datadir/${dataset}_probs_wsseeds_myelin.nii.gz" \
-e 0.05 0.0073 0.0073 \
-d 'int32' -f 'stack'
python $scriptdir/EM_stack2stack.py \
"$datadir/${dataset}_probs_wsdistance_myelin.h5" \
"$datadir/${dataset}_probs_wsdistance_myelin.nii.gz" \
-e 0.05 0.0073 0.0073 \
-d 'float' -f 'stack'
python $scriptdir/EM_stack2stack.py \
"$datadir/${dataset}_probs_wssegmentation_myelin.h5" \
"$datadir/${dataset}_probs_wssegmentation_myelin.nii.gz" \
-e 0.05 0.0073 0.0073 \
-d 'int32' -f 'stack'
python $scriptdir/EM_stack2stack.py \
"$datadir/${dataset}_probs_wssegmentation_axon.h5" \
"$datadir/${dataset}_probs_wssegmentation_axon.nii.gz" \
-e 0.05 0.0073 0.0073 \
-d 'int32' -f 'stack'







### select class and assign based on thresholding
# probindex = 0
# probthreshold = 0.5
# probvoxelcount = 10
# s = fs[fieldname_svox][:,:,:]
# p = fp[fieldname_prob][:,:,:,probindex] > probthreshold
# forward_map = np.zeros(np.max(s) + 1, 'bool')
# x = np.bincount(s[p]) >= probvoxelcount
# forward_map[0:len(x)] = x
# segments = forward_map[s]

### assign to the highest average prob in supervoxel
s = fs[fieldname_svox][:,:,:]
p = fp[fieldname_prob][:,:,:,:]
label_weighted = np.zeros([6,np.max(s) + 1], 'float')
for i in range(0,6):  # TODO proper iterator
    label_weighted[i,:] = np.bincount(np.ravel(s), weights=np.ravel(p[:,:,:,i]))

forward_map = np.argmax(label_weighted, axis=0) + 1
segments = forward_map[s]

# write segmentation
g = h5py.File(filepath_outp, 'w')
dset = g.create_dataset(fieldname_outp, segments.shape, dtype='int32')
g[fieldname_outp][:,:,:] = segments
g.close()

# exagerrated membrane skeleton (labels 1,3,5,6)
memlabels = [1,3,5,6]
memskel = np.zeros_like(s, dtype='bool')
for memlabel in memlabels:
    memskel[segments==memlabel] = True

label_im, nb_labels = label(~memskel)
filepath_labels = path.join(datadir, dataset + '_probs_slicsegmentation_labels.h5')
fieldname_labels = '/stack'
g = h5py.File(filepath_labels, 'w')
dset = g.create_dataset(fieldname_labels, label_im.shape, dtype='int32')
g[fieldname_labels][:,:,:] = label_im
g.close()

fp.close()
fs.close()


python $scriptdir/EM_stack2stack.py \
"$datadir/${dataset}_probs_slicsegmentation_average.h5" \
"$datadir/${dataset}_probs_slicsegmentation_average.nii.gz" \
-e 0.05 0.0073 0.0073 \
-d 'int32' -f 'stack'
python $scriptdir/EM_stack2stack.py \
"$datadir/${dataset}_probs_slicsegmentation_labels.h5" \
"$datadir/${dataset}_probs_slicsegmentation_labels.nii.gz" \
-e 0.05 0.0073 0.0073 \
-d 'int32' -f '/stack'




layer=3
python $scriptdir/EM_stack2stack.py \
"$datadir/${dataset}_probs${layer}_eed2.h5" \
"$datadir/${dataset}_probs${layer}_eed2.nii.gz" \
-e 0.05 0.0073 0.0073 \
-d 'float' -f 'stack'

### newer approach based on average
from os import path
import h5py
from mpi4py import MPI
import numpy as np
from scipy.ndimage.measurements import label

datadir = "/Users/michielk/oxdata/P01/EM/M3/M3_S1_GNU/pipeline_test"
pixprob_trainingset = "pixprob_training"
dataset = "segclass_training"

filepath_prob = path.join(datadir, dataset + '_probs.h5')
fieldname_prob = '/volume/predictions'
filepath_svox = path.join(datadir, dataset + '_slicvoxels.h5')
fieldname_svox = '/stack'
filepath_outp = path.join(datadir, dataset + '_probs_slicsegmentation_average.h5')
fieldname_outp = '/stack'

fp = h5py.File(filepath_prob, 'r')
fs = h5py.File(filepath_svox, 'r')


### to consider instead of watershed
# rw = random_walker(memprob, seeds, beta=130, mode='cg_mg', tol=0.001, copy=True, multichannel=False,
#                    return_full_prob=False, spacing=[0.05,0.0073,0.0073])











# STAGE3: agglomeration classifier
source activate neuroproof
neuroproof_graph_learn \
$datadir/supervoxels.h5 \
$datadir/$dataset.h5 \
$datadir/${dataset}_groundtruth.h5 \
--classifier-name $datadir/classifier_str1.xml \
--strategy-type 1

#
source activate neuroproof
neuroproof_graph_learn \
$datadir/training_sample2/slicvoxels.h5 \
$datadir/training_sample2/boundary_prediction.h5 \
$datadir/training_sample2/groundtruth.h5 \
--classifier-name $datadir/training_sample2/classifier_str1_slic.xml \
--strategy-type 1










# STAGE3
python /Users/michielk/workspace/EM/EM_series2stack.py \
$datadir/validation_sample/grayscale_maps \
$datadir/validation_sample/grayscale_maps.h5 \
-f 'stack' -m -o -r '*.png' -n 5

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
"$datadir/validation_sample/grayscale_maps.h5"
#python setup.py install

cp $datadir/validation_sample/grayscale_maps.h5 $datadir/validation_sample/supervoxels.h5
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

source activate neuroproof-devel
neuroproof_graph_predict \
$datadir/validation_sample/supervoxels.h5 \
$datadir/validation_sample/boundary_prediction.h5 \
$datadir/training_sample2/classifier_str1.xml \
--output-file $datadir/validation_sample/segmentation.h5 \
--graph-file $datadir/validation_sample/graph.json


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
