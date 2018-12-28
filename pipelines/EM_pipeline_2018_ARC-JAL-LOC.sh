###=========================================================================###
### prepare environment
###=========================================================================###
scriptdir=$HOME/workspace/EM
source $scriptdir/pipelines/datasets.sh
source $scriptdir/pipelines/functions.sh
source $scriptdir/pipelines/submission.sh

#compute_env='JAL'
#compute_env='ARC'
compute_env='LOCAL'
#compute_env='ARCB'
prep_environment $scriptdir $compute_env

# dataset='M3S1GNU'
# dataset='B-NT-S9-2a'
# dataset='B-NT-S10-2d_ROI_00'
# dataset='B-NT-S10-2d_ROI_02'
dataset='B-NT-S10-2f_ROI_00'
# dataset='B-NT-S10-2f_ROI_01'
# dataset='B-NT-S10-2f_ROI_02'

bs='0500' && prep_dataset $dataset $bs && echo ${#datastems[@]}

#echo -n -e "\033]0;$dataset\007"

source activate scikit-image-devel_0.13
export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8


###=========================================================================###
### convert and register  # setup for ARCUS-B
###=========================================================================###

### convert dm3 files (NOTE: paths set up in function 'dataset_parameters')
declare jid=''
scriptfile=$( dm3convert 'h' )
[[ -z $jid ]] && dep='' || dep="--dependency=after:$jid"
jid=$( sbatch $dep $scriptfile ) && jid=${jid##* }

### register slices (NOTE: does not handle montage, see earlier M3S1GNU processing)
declare jid=''
scriptfile=$( fiji_register 'h' )
[[ -z $jid ]] && dep='' || dep="--dependency=after:$jid"
jid=$( sbatch $dep $scriptfile ) && jid=${jid##* }

### convert the series of tifs to h5
declare opf='' ods='data'
scriptfile=$( tif2h5 'h' '' $ods )
[[ -z $jid ]] && dep='' || dep="--dependency=after:$jid"
JID=$( sbatch $dep $scriptfile ) && jid_data=${JID##* }

### create downsampled dataset
declare ipf='' ids='data' opf='' ods='data' \
    brfun='np.mean' brvol='' slab=20 memcpu=60000 wtime='02:00:00' vol_slice=''
scriptfile=$( blockreduce 'h' $ipf $ids $opf $ods $brfun '' $slab $memcpu $wtime $vol_slice )
[[ -z $JID ]] && dep='' || dep="--dependency=after:$JID"
jid=$( sbatch $dep $scriptfile ) && jid=${jid##* }

# threshold the data at 0 to get a dataset mask (maskDS)
declare ipf='' ids='data' opf='_masks_maskDS' ods='maskDS' \
    slab=20 arg='-g -l 0 -u 10000000'
scriptfile=$( prob2mask 'h' '' $ids $opf $ods $slab $arg )
[[ -z $JID ]] && dep='' || dep="--dependency=after:$JID"
jid=$( sbatch $dep $scriptfile ) && jid=${jid##* }

### create downsampled maskDS
declare ipf='_masks_maskDS' ids='maskDS' opf='_masks_maskDS' ods='maskDS' \
    brfun='np.amax' brvol='' slab=20 memcpu=60000 wtime='00:10:00' vol_slice=''
scriptfile=$( blockreduce 'h' $ipf $ids $opf $ods $brfun '' $slab $memcpu $wtime $vol_slice )
[[ -z $jid ]] && dep='' || dep="--dependency=after:$jid"
jid=$( sbatch $dep $scriptfile ) && jid=${jid##* }

### extract training data to train ilastik classifier
scriptfile=$( trainingdata 'h' )
[[ -z $JID ]] && dep='' || dep="--dependency=after:$JID"
jid=$( sbatch $dep $scriptfile ) && jid=${jid##* }


###=========================================================================###
### pixel classification: STAGE 2  # setup for ARCUS-B
###=========================================================================###

### apply ilastik classifier
declare ipf='' ids='data' opf='_probs' ods='volume/predictions' clf="pixclass_8class"
scriptfile=$( apply_ilastik 'h' '' $ids $opf $ods $clf )
jid=$( sbatch -p devel $dep $scriptfile ) && jid=${jid##* }

### correct element_size_um  # TODO: create function
scriptfile="$datadir/EM_corr_script.py"
write_ilastik_correct_attributes $scriptfile
jobname='correct_ilastik'
declare additions='' CONDA_ENV='' nodes=1 memcpu=3000 wtime='00:10:00' tasks=1 njobs=1 q='h'
scriptfile=$( single_job "python $scriptfile" )
[[ -z $jid ]] && dep='' || dep="--dependency=after:$jid"
jid=$( sbatch -p devel $dep $scriptfile ) && jid=${jid##* }

# TODO?: blockreduce '' 4 1

### split and sum
declare ipf='_probs' ids='volume/predictions' opf='_probs' ods='volume/predictions'
scriptfile=$( splitblocks 'h' 'a' $ipf $ids $opf $ods )
[[ -z $jid ]] && dep='' || dep="--dependency=after:$jid"
jid=$( sbatch $dep $scriptfile ) && jid=${jid##* }

# step=64
# for br in `seq 0 $step ${#datastems[@]}`; do
#     declare ipf='_probs' ids='volume/predictions' opf='_probs' ods='volume/predictions' blockrange="-r $br $((br+step))"
#     scriptfile=$( splitblocks 'h' 'a' $ipf $ids $opf $ods $blockrange )
#     [[ $br == 0 ]] && dep='' || dep="--dependency=after:$jid"
#     jid=$( sbatch -p devel $dep $scriptfile ) && jid=${jid##* }
# done


###=========================================================================###
### EED prep BARRIER # TODO: make into array jobs with one jid
# NOTE: sum_volumes cannot be in parallel, but the rest of the EED streams can
###=========================================================================###

### sum myelin (MM), mito (MT), myelin_inner (MI), myelin_outer (MO) compartments
declare ipf='_probs' ids='volume/predictions' opf='_probs_sum0247' ods='sum0247' vols='0 2 4 7'
sum_volumes '' 'a' $ipf $ids $opf $ods $vols

### sum myelinated ICS and unmyelinated ICS compartments
declare ipf='_probs' ids='volume/predictions' opf='_probs_sum16' ods='sum16' vols='1 6'
sum_volumes '' 'a' $ipf $ids $opf $ods $vols

### extract myelinated ICS compartment
declare ipf='_probs' ids='volume/predictions' opf='_probs_probMA' ods='probMA' vols='1'
sum_volumes '' 'a' $ipf $ids $opf $ods $vols


# ###=========================================================================###
# ### reduce MA prediction
# ###=========================================================================###
# declare ipf='_probs_probMA' ids='probMA' opf='_probs_probMA' ods='probMA' args='-d float16'
# mergeblocks 'h' '' '' $ipf $ids $opf $ods $args
# declare ipf='_probs_probMA' ids='probMA' opf='_probs_probMA' ods='probMA' \
#     brfun='np.mean' brvol='' slab=12 memcpu=60000 wtime='02:00:00' vol_slice=''
# blockreduce 'h' $ipf $ids $opf $ods $brfun '' $slab $memcpu $wtime $vol_slice


###=========================================================================###
### edge-enhancing diffusion  # setup for JALAPENO
# rsync -Pazv $host_arc:$datadir_arc/blocks_0500/*_sum*.h5 $datadir/blocks_0500/
# rsync -Pazv $host_arc:$datadir_arc/blocks_0500/*_probMA.h5 $datadir/blocks_0500/
# TODO: slabsize for blockreduce
###=========================================================================###
jid=''  # TODO: link to sum volumes
### smooth the summed probability map with edge-enhancing diffusion
declare ipf='_probs_sum0247' ids='sum0247' opf='_probs_eed_sum0247' ods='sum0247_eed' wtime='03:10:00'
scriptfile=$( eed 'h' 'a' $ipf $ids $opf $ods $wtime )
[[ -z $jid ]] && dep='' || dep="-j $jid"
JID=$( fsl_sub -q short.q $dep $scriptfile )
### merge the blocks
declare ipf='_probs_eed_sum0247' ids='sum0247_eed' opf='_probs_eed_sum0247' ods='sum0247_eed' args='-d float16'
scriptfile=$( mergeblocks 'h' '' '' $ipf $ids $opf $ods $args )
[[ -z $JID ]] && dep='' || dep="-j $JID"
jid=$( fsl_sub -q short.q $dep $scriptfile )
### downsample the volume
declare ipf='_probs_eed_sum0247' ids='sum0247_eed' opf='_probs_eed_sum0247' ods='sum0247_eed' \
    brfun='np.mean' brvol='' slab=12 memcpu=60000 wtime='02:00:00' vol_slice=''
scriptfile=$( blockreduce 'h' $ipf $ids $opf $ods $brfun '' $slab $memcpu $wtime $vol_slice )
[[ -z $jid ]] && dep='' || dep="-j $jid"
jid=$( fsl_sub -q short.q $dep $scriptfile )
### generate the raw myelin mask
declare ipf='_probs_eed_sum0247' ids='sum0247_eed' opf='_masks_maskMM' ods='maskMM' arg='-g -l 0.5 -s 2000 -d 1 -S'
scriptfile=$( prob2mask_datastems 'h' 'a' $ipf $ids $opf $ods $arg )
[[ -z $JID ]] && dep='' || dep="-j $JID"
jid=$( fsl_sub -q veryshort.q $dep $scriptfile )
### merge the blocks
declare ipf='_masks_maskMM' ids='maskMM_steps/raw' opf='_masks_maskMM' ods='maskMM_raw'
scriptfile=$( mergeblocks 'h' '' '' $ipf $ids $opf $ods $args )
[[ -z $jid ]] && dep='' || dep="-j $jid"
jid=$( fsl_sub -q veryshort.q $dep $scriptfile )
### remove small components and dilate
declare ipf='_masks_maskMM' ids='maskMM_raw' opf='_masks_maskMM' ods='maskMM' \
    slab=12 arg='-g -l 0 -u 0 -s 2000 -d 1'
scriptfile=$( prob2mask 'h' $ipf $ids $opf $ods $slab $arg )
[[ -z $jid ]] && dep='' || dep="-j $jid"
jid=$( fsl_sub -q veryshort.q $dep $scriptfile )
### downsample the volume
declare ipf='_masks_maskMM' ids='maskMM' opf='_masks_maskMM' ods='maskMM' \
    brfun='np.amax' brvol='' slab=27 memcpu=60000 wtime='00:10:00' vol_slice=''
scriptfile=$( blockreduce 'h' $ipf $ids $opf $ods $brfun '' $slab $memcpu $wtime $vol_slice )
[[ -z $jid ]] && dep='' || dep="-j $jid"
jid=$( fsl_sub -q veryshort.q $dep $scriptfile )



jid=''  # TODO: link to sum volumes
### smooth the summed probability map with edge-enhancing diffusion
declare ipf='_probs_sum16' ids='sum16' opf='_probs_eed_sum16' ods='sum16_eed' wtime='03:10:00'
scriptfile=$( eed 'h' 'a' $ipf $ids $opf $ods $wtime )
[[ -z $jid ]] && dep='' || dep="-j $jid"
JID=$( fsl_sub -q short.q $dep $scriptfile )
### merge the blocks
declare ipf='_probs_eed_sum16' ids='sum16_eed' opf='_probs_eed_sum16' ods='sum16_eed' args='-d float16'
scriptfile=$( mergeblocks 'h' '' '' $ipf $ids $opf $ods $args )
[[ -z $JID ]] && dep='' || dep="-j $JID"
jid=$( fsl_sub -q short.q $dep $scriptfile )
### downsample the volume
declare ipf='_probs_eed_sum16' ids='sum16_eed' opf='_probs_eed_sum16' ods='sum16_eed' \
    brfun='np.mean' brvol='' slab=12 memcpu=60000 wtime='02:00:00' vol_slice=''
scriptfile=$( blockreduce 'h' $ipf $ids $opf $ods $brfun '' $slab $memcpu $wtime $vol_slice )
[[ -z $jid ]] && dep='' || dep="-j $jid"
jid=$( fsl_sub -q short.q $dep $scriptfile )
### generate the ICS mask
declare ipf='_probs_eed_sum16' ids='sum16_eed' opf='_masks_maskICS' ods='maskICS' arg='-g -l 0.2'
scriptfile=$( prob2mask_datastems 'h' 'a' $ipf $ids $opf $ods $arg )
[[ -z $JID ]] && dep='' || dep="-j $JID"
jid=$( fsl_sub -q veryshort.q $dep $scriptfile )
### merge the blocks
declare ipf='_masks_maskICS' ids='maskICS' opf='_masks_maskICS' ods='maskICS'
scriptfile=$( mergeblocks 'h' '' '' $ipf $ids $opf $ods $args )
[[ -z $jid ]] && dep='' || dep="-j $jid"
jid=$( fsl_sub -q veryshort.q $dep $scriptfile )
### downsample the volume
declare ipf='_masks_maskICS' ids='maskICS' opf='_masks_maskICS' ods='maskICS' \
    brfun='np.amax' brvol='' slab=27 memcpu=60000 wtime='00:10:00' vol_slice=''
scriptfile=$( blockreduce 'h' $ipf $ids $opf $ods $brfun '' $slab $memcpu $wtime $vol_slice )
[[ -z $jid ]] && dep='' || dep="-j $jid"
jid=$( fsl_sub -q veryshort.q $dep $scriptfile )


jid=''  # TODO: link to sum volumes
### smooth the summed probability map with edge-enhancing diffusion
declare ipf='_probs_probMA' ids='probMA' opf='_probs_eed_probMA' ods='probMA_eed' wtime='03:10:00'
scriptfile=$( eed 'h' 'a' $ipf $ids $opf $ods $wtime )
[[ -z $jid ]] && dep='' || dep="-j $jid"
JID=$( fsl_sub -q short.q $dep $scriptfile )
### merge the blocks
declare ipf='_probs_eed_probMA' ids='probMA_eed' opf='_probs_eed_probMA' ods='probMA_eed' args='-d float16'
scriptfile=$( mergeblocks 'h' '' '' $ipf $ids $opf $ods $args )
[[ -z $JID ]] && dep='' || dep="-j $JID"
jid=$( fsl_sub -q short.q $dep $scriptfile )
### downsample the volume
declare ipf='_probs_eed_probMA' ids='probMA_eed' opf='_probs_eed_probMA' ods='probMA_eed' \
    brfun='np.mean' brvol='' slab=12 memcpu=60000 wtime='02:00:00' vol_slice=''
scriptfile=$( blockreduce 'h' $ipf $ids $opf $ods $brfun '' $slab $memcpu $wtime $vol_slice )
[[ -z $jid ]] && dep='' || dep="-j $jid"
jid=$( fsl_sub -q short.q $dep $scriptfile )
### generate the myelinated axon mask
declare ipf='_probs_eed_probMA' ids='probMA_eed' opf='_masks_maskMA' ods='maskMA' arg='-g -l 0.2'
scriptfile=$( prob2mask_datastems 'h' 'a' $ipf $ids $opf $ods $arg )
[[ -z $JID ]] && dep='' || dep="-j $JID"
jid=$( fsl_sub -q veryshort.q $dep $scriptfile )
### merge the blocks
declare ipf='_masks_maskMA' ids='maskMA' opf='_masks_maskMA' ods='maskMA'
scriptfile=$( mergeblocks 'h' '' '' $ipf $ids $opf $ods $args )
[[ -z $jid ]] && dep='' || dep="-j $jid"
jid=$( fsl_sub -q veryshort.q $dep $scriptfile )
### downsample the volume
declare ipf='_masks_maskMA' ids='maskMA' opf='_masks_maskMA' ods='maskMA' \
    brfun='np.amax' brvol='' slab=15 memcpu=60000 wtime='00:10:00' vol_slice=''
scriptfile=$( blockreduce 'h' $ipf $ids $opf $ods $brfun '' $slab $memcpu $wtime $vol_slice )
[[ -z $jid ]] && dep='' || dep="-j $jid"
jid=$( fsl_sub -q veryshort.q $dep $scriptfile )


# ###=========================================================================###
# ### prob2mask BARRIER: wait for all masks to complete
# # rsync -Pazv $host_arc:$datadir_arc/*_masks_maskDS.h5 $datadir
# ###=========================================================================###
#
# for dset in 'maskDS' 'maskMM' 'maskICS' 'maskMA'; do
#     h5copy -p -i ${dataset}_masks_$dset.h5 -s $dset -o ${dataset}_masks.h5 -d $dset
#     h5copy -p -i ${dataset_ds}_masks_$dset.h5 -s $dset -o ${dataset_ds}_masks.h5 -d $dset
# done


###=========================================================================###
### myelinated axon compartment
###=========================================================================###
### label ICS components in 2D
jid=''  # TODO: link to maskMM and _probs_eed_sum16.h5/sum16_eed
declare ipf='_masks_maskMM' ids='maskMM' opf='_labels_labelMA_core2D' ods='labelMA_core2D' meanint='dummy'
scriptfile=$( conncomp 'h' '2D' $dataset $ipf $ids $opf $ods $meanint )
[[ -z $jid ]] && dep='' || dep="-j $jid"
jid=$( fsl_sub -q short.q $dep $scriptfile )
### map all label properties to numpy vectors
declare ipf='_labels_labelMA_core2D' ids='labelMA_core2D' opf='_labels_labelMA' ods='dummy' meanint='_probs_eed_sum16.h5/sum16_eed'
scriptfile=$( conncomp 'h' '2Dfilter' $dataset $ipf $ids $opf $ods $meanint )
[[ -z $jid ]] && dep='' || dep="-j $jid"
jid=$( fsl_sub -q long.q $dep $scriptfile )
# TODO: train and apply full size classifier

### label ICS components in 2D
jid=''  # TODO: link to *ds7_masks_maskMM and *ds7_probs_eed_sum16.h5/sum16_eed
declare ipf='_masks_maskMM' ids='maskMM' opf='_labels_labelMA_core2D' ods='labelMA_core2D' meanint='dummy'
scriptfile=$( conncomp 'h' '2D' $dataset_ds $ipf $ids $opf $ods $meanint )
[[ -z $jid ]] && dep='' || dep="-j $jid"
jid=$( fsl_sub -q short.q $dep $scriptfile )
### map all label properties to numpy vectors
declare ipf='_labels_labelMA_core2D' ids='labelMA_core2D' opf='_labels_labelMA' ods='dummy' meanint='_probs_eed_sum16.h5/sum16_eed'
scriptfile=$( conncomp 'h' '2Dfilter' $dataset_ds $ipf $ids $opf $ods $meanint )
[[ -z $jid ]] && dep='' || dep="-j $jid"
jid=$( fsl_sub -q short.q $dep $scriptfile )
### map the properties to volumes for inspection
declare ipf='_labels_labelMA_core2D' ids='labelMA_core2D' opf='_labels_labelMA' ods='dummy' meanint='dummy'
scriptfile=$( conncomp 'h' '2Dprops' $dataset_ds $ipf $ids $opf $ods $meanint )
[[ -z $jid ]] && dep='' || dep="-j $jid"
jid=$( fsl_sub -q veryshort.q $dep $scriptfile )
### apply 2D label classifier to the labels to identify MA labels
declare ipf='_labels_labelMA_core2D' ids='labelMA_core2D' opf='_labels_labelMA' ods='labelMA_pred' meanint='dummy' args='-A 3000 -I 0.8' clfpath="$scriptdir/clf.pkl" scalerpath="$scriptdir/scaler.pkl"
scriptfile=$( conncomp 'h' 'test' $dataset_ds $ipf $ids $opf $ods $meanint $clfpath $scalerpath $args )
[[ -z $jid ]] && dep='' || dep="-j $jid"
jid=$( fsl_sub -q long.q $dep $scriptfile )
### aggregate 2D labels to 3D labels
declare ipf='_labels_labelMA' ids='labelMA_pred' opf='_labels_labelMA' ods='labelMA_3Dlabeled' meanint='dummy'
scriptfile=$( conncomp 'h' '2Dto3D' $dataset_ds $ipf $ids $opf $ods $meanint )
[[ -z $jid ]] && dep='' || dep="-j $jid"
jid=$( fsl_sub -q veryshort.q $dep $scriptfile )

# TODO: might want to keep all this in opf='_labels_labelMA_core2D'


python $scriptdir/wmem/remap_labels.py \
$datadir/${dataset_ds}_labels_labelMA_core2D.h5/labelMA_pred_nocore3D \
$datadir/${dataset_ds}_labels_labelMA_core2D.h5/labelMA_pred_nocore3D_proofread \
-d $datadir/${dataset_ds}_labels_labelMA_core2D_labelMA_pred_nocore3D_delete.txt
h52nii '' $dataset_ds '_labels_labelMA_core2D' 'labelMA_pred_nocore3D_proofread' '' '' '-i zyx -o xyz -d uint32'


### aggregate 2D labels to 3D labels (merge_slicelabels alternative)
q=2; o=0.50
mpiexec -n 6 python $scriptdir/wmem/merge_slicelabels.py -S \
$datadir/${dataset_ds}_labels_labelMA_core2D_test.h5/labelMA_pred_nocore3D \
$datadir/${dataset_ds}_labels_labelMA_core2D_test.h5/labelMA_pred_nocore3D_2Dmerge_q${q}-o${o} \
-M -m 'MAstitch' -d 0 \
-q $q -o $o -p

h52nii '' $dataset_ds '_labels_labelMA_core2D_test' "labelMA_pred_nocore3D_2Dmerge_q${q}-o${o}" '' '' '-i zyx -o xyz -d uint16'
h52nii '' $dataset_ds '_labels_labelMA_core2D_test' "labelMA_pred_nocore3D_2Dmerge_q${q}-o${o}_steps/stitched" '' '' '-i zyx -o xyz -d uint16'



###=========================================================================###
### PROOFREADING / BRUSHING UP myelinated axon compartment
###=========================================================================###

### easy 3D labeling
python $scriptdir/wmem/connected_components.py \
$datadir/${dataset_ds}_masks_maskMM.h5/maskMM \
$datadir/${dataset_ds}_labels_labelMA_core3D.h5/labelMA_core3D \
-m '3D' -q 5000 -a 500
h52nii '' $dataset_ds '_labels_labelMA_core3D' 'labelMA_core3D' '' '' '-i zyx -o xyz -d uint16'
python $scriptdir/wmem/remap_labels.py \
$datadir/${dataset_ds}_labels_labelMA_core3D.h5/labelMA_core3D \
$datadir/${dataset_ds}_labels_labelMA_core3D.h5/labelMA_core3D_proofread \
-d $datadir/${dataset_ds}_labels_labelMA_core3D_delete.txt
h52nii '' $dataset_ds '_labels_labelMA_core3D' 'labelMA_core3D_proofread' '' '' '-i zyx -o xyz -d uint16'
# # split in the labels that do and don't traverse the volume
python -W ignore $scriptdir/wmem/nodes_of_ranvier.py -S \
$datadir/${dataset_ds}_labels_labelMA_core3D.h5/labelMA_core3D \
$datadir/${dataset_ds}_labels_labelMA_core3D_NoR.h5/labelMA_core3D_NoR \
--boundarymask $datadir/${dataset_ds}_masks_maskDS.h5/maskDS
h52nii '' $dataset_ds '_labels_labelMA_core3D_NoR' "labelMA_core3D_NoR_steps/boundarymask" '' '' '-i zyx -o xyz'
h52nii '' $dataset_ds '_labels_labelMA_core3D_NoR' "labelMA_core3D_NoR_steps/labels_tv" '' '' '-i zyx -o xyz -d uint16'
h52nii '' $dataset_ds '_labels_labelMA_core3D_NoR' "labelMA_core3D_NoR_steps/labels_nt" '' '' '-i zyx -o xyz -d uint16'




### masking 3Dlabeled pred with core3D
# TODO: robust handle mapping of relabeling, for when proofreading requires going back to 2Dcore labels
import os
from wmem import utils
import numpy as np
from skimage.segmentation import relabel_sequential
### replace labels in 'labelMA_3Dlabeled_filtered' with overlapping labels in 'labelMA_core3D_NoR_steps/labels_nt'

datadir = '/Users/michielk/oxdata/P01/EM/Myrf_01/SET-B/B-NT-S10-2f_ROI_00'

h5_fname = 'B-NT-S10-2f_ROI_00ds7_labels_labelMA_core3D.h5'
h5_dset = 'labelMA_core3D_proofread'
h5path_in1 = os.path.join(datadir, h5_fname, h5_dset)
h5file_in1, ds_in1, el, al = utils.h5_load(h5path_in1)

#h5_fname = 'B-NT-S10-2f_ROI_00ds7_labels_labelMA.h5'
#h5_dset = 'labelMA_3Dlabeled'
#h5path_in2 = os.path.join(datadir, h5_fname, h5_dset)
#h5file_in2, ds_in2, el, al = utils.h5_load(h5path_in2)

#h5_fname = 'B-NT-S10-2f_ROI_00ds7_labels_labelMA_3Dlabeled_filtered.h5'
#h5_dset = 'labelMA_3Dlabeled_filtered'
#h5path_out1 = os.path.join(datadir, h5_fname, h5_dset)
#h5file_out1, ds_out1 = utils.h5_write(None, ds_in2.shape, ds_in2.dtype, h5path_out1, element_size_um=el, axislabels=al)

#h5_dset = 'labelMA_3Dlabeled_filtered_relabeled'
#h5path_out2 = os.path.join(datadir, h5_fname, h5_dset)
#h5file_out2, ds_out2 = utils.h5_write(None, ds_in2.shape, ds_in2.dtype, h5path_out2, element_size_um=el, axislabels=al)

h5_fname = 'B-NT-S10-2f_ROI_00ds7_labels_labelMA_core2D_test.h5'
h5_dset = 'labelMA_pred_nocore3D_2Dmerge_q2-o0.50'
h5path_in2 = os.path.join(datadir, h5_fname, h5_dset)
h5file_in2, ds_in2, el, al = utils.h5_load(h5path_in2)

h5_fname = 'B-NT-S10-2f_ROI_00ds7_labels_labelMA_core2D_test.h5'
h5_dset = 'labelMA_pred_nocore3D_2Dmerge_q2-o0.50_relabeled'
h5path_out2 = os.path.join(datadir, h5_fname, h5_dset)
h5file_out2, ds_out2 = utils.h5_write(None, ds_in2.shape, ds_in2.dtype, h5path_out2, element_size_um=el, axislabels=al)

labels_3D = ds_in1[:]
labels_2D = ds_in2[:]

mask_3D = ds_in1[:].astype('bool')
maxlabel_3D = np.amax(np.unique(labels_3D))
labels_2D[mask_3D] = 0
#ds_out1[:] = labels_2D

relabels_2D = relabel_sequential(labels_2D, offset=maxlabel_3D+1)[0]
relabels_2D[mask_3D] = 0
ds_out2[:] = relabels_2D

h5file_in1.close()
h5file_in2.close()
h5file_out1.close()
h5file_out2.close()

h52nii '' $dataset_ds '_labels_labelMA_core2D_test' 'labelMA_pred_nocore3D_2Dmerge_q2-o0.50_relabeled' '' '' '-i zyx -o xyz -d uint16'

#h52nii '' $dataset_ds '_labels_labelMA_3Dlabeled_filtered' 'labelMA_3Dlabeled_filtered' '' '' '-i zyx -o xyz -d uint16'
#h52nii '' $dataset_ds '_labels_labelMA_3Dlabeled_filtered' 'labelMA_3Dlabeled_filtered_relabeled' '' '' '-i zyx -o xyz -d uint16'


### filtering labels on height

def filter_on_heigth(labels, min_height, ls_short=set([])):
    rp_nt = regionprops(labels)
    for prop in rp_nt:
        if prop.bbox[3]-prop.bbox[0] <= min_height:
            ls_short |= set([prop.label])
    print('number of short labels: {}'.format(len(ls_short)))
    return ls_short

height = 10

import os
from wmem import utils
import numpy as np
from skimage.segmentation import relabel_sequential
from skimage.measure import label, regionprops

datadir = '/Users/michielk/oxdata/P01/EM/Myrf_01/SET-B/B-NT-S10-2f_ROI_00'

h5_fname = 'B-NT-S10-2f_ROI_00ds7_labels_labelMA_core3D.h5'
h5_dset = 'labelMA_core3D_proofread'
h5path_in1 = os.path.join(datadir, h5_fname, h5_dset)
h5file_in1, ds_in1, el, al = utils.h5_load(h5path_in1)

h5_fname = 'B-NT-S10-2f_ROI_00ds7_labels_labelMA_3Dlabeled_filtered.h5'
h5_dset = 'labelMA_3Dlabeled_filtered_relabeled'
h5path_in2 = os.path.join(datadir, h5_fname, h5_dset)
h5file_in2, ds_in2, el, al = utils.h5_load(h5path_in2)

h5_dset = 'labelMA_pred_nocore3D_2Dmerge_q2-o0.50_relabeled_sizefilters_h{:02d}'.format(height)
#h5_dset = 'labelMA_3Dlabeled_filtered_relabeled_sizefilters_h{:02d}'.format(height)
h5path_out = os.path.join(datadir, h5_fname, h5_dset)
h5file_out, ds_out = utils.h5_write(None, ds_in2.shape, ds_in2.dtype, h5path_out, element_size_um=el, axislabels=al)

mask_3D = ds_in1[:].astype('bool')

h5root = h5file_in2.filename.split('.h5')[0]
lsroot = '{}_{}'.format(h5root, ds_out.name[1:])

labels = ds_in2[:]
labels[mask_3D] = 0  # should already be 0, (or: previous step can be skipped but implement the relabeling?)
ulabels = np.unique(labels)
maxlabel = np.amax(ulabels)
labelset = set(ulabels)
print("number of labels in labelvolume: {}".format(len(labelset)))

ls_short = filter_on_heigth(labels, height)
utils.write_labelsets({0: ls_short}, lsroot, ['txt', 'pickle'])

labelset -= ls_short
fw = np.zeros(maxlabel + 1, dtype='i')
for l in labelset:
    fw[l] = l

ds_out[:] = fw[labels]
h5file_out.close()

h5path_out = os.path.join(datadir, h5_fname, h5_dset + '_steps/short')
h5file_out, ds_out = utils.h5_write(None, ds_in2.shape, ds_in2.dtype, h5path_out, element_size_um=el, axislabels=al)
fw = np.zeros(maxlabel + 1, dtype='i')
for l in ls_short:
    fw[l] = l

ds_out[:] = fw[labels]

h5file_in1.close()
h5file_in2.close()
h5file_out.close()

h52nii '' $dataset_ds '_labels_labelMA_core2D_test' 'labelMA_pred_nocore3D_2Dmerge_q2-o0.50_relabeled_sizefilters_h10' '' '' '-i zyx -o xyz -d uint16'
h52nii '' $dataset_ds '_labels_labelMA_core2D_test' 'labelMA_pred_nocore3D_2Dmerge_q2-o0.50_relabeled_sizefilters_h10_steps/short' '' '' '-i zyx -o xyz -d uint16'

h52nii '' $dataset_ds '_labels_labelMA_3Dlabeled_filtered' 'labelMA_3Dlabeled_filtered_relabeled_sizefilters_h10' '' '' '-i zyx -o xyz -d uint16'
h52nii '' $dataset_ds '_labels_labelMA_3Dlabeled_filtered' 'labelMA_3Dlabeled_filtered_relabeled_sizefilters_h10_steps/short' '' '' '-i zyx -o xyz -d uint16'


### after proofreading
python $scriptdir/wmem/remap_labels.py \
$datadir/${dataset_ds}_labels_labelMA_3Dlabeled_filtered.h5/labelMA_3Dlabeled_filtered_relabeled_sizefilters_h10 \
$datadir/${dataset_ds}_labels_labelMA_3Dlabeled_filtered.h5/labelMA_3Dlabeled_filtered_relabeled_sizefilters_h10_proofread \
-d $datadir/${dataset_ds}_labels_labelMA_3Dlabeled_filtered_labelMA_3Dlabeled_filtered_relabeled_sizefilters_h10_delete.txt
h52nii '' $dataset_ds '_labels_labelMA_3Dlabeled_filtered' "labelMA_3Dlabeled_filtered_relabeled_sizefilters_h10_proofread" '' '' '-i zyx -o xyz -d uint16'










### remove outliers from 3Dlabeled
- size
- amount of overlap
- shape
- centre of mass


-- take out core3D_proofread first and --relabel
-- take out short
-- per label:
--- find all 2D labels
---- load B-NT-S10-2f_ROI_00ds7_labels_labelMA_3Dlabeled_filtered.h5/labelMA_3Dlabeled_filtered_relabeled_noshort as label3D
---- load B-NT-S10-2f_ROI_00ds7_labels_labelMA_core2D.h5/labelMA_core2D as label2D


source activate scikit-image-devel_0.13
export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8
python

import os
from wmem import utils
import numpy as np
from skimage.measure import regionprops

datadir = '/Users/michielk/oxdata/P01/EM/Myrf_01/SET-B/B-NT-S10-2f_ROI_00'

h5_fname = 'B-NT-S10-2f_ROI_00ds7_labels_labelMA_core3D.h5'
h5_dset = 'labelMA_core3D_proofread'
h5path_in0 = os.path.join(datadir, h5_fname, h5_dset)
h5file_in0, ds_in0, el, al = utils.h5_load(h5path_in0)
mask = ds_in0[:] > 0
h5file_in0.close()

h5_fname = 'B-NT-S10-2f_ROI_00ds7_labels_labelMA_core2D.h5'
h5_dset = 'labelMA_core2D'
h5path_in1 = os.path.join(datadir, h5_fname, h5_dset)
h5file_in1, ds_in1, el, al = utils.h5_load(h5path_in1)

h5_fname = 'B-NT-S10-2f_ROI_00ds7_labels_labelMA_3Dlabeled_filtered.h5'
h5_dset = 'labelMA_3Dlabeled_filtered_relabeled_noshort'
h5path_in2 = os.path.join(datadir, h5_fname, h5_dset)
h5file_in2, ds_in2, el, al = utils.h5_load(h5path_in2)

h5_fname = 'B-NT-S10-2f_ROI_00ds7_labels_labelMA.h5'
h5_dset = 'labelMA_pred'
h5path_in3 = os.path.join(datadir, h5_fname, h5_dset)
h5file_in3, ds_in3, el, al = utils.h5_load(h5path_in3)

label2D = ds_in1[:]
label3D = ds_in2[:]
pred = ds_in3[:]

label2D[mask] = 0
label3D[mask] = 0
pred[mask] = 0

h5_fname = 'B-NT-S10-2f_ROI_00ds7_labels_labelMA_core2D_test.h5'
h5_dset = 'labelMA_core2D_nocore3D'
h5path_out = os.path.join(datadir, h5_fname, h5_dset)
h5file_out, ds_out = utils.h5_write(None, ds_in1.shape, label2D.dtype, h5path_out, element_size_um=el, axislabels=al)
ds_out[:] = np.squeeze(label2D)
h5file_out.close()
h5_fname = 'B-NT-S10-2f_ROI_00ds7_labels_labelMA_core2D_test.h5'
h5_dset = 'labelMA_3Dlabeled_filtered_relabeled_noshort_nocore3D'
h5path_out = os.path.join(datadir, h5_fname, h5_dset)
h5file_out, ds_out = utils.h5_write(None, ds_in2.shape, label3D.dtype, h5path_out, element_size_um=el, axislabels=al)
ds_out[:] = np.squeeze(label3D)
h5file_out.close()
h5_fname = 'B-NT-S10-2f_ROI_00ds7_labels_labelMA_core2D_test.h5'
h5_dset = 'labelMA_pred_nocore3D'
h5path_out = os.path.join(datadir, h5_fname, h5_dset)
h5file_out, ds_out = utils.h5_write(None, ds_in3.shape, pred.dtype, h5path_out, element_size_um=el, axislabels=al)
ds_out[:] = np.squeeze(pred)
h5file_out.close()


#h52nii '' $dataset_ds '_labels_labelMA_core2D_test' 'labelMA_core2D_nocore3D' '' '' '-i zyx -o xyz -d uint32'
#h52nii '' $dataset_ds '_labels_labelMA_core2D_test' 'labelMA_3Dlabeled_filtered_relabeled_noshort_nocore3D' '' '' '-i zyx -o xyz -d uint32'
#h52nii '' $dataset_ds '_labels_labelMA_core2D_test' 'labelMA_pred_nocore3D' '' '' '-i zyx -o xyz -d uint32'


# find labelsets from the 2D labelvolume that are contained in the 3D labels
labelsets = {}
rp = regionprops(label3D, label2D)
for prop in rp:
    print(prop.label)
    labels2D = np.unique(prop.intensity_image[prop.image])
    labelsets[prop.label] = set(labels2D)

h5root = 'B-NT-S10-2f_ROI_00ds7_labels_labelMA_3Dlabeled_filtered'
lsroot = '{}_{}_{}'.format(h5root, h5_dset, '2Dlabels')
utils.write_labelsets(labelsets, lsroot, ['pickle'])


# Load the 2Dprops data.
nlabels = 794867
bname = os.path.join(datadir, 'B-NT-S10-2f_ROI_00ds7_labels_labelMA')
map_propnames = ['label', 'area', 'eccentricity', 'mean_intensity', 'solidity', 'extent', 'euler_number']
X_train = np.zeros([nlabels, len(map_propnames)])
for i, propname in enumerate(map_propnames):
    nppath = '{}_{}.npy'.format(bname, propname)
    X_train[:, i] = np.load(nppath)  # include the background label


# Create a forward map marking outlier labels
fw = np.zeros([nlabels, 1], dtype='bool')
#fw = np.zeros([nlabels, 1])
for lsk, lsv in labelsets.items():
    #print(lsk)
    idxs = []
    feats = np.zeros([len(lsv), len(map_propnames)])
    for i, slc in enumerate(sorted(lsv)):  # NOTE: sorted relies on increasing labelnr over slices; important for neighbour eval
        feats[i, :] = X_train[slc, :]
        #fw[slc] = feats[i, 1]
        #fw[slc] = True
    if lsk in set([4759, 20078, 16650, 4892, 7424, 12384]):
        areafeat = feats[:, 1]
        area_rel_rsn = np.divide(areafeat, np.roll(areafeat, -1))  # if large, current label is much bigger than right-side neighbour
        area_rel_rsn[-1] = 0 # idx_max meaningless
        area_rel_lsn = np.divide(areafeat[::-1], np.roll(areafeat[::-1], -1))   # if large, current label is much bigger than left-side neighbour
        area_rel_lsn[-1] = 0 # idx_max meaningless
        thr = 5
        thr_rsn = area_rel_rsn > thr
        thr_lsn = area_rel_lsn[::-1] > thr
        thresholded = thr_rsn | thr_lsn
        idxs = np.argwhere(thresholded)
        lsk
        pp.pprint(zip(areafeat, thresholded)), idxs
        #med = np.median(areafeat[:, 1])
        #idxs = np.argwhere(areafeat[:, 1] > 10 * med)  # this just marks the belly of a thickening axon
        if idxs.any():
            idxsflat = [idx for idx1 in idxs for idx in idx1]
            for idx in idxsflat:
                dellabel = feats[idx, 0]
                fw[dellabel] = True

mp = fw[ds_in1[:]]

h5_fname = 'B-NT-S10-2f_ROI_00ds7_labels_labelMA_core2D_test.h5'
h5_dset = 'labelMA_core2D_marked'
h5path_out = os.path.join(datadir, h5_fname, h5_dset)
h5file_out, ds_out = utils.h5_write(None, ds_in1.shape, mp.dtype, h5path_out, element_size_um=el, axislabels=al)
ds_out[:] = np.squeeze(mp)
h5file_out.close()

h5file_in1.close()
h5file_in2.close()

h52nii '' $dataset_ds '_labels_labelMA_core2D_test' 'labelMA_core2D_marked' '' '' '-i zyx -o xyz -d uint8'




np.delete(X_train, obj, axis=None)[source]


#mask = label3D != label
#labels2D = np.unique(label2D[mask])



--- look at 2D label properties
B-NT-S10-2f_ROI_00ds7_labels_labelMA_area.npy
...




















### merging interrupted labels  # TODO: make this one step with watershed merge
source activate scikit-image-devel_0.13
q=5; o=0.50
mpiexec -n 6 python $scriptdir/wmem/merge_slicelabels.py -S \
$datadir/${dataset_ds}_labels_labelMA_3Dlabeled_filtered.h5/labelMA_3Dlabeled_filtered_relabeled_sizefilters_h10_proofread \
$datadir/${dataset_ds}_labels_labelMA_3Dlabeled_filtered.h5/labelMA_3Dmerge_h10pf_q${q}-o${o} \
-M -m 'MAstitch' -d 0 \
-q $q -o $o -p

export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8
h52nii '' $dataset_ds '_labels_labelMA_3Dlabeled_filtered' "labelMA_3Dmerge_h10pf_q${q}-o${o}" '' '' '-i zyx -o xyz -d uint16'
h52nii '' $dataset_ds '_labels_labelMA_3Dlabeled_filtered' "labelMA_3Dmerge_h10pf_q${q}-o${o}_steps/stitched" '' '' '-i zyx -o xyz -d uint16'






# TODO: proofread stitched, make more conservative than -o 0.50, or skip step
# q=2; o=0.80  # NOTE: no merge results
q=3; o=0.80
q=5; o=0.50
mpiexec -n 6 python $scriptdir/wmem/merge_slicelabels.py -S \
$datadir/${dataset_ds}_labels_labelMA_3Dlabeled_filtered.h5/labelMA_3Dlabeled_filtered_relabeled \
$datadir/${dataset_ds}_labels_labelMA_3Dlabeled_filtered.h5/labelMA_3Dmerge_q${q}-o${o} \
-M -m 'MAstitch' -d 0 \
-q $q -o $o -p
### mapping (TODO: can fill holes between labels as well)
# python $scriptdir/wmem/merge_slicelabels.py -S \
# $datadir/${dataset_ds}_labels_labelMA_3Dlabeled_filtered.h5/labelMA_3Dlabeled_filtered_relabeled \
# $datadir/${dataset_ds}_labels_labelMA_3Dlabeled_filtered.h5/labelMA_3Dmerge_q${q}-o${o} \
# --maskMM $datadir/${dataset_ds}_masks_maskMM.h5/maskMM \
# -m 'MAfwmap' -d 0
h52nii '' $dataset_ds '_labels_labelMA_3Dlabeled_filtered' "labelMA_3Dmerge_q${q}-o${o}" '' '' '-i zyx -o xyz -d uint16'
h52nii '' $dataset_ds '_labels_labelMA_3Dlabeled_filtered' "labelMA_3Dmerge_q${q}-o${o}_steps/stitched" '' '' '-i zyx -o xyz -d uint16'

python $scriptdir/wmem/remap_labels.py \
$datadir/${dataset_ds}_labels_labelMA_3Dlabeled_filtered.h5/labelMA_3Dmerge_q${q}-o${o} \
$datadir/${dataset_ds}_labels_labelMA_3Dlabeled_filtered.h5/labelMA_3Dmerge_q${q}-o${o}_proofread \
-d $datadir/${dataset_ds}_labels_labelMA_3Dlabeled_filtered_labelMA_3Dmerge_q3-o0.80_steps-stitched_delete.txt
python $scriptdir/wmem/remap_labels.py \
$datadir/${dataset_ds}_labels_labelMA_3Dlabeled_filtered.h5/labelMA_3Dmerge_q${q}-o${o}_steps/stitched \
$datadir/${dataset_ds}_labels_labelMA_3Dlabeled_filtered.h5/labelMA_3Dmerge_q${q}-o${o}_steps/stitched_proofread \
-d $datadir/${dataset_ds}_labels_labelMA_3Dlabeled_filtered_labelMA_3Dmerge_q3-o0.80_steps-stitched_delete.txt
h52nii '' $dataset_ds '_labels_labelMA_3Dlabeled_filtered' "labelMA_3Dmerge_q${q}-o${o}_proofread" '' '' '-i zyx -o xyz -d uint16'
h52nii '' $dataset_ds '_labels_labelMA_3Dlabeled_filtered' "labelMA_3Dmerge_q${q}-o${o}_steps/stitched_proofread" '' '' '-i zyx -o xyz -d uint16'




### merging interrupted labels
python -W ignore $scriptdir/wmem/nodes_of_ranvier.py -S \
$datadir/${dataset_ds}_labels_labelMA_3Dlabeled_filtered.h5/labelMA_3Dmerge_q${q}-o${o}_proofread \
$datadir/${dataset_ds}_labels_labelMA_3Dlabeled_filtered.h5/labelMA_3Dmerge_q${q}-o${o}_NoR \
-s 500 \
--boundarymask $datadir/${dataset_ds}_masks_maskDS.h5/maskDS \
-m 'watershed' -r 20 10 10 \
--data $datadir/${dataset_ds}_probs_eed_probMA.h5/probMA_eed  #--maskMM $datadir/${dataset_ds}_masks_maskMM.h5/maskMM
# -m 'watershed' -r 30 50 50 --data $datadir/$dataset_ds.h5/data --maskMM $datadir/${dataset_ds}_masks_maskMM.h5/maskMM

# NOTE: there are still quite a few small labels that should be included at -s 500
h52nii '' $dataset_ds '_labels_labelMA_3Dlabeled_filtered' "labelMA_3Dmerge_q${q}-o${o}_NoR_steps/smalllabelmask" '' '' '-i zyx -o xyz -d uint8'
h52nii '' $dataset_ds '_labels_labelMA_3Dlabeled_filtered' "labelMA_3Dmerge_q${q}-o${o}_NoR_steps/boundarymask" '' '' '-i zyx -o xyz'
h52nii '' $dataset_ds '_labels_labelMA_3Dlabeled_filtered' "labelMA_3Dmerge_q${q}-o${o}_NoR_steps/labels_tv" '' '' '-i zyx -o xyz -d uint16'
# 1170 2689 3791 (NOTE not these labels anymore) need edits; then (fill and) accept as labels
h52nii '' $dataset_ds '_labels_labelMA_3Dlabeled_filtered' "labelMA_3Dmerge_q${q}-o${o}_NoR_steps/largelabels" '' '' '-i zyx -o xyz -d uint16'
h52nii '' $dataset_ds '_labels_labelMA_3Dlabeled_filtered' "labelMA_3Dmerge_q${q}-o${o}_NoR_steps/labels_nt" '' '' '-i zyx -o xyz -d uint16'
h52nii '' $dataset_ds '_labels_labelMA_3Dlabeled_filtered' "labelMA_3Dmerge_q${q}-o${o}_NoR_steps/filled" '' '' '-i zyx -o xyz -d uint16'
h52nii '' $dataset_ds '_labels_labelMA_3Dlabeled_filtered' "labelMA_3Dmerge_q${q}-o${o}_NoR" '' '' '-i zyx -o xyz -d uint16'

# tv quick proofread (5 min; old): 15659 3059 2921 23093 4972 23794 9883 9557
# tv quick proofread (5 min): (21596 19475 9883) 4972 20130 1639 1401 3081 15659 119 23795 2600 11084 3059 4972


### agglomerate watershedMA




###=========================================================================###
### slicvoxels
###=========================================================================###
# NOTE: SLIC devel LOCAL on 'B-NT-S10-2f_ROI_01'
# NOTE: SLIC blocks_0500 DONE on ARC for datasets 'M3S1GNU' and 'B-NT-S10-2f_ROI_01'
# NOTE: these have not been merged or checked

declare l=9000 c=0.20 s=0.03
declare ipf='_probs' ids='volume/predictions' opf="_slic_slic4D_${l}_${c}_${s}" ods="slic4D_${l}_${c}_${s}"
scriptfile=$( slicvoxels 'h' 'a' $ipf $ids $opf $ods $l $c $s 6 )

jid=
[[ -z $jid ]] && dep='' || dep="--dependency=after:$jid"
for i in `seq -f "%03g" 0 56`; do
scriptfile=EM_sb_slic_slic4D_9000_0.20_0.03_$i.sh
jid=$( sbatch $dep $scriptfile ) && jid=${jid##* }
done

jid=
[[ -z $jid ]] && dep='' || dep="--dependency=after:$jid"
for i in `seq -f "%03g" 0 50`; do
scriptfile=EM_sb_slic_slic4D_9000_0.20_0.03_$i.sh
jid=$( sbatch -p devel $dep $scriptfile ) && jid=${jid##* }
done
for i in `seq 1644139 1644189`; do
scontrol update jobid=$i partition=devel MinMemoryCPU=60000 TimeLimit=00:10:00
done


###=========================================================================###
### TODO: refine maskMM
###=========================================================================###


###=========================================================================###
### TODO: separate sheaths maskMM
###=========================================================================###


###=========================================================================###
### TODO: create maskMT
###=========================================================================###


###=========================================================================###
### TODO: watershed outside maskMM
###=========================================================================###


###=========================================================================###
### TODO: PROOFREADING myelinated axon compartment
###=========================================================================###
### agglomerate watershedMA


###=========================================================================###
### TODO: Neuroproof training
###=========================================================================###


###=========================================================================###
### TODO: Neuroproof agglomeration
###=========================================================================###
