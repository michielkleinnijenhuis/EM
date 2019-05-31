###=========================================================================###
### prepare environment
###=========================================================================###
scriptdir=$HOME/workspace/EM
source $scriptdir/pipelines/datasets.sh
source $scriptdir/pipelines/functions.sh
source $scriptdir/pipelines/submission.sh
export PYTHONPATH=$PYTHONPATH:$HOME/workspace/pyDM3reader  # FIXME
export PYTHONPATH=$PYTHONPATH:$HOME/workspace/maskSLIC  # FIXME

#compute_env='JAL'
#compute_env='ARC'
#compute_env='ARCB'
# compute_env='LOCAL'
compute_env='RIOS013'
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
# FIXME: a new maskMM was introduced (inserted from scratch_EM6.py and scratch_EM9.py)
###=========================================================================###
### generate the raw myelin mask
declare ipf='' ids='data' opf='_masks_maskMM' ods='maskMM' slab=12 arg='-g -l -1 -u 26000 -d 1'
scriptfile=$( prob2mask 'h' '' $ids $opf $ods $slab $arg )
. $scriptfile
### remove small objects from mask
ipf='_masks_maskMM' ids='maskMM' opf=$ipf ods="${ids}_PP"
scriptfile="$datadir/EM_maskMM_PP.py"
echo "from wmem import prob2mask" > $scriptfile
echo "prob2mask.preprocess_mask(image_in='$datadir/$dataset_ds$ipf.h5/$ids', outputpath='$datadir/$dataset_ds$opf.h5/$ods', min_size=1000)" >> $scriptfile
python -W ignore $scriptfile


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

# TODO: might want to keep all this in opf='_labels_labelMA_core2D'; not in '_labels_labelMA'


###=========================================================================###
### 3D labeling
###=========================================================================###
core3D='_labels_labelMA_core3D_test'

### easy 3D labeling
ipf='_masks_maskMM'
ids='maskMM'
opf="$core3D"
ods='labelMA_core3D'
args='-q 5000 -a 500'
scriptfile=$( conncomp_3D 'h' $dataset_ds $ipf $ids $opf $ods $args )
. "$scriptfile"

### split in the labels that do (tv) and don't traverse the volume (nt)
ipf="$core3D"
ids='labelMA_core3D'
opf="$ipf"
ods="${ids}_NoR"
args="--boundarymask $datadir/${dataset_ds}_masks_maskDS.h5/maskDS"
scriptfile=$( NoR 'h' $dataset_ds $ipf $ids $opf $ods $args )
. "$scriptfile"

### delete labels from labelMA_core3D through manual proofreading
# NOTE: PROOFREADING done through labels_nt
ipf="$core3D"
ids='labelMA_core3D'
opf="$ipf"
ods="${ids}_proofread"
args="-d $datadir/${dataset_ds}${ipf}_${ids}_delete.txt"
scriptfile=$( remap 'h' $dataset_ds $ipf $ids $opf $ods $args )
. "$scriptfile"

### split the proofread volume in labels through-volume (tv) / in-volume (nt)
ipf="$core3D"
ids='labelMA_core3D_proofread'
opf="$ipf"
ods="${ids}_NoR"
args="--boundarymask $datadir/${dataset_ds}_masks_maskDS.h5/maskDS"
scriptfile=$( NoR 'h' $dataset_ds $ipf $ids $opf $ods $args )
. "$scriptfile"

### proofreading the through-volume labels
# NOTE: labels_tv and labels_nt not yet correct: labels_nt has many that are actually traversing
ipf="$core3D"
ids='labelMA_core3D_proofread'
opf="$ipf"
ods="${ids}_NoR"
scriptfile="$datadir/EM_corrNoR.py"
echo "from wmem import nodes_of_ranvier" > $scriptfile
echo "nodes_of_ranvier.correct_NoR(image_in='$datadir/$dataset_ds$ipf.h5/$ids')" >> $scriptfile
python -W ignore $scriptfile

# TODO: improve nodes_of_ranvier.py module for splitting tv/nt such that this proofreading is unnecessary


###=========================================================================###
### 2D labeling postprocessing
###=========================================================================###
core2D='_labels_labelMA_core2D_test'

### mask the predicted 2D labels with finished 3D labels
ipf1='_labels_labelMA'
ids1='labelMA_pred'
ipf2='_labels_labelMA_core3D'
ids2='labelMA_core3D_proofread'
opf="$core2D"
ods='labelMA_pred_nocore3D'
args='-m mask'
scriptfile=$( combine_labels 'h' $dataset_ds $args )
. "$scriptfile"

### delete 2D labels from volume
ipf="$core2D"
ids='labelMA_pred_nocore3D'
opf="$ipf"
ods="$ids"
python $scriptdir/wmem/stack2stack.py \
    "$datadir/${dataset_ds}${ipf}_${ids}_nii_delete.nii.gz" \
    "$datadir/${dataset_ds}${opf}.h5/${ods}_nii_delete" -i xyz -o zyx -d uint8
scriptfile="$datadir/EM_delete_nii.py"
echo "from wmem import remap_labels" > $scriptfile
echo "remap_labels.delete_nii(image_in='$datadir/$dataset_ds$ipf.h5/$ids')" >> $scriptfile
python -W ignore $scriptfile
# NOTE: volume *_delete_nii.nii.gz not on RIOS013; only in <>_labels_labelMA_core2D.h5

### proofread the 2D labels
ipf="$core2D"
ids='labelMA_pred_nocore3D'
opf="$ipf"
ods="${ids}_proofread"
args="-d $datadir/${dataset_ds}${ipf}_${ids}_delete.txt $datadir/${dataset_ds}${ipf}_${ids}_nii_delete.txt"
scriptfile=$( remap 'h' $dataset_ds $ipf $ids $opf $ods $args )
. "$scriptfile"

### aggregate 2D labels to 3D labels (merge_slicelabels alternative)
# TODO: might go for conservative settings of q/o and proofread from there (but with well-proofread 2D labels and q=2 we can relax the o=0.50)
q=2; o=0.50;
ipf="$core2D"
ids='labelMA_pred_nocore3D_proofread'
opf="$ipf"
ods="${ids}_2Dmerge_q${q}-o${o}"
args="-M -m 'MAstitch' -d 0 -q $q -o $o"
scriptfile=$( merge_slicelabels_mpi 'h' $dataset_ds $args )
. "$scriptfile"

### map the aggregated 2D labels
ipf="$core2D"
ids='labelMA_pred_nocore3D_proofread'
opf="$ipf"
ods="${ids}_2Dmerge_q${q}-o${o}"
args="-m MAfwmap"
scriptfile=$( merge_slicelabels 'h' $dataset_ds $args )
. "$scriptfile"

### fill gaps in aggregated labels
ipf="$core2D"
ids="labelMA_pred_nocore3D_proofread_2Dmerge_q${q}-o${o}"
opf="$ipf"
ods="${ids}_closed"
args="--maskMM $datadir/${dataset_ds}_masks_maskMM.h5/maskMM_PP -m MAfilter -l 1 0 0 -S"
scriptfile=$( merge_slicelabels 'h' $dataset_ds $args )
. "$scriptfile"

### aggregate labels by overlap (threshold_overlap=0.20; offsets=2)
ipf1="$core2D"
ids="labelMA_pred_nocore3D_proofread_2Dmerge_q${q}-o${o}_closed"
opf="$ipf"
ods="${ids}_ns"
args="-m 'neighbours_slices' -q 2 -o 0.20 -M -S"
scriptfile=$( merge_labels_ws 'h' $dataset_ds $ipf $ids $opf $ods $args )
. "$scriptfile"

### merge aggregated 2D labels with core3D_nt
ipf1="$core2D"
ids1="labelMA_pred_nocore3D_proofread_2Dmerge_q${q}-o${o}_closed_ns"
ipf2="$core3D"
ids2='labelMA_core3D_proofread_NoR_steps/labels_nt'
opf='_labels_labelMA_comb_test'
ods='labelMA_nt'
args='-m add'
scriptfile=$( combine_labels 'h' $dataset_ds $args )
. "$scriptfile"


























###=========================================================================###
### TODO: figure out if and/or where to insert _probs_probMA 3Dlabels
###=========================================================================###

### threshold probability image

# TODO: determine best threshold
# thr=0.5
# ipf='_probs_eed_probMA' ids='probMA_eed'
# opf='_masks_maskMA' ods="probMA_eed_thr${thr}"
# python -W ignore $scriptdir/wmem/prob2mask.py \
# "$datadir/${dataset_ds}${ipf}.h5/${ids}" \
# "$datadir/${dataset_ds}${opf}.h5/${ods}" \
# -l $thr -u 2
# h52nii '' "$dataset_ds" "${opf}" "${ods}" '' '' '-i zyx -o xyz -d uint8'
# ### label the mask  # FIXME: this would expect a myelin mask not an MA mask
# ipf='_masks_maskMA' ids="probMA_eed_thr${thr}"
# opf='_labels_labelMA_core3D' ods="probMA_eed_thr${thr}_labeled"
# python $scriptdir/wmem/connected_components.py \
# $datadir/${dataset_ds}${ipf}.h5/${ids} \
# $datadir/${dataset_ds}${opf}.h5/${ods} \
# -m '3D' -q 5000 -a 500
# h52nii '' "${dataset_ds}" "${opf}" "${ods}" '' '' '-i zyx -o xyz -d uint16'

def label_probMA_eed():
    import os
    import numpy as np
    #from scipy.ndimage import label
    from skimage.measure import label
    from skimage.segmentation import relabel_sequential
    from wmem import utils, LabelImage, prob2mask
    relabel_from = 1604
    datadir = '/Users/michielk/oxdata/P01/EM/Myrf_01/SET-B/B-NT-S10-2f_ROI_00'
    dataset = 'B-NT-S10-2f_ROI_00ds7'
    ipf, ids ='_probs_eed_probMA', 'probMA_eed'
    opf, ods = '_labels_labelMA_core3D', 'probMA_eed_thr0.5_labeled'
    image_in = os.path.join(datadir, '{}{}.h5/{}'.format(dataset, ipf, ids))
    im = prob2mask.prob2mask(image_in, lower_threshold=0.5, upper_threshold=2)
    outputpath = os.path.join(datadir, '{}{}.h5/{}'.format(dataset, opf, ods))
    mo = LabelImage(outputpath, **im.get_props(dtype='uint32'))
    mo.create()
    # mo.write(label(im.ds)[0])  # scipy
    mo.write(relabel_sequential(label(im.ds), relabel_from)[0])  # scikit image
    im.close()
    mo.close()

opf='_labels_labelMA_core3D' ods="probMA_eed_thr0.5_labeled"
h52nii '' "$dataset_ds" "${opf}" "${ods}" '' '' '-i zyx -o xyz -d uint16'




###=========================================================================###
### 2D labeling postprocessing
###=========================================================================###

# FIXME 1?: how did I get to labelMA_pred_nocore3D (were 2D labels relabeled from maxlabel_3D+1?); see scratchEM8.py: PROBABLY NOT!
# FIXME 2: probably not used the proofread version of '_labels_labelMA_core3D.h5/labelMA_core3D'; see scratchEM8.py
# FIX 2 DONE

### mask the predicted 2D labels with finished 3D labels
ipf1='_labels_labelMA' ids1='labelMA_pred'
ipf2='_labels_labelMA_core3D' ids2='labelMA_core3D_proofread'
opf='_labels_labelMA_core2D_testmask' ods='labelMA_pred_nocore3D'
python $scriptdir/wmem/combine_labels.py \
    "$datadir/${dataset_ds}${ipf1}.h5/${ids1}" \
    "$datadir/${dataset_ds}${ipf2}.h5/${ids2}" \
    "$datadir/${dataset_ds}${opf}.h5/${ods}" -m 'mask'
opf='_labels_labelMA_core2D_testmask' ods='labelMA_pred_nocore3D'
h52nii '' "${dataset_ds}" "${opf}" "${ods}" '' '' '-i zyx -o xyz -d uint16'

# fslmaths B-NT-S10-2f_ROI_00ds7_labels_labelMA_core2D_labelMA_pred_nocore3D_nii_delete.nii.gz -add B-NT-S10-2f_ROI_00ds7_labels_labelMA_core2D_labelMA_pred_nocore3D_delete_OLDNAME.nii.gz -bin B-NT-S10-2f_ROI_00ds7_labels_labelMA_core2D_labelMA_pred_nocore3D_nii_delete_new.nii.gz

ipf='_labels_labelMA_core2D' ids='labelMA_pred_nocore3D_nii_delete'
opf='_labels_labelMA_core2D' ods='labelMA_pred_nocore3D_nii_delete'
python $scriptdir/wmem/stack2stack.py \
    "$datadir/${dataset_ds}${ipf}_${ids}.nii.gz" \
    "$datadir/${dataset_ds}${opf}.h5/${ods}" -i xyz -o zyx -d uint8

def delete_nii():
    import os
    from wmem import utils, MaskImage, LabelImage
    import numpy as np
    datadir = '/Users/michielk/oxdata/P01/EM/Myrf_01/SET-B/B-NT-S10-2f_ROI_00'
    h5_fname = 'B-NT-S10-2f_ROI_00ds7_labels_labelMA_core2D.h5'
    h5_dset = 'labelMA_pred_nocore3D'
    h5_path = os.path.join(datadir, h5_fname, h5_dset)
    pred2D = LabelImage(h5_path)
    pred2D.load()
    nii_fname = 'B-NT-S10-2f_ROI_00ds7_labels_labelMA_core2D'
    nii_dset = 'labelMA_pred_nocore3D_nii_delete'
    nii_path = os.path.join(datadir, '{}_{}.nii.gz'.format(nii_fname, nii_dset))
    mask = MaskImage(nii_path)
    mask.load()

    m = np.transpose(mask.ds[:].astype('bool'))
    labs = pred2D.ds[:]
    labs_masked = labs[m]
    labels = np.unique(labs_masked)
    labelsets = {0: set(list(labels))}

    comps = pred2D.split_path()
    lsroot = '{}_{}_nii_delete'.format(comps['base'], comps['dset'])
    utils.write_labelsets(labelsets, lsroot, ['pickle', 'txt'])

    pred2D.close()
    mask.close()

### proofread the 2D labels
ipf='_labels_labelMA_core2D' ids='labelMA_pred_nocore3D'
opf='_labels_labelMA_core2D' ods='labelMA_pred_nocore3D_proofread'
python $scriptdir/wmem/remap_labels.py \
$datadir/${dataset_ds}${ipf}.h5/${ids} \
$datadir/${dataset_ds}${opf}.h5/${ods} \
-d $datadir/${dataset_ds}${ipf}_${ids}_delete.txt $datadir/${dataset_ds}${ipf}_${ids}_nii_delete.txt
h52nii '' "${dataset_ds}" "${opf}" "${ods}" '' '' '-i zyx -o xyz -d uint16'

# ### select the 2D labels overlapping with probMA_labeled and map them
# NOTE: many useful 2D labels excluded; first aggregate (conservatively)
# r=0.01
# ipf='_labels_labelMA_core3D' ids="probMA_eed_thr0.5_labeled"
# spf='_labels_labelMA_core2D' sds='labelMA_pred_nocore3D_proofread'
# opf='_labels_labelMA_core2D_test' ods="${lds}_${ids}"
# mpiexec -n 4 python $scriptdir/wmem/agglo_from_labelmask.py \
# "${dataset_ds}${ipf}.h5/${ids}" \
# "${dataset_ds}${spf}.h5/${sds}" \
# "${dataset_ds}${opf}.h5/${ods}" \
# -r $r -M
# h52nii '' "${dataset_ds}" "${opf}" "${ods}" '' '' '-i zyx -o xyz -d uint16'


### aggregate 2D labels to 3D labels (merge_slicelabels alternative)
# TODO: might go for conservative settings of q/o and proofread from there
# (but with well-proofread 2D labels and q=2 we can relax the o=0.50)
q=2; o=0.50
ipf='_labels_labelMA_core2D' ids='labelMA_pred_nocore3D_proofread'
opf='_labels_labelMA_core2D' ods="labelMA_pred_nocore3D_proofread_2Dmerge_q${q}-o${o}"
mpiexec -n 6 python $scriptdir/wmem/merge_slicelabels.py \
$datadir/${dataset_ds}${ipf}.h5/${ids} \
$datadir/${dataset_ds}${opf}.h5/${ods} \
-M -m 'MAstitch' -d 0 -q $q -o $o
### map the aggregated 2D labels (FIXME: check if can I simply use the -p flag above (, but it can be zipped and chunked when not using MPI))
ipf='_labels_labelMA_core2D' ids='labelMA_pred_nocore3D_proofread'
opf='_labels_labelMA_core2D' ods="labelMA_pred_nocore3D_proofread_2Dmerge_q${q}-o${o}"
python $scriptdir/wmem/merge_slicelabels.py \
$datadir/${dataset_ds}${ipf}.h5/${ids} \
$datadir/${dataset_ds}${opf}.h5/${ods} \
-m 'MAfwmap'
h52nii '' "${dataset_ds}" "${opf}" "${ods}" '' '' '-i zyx -o xyz -d uint16'
# h52nii '' "${dataset_ds}" "${opf}" "${ods}_steps/stitched" '' '' '-i zyx -o xyz -d uint16'

### fill gaps in aggregated labels
ipf='_labels_labelMA_core2D' ids="labelMA_pred_nocore3D_proofread_2Dmerge_q${q}-o${o}"
opf='_labels_labelMA_core2D' ods="labelMA_pred_nocore3D_proofread_2Dmerge_q${q}-o${o}_closed"
python $scriptdir/wmem/merge_slicelabels.py -S \
$datadir/${dataset_ds}${ipf}.h5/${ids} \
$datadir/${dataset_ds}${opf}.h5/${ods} \
--maskMM $datadir/${dataset_ds}_masks_maskMM.h5/maskMM_PP \
-m 'MAfilter' -l 1 0 0  # 3 1 1 # 6 1 1
h52nii '' "${dataset_ds}" "${opf}" "${ods}" '' '' '-i zyx -o xyz -d uint16'
### aggregate labels by overlap (threshold_overlap=0.20; offsets=2) TODO
ipf='_labels_labelMA_core2D' ids="labelMA_pred_nocore3D_proofread_2Dmerge_q${q}-o${o}_closed"
opf='_labels_labelMA_core2D' ods="labelMA_pred_nocore3D_proofread_2Dmerge_q${q}-o${o}_closed_ns"
mpiexec -n 6 python -W ignore $scriptdir/wmem/merge_labels.py -M \
$datadir/${dataset_ds}${ipf}.h5/${ids} \
$datadir/${dataset_ds}${opf}.h5/${ods} \
-m 'neighbours_slices' -q 2 -o 0.20
h52nii '' "${dataset_ds}" "${opf}" "${ods}" '' '' '-i zyx -o xyz -d uint16'


### relabel the aggregated 2D labels starting at the maxlabel of labelMA_core3D_proofread
# NOTE: no need to relabel when combined with probMA_labeled
# maxlabel_core3D=1604
# ipf='_labels_labelMA_core2D' ids="labelMA_pred_nocore3D_proofread_2Dmerge_q${q}-o${o}"
# opf='_labels_labelMA_core2D' ods="labelMA_pred_nocore3D_proofread_2Dmerge_q${q}-o${o}_relabeled"
# python $scriptdir/wmem/merge_slicelabels.py -S \
# $datadir/${dataset_ds}${ipf}.h5/${ids} \
# $datadir/${dataset_ds}${opf}.h5/${ods} \
# -M -m 'MAfilter' -r $maxlabel_core3D
# h52nii '' "${dataset_ds}" "${opf}" "${ods}" '' '' '-i zyx -o xyz -d uint16'


### fill (doesn't seem relevant here)
# ipf='_labels_labelMA_comb' ids="labelMA_nt_aux3D"
# opf='_labels_labelMA_comb' ods="labelMA_nt_aux3D_filledm2"
# methods='2'
# python $scriptdir/wmem/fill_holes.py -S \
# $datadir/${dataset_ds}${ipf}.h5/${ids} \
# $datadir/${dataset_ds}${opf}.h5/${ods} \
# -m ${methods} -s 9 9 9
# h52nii '' "${dataset_ds}" "${opf}" "${ods}" '' '' '-i zyx -o xyz -d uint16'
### TODO: aggregate labels by overlap (threshold_overlap=0.80; offsets=3)
# ipf='_labels_labelMA_comb' ids="labelMA_nt_aux3D"
# opf='_labels_labelMA_comb' ods="labelMA_nt_aux3D_ns"
# mpiexec -n 6 python -W ignore $scriptdir/wmem/merge_labels.py -M \
# $datadir/${dataset_ds}${ipf}.h5/${ids} \
# $datadir/${dataset_ds}${opf}.h5/${ods} \
# -m 'neighbours_slices' -q 3 -o 0.80
# h52nii '' "${dataset_ds}" "${opf}" "${ods}" '' '' '-i zyx -o xyz -d uint16'

###=========================================================================###
### 2D+3D in-volume labels
###=========================================================================###

### TODO: first merge with core3D_nt??
### merge aggregated 2D labels with core3D_nt
def merge_2D_with_3D():
    import os
    from wmem import utils, LabelImage
    import numpy as np
    datadir = '/Users/michielk/oxdata/P01/EM/Myrf_01/SET-B/B-NT-S10-2f_ROI_00'
    h5_fname = 'B-NT-S10-2f_ROI_00ds7_labels_labelMA_core3D.h5'
    h5_dset = 'labelMA_core3D_proofread_NoR_steps/labels_nt'
    h5_path = os.path.join(datadir, h5_fname, h5_dset)
    core3D_nt = LabelImage(h5_path)
    core3D_nt.load()
    h5_fname = 'B-NT-S10-2f_ROI_00ds7_labels_labelMA_core2D.h5'
    h5_dset = 'labelMA_pred_nocore3D_proofread_2Dmerge_q2-o0.50_closed_ns'
    h5_path = os.path.join(datadir, h5_fname, h5_dset)
    core2D_nt = LabelImage(h5_path)
    core2D_nt.load()
    h5_fname = 'B-NT-S10-2f_ROI_00ds7_labels_labelMA_comb.h5'
    h5_dset = 'labelMA_nt'
    h5_path = os.path.join(datadir, h5_fname, h5_dset)
    mo = LabelImage(h5_path, **core3D_nt.get_props())
    mo.create()
    mo.write(core3D_nt.ds[:] + core2D_nt.ds[:])
    mo.close()
    core3D_nt.close()
    core2D_nt.close()

opf='_labels_labelMA_comb' ods='labelMA_nt'
h52nii '' "${dataset_ds}" "${opf}" "${ods}" '' '' '-i zyx -o xyz -d uint16'


###############################################################################
### select the 3D labels overlapping with probMA_labeled and map them       ###
###############################################################################
r=0.00
ipf='_labels_labelMA_core3D' ids="probMA_eed_thr0.5_labeled"
spf='_labels_labelMA_comb' sds="labelMA_nt"
opf='_labels_labelMA_comb' ods="${sds}_${ids}"
# mpiexec -n 4
python $scriptdir/wmem/agglo_from_labelmask.py \
"${dataset_ds}${ipf}.h5/${ids}" \
"${dataset_ds}${spf}.h5/${sds}" \
"${dataset_ds}${opf}.h5/${ods}" \
-r $r # -M


def map_axons():
    import os
    from wmem import utils, LabelImage
    import numpy as np
    import glob
    import pickle
    datadir = '/Users/michielk/oxdata/P01/EM/Myrf_01/SET-B/B-NT-S10-2f_ROI_00'
    h5_fname = 'B-NT-S10-2f_ROI_00ds7_labels_labelMA_core3D.h5'
    h5_dset = 'probMA_eed_thr0.5_labeled'
    h5_path = os.path.join(datadir, h5_fname, h5_dset)
    axons = utils.get_image(h5_path, imtype='Label')
    h5_fname = 'B-NT-S10-2f_ROI_00ds7_labels_labelMA_comb.h5'
    h5_dset = 'labelMA_nt'
    h5_path = os.path.join(datadir, h5_fname, h5_dset)
    svoxs = utils.get_image(h5_path, imtype='Label')
    h5_fname = 'B-NT-S10-2f_ROI_00ds7_labels_labelMA_comb.h5'
    h5_dset = 'labelMA_nt_probMA_eed_thr0.5_labeled'
    outputpath = os.path.join(datadir, h5_fname, h5_dset)
    props = svoxs.get_props(protective=False)

    def invert_labelsets(labelsets):
        labelsets_inv = {}  # key_svox: {set_axon} (one-to-one mapping when labelsets_new has no doubles)
        for lsk in sorted(labelsets.iterkeys()):
            lsv = labelsets[lsk]
            for l in lsv:
                if l in labelsets_inv.keys():
                    labelsets_inv[l].add(lsk)
                else:
                    labelsets_inv[l] = set([lsk])
        return labelsets_inv

    # FIXME: handle doubles better (now mapped to lowest axon label)
    lsfile = os.path.join(datadir, 'B-NT-S10-2f_ROI_00ds7_labels_labelMA_comb_labelMA_nt_probMA_eed_thr0.5_labeled_host-MKMBPr.local_rank-00.pickle')
    labelsets = svoxs.read_labelsets(lsfile)  # key_axon: {set_svoxs} (contains double svoxs in values)
    labelsets_inv = invert_labelsets(labelsets)  # key_svox: {set_axons} (contains double axons in values)

    # axon mapping (any axons that has doubles needs to be mapped to the lowest-labeled)
    labelsets_inv_nodoubles = {}
    for lsk in sorted(labelsets_inv.iterkeys()):
        labelsets_inv_nodoubles = utils.classify_label_set(labelsets_inv_nodoubles, labelsets_inv[lsk], lsk)
        axonmapping = {}  # key_axon-first: {set_axons}
    for lsk, lsv in labelsets_inv_nodoubles.items():
        lsvlist = sorted(list(lsv))
        axonmapping[lsvlist[0]] = set(lsvlist)

    # svoxs mapping
    labelsets_nodoubles = {}  # key_axon: {set_svoxs} (no doubles, labelset with double integrated to lowest-labeled axon)
    for lsk in sorted(labelsets.iterkeys()):
        labelset = labelsets[lsk]
        labelsets_nodoubles = utils.classify_label_set(labelsets_nodoubles, labelset, lsk)

    svoxs_fw = svoxs.forward_map(labelsets=labelsets_nodoubles, from_empty=False)
    mo2 = LabelImage(outputpath + '_svoxs', **props)
    mo2.create()
    mo2.write(svoxs_fw)
    mo2.close()
    svoxs_fw = svoxs.forward_map(labelsets=labelsets_nodoubles, from_empty=True)
    mo4 = LabelImage(outputpath + '_svoxs_stitched', **props)
    mo4.create()
    mo4.write(svoxs_fw)
    mo4.close()
    axons_fw = axons.forward_map(labelsets=axonmapping, from_empty=True)
    mo3 = LabelImage(outputpath + '_axons', **props)
    mo3.create()
    mo3.write(axons_fw)
    mo3.close()
    m = axons_fw[:]
    maskA = m > 0
    m[~maskA] = svoxs_fw[~maskA]
    mo = LabelImage(outputpath, **props)
    mo.create()
    mo.write(m)
    mo.close()

h52nii '' "${dataset_ds}" "${opf}" "${ods}" '' '' '-i zyx -o xyz -d uint16'
h52nii '' "${dataset_ds}" "${opf}" "${ods}_axons" '' '' '-i zyx -o xyz -d uint16'
h52nii '' "${dataset_ds}" "${opf}" "${ods}_svoxs" '' '' '-i zyx -o xyz -d uint16'
h52nii '' "${dataset_ds}" "${opf}" "${ods}_svoxs_stitched" '' '' '-i zyx -o xyz -d uint16'


fill_connected_labels(datadir='/Users/michielk/oxdata/P01/EM/Myrf_01/SET-B/B-NT-S10-2f_ROI_00',
                      dataset='B-NT-S10-2f_ROI_00ds7',
                      h5_dset_in='labelMA_nt_probMA_eed_thr0.5_labeled_svoxs_stitched',
                      searchradius=[40, 10, 10], use_axons=True,
                      between=True, to_border=True)

opf='_labels_labelMA_comb' ods="labelMA_nt_probMA_eed_thr0.5_labeled_svoxs_stitched_filled"
h52nii '' "${dataset_ds}" "${opf}" "${ods}" '' '' '-i zyx -o xyz -d uint16'



ipf='_labels_labelMA_comb' ids="labelMA_nt_probMA_eed_thr0.5_labeled_svoxs_stitched_filled"
opf='_labels_labelMA_comb' ods="labelMA_nt_probMA_eed_thr0.5_labeled_svoxs_stitched_filled_NoR"
python -W ignore $scriptdir/wmem/nodes_of_ranvier.py -S \
$datadir/${dataset_ds}${ipf}.h5/${ids} \
$datadir/${dataset_ds}${opf}.h5/${ods} \
--boundarymask $datadir/${dataset_ds}_masks_maskDS.h5/maskDS
#number of labels in labelvolume: 4844
#number of short labels: 0
#number of large, long labels: 4844
#number of large, long in-volume labels: 4439
#number of large, long through-volume labels: 405
h52nii '' "${dataset_ds}" "${opf}" "${ods}_steps/labels_tv" '' '' '-i zyx -o xyz -d uint16'
h52nii '' "${dataset_ds}" "${opf}" "${ods}_steps/labels_nt" '' '' '-i zyx -o xyz -d uint16'  # strange that maxlabel is 32148??


###############################################################################
### select the 3D labels overlapping with probMA_labeled and map them END   ###
###############################################################################


def create_mask(datadir='/Users/michielk/oxdata/P01/EM/Myrf_01/SET-B/B-NT-S10-2f_ROI_00'):
    import os
    from wmem import utils, LabelImage, MaskImage
    import numpy as np
    h5_fname = 'B-NT-S10-2f_ROI_00ds7_labels_labelMA_core3D_tv.h5'
    h5_dset = 'labelMA_core3D_proofread_NoR_steps/labels_tv'
    h5_path = os.path.join(datadir, h5_fname, h5_dset)
    core3D_tv = LabelImage(h5_path)
    core3D_tv.load()
    h5_fname = 'B-NT-S10-2f_ROI_00ds7_masks_maskDS.h5'
    h5_dset = 'maskDS'
    h5_path = os.path.join(datadir, h5_fname, h5_dset)
    maskDS = MaskImage(h5_path)
    maskDS.load()
    h5_fname = 'B-NT-S10-2f_ROI_00ds7_masks_maskWS.h5'
    h5_dset = 'maskMM_PP'
    h5_path = os.path.join(datadir, h5_fname, h5_dset)
    maskMM = MaskImage(h5_path)
    maskMM.load()
    h5_fname = 'B-NT-S10-2f_ROI_00ds7_masks_maskWS.h5'
    h5_dset = 'maskWS_iter0'
    h5_path = os.path.join(datadir, h5_fname, h5_dset)
    mo = MaskImage(h5_path, **maskDS.get_props())
    mo.create()
    mask = maskDS.ds[:, :, :]
    mask[core3D_tv.ds[:,:,:] != 0] = 0
    mask[maskMM.ds[:,:,:] == 1] = 0
    mo.write(mask)
    mo.close()
    core3D_tv.close()
    maskDS.close()
create_mask()
# create_mask('/data/ndcn-fmrib-water-brain/ndcn0180/EM/Myrf_01/SET-B/B-NT-S10-2f_ROI_00')

opf='_masks_maskWS' ods='maskWS_iter0'
h52nii '' "$dataset_ds" "${opf}" "${ods}" '' '' '-i zyx -o xyz -d uint8'

### watershed merge
# ipf='_labels_labelMA_comb' ids="labelMA_nt"
# opf='_labels_labelMA_comb' ods="${ids}_ws10"
# dpf='' dds='data'  #dpf='_probs1_eed2' dds='probs_eed'  #
# # mpf='_masks_maskMM' mds='maskMM_PP'
# mpf='_masks_maskWS' mds='maskWS'
# # opf='_labels_labelMA_comb' ods="labelMA_nt_probMA_nt_wsmerge_${dds}_${mds}_test"
# mpiexec -n 6 python -W ignore $scriptdir/wmem/merge_labels.py -S \
# $datadir/${dataset_ds}${ipf}.h5/${ids} \
# $datadir/${dataset_ds}${opf}.h5/${ods} \
# --data $datadir/${dataset_ds}${dpf}.h5/${dds} \
# --maskDS $datadir/${dataset_ds}${mpf}.h5/${mds} \
# -m 'watershed' -r 10 10 10 -M

ipf='_labels_labelMA_comb' ids="labelMA_nt"
opf='_labels_labelMA_comb' ods="${ids}_ws10"
dpf='' dds='data'
mpf='_masks_maskWS' mds='maskWS'
args="--data $datadir/${dataset_ds}${dpf}.h5/${dds} --maskDS $datadir/${dataset_ds}${mpf}.h5/${mds} -m 'watershed' -r 10 10 10 -M"
scriptfile=$( merge_labels_ws 'h' $dataset_ds $ipf $ids $opf $ods $args )
h52nii '' "$dataset_ds" "${opf}" "${ods}" '' '' '-i zyx -o xyz -d uint16'
h52nii '' "$dataset_ds" "${opf}" "${ods}_steps/stitched" '' '' '-i zyx -o xyz -d uint16'


from wmem import merge_labels
merge_labels.fill_connected_labels(datadir='/Users/michielk/oxdata/P01/EM/Myrf_01/SET-B/B-NT-S10-2f_ROI_00',
                                   dataset='B-NT-S10-2f_ROI_00ds7',
                                   h5_dset_in='labelMA_nt_ws10',
                                   searchradius=[10, 10, 10], use_axons=False,
                                   between=True, to_border=False)

opf='_labels_labelMA_comb' ods="labelMA_nt_ws10_between"
h52nii '' "$dataset_ds" "${opf}" "${ods}" '' '' '-i zyx -o xyz -d uint16'

ipf='_labels_labelMA_comb' ids='labelMA_nt_ws10_between'
opf='_labels_labelMA_comb' ods="${ids}_NoR"
python -W ignore $scriptdir/wmem/nodes_of_ranvier.py -S \
$datadir/${dataset_ds}${ipf}.h5/${ids} \
$datadir/${dataset_ds}${opf}.h5/${ods} \
--boundarymask $datadir/${dataset_ds}_masks_maskDS.h5/maskDS -s 500 -R
h52nii '' "${dataset_ds}" "${opf}" "${ods}_steps/labels_tv" '' '' '-i zyx -o xyz -d uint16'
h52nii '' "${dataset_ds}" "${opf}" "${ods}_steps/labels_nt" '' '' '-i zyx -o xyz -d uint16'
h52nii '' "${dataset_ds}" "${opf}" "${ods}_steps/smalllabelmask" '' '' '-i zyx -o xyz -d uint8'
# number of labels in labelvolume: 5700
# number of small labels: 2190
# number of short labels: 0
# number of large, long labels: 3510
# number of large, long in-volume labels: 2985
# number of large, long through-volume labels: 525

# FIXME: quick run-through of finished label from ws10: 10min
# label 3588 spills over much of it's length: why? (co=191,855,51)
# label 2469 failed to fill (co=62,1000,151)
# small bit of label 1489 (co=413,323,68) also at (co=518,308,86)
# label 1522 weirdly traverses: (co=603,436,58)
# label 1237 (co=794,315,53) spills weirdly
# remove partial border label at (co=547,39,165)
# border label 26 at (co=550,15,18) is partial
# border label 3.2xxx at (co=819,33,91) is partial
# border label 92 at (co=1056,37,54) is partial
# border label 4770 at (co=1115,1144,20) is partial
# border label 6234 at (co=1177,357,7) is partial
# border label 9114 at (co=1177,1037,7) is partial


def create_mask(datadir='/Users/michielk/oxdata/P01/EM/Myrf_01/SET-B/B-NT-S10-2f_ROI_00',
                dataset='B-NT-S10-2f_ROI_00ds7',
                h5_dset_in='labelMA_nt_ws10_between'):
    import os
    from wmem import utils, LabelImage, MaskImage
    import numpy as np
    h5_fname = '{}_{}.h5'.format(dataset, 'labels_labelMA_comb')
    h5_dset = '{}_NoR_steps/labels_tv'.format(h5_dset_in)
    h5_path = os.path.join(datadir, h5_fname, h5_dset)
    tv = LabelImage(h5_path)
    tv.load()
    h5_fname = 'B-NT-S10-2f_ROI_00ds7_masks_maskWS.h5'
    h5_dset = 'maskWS_iter0'
    h5_path = os.path.join(datadir, h5_fname, h5_dset)
    maskWS = MaskImage(h5_path)
    maskWS.load()
    h5_fname = 'B-NT-S10-2f_ROI_00ds7_masks_maskWS.h5'
    h5_dset = 'maskWS_iter1'
    h5_path = os.path.join(datadir, h5_fname, h5_dset)
    mo = MaskImage(h5_path, **maskWS.get_props())
    mo.create()
    mask = maskWS.ds[:, :, :]
    mask[tv.ds[:,:,:] != 0] = 0
    mo.write(mask)
    mo.close()
    tv.close()
    maskWS.close()
create_mask()

opf='_masks_maskWS' ods='maskWS_iter1'
h52nii '' "$dataset_ds" "${opf}" "${ods}" '' '' '-i zyx -o xyz -d uint8'

srz=40
ipf='_labels_labelMA_comb' ids="labelMA_nt_ws10_between_NoR_steps/labels_nt"
opf='_labels_labelMA_comb' ods="labelMA_nt_ws$srz"
dpf='' dds='data'
mpf='_masks_maskWS' mds='maskWS_iter1'
args="--data $datadir/${dataset_ds}${dpf}.h5/${dds} --maskDS $datadir/${dataset_ds}${mpf}.h5/${mds} -m 'watershed' -r $srz 10 10 -M"
scriptfile=$( merge_labels_ws 'h' $dataset_ds $ipf $ids $opf $ods $args )
h52nii '' "$dataset_ds" "${opf}" "${ods}" '' '' '-i zyx -o xyz -d uint16'
h52nii '' "$dataset_ds" "${opf}" "${ods}_steps/stitched" '' '' '-i zyx -o xyz -d uint16'

from wmem import merge_labels
merge_labels.fill_connected_labels(datadir='/Users/michielk/oxdata/P01/EM/Myrf_01/SET-B/B-NT-S10-2f_ROI_00',
                                   dataset='B-NT-S10-2f_ROI_00ds7',
                                   h5_dset_in='labelMA_nt_ws40',
                                   searchradius=[40, 10, 10], use_axons=False,
                                   between=True, to_border=False)

opf='_labels_labelMA_comb' ods="labelMA_nt_ws40_between"
h52nii '' "$dataset_ds" "${opf}" "${ods}" '' '' '-i zyx -o xyz -d uint16'

ipf='_labels_labelMA_comb' ids='labelMA_nt_ws40_between'
opf='_labels_labelMA_comb' ods="${ids}_NoR"
python -W ignore $scriptdir/wmem/nodes_of_ranvier.py -S \
$datadir/${dataset_ds}${ipf}.h5/${ids} \
$datadir/${dataset_ds}${opf}.h5/${ods} \
--boundarymask $datadir/${dataset_ds}_masks_maskDS.h5/maskDS -s 500 -R
h52nii '' "${dataset_ds}" "${opf}" "${ods}_steps/labels_tv" '' '' '-i zyx -o xyz -d uint16'
h52nii '' "${dataset_ds}" "${opf}" "${ods}_steps/labels_nt" '' '' '-i zyx -o xyz -d uint16'
h52nii '' "${dataset_ds}" "${opf}" "${ods}_steps/smalllabelmask" '' '' '-i zyx -o xyz -d uint8'
# number of labels in labelvolume: 1991
# number of small labels: 0
# number of short labels: 0
# number of large, long labels: 1991
# number of large, long in-volume labels: 1555
# number of large, long through-volume labels: 436

# FIXME: quick run-through of finished label from ws40: 20min
# label 5.4xxx at (co=208,222,28) not well filled
# label 572 at (co=261,229,166) spills a bit
#! label 1221 at (co=268,317,32) connects through UA space
#! label 2918 at (co=415,686,114) connects to multiple segments
# label 2429 at (co=471,577,114) has a poorly filled NoR
# label 1833 at (co=467,438,60) has a poorly filled NoR
# label 633 includes 2D labels outside of MA space at (co=496,141,139)
# label 1606 at (co=595,371,55) connects to the wrong segment
# label 567 at (co=745,109,89) merge and spills badly
# label 690 at (co=841,171,90) connects to the wrong segment
# label 2474 includes 2D labels outside of MA space at (co=966,620,179)
# label 222 includes 2D labels outside of MA space at (co=980,83,150)
# label 2488 at (co=1100,656,133) connects to the wrong segment

def create_mask(datadir='/Users/michielk/oxdata/P01/EM/Myrf_01/SET-B/B-NT-S10-2f_ROI_00',
                dataset='B-NT-S10-2f_ROI_00ds7',
                h5_dset_in='labelMA_nt_ws40_between'):
    import os
    from wmem import utils, LabelImage, MaskImage
    import numpy as np
    h5_fname = '{}_{}.h5'.format(dataset, 'labels_labelMA_comb')
    h5_dset = '{}_NoR_steps/labels_tv'.format(h5_dset_in)
    h5_path = os.path.join(datadir, h5_fname, h5_dset)
    tv = LabelImage(h5_path)
    tv.load()
    h5_fname = 'B-NT-S10-2f_ROI_00ds7_masks_maskWS.h5'
    h5_dset = 'maskWS_iter1'
    h5_path = os.path.join(datadir, h5_fname, h5_dset)
    maskWS = MaskImage(h5_path)
    maskWS.load()
    h5_fname = 'B-NT-S10-2f_ROI_00ds7_masks_maskWS.h5'
    h5_dset = 'maskWS_iter2'
    h5_path = os.path.join(datadir, h5_fname, h5_dset)
    mo = MaskImage(h5_path, **maskWS.get_props())
    mo.create()
    mask = maskWS.ds[:, :, :]
    mask[tv.ds[:,:,:] != 0] = 0
    mo.write(mask)
    mo.close()
    tv.close()
    maskWS.close()
create_mask()

opf='_masks_maskWS' ods='maskWS_iter2'
h52nii '' "$dataset_ds" "${opf}" "${ods}" '' '' '-i zyx -o xyz -d uint8'


srz=80
ipf='_labels_labelMA_comb' ids="labelMA_nt_ws40_between_NoR_steps/labels_nt"
opf='_labels_labelMA_comb' ods="labelMA_nt_ws$srz"
dpf='' dds='data'
mpf='_masks_maskWS' mds='maskWS_iter2'
args="--data $datadir/${dataset_ds}${dpf}.h5/${dds} --maskDS $datadir/${dataset_ds}${mpf}.h5/${mds} -m 'watershed' -r $srz 10 10 -M"
scriptfile=$( merge_labels_ws 'h' $dataset_ds $ipf $ids $opf $ods $args )
h52nii '' "$dataset_ds" "${opf}" "${ods}" '' '' '-i zyx -o xyz -d uint16'
h52nii '' "$dataset_ds" "${opf}" "${ods}_steps/stitched" '' '' '-i zyx -o xyz -d uint16'

from wmem import merge_labels
merge_labels.fill_connected_labels(datadir='/Users/michielk/oxdata/P01/EM/Myrf_01/SET-B/B-NT-S10-2f_ROI_00',
                                   dataset='B-NT-S10-2f_ROI_00ds7',
                                   h5_dset_in='labelMA_nt_ws80',
                                   searchradius=[80, 10, 10], use_axons=False,
                                   between=True, to_border=False)

opf='_labels_labelMA_comb' ods="labelMA_nt_ws80_between"
h52nii '' "$dataset_ds" "${opf}" "${ods}" '' '' '-i zyx -o xyz -d uint16'

ipf='_labels_labelMA_comb' ids='labelMA_nt_ws80_between'
opf='_labels_labelMA_comb' ods="${ids}_NoR"
python -W ignore $scriptdir/wmem/nodes_of_ranvier.py -S \
$datadir/${dataset_ds}${ipf}.h5/${ids} \
$datadir/${dataset_ds}${opf}.h5/${ods} \
--boundarymask $datadir/${dataset_ds}_masks_maskDS.h5/maskDS -s 500 -R
h52nii '' "${dataset_ds}" "${opf}" "${ods}_steps/labels_tv" '' '' '-i zyx -o xyz -d uint16'
h52nii '' "${dataset_ds}" "${opf}" "${ods}_steps/labels_nt" '' '' '-i zyx -o xyz -d uint16'
h52nii '' "${dataset_ds}" "${opf}" "${ods}_steps/smalllabelmask" '' '' '-i zyx -o xyz -d uint8'
# number of labels in labelvolume: 1419
# number of small labels: 0
# number of short labels: 0
# number of large, long labels: 1419
# number of large, long in-volume labels: 1367
# number of large, long through-volume labels: 52

# FIXME: quick run-through of finished label from ws80: 20min
# label 1778 connects through UA space at (co=115,440,20)
# label 3233 connects through UA space at (co=340,797,29)
# label 3.6xxx connects badly at (co=466,1030,89)
# border label 204 is not filled and is partial at (co=514,69,164)
# label 3.5xxx connects through UA space at (co=777,502,87)
# label 2.5xxx as badly doubly connected at (co=914,1039,106)
# label 639 spills at (co=916,179,180)
# label 1998 is messily connected at (co=1154,422,75)




# TODO: perhaps first remove labels that can be filled to border etc.; or do it from iter0...
# from wmem import merge_labels
# merge_labels.fill_connected_labels(datadir='/Users/michielk/oxdata/P01/EM/Myrf_01/SET-B/B-NT-S10-2f_ROI_00',
#                                    dataset='B-NT-S10-2f_ROI_00ds7',
#                                    h5_dset_in='labelMA_nt_ws80',
#                                    searchradius=[80, 10, 10], use_axons=False,
#                                    between=False, to_border=True)



def create_mask(datadir='/Users/michielk/oxdata/P01/EM/Myrf_01/SET-B/B-NT-S10-2f_ROI_00',
                dataset='B-NT-S10-2f_ROI_00ds7',
                h5_dset_in='labelMA_nt_ws80_between'):
    import os
    from wmem import utils, LabelImage, MaskImage
    import numpy as np
    h5_fname = '{}_{}.h5'.format(dataset, 'labels_labelMA_comb')
    h5_dset = '{}_NoR_steps/labels_tv'.format(h5_dset_in)
    h5_path = os.path.join(datadir, h5_fname, h5_dset)
    tv = LabelImage(h5_path)
    tv.load()
    h5_fname = 'B-NT-S10-2f_ROI_00ds7_masks_maskWS.h5'
    h5_dset = 'maskWS_iter2'
    h5_path = os.path.join(datadir, h5_fname, h5_dset)
    maskWS = MaskImage(h5_path)
    maskWS.load()
    h5_fname = 'B-NT-S10-2f_ROI_00ds7_masks_maskWS.h5'
    h5_dset = 'maskWS_iter3'
    h5_path = os.path.join(datadir, h5_fname, h5_dset)
    mo = MaskImage(h5_path, **maskWS.get_props())
    mo.create()
    mask = maskWS.ds[:, :, :]
    mask[tv.ds[:,:,:] != 0] = 0
    mo.write(mask)
    mo.close()
    tv.close()
    maskWS.close()
create_mask()

opf='_masks_maskWS' ods='maskWS_iter3'
h52nii '' "$dataset_ds" "${opf}" "${ods}" '' '' '-i zyx -o xyz -d uint8'

# FIXME: the maskless merge needs an adaptation (e.g. restrict z-range to 40?) or a manual check: erroneous merges
srz=81
ipf='_labels_labelMA_comb' ids="labelMA_nt_ws80_between_NoR_steps/labels_nt"
opf='_labels_labelMA_comb' ods="labelMA_nt_ws$srz"
dpf='' dds='data'
mpf='_masks_maskWS' mds='maskWS_iter3'
args="--data $datadir/${dataset_ds}${dpf}.h5/${dds} -m 'watershed' -r $srz 10 10 -M"
scriptfile=$( merge_labels_ws 'h' $dataset_ds $ipf $ids $opf $ods $args )
h52nii '' "$dataset_ds" "${opf}" "${ods}" '' '' '-i zyx -o xyz -d uint16'
h52nii '' "$dataset_ds" "${opf}" "${ods}_steps/stitched" '' '' '-i zyx -o xyz -d uint16'

from wmem import merge_labels
merge_labels.fill_connected_labels(datadir='/Users/michielk/oxdata/P01/EM/Myrf_01/SET-B/B-NT-S10-2f_ROI_00',
                                   dataset='B-NT-S10-2f_ROI_00ds7',
                                   h5_dset_in='labelMA_nt_ws81',
                                   searchradius=[81, 10, 10], use_axons=False,
                                   between=True, to_border=False)

opf='_labels_labelMA_comb' ods="labelMA_nt_ws81_between"
h52nii '' "$dataset_ds" "${opf}" "${ods}" '' '' '-i zyx -o xyz -d uint16'

ipf='_labels_labelMA_comb' ids='labelMA_nt_ws81_between'
opf='_labels_labelMA_comb' ods="${ids}_NoR"
python -W ignore $scriptdir/wmem/nodes_of_ranvier.py -S \
$datadir/${dataset_ds}${ipf}.h5/${ids} \
$datadir/${dataset_ds}${opf}.h5/${ods} \
--boundarymask $datadir/${dataset_ds}_masks_maskDS.h5/maskDS -s 500 -R
h52nii '' "${dataset_ds}" "${opf}" "${ods}_steps/labels_tv" '' '' '-i zyx -o xyz -d uint16'
h52nii '' "${dataset_ds}" "${opf}" "${ods}_steps/labels_nt" '' '' '-i zyx -o xyz -d uint16'
h52nii '' "${dataset_ds}" "${opf}" "${ods}_steps/smalllabelmask" '' '' '-i zyx -o xyz -d uint8'
# number of labels in labelvolume: 1171
# number of small labels: 0
# number of short labels: 0
# number of large, long labels: 1171
# number of large, long in-volume labels: 1098
# number of large, long through-volume labels: 73




# FIXME: this doesn't go well yet; doesn't use cylinder
## test on 1518:
from wmem import merge_labels
merge_labels.fill_connected_labels(datadir='/Users/michielk/oxdata/P01/EM/Myrf_01/SET-B/B-NT-S10-2f_ROI_00',
                                   dataset='B-NT-S10-2f_ROI_00ds7',
                                   h5_dset_in='labelMA_nt_ws81_between',
                                   searchradius=[81, 30, 30], use_axons=False,
                                   between=False, to_border=True)

opf='_labels_labelMA_comb' ods="labelMA_nt_ws81_between_toborder"
h52nii '' "$dataset_ds" "${opf}" "${ods}" '' '' '-i zyx -o xyz -d uint16'

ipf='_labels_labelMA_comb' ids="labelMA_nt_ws81_between_toborder"
opf='_labels_labelMA_comb' ods="${ids}_NoR"
python -W ignore $scriptdir/wmem/nodes_of_ranvier.py -S \
$datadir/${dataset_ds}${ipf}.h5/${ids} \
$datadir/${dataset_ds}${opf}.h5/${ods} \
--boundarymask $datadir/${dataset_ds}_masks_maskDS.h5/maskDS -s 500 -R
h52nii '' "${dataset_ds}" "${opf}" "${ods}_steps/labels_tv" '' '' '-i zyx -o xyz -d uint16'
h52nii '' "${dataset_ds}" "${opf}" "${ods}_steps/labels_nt" '' '' '-i zyx -o xyz -d uint16'
h52nii '' "${dataset_ds}" "${opf}" "${ods}_steps/smalllabelmask" '' '' '-i zyx -o xyz -d uint8'
# NEW :(
# number of labels in labelvolume: 1171
# number of small labels: 0
# number of short labels: 0
# number of large, long labels: 1171
# number of large, long in-volume labels: 937
# number of large, long through-volume labels: 234

# number of labels in labelvolume: 1171
# number of small labels: 0
# number of short labels: 0
# number of large, long labels: 1171
# number of large, long in-volume labels: 654
# number of large, long through-volume labels: 517





ipf1='_labels_labelMA_core3D' ids1='labelMA_core3D_proofread_NoR_steps/labels_tv' ipf2='_labels_labelMA_comb' ids2='labelMA_nt_ws10_between_NoR_steps/labels_tv'
ipf1='_labels_labelMA_2D3D' ids1='labelMA_tv' ipf2='_labels_labelMA_comb' ids2='labelMA_nt_ws40_between_NoR_steps/labels_tv'
ipf1='_labels_labelMA_2D3D' ids1='labelMA_tv' ipf2='_labels_labelMA_comb' ids2='labelMA_nt_ws80_between_NoR_steps/labels_tv'
ipf1='_labels_labelMA_2D3D' ids1='labelMA_tv' ipf2='_labels_labelMA_comb' ids2='labelMA_nt_ws81_between_toborder_NoR_steps/labels_tv'
opf='_labels_labelMA_2D3D' ods='labelMA_tv'
python $scriptdir/wmem/combine_labels.py \
"${datadir}/${dataset_ds}${ipf1}.h5/${ids1}" \
"${datadir}/${dataset_ds}${ipf2}.h5/${ids2}" \
"${datadir}/${dataset_ds}${opf}.h5/${ods}" -m 'add'
h52nii '' "${dataset_ds}" "${opf}" "${ods}" '' '' '-i zyx -o xyz -d uint16'



### using searchregion of z=40
# number of labels in labelvolume: 4441
# number of short labels: 0
# number of large, long labels: 4441
# number of large, long in-volume labels: 3550
# number of large, long through-volume labels: 891
# number of labels in labelvolume: 4441
# number of small labels: 1932
# number of short labels: 0
# number of large, long labels: 2509
# number of large, long in-volume labels: 1637
# number of large, long through-volume labels: 872


# FIXME: there seem to be many labels that are done not marked by nodes_of_ranvier code; or that are not finished by fill_connected_labels => connect_to_border while they could/should; this is probably due to conservative nature of having the cylinder surround as negative label for nodes of ranvier that overlap the border; this fill the axon just part-way
# TODO: force-fill option?
# TODO: remove small labels
# notice label 1324 is a doubly connected (Y-shaped) label


fill_connected_labels(datadir='/Users/michielk/oxdata/P01/EM/Myrf_01/SET-B/B-NT-S10-2f_ROI_00',
                      dataset='B-NT-S10-2f_ROI_00ds7',
                      h5_dset_in='labelMA_nt_ws',
                      searchradius=[40, 10, 10], use_axons=False,
                      between=True, to_border=False)
fill_connected_labels(datadir='/Users/michielk/oxdata/P01/EM/Myrf_01/SET-B/B-NT-S10-2f_ROI_00',
                      dataset='B-NT-S10-2f_ROI_00ds7',
                      h5_dset_in='labelMA_nt_ws_between',
                      searchradius=[40, 30, 30], use_axons=False,
                      between=False, to_border=True)

opf='_labels_labelMA_comb' ods="labelMA_nt_ws_between"
h52nii '' "$dataset_ds" "${opf}" "${ods}" '' '' '-i zyx -o xyz -d uint16'
opf='_labels_labelMA_comb' ods="labelMA_nt_ws_between_toborder"
h52nii '' "$dataset_ds" "${opf}" "${ods}" '' '' '-i zyx -o xyz -d uint16'



### iteration 2: first identify, then watershed merge not using mask! works quite well...
# ipf='_labels_labelMA_comb' ids="labelMA_nt_ws_filled"
ipf='_labels_labelMA_comb' ids='labelMA_nt_ws_between_toborder'
opf='_labels_labelMA_comb' ods="${ids}_NoR"
python -W ignore $scriptdir/wmem/nodes_of_ranvier.py -S \
$datadir/${dataset_ds}${ipf}.h5/${ids} \
$datadir/${dataset_ds}${opf}.h5/${ods} \
--boundarymask $datadir/${dataset_ds}_masks_maskDS.h5/maskDS
h52nii '' "${dataset_ds}" "${opf}" "${ods}_steps/labels_tv" '' '' '-i zyx -o xyz -d uint16'
h52nii '' "${dataset_ds}" "${opf}" "${ods}_steps/labels_nt" '' '' '-i zyx -o xyz -d uint16'

ipf='_labels_labelMA_comb' ids="labelMA_nt_ws_between_toborder_NoR_steps/labels_nt"
opf='_labels_labelMA_comb' ods="${ids}_ws"
dpf='' dds='data'  #dpf='_probs1_eed2' dds='probs_eed'  #
mpf='_masks_maskMM' mds='maskMM_PP'
mpiexec -n 6 python -W ignore $scriptdir/wmem/merge_labels.py -S \
$datadir/${dataset_ds}${ipf}.h5/${ids} \
$datadir/${dataset_ds}${opf}.h5/${ods} \
--data $datadir/${dataset_ds}${dpf}.h5/${dds} \
--maskMM $datadir/${dataset_ds}${mpf}.h5/${mds} \
-m 'watershed' -r 80 10 10 -M
h52nii '' "$dataset_ds" "${opf}" "${ods}" '' '' '-i zyx -o xyz -d uint16'
h52nii '' "$dataset_ds" "${opf}" "${ods}_steps/stitched" '' '' '-i zyx -o xyz -d uint16'

# TODO: maybe not use mask here
fill_connected_labels(datadir='/Users/michielk/oxdata/P01/EM/Myrf_01/SET-B/B-NT-S10-2f_ROI_00',
                      dataset='B-NT-S10-2f_ROI_00ds7',
                      h5_dset_in='labelMA_nt_ws_between_toborder_NoR_steps/labels_nt_ws',
                      searchradius=[80, 10, 10], use_axons=False,
                      between=True, to_border=False)
fill_connected_labels(datadir='/Users/michielk/oxdata/P01/EM/Myrf_01/SET-B/B-NT-S10-2f_ROI_00',
                      dataset='B-NT-S10-2f_ROI_00ds7',
                      h5_dset_in='labelMA_nt_ws_between_toborder_NoR_steps/labels_nt_ws_between',
                      searchradius=[80, 30, 30], use_axons=False,
                      between=False, to_border=True)

opf='_labels_labelMA_comb' ods='labelMA_nt_ws_between_toborder_NoR_steps/labels_nt_ws_between_toborder'
h52nii '' "$dataset_ds" "${opf}" "${ods}" '' '' '-i zyx -o xyz -d uint16'


# quick manual copy to THIS ids:
ipf='_labels_labelMA_comb' ids='labels_nt_ws_filled'
opf='_labels_labelMA_comb' ods="${ids}_NoR"
python -W ignore $scriptdir/wmem/nodes_of_ranvier.py -S \
$datadir/${dataset_ds}${ipf}.h5/${ids} \
$datadir/${dataset_ds}${opf}.h5/${ods} \
--boundarymask $datadir/${dataset_ds}_masks_maskDS.h5/maskDS
h52nii '' "${dataset_ds}" "${opf}" "${ods}_steps/labels_tv" '' '' '-i zyx -o xyz -d uint16'
h52nii '' "${dataset_ds}" "${opf}" "${ods}_steps/labels_nt" '' '' '-i zyx -o xyz -d uint16'


# testing fill_centerline for to_border: doesn't fly
fill_connected_labels(datadir='/Users/michielk/oxdata/P01/EM/Myrf_01/SET-B/B-NT-S10-2f_ROI_00',
                      dataset='B-NT-S10-2f_ROI_00ds7',
                      h5_dset_in='labels_nt_ws_filled',
                      searchradius=[40, 10, 10], use_axons=False,
                      between=False, to_border=True)

opf='_labels_labelMA_comb' ods='labels_nt_ws_filled_filled'
h52nii '' "$dataset_ds" "${opf}" "${ods}" '' '' '-i zyx -o xyz -d uint16'


# quick manual copy to THIS ids:
ipf='_labels_labelMA_comb' ids='labels_nt_ws_filled_filled'
opf='_labels_labelMA_comb' ods="${ids}_NoR"
python -W ignore $scriptdir/wmem/nodes_of_ranvier.py -S \
$datadir/${dataset_ds}${ipf}.h5/${ids} \
$datadir/${dataset_ds}${opf}.h5/${ods} \
--boundarymask $datadir/${dataset_ds}_masks_maskDS.h5/maskDS -s 500 -R
h52nii '' "${dataset_ds}" "${opf}" "${ods}_steps/labels_tv" '' '' '-i zyx -o xyz -d uint16'
h52nii '' "${dataset_ds}" "${opf}" "${ods}_steps/labels_nt" '' '' '-i zyx -o xyz -d uint16'

## without removing small
# number of labels in labelvolume: 1939
# number of short labels: 0
# number of large, long labels: 1939
# number of large, long in-volume labels: 1694
# number of large, long through-volume labels: 245
## removing small
# number of labels in labelvolume: 1939
# number of small labels: 1027
# number of short labels: 0
# number of large, long labels: 912
# number of large, long in-volume labels: 667
# number of large, long through-volume labels: 245
## removing small from filled_filled (force fill to border)
# number of labels in labelvolume: 1933
# number of small labels: 988
# number of short labels: 0
# number of large, long labels: 945
# number of large, long in-volume labels: 475
# number of large, long through-volume labels: 470




# r=0.00
# ipf='_labels_labelMA_core3D' ids="probMA_eed_thr0.5_labeled"
# spf='_labels_labelMA_comb' sds="labelMA_nt_ws_filled_NoR_steps/labels_nt"
# opf='_labels_labelMA_comb' ods="labelMA_nt_ws_filled_NoR_nt_probMA"
# python $scriptdir/wmem/agglo_from_labelmask.py \
# "${dataset_ds}${ipf}.h5/${ids}" \
# "${dataset_ds}${spf}.h5/${sds}" \
# "${dataset_ds}${opf}.h5/${ods}" \
# -r $r
#
#
#
# import os
# from wmem import utils, LabelImage
# import numpy as np
# import glob
# import pickle
# def map_axons(datadir='/Users/michielk/oxdata/P01/EM/Myrf_01/SET-B/B-NT-S10-2f_ROI_00',
#               h5_dset_in='labelMA_nt_ws_filled_NoR_nt_probMA'):
#     h5_fname = 'B-NT-S10-2f_ROI_00ds7_labels_labelMA_core3D.h5'
#     h5_dset = 'probMA_eed_thr0.5_labeled'
#     h5_path = os.path.join(datadir, h5_fname, h5_dset)
#     axons = utils.get_image(h5_path, imtype='Label')
#     h5_fname = 'B-NT-S10-2f_ROI_00ds7_labels_labelMA_comb.h5'
#     # h5_dset = 'labelMA_nt'
#     h5_path = os.path.join(datadir, h5_fname, h5_dset_in)
#     svoxs = utils.get_image(h5_path, imtype='Label')
#     h5_fname = 'B-NT-S10-2f_ROI_00ds7_labels_labelMA_comb.h5'
#     h5_dset_out = '{}_labeled'.format(h5_dset_in)
#     outputpath = os.path.join(datadir, h5_fname, h5_dset_out)
#     props = svoxs.get_props(protective=False)
#
#     def invert_labelsets(labelsets):
#         labelsets_inv = {}  # key_svox: {set_axon} (one-to-one mapping when labelsets_new has no doubles)
#         for lsk in sorted(labelsets.iterkeys()):
#             lsv = labelsets[lsk]
#             for l in lsv:
#                 if l in labelsets_inv.keys():
#                     labelsets_inv[l].add(lsk)
#                 else:
#                     labelsets_inv[l] = set([lsk])
#         return labelsets_inv
#
#     # FIXME: handle doubles better (now mapped to lowest axon label)
#     comps = svoxs.split_path()
#     lsfile = '{}_{}_host-MKMBPr.local_rank-00.pickle'.format(comps['base'], comps['dset'])
#     # lsfile = os.path.join(datadir, 'B-NT-S10-2f_ROI_00ds7_labels_labelMA_comb_{}_host-MKMBPr.local_rank-00.pickle'.format(h5_dset_out))
#     labelsets = svoxs.read_labelsets(lsfile)  # key_axon: {set_svoxs} (contains double svoxs in values)
#     labelsets_inv = invert_labelsets(labelsets)  # key_svox: {set_axons} (contains double axons in values)
#
#     # axon mapping (any axons that has doubles needs to be mapped to the lowest-labeled)
#     labelsets_inv_nodoubles = {}
#     for lsk in sorted(labelsets_inv.iterkeys()):
#         labelsets_inv_nodoubles = utils.classify_label_set(labelsets_inv_nodoubles, labelsets_inv[lsk], lsk)
#         axonmapping = {}  # key_axon-first: {set_axons}
#     for lsk, lsv in labelsets_inv_nodoubles.items():
#         lsvlist = sorted(list(lsv))
#         axonmapping[lsvlist[0]] = set(lsvlist)
#
#     # svoxs mapping
#     labelsets_nodoubles = {}  # key_axon: {set_svoxs} (no doubles, labelset with double integrated to lowest-labeled axon)
#     for lsk in sorted(labelsets.iterkeys()):
#         labelset = labelsets[lsk]
#         labelsets_nodoubles = utils.classify_label_set(labelsets_nodoubles, labelset, lsk)
#
#     svoxs_fw = svoxs.forward_map(labelsets=labelsets_nodoubles, from_empty=False)
#     mo2 = LabelImage(outputpath + '_svoxs', **props)
#     mo2.create()
#     mo2.write(svoxs_fw)
#     mo2.close()
#     svoxs_fw = svoxs.forward_map(labelsets=labelsets_nodoubles, from_empty=True)
#     mo4 = LabelImage(outputpath + '_svoxs_stitched', **props)
#     mo4.create()
#     mo4.write(svoxs_fw)
#     mo4.close()
#     axons_fw = axons.forward_map(labelsets=axonmapping, from_empty=True)
#     mo3 = LabelImage(outputpath + '_axons', **props)
#     mo3.create()
#     mo3.write(axons_fw)
#     mo3.close()
#     m = axons_fw[:]
#     maskA = m > 0
#     m[~maskA] = svoxs_fw[~maskA]
#     mo = LabelImage(outputpath, **props)
#     mo.create()
#     mo.write(m)
#     mo.close()
#
# map_axons(datadir='/Users/michielk/oxdata/P01/EM/Myrf_01/SET-B/B-NT-S10-2f_ROI_00',
#           h5_dset_in='labelMA_nt_ws_filled_NoR_nt_probMA')
#
# h52nii '' "${dataset_ds}" "${opf}" "${ods}" '' '' '-i zyx -o xyz -d uint16'
# h52nii '' "${dataset_ds}" "${opf}" "${ods}_axons" '' '' '-i zyx -o xyz -d uint16'
# h52nii '' "${dataset_ds}" "${opf}" "${ods}_svoxs" '' '' '-i zyx -o xyz -d uint16'
# h52nii '' "${dataset_ds}" "${opf}" "${ods}_svoxs_stitched" '' '' '-i zyx -o xyz -d uint16'


total = core3D_tv +






fill_connected_labels(datadir='/Users/michielk/oxdata/P01/EM/Myrf_01/SET-B/B-NT-S10-2f_ROI_00',
                      dataset='B-NT-S10-2f_ROI_00ds7',
                      h5_dset_in='labelMA_nt_probMA_nt_wsmerge_data_maskMM_PP_test_steps/stitched',
                      searchradius=[40, 10, 10], use_axons=False)

h52nii '' "$dataset_ds" "${opf}" "${ods}_steps/stitched_filled" '' '' '-i zyx -o xyz -d uint16'

fill_connected_labels(datadir='/Users/michielk/oxdata/P01/EM/Myrf_01/SET-B/B-NT-S10-2f_ROI_00',
                      dataset='B-NT-S10-2f_ROI_00ds7',
                      h5_dset_in='labelMA_nt_ws',
                      searchradius=[40, 10, 10], use_axons=False)



import os
from wmem import utils, LabelImage, merge_labels
import numpy as np
from skimage.measure import label, regionprops
from skimage.morphology import watershed
import glob
import pickle
datadir = '/Users/michielk/oxdata/P01/EM/Myrf_01/SET-B/B-NT-S10-2f_ROI_00'

h5_fname = 'B-NT-S10-2f_ROI_00ds7_labels_labelMA_comb.h5'
h5_dset = "labelMA_nt_probMA_eed_thr0.5_labeled_svoxs_stitched_filled_NoR_steps/labels_nt"
h5_path = os.path.join(datadir, h5_fname, h5_dset)
im = utils.get_image(h5_path, imtype='Label')

h5_fname = 'B-NT-S10-2f_ROI_00ds7_labels_labelMA_comb.h5'
h5_dset = 'labelMA_nt_probMA_nt_wsmerge_data_maskMM_PP'
outputpath = os.path.join(datadir, h5_fname, h5_dset)

merge_labels.map_labels(im, None, None, outputpath, True, False)

im.close()

opf='_labels_labelMA_comb' ods="labelMA_nt_probMA_nt_wsmerge_data_maskMM_PP"
h52nii '' "$dataset_ds" "${opf}" "${ods}" '' '' '-i zyx -o xyz -d uint16'
h52nii '' "$dataset_ds" "${opf}" "${ods}_steps/stitched" '' '' '-i zyx -o xyz -d uint16'

# TODO: connect split labels
# TODO: combine with probMA axons (there are some axons not connected after watershed fill)









# fill holes in MA axons
def make_maskMMplus():
    import os
    from wmem import parse, utils, Image, MaskImage, LabelImage
    datadir = '/Users/michielk/oxdata/P01/EM/Myrf_01/SET-B/B-NT-S10-2f_ROI_00'
    # TODO: consider a more strict mask here
    h5path_in = os.path.join(datadir, 'B-NT-S10-2f_ROI_00ds7_masks_maskMM.h5/maskMM_PP')
    im1 = MaskImage(h5path_in)
    im1.load()
    h5path_in = os.path.join(datadir, 'B-NT-S10-2f_ROI_00ds7_labels_labelMA_core3D.h5/labelMA_core3D_proofread_NoR_steps/labels_tv')
    im2 = MaskImage(h5path_in)
    im2.load()
    h5path_in = os.path.join(datadir, 'B-NT-S10-2f_ROI_00ds7_labels_labelMA_comb.h5/labelMA_nt')
    im3 = MaskImage(h5path_in)
    im3.load()
    h5path_out = os.path.join(datadir, 'B-NT-S10-2f_ROI_00ds7_masks_maskMM.h5/maskMM_PP_MA')
    mo = MaskImage(h5path_out, shape=im1.dims, elsize=im1.elsize, axlab=im1.axlab, dtype=im1.dtype, protective=False)
    mo.create()
    data = (im1.ds[:] + im2.ds[:] + im3.ds[:]) > 0
    mo.write(data=data)
    mo.close()
    im3.close()
    im2.close()
    im1.close()

h52nii '' $dataset_ds '_masks_maskMM' 'maskMM_PP_MA' '' '' '-i zyx -o xyz -d uint8'

ipf='_masks_maskMM' ids='maskMM_PP_MA'
opf='_labels_labelMA_comb' ods='labelMA_aux3D'
python $scriptdir/wmem/connected_components.py \
$datadir/${dataset_ds}${ipf}.h5/${ids} \
$datadir/${dataset_ds}${opf}.h5/${ods} \
-m '3D' -q 5000 -a 50
h52nii '' "${dataset_ds}" "${opf}" "${ods}" '' '' '-i zyx -o xyz -d uint16'

### FIXME: too many faulty labels outside MA

### merge labelMA_aux3D_relabeled into labelMA_nt
def merge_aux3D():
    import os
    from wmem import utils, LabelImage
    import numpy as np
    datadir = '/Users/michielk/oxdata/P01/EM/Myrf_01/SET-B/B-NT-S10-2f_ROI_00'
    h5_fname = 'B-NT-S10-2f_ROI_00ds7_labels_labelMA_comb.h5'
    h5_dset = 'labelMA_nt'
    h5_path = os.path.join(datadir, h5_fname, h5_dset)
    im1 = LabelImage(h5_path)
    im1.load()
    # 23142 labels (max: 832496) in volume /Users/michielk/oxdata/P01/EM/Myrf_01/SET-B/B-NT-S10-2f_ROI_00/B-NT-S10-2f_ROI_00ds7_labels_labelMA_comb.h5/labelMA_nt
    h5_fname = 'B-NT-S10-2f_ROI_00ds7_labels_labelMA_comb.h5'
    h5_dset = 'labelMA_aux3D'
    h5_path = os.path.join(datadir, h5_fname, h5_dset)
    im2 = LabelImage(h5_path)
    im2.load()
    # 6791 labels (max: 6791) in volume /Users/michielk/oxdata/P01/EM/Myrf_01/SET-B/B-NT-S10-2f_ROI_00/B-NT-S10-2f_ROI_00ds7_labels_labelMA_comb.h5/labelMA_aux3D
    h5_fname = 'B-NT-S10-2f_ROI_00ds7_labels_labelMA_comb.h5'
    h5_dset = 'labelMA_nt_aux3D'
    h5_path = os.path.join(datadir, h5_fname, h5_dset)
    mo = LabelImage(h5_path, **im1.get_props())
    mo.create()
    relabel_from = im1.maxlabel + 1
    aux = im2.ds[:]
    auxmask = aux != 0
    aux[auxmask] += relabel_from
    mo.ds[:] = im1.ds[:] + aux
    mo.close()
    im1.close()
    im2.close()

h52nii '' $dataset_ds '_labels_labelMA_comb' 'labelMA_nt_aux3D' '' '' '-i zyx -o xyz -d uint16'

# # this is just playing to see how much was gained
# python -W ignore $scriptdir/wmem/connected_components.py -S \
# $datadir/${dataset_ds}_labels_labelMA_comb.h5/labelMA_nt_ns_closed_ns_aux3D \
# $datadir/${dataset_ds}_labels_labelMA_comb.h5/labelMA_nt_ns_closed_ns_aux3D_3Dlabeled \
# -m '2Dto3D' -d 0
# h52nii '' $dataset_ds '_labels_labelMA_comb' 'labelMA_nt_ns_closed_ns_aux3D_3Dlabeled' '' '' '-i zyx -o xyz -d uint16'
# mpiexec -n 6 python -W ignore $scriptdir/wmem/merge_labels.py -S -M \
# $datadir/${dataset_ds}_labels_labelMA_comb.h5/labelMA_nt_ns_closed_ns_aux3D_3Dlabeled \
# $datadir/${dataset_ds}_labels_labelMA_comb.h5/labelMA_nt_ns_closed_ns_aux3D_3Dlabeled_ws_maskMM \
# --data $datadir/${dataset_ds}_probs1_eed2.h5/probs_eed \
# --maskMM $datadir/${dataset_ds}_masks_maskMM.h5/maskMM_PP \
# -m 'watershed'
# h52nii '' $dataset_ds '_labels_labelMA_comb' "labelMA_nt_ns_closed_ns_aux3D_3Dlabeled_ws_maskMM" '' '' '-i zyx -o xyz -d uint16'
# h52nii '' $dataset_ds '_labels_labelMA_comb' "labelMA_nt_ns_closed_ns_aux3D_3Dlabeled_ws_maskMM_steps/stitched" '' '' '-i zyx -o xyz -d uint16'
# h52nii '' $dataset_ds '_labels_labelMA_comb' "labelMA_nt_ns_closed_ns_aux3D_3Dlabeled_ws_maskMM_steps/filled" '' '' '-i zyx -o xyz -d uint16'
# python -W ignore $scriptdir/wmem/nodes_of_ranvier.py -S \
# $datadir/${dataset_ds}_labels_labelMA_comb.h5/labelMA_nt_ns_closed_ns_aux3D_3Dlabeled \
# $datadir/${dataset_ds}_labels_labelMA_comb.h5/labelMA_nt_ns_closed_ns_aux3D_3Dlabeled_NoR \
# --boundarymask $datadir/${dataset_ds}_masks_maskDS.h5/maskDS
# #number of labels in labelvolume: 7961
# #number of short labels: 0
# #number of large, long labels: 7961
# #number of large, long in-volume labels: 7298
# #number of large, long through-volume labels: 663
# h52nii '' $dataset_ds '_labels_labelMA_comb' "labelMA_nt_ns_closed_ns_aux3D_3Dlabeled_NoR_steps/labels_tv" '' '' '-i zyx -o xyz -d uint16'
# h52nii '' $dataset_ds '_labels_labelMA_comb' "labelMA_nt_ns_closed_ns_aux3D_3Dlabeled_NoR_steps/labels_nt" '' '' '-i zyx -o xyz -d uint16'
# mpiexec -n 6 python -W ignore $scriptdir/wmem/merge_labels.py -S -M \
# $datadir/${dataset_ds}_labels_labelMA_comb.h5/labelMA_nt_ns_closed_ns_aux3D_3Dlabeled_NoR_steps/labels_tv \
# $datadir/${dataset_ds}_labels_labelMA_comb.h5/labelMA_nt_ns_closed_ns_aux3D_3Dlabeled_NoR_steps_tv_ws_maskMM \
# --data $datadir/${dataset_ds}_probs1_eed2.h5/probs_eed \
# --maskMM $datadir/${dataset_ds}_masks_maskMM.h5/maskMM_PP \
# -m 'watershed'
# h52nii '' $dataset_ds '_labels_labelMA_comb' "labelMA_nt_ns_closed_ns_aux3D_3Dlabeled_NoR_steps_tv_ws_maskMM" '' '' '-i zyx -o xyz -d uint16'
# h52nii '' $dataset_ds '_labels_labelMA_comb' "labelMA_nt_ns_closed_ns_aux3D_3Dlabeled_NoR_steps_tv_ws_maskMM_steps/stitched" '' '' '-i zyx -o xyz -d uint16'
# h52nii '' $dataset_ds '_labels_labelMA_comb' "labelMA_nt_ns_closed_ns_aux3D_3Dlabeled_NoR_steps_tv_ws_maskMM_steps/filled" '' '' '-i zyx -o xyz -d uint16'










ipf='_labels_labelMA_comb' ids="${sds}_${ids}"
opf='_labels_labelMA_comb' ods="${sds}_${ids}_NoR"
python -W ignore $scriptdir/wmem/nodes_of_ranvier.py -S \
$datadir/${dataset_ds}${ipf}.h5/${ids} \
$datadir/${dataset_ds}${opf}.h5/${ods} \
--boundarymask $datadir/${dataset_ds}_masks_maskDS.h5/maskDS
h52nii '' "${dataset_ds}" "${opf}" "${ods}_steps/labels_tv" '' '' '-i zyx -o xyz -d uint16'
h52nii '' "${dataset_ds}" "${opf}" "${ods}_steps/labels_nt" '' '' '-i zyx -o xyz -d uint16'

### TODO: fill label gaps
# -- on high-res to fill thin axon segments
### TODO: connect nt

# TODO: investigate the (now filled) gaps in the 2D labels to extract the NoR
# - consider gap length, surrounding myelin, ...
# - handle spilling (project a centreline/cylinder from end to end)

ipf='_labels_labelMA_comb' ids="labelMA_nt_aux3D_probMA_eed_thr0.5_labeled"
opf='_labels_labelMA_comb' ods="labelMA_nt_aux3D_probMA_eed_thr0.5_labeled_wsmerge"
mpiexec -n 6 python -W ignore $scriptdir/wmem/merge_labels.py -S -M \
$datadir/${dataset_ds}${ipf}.h5/${ids} \
$datadir/${dataset_ds}${opf}.h5/${ods} \
--data $datadir/${dataset_ds}_probs1_eed2.h5/probs_eed \
--maskMM $datadir/${dataset_ds}_masks_maskMM.h5/maskMM_PP \
-m 'watershed'
h52nii '' "$dataset_ds" "${opf}" "${ods}" '' '' '-i zyx -o xyz -d uint16'
h52nii '' "$dataset_ds" "${opf}" "${ods}_steps/stitched" '' '' '-i zyx -o xyz -d uint16'
h52nii '' "$dataset_ds" "${opf}" "${ods}_steps/filled" '' '' '-i zyx -o xyz -d uint16'







def label_comb():
    import os
    import numpy as np
    from wmem import utils, LabelImage
    datadir = '/Users/michielk/oxdata/P01/EM/Myrf_01/SET-B/B-NT-S10-2f_ROI_00'
    dataset = 'B-NT-S10-2f_ROI_00ds7'
    ipf, ids ='_labels_labelMA_core3D', "labelMA_core3D_proofread_NoR_steps/labels_tv"
    im1 = LabelImage(os.path.join(datadir, '{}{}.h5/{}'.format(dataset, ipf, ids)))
    im1.load()
    im1ds = im1.ds[:]
    im1.close()
    ipf, ids ='_labels_labelMA_comb', "labelMA_nt_aux3D_probMA_eed_thr0.5_labeled"
    im2 = LabelImage(os.path.join(datadir, '{}{}.h5/{}'.format(dataset, ipf, ids)))
    im2.load()
    im2ds = im2.ds[:]
    im2.close()
    opf, ods = '_labels_labelMA_comb', 'all_labels'
    outputpath = os.path.join(datadir, '{}{}.h5/{}'.format(dataset, opf, ods))
    all_labels = im1ds + im2ds
    mo = LabelImage(outputpath, **im1.get_props(dtype='uint32'))
    mo.create()
    mo.write(all_labels)
    mo.close()

opf='_labels_labelMA_comb' ods='all_labels'
h52nii '' "${dataset_ds}" "${opf}" "${ods}" '' '' '-i zyx -o xyz -d uint16'

def label_comb():
    import os
    import numpy as np
    from skimage.measure import label
    from wmem import utils, LabelImage
    datadir = '/Users/michielk/oxdata/P01/EM/Myrf_01/SET-B/B-NT-S10-2f_ROI_00'
    dataset = 'B-NT-S10-2f_ROI_00ds7'
    ipf, ids ='_labels_labelMA_comb', "all_labels"
    im2 = LabelImage(os.path.join(datadir, '{}{}.h5/{}'.format(dataset, ipf, ids)))
    im2.load()
    im2ds = im2.ds[:]
    im2.close()
    opf, ods = '_labels_labelMA_comb', 'all_labels'
    outputpath = os.path.join(datadir, '{}{}.h5/{}'.format(dataset, opf, ods))
    mask = im2ds > 0
    labels = label(mask)
    mo = LabelImage(outputpath, **im2.get_props(dtype='uint32'))
    mo.create()
    mo.write(labels)
    mo.close()

opf='_labels_labelMA_comb' ods='all_labels'
h52nii '' "${dataset_ds}" "${opf}" "${ods}" '' '' '-i zyx -o xyz -d uint16'

ipf='_labels_labelMA_comb' ids="all_labels"
opf='_labels_labelMA_comb' ods="all_labels_NoR"
python -W ignore $scriptdir/wmem/nodes_of_ranvier.py -S \
$datadir/${dataset_ds}${ipf}.h5/${ids} \
$datadir/${dataset_ds}${opf}.h5/${ods} \
--boundarymask $datadir/${dataset_ds}_masks_maskDS.h5/maskDS
h52nii '' "${dataset_ds}" "${opf}" "${ods}_steps/labels_tv" '' '' '-i zyx -o xyz -d uint16'
h52nii '' "${dataset_ds}" "${opf}" "${ods}_steps/labels_nt" '' '' '-i zyx -o xyz -d uint16'


ipf='_labels_labelMA_comb' ids="all_labels"
opf='_labels_labelMA_comb' ods="all_labels_filledm2"
q=2; o=0.50; methods='2'
python $scriptdir/wmem/fill_holes.py -S \
$datadir/${dataset_ds}${ipf}.h5/${ids} \
$datadir/${dataset_ds}${opf}.h5/${ods} \
-m ${methods} -s 9 9 9
h52nii '' $dataset_ds ${opf} ${ods} '' '' '-i zyx -o xyz -d uint16'













#python $scriptdir/wmem/merge_slicelabels.py -S \
#$datadir/${dataset_ds}_labels_labelMA_comb.h5/labelMA_aux3D \
#$datadir/${dataset_ds}_labels_labelMA_comb.h5/labelMA_aux3D_relabeled \
#-M -m 'MAfilter' -r 26114
#h52nii '' $dataset_ds '_labels_labelMA_comb' 'labelMA_aux3D_relabeled' '' '' '-i zyx -o xyz -d uint16'













python -W ignore $scriptdir/wmem/nodes_of_ranvier.py -S \
$datadir/${dataset_ds}_labels_labelMA_comb.h5/labelMA_nt_ns_closed_ns \
$datadir/${dataset_ds}_labels_labelMA_comb.h5/labelMA_nt_ns_closed_ns_NoR \
--boundarymask $datadir/${dataset_ds}_masks_maskDS.h5/maskDS
#number of labels in labelvolume: 11823
#number of short labels: 0
#number of large, long labels: 11823
#number of large, long in-volume labels: 11601
#number of large, long through-volume labels: 222
h52nii '' $dataset_ds '_labels_labelMA_comb' "labelMA_nt_ns_closed_ns_NoR_steps/labels_tv" '' '' '-i zyx -o xyz -d uint16'
h52nii '' $dataset_ds '_labels_labelMA_comb' "labelMA_nt_ns_closed_ns_NoR_steps/labels_nt" '' '' '-i zyx -o xyz -d uint16'
mpiexec -n 6 python -W ignore $scriptdir/wmem/merge_labels.py -S -M \
$datadir/${dataset_ds}_labels_labelMA_comb.h5/labelMA_nt_ns_closed_ns_NoR_steps/labels_tv \
$datadir/${dataset_ds}_labels_labelMA_comb.h5/labelMA_nt_ns_closed_ns_NoR_tv_ws_maskMM \
--data $datadir/${dataset_ds}_probs1_eed2.h5/probs_eed \
--maskMM $datadir/${dataset_ds}_masks_maskMM.h5/maskMM_PP \
-m 'watershed'
h52nii '' $dataset_ds '_labels_labelMA_comb' "labelMA_nt_ns_closed_ns_NoR_tv_ws_maskMM" '' '' '-i zyx -o xyz -d uint16'
h52nii '' $dataset_ds '_labels_labelMA_comb' "labelMA_nt_ns_closed_ns_NoR_tv_ws_maskMM_steps/stitched" '' '' '-i zyx -o xyz -d uint16'
h52nii '' $dataset_ds '_labels_labelMA_comb' "labelMA_nt_ns_closed_ns_NoR_tv_ws_maskMM_steps/filled" '' '' '-i zyx -o xyz -d uint16'


mpiexec -n 6 python -W ignore $scriptdir/wmem/merge_labels.py -S -M \
$datadir/${dataset_ds}_labels_labelMA_comb.h5/labelMA_nt_ns_closed_ns \
$datadir/${dataset_ds}_labels_labelMA_comb.h5/labelMA_nt_ns_closed_ns_ws_maskMM \
--data $datadir/${dataset_ds}_probs1.h5/probMA \
--maskMM $datadir/${dataset_ds}_masks_maskMM.h5/maskMM_PP \
-m 'watershed' -r 20 10 10

h52nii '' $dataset_ds '_labels_labelMA_comb' "labelMA_nt_ns_closed_ns_NoR_tv_ws_maskMM" '' '' '-i zyx -o xyz -d uint16'
h52nii '' $dataset_ds '_labels_labelMA_comb' "labelMA_nt_ns_closed_ns_NoR_tv_ws_maskMM_steps/stitched" '' '' '-i zyx -o xyz -d uint16'
h52nii '' $dataset_ds '_labels_labelMA_comb' "labelMA_nt_ns_closed_ns_NoR_tv_ws_maskMM_steps/filled" '' '' '-i zyx -o xyz -d uint16'


### try doing a 3D watershed with comb_nt as seeds; only accept small-ish labels
# (labels spilling out of the MA might tend to be BIG)

### AND/OR





python -W ignore $scriptdir/wmem/nodes_of_ranvier.py \
$datadir/${dataset_ds}_labels_labelMA_comb.h5/labelMA_nt \
$datadir/${dataset_ds}_labels_labelMA_comb.h5/labelMA_nt_NoR_ns \
--boundarymask $datadir/${dataset_ds}_masks_maskDS.h5/maskDS \
-m 'neighbours_slices'
#number of labels in labelvolume: 24655
#number of short labels: 0
#number of large, long labels: 24655
#number of large, long in-volume labels: 24548
#number of large, long through-volume labels: 107
h52nii '' $dataset_ds '_labels_labelMA_comb' "labelMA_nt_NoR_ns" '' '' '-i zyx -o xyz -d uint16'

### fill gaps in aggregated labels (TODO)
python $scriptdir/wmem/merge_slicelabels.py -S \
$datadir/${dataset_ds}_labels_labelMA_comb.h5/labelMA_nt_NoR_ns \
$datadir/${dataset_ds}_labels_labelMA_comb.h5/labelMA_nt_NoR_ns_filled-ms \
--maskMM $datadir/${dataset_ds}_masks_maskMM.h5/maskMM_PP \
-m 'MAfilter' -l 1 0 0  # 3 1 1
h52nii '' $dataset_ds '_labels_labelMA_comb' "labelMA_nt_NoR_ns_filled-ms" '' '' '-i zyx -o xyz -d uint16'

# TODO: clean up disconnected pieces?

### aggregate labels by overlap (threshold_overlap=0.20; offsets=2)
python -W ignore $scriptdir/wmem/nodes_of_ranvier.py \
$datadir/${dataset_ds}_labels_labelMA_comb.h5/labelMA_nt_NoR_ns_filled-ms \
$datadir/${dataset_ds}_labels_labelMA_comb.h5/labelMA_nt_NoR_ns_filled-ms-ns \
--boundarymask $datadir/${dataset_ds}_masks_maskDS.h5/maskDS \
-m 'neighbours_slices'
h52nii '' $dataset_ds '_labels_labelMA_comb' "labelMA_nt_NoR_ns_filled-ms-ns" '' '' '-i zyx -o xyz -d uint16'

python -W ignore $scriptdir/wmem/merge_labels.py \
$datadir/${dataset_ds}_labels_labelMA_comb.h5/labelMA_nt_NoR_ns_filled-ms \
$datadir/${dataset_ds}_labels_labelMA_comb.h5/labelMA_nt_NoR_ns_filled-ms-ns \
--boundarymask $datadir/${dataset_ds}_masks_maskDS.h5/maskDS \
-m 'neighbours_slices'
h52nii '' $dataset_ds '_labels_labelMA_comb' "labelMA_nt_NoR_ns_filled-ms-ns" '' '' '-i zyx -o xyz -d uint16'





### aggregate labels by watershed after fill
python -W ignore $scriptdir/wmem/nodes_of_ranvier.py -S \
$datadir/${dataset_ds}_labels_labelMA_comb.h5/labelMA_nt_NoR_ns_filled-ms \
$datadir/${dataset_ds}_labels_labelMA_comb.h5/labelMA_nt_NoR_ns_filled-ms_ws_maskMM \
--boundarymask $datadir/${dataset_ds}_masks_maskDS.h5/maskDS \
--data $datadir/${dataset_ds}_probs1_eed2.h5/probs_eed \
--maskMM $datadir/${dataset_ds}_masks_maskMM.h5/maskMM_PP \
-m 'watershed'
#number of labels in labelvolume: 16951
#number of short labels: 0
#number of large, long labels: 16951
#number of large, long in-volume labels: 16884
#number of large, long through-volume labels: 67
h52nii '' $dataset_ds '_labels_labelMA_comb' "labelMA_nt_NoR_ns_filled-ms_ws_maskMM" '' '' '-i zyx -o xyz -d uint16'
h52nii '' $dataset_ds '_labels_labelMA_comb' "labelMA_nt_NoR_ns_filled-ms_ws_maskMM_steps/filled" '' '' '-i zyx -o xyz -d uint16'

### aggregate labels by conncomp after fill (TODO: first apply maskMM to fill result)
python -W ignore $scriptdir/wmem/nodes_of_ranvier.py \
$datadir/${dataset_ds}_labels_labelMA_comb.h5/labelMA_nt_NoR_ns_filled-ms \
$datadir/${dataset_ds}_labels_labelMA_comb.h5/labelMA_nt_NoR_ns_filled-ms_cc \
--boundarymask $datadir/${dataset_ds}_masks_maskDS.h5/maskDS \
-m 'conncomp'
#number of labels in labelvolume: 16951
#number of short labels: 0
#number of large, long labels: 16951
#number of large, long in-volume labels: 16884
#number of large, long through-volume labels: 67
h52nii '' $dataset_ds '_labels_labelMA_comb' "labelMA_nt_NoR_ns_filled-ms_cc" '' '' '-i zyx -o xyz -d uint16'

### aggregate labels by overlap (threshold_overlap=0.80; for j in [1, 2, 3]:)
python -W ignore $scriptdir/wmem/nodes_of_ranvier.py \
$datadir/${dataset_ds}_labels_labelMA_comb.h5/labelMA_nt_NoR_ns_filled-ms \
$datadir/${dataset_ds}_labels_labelMA_comb.h5/labelMA_nt_NoR_ns_filled-ms_ns \
--boundarymask $datadir/${dataset_ds}_masks_maskDS.h5/maskDS \
-m 'neighbours_slices'
#number of labels in labelvolume: 16951
#number of short labels: 0
#number of large, long labels: 16951
#number of large, long in-volume labels: 16884
#number of large, long through-volume labels: 67
h52nii '' $dataset_ds '_labels_labelMA_comb' "labelMA_nt_NoR_ns_filled-ms_ns" '' '' '-i zyx -o xyz -d uint16'



python $scriptdir/wmem/merge_slicelabels.py \
$datadir/${dataset_ds}_labels_labelMA_comb.h5/labelMA_nt_NoR_ns \
$datadir/${dataset_ds}_labels_labelMA_comb.h5/labelMA_nt_NoR_ns_relabeled \
-M -m 'MAfilter' -r 1
h52nii '' $dataset_ds '_labels_labelMA_comb' "labelMA_nt_NoR_ns_relabeled" '' '' '-i zyx -o xyz -d uint16'



#python -W ignore $scriptdir/wmem/nodes_of_ranvier.py -S \
#$datadir/${dataset_ds}_labels_labelMA_core2D.h5/labelMA_pred_nocore3D_proofread_2Dmerge_q${q}-o${o} \
#$datadir/${dataset_ds}_labels_labelMA_core2D.h5/labelMA_pred_nocore3D_proofread_2Dmerge_q${q}-o${o}_NoR_cc \
#--boundarymask $datadir/${dataset_ds}_masks_maskDS.h5/maskDS \
#-m 'neighbours_slices'
#h52nii '' $dataset_ds '_labels_labelMA_core2D' "labelMA_pred_nocore3D_proofread_2Dmerge_q${q}-o${o}_NoR_cc" '' '' '-i zyx -o xyz -d uint16'
#h52nii '' $dataset_ds '_labels_labelMA_core2D' "labelMA_pred_nocore3D_proofread_2Dmerge_q${q}-o${o}_NoR_cc_steps/labels_nt" '' '' '-i zyx -o xyz -d uint16'
#h52nii '' $dataset_ds '_labels_labelMA_core2D' "labelMA_pred_nocore3D_proofread_2Dmerge_q${q}-o${o}_NoR_cc_steps/labels_tv" '' '' '-i zyx -o xyz -d uint16'

### fill gaps in aggregated labels (TODO)
python $scriptdir/wmem/merge_slicelabels.py -S \
$datadir/${dataset_ds}_labels_labelMA_core2D.h5/labelMA_pred_nocore3D_proofread_2Dmerge_q${q}-o${o}_NoR_cc \
$datadir/${dataset_ds}_labels_labelMA_core2D.h5/labelMA_pred_nocore3D_proofread_2Dmerge_q${q}-o${o}_NoR_cc_filled-ms \
-m 'MAfilter' -l 1 0 0  # 3 1 1
h52nii '' $dataset_ds '_labels_labelMA_core2D' "labelMA_pred_nocore3D_proofread_2Dmerge_q${q}-o${o}_NoR_cc_filled-ms" '' '' '-i zyx -o xyz -d uint16'




q=2; o=0.50; methods='2'
python $scriptdir/wmem/fill_holes.py -S \
$datadir/${dataset_ds}_labels_labelMA_core2D.h5/labelMA_pred_nocore3D_proofread_2Dmerge_q${q}-o${o}_NoR_cc \
$datadir/${dataset_ds}_labels_labelMA_core2D.h5/labelMA_pred_nocore3D_proofread_2Dmerge_q${q}-o${o}_NoR_cc_filledm2 \
-m ${methods} -s 9 9 9
h52nii '' $dataset_ds '_labels_labelMA_core2D' "labelMA_pred_nocore3D_proofread_2Dmerge_q${q}-o${o}_NoR_cc_filledm2" '' '' '-i zyx -o xyz -d uint16'


methods='42'; close=9
python $scriptdir/wmem/fill_holes.py -S \
$datadir/${dataset_ds}_labels_labelMA_core2D.h5/labelMA_pred_nocore3D_proofread_2Dmerge_q${q}-o${o} \
$datadir/${dataset_ds}_labels_labelMA_core2D.h5/labelMA_pred_nocore3D_proofread_2Dmerge_q${q}-o${o}_filled \
-w ${methods} \
--maskDS $datadir/${dataset_ds}_masks_maskDS.h5/maskDS \
--maskMM $datadir/${dataset_ds}_masks_maskMM.h5/maskMM_PP \
--maskMX $datadir/${dataset_ds}_masks_maskMM.h5/maskMM-0.02 \
--maskMA '_maskMA' 'stack'







###=========================================================================###
### PROOFREADING / BRUSHING UP myelinated axon compartment
###=========================================================================###

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
for i in `seq -f "%03g" 0 48`; do
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
PREFIX="${CONDA_PATH}/envs/neuroproof-test"
NPdir="${HOME}/workspace/Neuroproof_minimal"
datadir="${DATA}/EM/Neuroproof/M3_S1_GNU_NP" && cd $datadir


###=========================================================================###
### TODO: Neuroproof agglomeration
###=========================================================================###
