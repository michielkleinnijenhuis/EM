###=========================================================================###
### prepare environment
###=========================================================================###
scriptdir="$HOME/workspace/EM"
. "$scriptdir/pipelines/datasets.sh"
. "$scriptdir/pipelines/functions.sh"
. "$scriptdir/pipelines/submission.sh"
export PYTHONPATH="$PYTHONPATH:$HOME/workspace/pyDM3reader"  # FIXME
export PYTHONPATH="$PYTHONPATH:$HOME/workspace/maskSLIC"  # FIXME

#compute_env='JAL'
#compute_env='ARC'
#compute_env='ARCB'
# compute_env='LOCAL'
compute_env='RIOS013'
# compute_env='HPC'
prep_environment "$scriptdir" "$compute_env"

# dataset='M3S1GNU'
# dataset='B-NT-S9-2a'
# dataset='B-NT-S10-2d_ROI_00'
# dataset='B-NT-S10-2d_ROI_02'
dataset='B-NT-S10-2f_ROI_00'
# dataset='B-NT-S10-2f_ROI_01'
# dataset='B-NT-S10-2f_ROI_02'

bs='0500' && prep_dataset "$dataset" "$bs" && echo "${#datastems[@]}"

#echo -n -e "\033]0;$dataset\007"

export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8

core3D='_labels_labelMA_core3D_test'
core2D='_labels_labelMA_core2D_test'
comb='_labels_labelMA_comb_test'

###=========================================================================###
### convert and register  # setup for ARCUS-B
###=========================================================================###

### convert dm3 files (NOTE: paths set up in function 'dataset_parameters')
jid=''
scriptfile=$( dm3convert 'h' )
[[ -z $jid ]] && dep='' || dep="--dependency=after:$jid"
jid=$( sbatch $dep $scriptfile ) && jid=${jid##* }

### register slices (NOTE: does not handle montage, see earlier M3S1GNU processing)
jid=''
scriptfile=$( fiji_register 'h' )
[[ -z $jid ]] && dep='' || dep="--dependency=after:$jid"
jid=$( sbatch $dep $scriptfile ) && jid=${jid##* }

### convert the series of tifs to h5
opf=''
ods='data'
scriptfile=$( tif2h5 'h' '' $ods )
[[ -z $jid ]] && dep='' || dep="--dependency=after:$jid"
JID=$( sbatch $dep $scriptfile ) && jid_data=${JID##* }

### create downsampled dataset
ipf=''
ids='data'
opf="$ipf"
ods="$ids"
brfun='np.mean'
brvol=''
slab=20
memcpu=60000
wtime='02:00:00'
vol_slice=''
scriptfile=$( blockreduce 'h' $ipf $ids $opf $ods $brfun '' $slab $memcpu $wtime $vol_slice )
[[ -z $JID ]] && dep='' || dep="--dependency=after:$JID"
jid=$( sbatch $dep $scriptfile ) && jid=${jid##* }

# threshold the data at 0 to get a dataset mask (maskDS)
ipf=''
ids='data'
opf='_masks_maskDS'
ods='maskDS'
slab=20
arg='-g -l 0 -u 10000000'
scriptfile=$( prob2mask 'h' '' $ids $opf $ods $slab $arg )
[[ -z $JID ]] && dep='' || dep="--dependency=after:$JID"
jid=$( sbatch $dep $scriptfile ) && jid=${jid##* }

### create downsampled maskDS
ipf='_masks_maskDS'
ids='maskDS'
opf="$ipf"
ods="$ids"
brfun='np.amax'
brvol=''
slab=20
memcpu=60000
wtime='00:10:00'
vol_slice=''
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
ipf=''
ids='data'
opf='_probs'
ods='volume/predictions'
clf="pixclass_8class"
scriptfile=$( apply_ilastik 'h' '' $ids $opf $ods $clf )
jid=$( sbatch -p devel $dep $scriptfile ) && jid=${jid##* }

### correct element_size_um  # TODO: create function
scriptfile="$datadir/EM_corr_script.py"
write_ilastik_correct_attributes "$scriptfile"
jobname='correct_ilastik'
additions=''
CONDA_ENV=''
nodes=1
memcpu=3000
wtime='00:10:00'
tasks=1
njobs=1
q='h'
scriptfile=$( single_job python $scriptfile )
[[ -z $jid ]] && dep='' || dep="--dependency=after:$jid"
jid=$( sbatch -p devel $dep $scriptfile ) && jid=${jid##* }

# TODO?: blockreduce '' 4 1

### split and sum
ipf='_probs'
ids='volume/predictions'
opf="$ipf"
ods="$ids"
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
ipf='_probs'
ids='volume/predictions'
opf='_probs_sum0247'
ods='sum0247'
vols='0 2 4 7'
# sum_volumes 'h' 'a' $ipf $ids $opf $ods $vols
scriptfile=$( sum_volumes 'h' 'a' $ipf $ids $opf $ods $vols )
. "$scriptfile"

### sum myelinated ICS and unmyelinated ICS compartments
ipf='_probs'
ids='volume/predictions'
opf='_probs_sum16'
ods='sum16'
vols='1 6'
# sum_volumes '' 'a' $ipf $ids $opf $ods $vols
scriptfile=$( sum_volumes 'h' 'a' $ipf $ids $opf $ods $vols )
. "$scriptfile"

### extract myelinated ICS compartment
ipf='_probs'
ids='volume/predictions'
opf='_probs_probMA'
ods='probMA'
vols='1'
# sum_volumes '' 'a' $ipf $ids $opf $ods $vols
scriptfile=$( sum_volumes 'h' 'a' $ipf $ids $opf $ods $vols )
. "$scriptfile"


# ###=========================================================================###
# ### reduce MA prediction
# ###=========================================================================###
# ipf='_probs_probMA'
# ids='probMA'
# opf="$ipf"
# ods="$ids"
# args='-d float16'
# # mergeblocks 'h' '' '' $ipf $ids $opf $ods $args
# scriptfile=$( mergeblocks 'h' '' '' $ipf $ids $opf $ods $args )
# . "$scriptfile"

# ipf='_probs_probMA'
# ids='probMA'
# opf="$ipf"
# ods="$ids"
# brfun='np.mean'
# brvol=''
# slab=12
# memcpu=60000
# wtime='02:00:00'
# vol_slice=''
# # blockreduce 'h' $ipf $ids $opf $ods $brfun '' $slab $memcpu $wtime $vol_slice
# scriptfile=$( blockreduce 'h' $ipf $ids $opf $ods $brfun '' $slab $memcpu $wtime $vol_slice )
# [[ -z $jid ]] && dep='' || dep="--dependency=after:$jid"
# jid=$( sbatch $dep $scriptfile ) && jid=${jid##* }


###=========================================================================###
### edge-enhancing diffusion  # setup for JALAPENO
# rsync -Pazv $host_arc:$datadir_arc/blocks_0500/*_sum*.h5 $datadir/blocks_0500/
# rsync -Pazv $host_arc:$datadir_arc/blocks_0500/*_probMA.h5 $datadir/blocks_0500/
# TODO: slabsize for blockreduce
###=========================================================================###
jid=''  # TODO: link to sum volumes
### smooth the summed probability map with edge-enhancing diffusion
ipf='_probs_sum0247'
ids='sum0247'
opf='_probs_eed_sum0247'
ods='sum0247_eed'
wtime='03:10:00'
scriptfile=$( eed 'h' 'a' $ipf $ids $opf $ods $wtime )
[[ -z $jid ]] && dep='' || dep="-j $jid"
JID=$( fsl_sub -q short.q $dep $scriptfile )
### merge the blocks
ipf='_probs_eed_sum0247'
ids='sum0247_eed'
opf="$ipf"
ods="$ids"
args='-d float16'
scriptfile=$( mergeblocks 'h' '' '' $ipf $ids $opf $ods $args )
[[ -z $JID ]] && dep='' || dep="-j $JID"
jid=$( fsl_sub -q short.q $dep $scriptfile )
### downsample the volume
ipf='_probs_eed_sum0247'
ids='sum0247_eed'
opf="$ipf"
ods="$ids"
brfun='np.mean'
brvol=''
slab=12
memcpu=60000
wtime='02:00:00'
vol_slice=''
scriptfile=$( blockreduce 'h' $ipf $ids $opf $ods $brfun '' $slab $memcpu $wtime $vol_slice )
[[ -z $jid ]] && dep='' || dep="-j $jid"
jid=$( fsl_sub -q short.q $dep $scriptfile )
### generate the raw myelin mask (NOTE: replaced in favour of data threshold)
ipf='_probs_eed_sum0247'
ids='sum0247_eed'
opf='_masks_maskMM'
ods='maskMM'
args='-g -l 0.5 -s 2000 -d 1 -S'
scriptfile=$( prob2mask_datastems 'h' 'a' $ipf $ids $opf $ods $args )
[[ -z $JID ]] && dep='' || dep="-j $JID"
jid=$( fsl_sub -q veryshort.q $dep $scriptfile )
### merge the blocks
ipf='_masks_maskMM'
ids='maskMM_steps/raw'
opf='_masks_maskMM'
ods='maskMM_raw'
args=''  # FIXME: args seemed to be missing here: args='-d float16'?
scriptfile=$( mergeblocks 'h' '' '' $ipf $ids $opf $ods $args )
[[ -z $jid ]] && dep='' || dep="-j $jid"
jid=$( fsl_sub -q veryshort.q $dep $scriptfile )
### remove small components and dilate
ipf='_masks_maskMM'
ids='maskMM_raw'
opf='_masks_maskMM'
ods='maskMM'
slab=12
args='-g -l 0 -u 0 -s 2000 -d 1'
scriptfile=$( prob2mask 'h' $ipf $ids $opf $ods $slab $args )
[[ -z $jid ]] && dep='' || dep="-j $jid"
jid=$( fsl_sub -q veryshort.q $dep $scriptfile )
### downsample the volume
ipf='_masks_maskMM'
ids='maskMM'
opf="$ipf"
ods="$ids"
brfun='np.amax'
brvol=''
slab=27
memcpu=60000
wtime='00:10:00'
vol_slice=''
scriptfile=$( blockreduce 'h' $ipf $ids $opf $ods $brfun '' $slab $memcpu $wtime $vol_slice )
[[ -z $jid ]] && dep='' || dep="-j $jid"
jid=$( fsl_sub -q veryshort.q $dep $scriptfile )


jid=''  # TODO: link to sum volumes
### smooth the summed probability map with edge-enhancing diffusion
ipf='_probs_sum16'
ids='sum16'
opf='_probs_eed_sum16'
ods='sum16_eed'
wtime='03:10:00'
scriptfile=$( eed 'h' 'a' $ipf $ids $opf $ods $wtime )
[[ -z $jid ]] && dep='' || dep="-j $jid"
JID=$( fsl_sub -q short.q $dep $scriptfile )
### merge the blocks
ipf='_probs_eed_sum16'
ids='sum16_eed'
opf="$ipf"
ods="$ids"
args='-d float16'
scriptfile=$( mergeblocks 'h' '' '' $ipf $ids $opf $ods $args )
[[ -z $JID ]] && dep='' || dep="-j $JID"
jid=$( fsl_sub -q short.q $dep $scriptfile )
### downsample the volume
ipf='_probs_eed_sum16'
ids='sum16_eed'
opf="$ipf"
ods="$ids"
brfun='np.mean'
brvol=''
slab=12
memcpu=60000
wtime='02:00:00'
vol_slice=''
scriptfile=$( blockreduce 'h' $ipf $ids $opf $ods $brfun '' $slab $memcpu $wtime $vol_slice )
[[ -z $jid ]] && dep='' || dep="-j $jid"
jid=$( fsl_sub -q short.q $dep $scriptfile )
### generate the ICS mask
ipf='_probs_eed_sum16'
ids='sum16_eed'
opf='_masks_maskICS'
ods='maskICS'
args='-g -l 0.2'
scriptfile=$( prob2mask_datastems 'h' 'a' $ipf $ids $opf $ods $args )
[[ -z $JID ]] && dep='' || dep="-j $JID"
jid=$( fsl_sub -q veryshort.q $dep $scriptfile )
### merge the blocks
ipf='_masks_maskICS'
ids='maskICS'
opf="$ipf"
ods="$ids"
args=''  # FIXME: args seemed to be missing here: args='-d float16'?
scriptfile=$( mergeblocks 'h' '' '' $ipf $ids $opf $ods $args )
[[ -z $jid ]] && dep='' || dep="-j $jid"
jid=$( fsl_sub -q veryshort.q $dep $scriptfile )
### downsample the volume
ipf='_masks_maskICS'
ids='maskICS'
opf="$ipf"
ods="$ids"
brfun='np.amax'
brvol=''
slab=27
memcpu=60000
wtime='00:10:00'
vol_slice=''
scriptfile=$( blockreduce 'h' $ipf $ids $opf $ods $brfun '' $slab $memcpu $wtime $vol_slice )
[[ -z $jid ]] && dep='' || dep="-j $jid"
jid=$( fsl_sub -q veryshort.q $dep $scriptfile )


jid=''  # TODO: link to sum volumes
### smooth the summed probability map with edge-enhancing diffusion
ipf='_probs_probMA'
ids='probMA'
opf='_probs_eed_probMA'
ods='probMA_eed'
wtime='03:10:00'
scriptfile=$( eed 'h' 'a' $ipf $ids $opf $ods $wtime )
[[ -z $jid ]] && dep='' || dep="-j $jid"
JID=$( fsl_sub -q short.q $dep $scriptfile )
### merge the blocks
ipf='_probs_eed_probMA'
ids='probMA_eed'
opf="$ipf"
ods="$ids"
args='-d float16'
scriptfile=$( mergeblocks 'h' '' '' $ipf $ids $opf $ods $args )
[[ -z $JID ]] && dep='' || dep="-j $JID"
jid=$( fsl_sub -q short.q $dep $scriptfile )
### downsample the volume
ipf='_probs_eed_probMA'
ids='probMA_eed'
opf="$ipf"
ods="$ids"
brfun='np.mean'
brvol=''
slab=12
memcpu=60000
wtime='02:00:00'
vol_slice=''
scriptfile=$( blockreduce 'h' $ipf $ids $opf $ods $brfun '' $slab $memcpu $wtime $vol_slice )
[[ -z $jid ]] && dep='' || dep="-j $jid"
jid=$( fsl_sub -q short.q $dep $scriptfile )
### generate the myelinated axon mask
ipf='_probs_eed_probMA'
ids='probMA_eed'
opf='_masks_maskMA'
ods='maskMA'
args='-g -l 0.2'
scriptfile=$( prob2mask_datastems 'h' 'a' $ipf $ids $opf $ods $args )
[[ -z $JID ]] && dep='' || dep="-j $JID"
jid=$( fsl_sub -q veryshort.q $dep $scriptfile )
### merge the blocks
ipf='_masks_maskMA'
ids='maskMA'
opf="$ipf"
ods="$ids"
args=''  # FIXME: args seemed to be missing here: args='-d float16'?
scriptfile=$( mergeblocks 'h' '' '' $ipf $ids $opf $ods $args )
[[ -z $jid ]] && dep='' || dep="-j $jid"
jid=$( fsl_sub -q veryshort.q $dep $scriptfile )
### downsample the volume
ipf='_masks_maskMA'
ids='maskMA'
opf="$ipf"
ods="$ids"
brfun='np.amax'
brvol=''
slab=15
memcpu=60000
wtime='00:10:00'
vol_slice=''
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
# NB: a new maskMM was introduced (inserted from scratch_EM6.py and scratch_EM9.py)
###=========================================================================###
### generate the raw myelin mask
ipf=''
ids='data'
opf='_masks_maskMM'
ods='maskMM'
slab=12
args='-g -l -1 -u 26000 -d 1'
scriptfile=$( prob2mask 'h' '' $ids $opf $ods $slab $args )
. "$scriptfile"
### remove small objects from mask
ipf='_masks_maskMM'
ids='maskMM'
opf="$ipf"
ods="${ids}_PP"
scriptfile="$datadir/EM_maskMM_PP.py"
echo "from wmem import prob2mask" > "$scriptfile"
echo "prob2mask.preprocess_mask(image_in='$datadir/$dataset_ds$ipf.h5/$ids', outputpath='$datadir/$dataset_ds$opf.h5/$ods', min_size=1000)" >> "$scriptfile"
python -W ignore "$scriptfile"


###=========================================================================###
### 3D labeling
###=========================================================================###

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
echo "from wmem import nodes_of_ranvier" > "$scriptfile"
echo "nodes_of_ranvier.correct_NoR(image_in='$datadir/$dataset_ds$ipf.h5/$ids')" >> "$scriptfile"
python -W ignore "$scriptfile"

# TODO: improve nodes_of_ranvier.py module for splitting tv/nt such that this proofreading is unnecessary


###=========================================================================###
### myelinated axon compartment
###=========================================================================###

### label ICS components in 2D
jid=''  # TODO: link to maskMM and _probs_eed_sum16.h5/sum16_eed
ipf='_masks_maskMM'
ids='maskMM'
opf="$core2D"
ods='labelMA_core2D'
meanint='dummy'
scriptfile=$( conncomp 'h' '2D' $dataset $ipf $ids $opf $ods $meanint )
[[ -z $jid ]] && dep='' || dep="-j $jid"
jid=$( fsl_sub -q short.q $dep $scriptfile )
### map all label properties to numpy vectors
ipf="$core2D"
ids='labelMA_core2D'
opf='_labels_labelMA'
ods='dummy'
meanint='_probs_eed_sum16.h5/sum16_eed'
scriptfile=$( conncomp 'h' '2Dfilter' $dataset $ipf $ids $opf $ods $meanint )
[[ -z $jid ]] && dep='' || dep="-j $jid"
jid=$( fsl_sub -q long.q $dep $scriptfile )
# TODO: train and apply full size classifier

### label ICS components in 2D
jid=''  # TODO: link to *ds7_masks_maskMM and *ds7_probs_eed_sum16.h5/sum16_eed
ipf='_masks_maskMM'
ids='maskMM'
opf="$core2D"
ods='labelMA_core2D'
meanint='dummy'
scriptfile=$( conncomp 'h' '2D' $dataset_ds $ipf $ids $opf $ods $meanint )
[[ -z $jid ]] && dep='' || dep="-j $jid"
jid=$( fsl_sub -q short.q $dep $scriptfile )
### map all label properties to numpy vectors
ipf="$core2D"
ids='labelMA_core2D'
opf='_labels_labelMA'
ods='dummy'
meanint='_probs_eed_sum16.h5/sum16_eed'
scriptfile=$( conncomp 'h' '2Dfilter' $dataset_ds $ipf $ids $opf $ods $meanint )
[[ -z $jid ]] && dep='' || dep="-j $jid"
jid=$( fsl_sub -q short.q $dep $scriptfile )
### map the properties to volumes for inspection
ipf="$core2D"
ids='labelMA_core2D'
opf='_labels_labelMA'
ods='dummy'
meanint='dummy'
scriptfile=$( conncomp 'h' '2Dprops' $dataset_ds $ipf $ids $opf $ods $meanint )
[[ -z $jid ]] && dep='' || dep="-j $jid"
jid=$( fsl_sub -q veryshort.q $dep $scriptfile )
### apply 2D label classifier to the labels to identify MA labels
ipf="$core2D"
ids='labelMA_core2D'
opf='_labels_labelMA'
ods='labelMA_pred'
meanint='dummy'
clfpath="$scriptdir/clf.pkl"
scalerpath="$scriptdir/scaler.pkl"
args='-A 3000 -I 0.8'
scriptfile=$( conncomp 'h' 'test' $dataset_ds $ipf $ids $opf $ods $meanint $clfpath $scalerpath $args )
[[ -z $jid ]] && dep='' || dep="-j $jid"
jid=$( fsl_sub -q long.q $dep $scriptfile )
### aggregate 2D labels to 3D labels
ipf='_labels_labelMA'
ids='labelMA_pred'
opf='_labels_labelMA'
ods='labelMA_3Dlabeled'
meanint='dummy'
scriptfile=$( conncomp 'h' '2Dto3D' $dataset_ds $ipf $ids $opf $ods $meanint )
[[ -z $jid ]] && dep='' || dep="-j $jid"
jid=$( fsl_sub -q veryshort.q $dep $scriptfile )

# TODO: might want to keep all this in opf='_labels_labelMA_core2D'; not in '_labels_labelMA'


###=========================================================================###
### 2D labeling postprocessing
###=========================================================================###

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
# NB: mpi-enabled version not functioning properly!
# q=2; o=0.50;
# ipf="$core2D"
# ids='labelMA_pred_nocore3D_proofread'
# opf="$ipf"
# ods="${ids}_2Dmerge_q${q}-o${o}"
# args="-M -m 'MAstitch' -d 0 -q $q -o $o"
# scriptfile=$( merge_slicelabels_mpi 'h' $dataset_ds $args )
# . "$scriptfile"
q=2; o=0.50;
ipf="$core2D"
ids='labelMA_pred_nocore3D_proofread'
opf="$ipf"
ods="${ids}_2Dmerge_q${q}-o${o}"
args="-M -m 'MAstitch' -d 0 -q $q -o $o"
scriptfile=$( merge_slicelabels 'h' $dataset_ds $args )
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
# FIXME?: mpi-enabled version not functioning properly!
# ipf="$core2D"
# ids="labelMA_pred_nocore3D_proofread_2Dmerge_q${q}-o${o}_closed"
# opf="$ipf"
# ods="${ids}_ns"
# args="-m 'neighbours_slices' -q 2 -o 0.20 -M -S"
# scriptfile=$( merge_labels_ws 'h' $dataset_ds $ipf $ids $opf $ods $args )
# . "$scriptfile"
# NOTE: mpi disabled for env RIOS013
ipf="$core2D"
ids="labelMA_pred_nocore3D_proofread_2Dmerge_q${q}-o${o}_closed"
opf="$ipf"
ods="${ids}_ns"
args="-m 'neighbours_slices' -q 2 -o 0.20 -S"
scriptfile=$( merge_labels_ws 'h' $dataset_ds $ipf $ids $opf $ods $args )
. "$scriptfile"

### relabel the aggregated 2D labels starting at the maxlabel of labelMA_core3D_proofread
maxlabel_core3D=1604
ipf="$core2D"
ids="labelMA_pred_nocore3D_proofread_2Dmerge_q${q}-o${o}_closed_ns"
opf="$ipf"
ods="${ids}_relabeled"
args="-m 'MAfilter' -r $maxlabel_core3D"
scriptfile=$( merge_slicelabels 'h' $dataset_ds $args )
. "$scriptfile"

### merge aggregated 2D labels with core3D_nt
ipf1="$core2D"
ids1="labelMA_pred_nocore3D_proofread_2Dmerge_q${q}-o${o}_closed_ns_relabeled"
ipf2="$core3D"
ids2='labelMA_core3D_proofread_NoR_steps/labels_nt'
opf='_labels_labelMA_comb_test'
ods='labelMA_nt'
args='-m add'
scriptfile=$( combine_labels 'h' $dataset_ds $args )
. "$scriptfile"











###=========================================================================###
### ....
###=========================================================================###

### making sure labels are contiguous
ipf="$comb"
ids='labelMA_nt'
opf="$ipf"
ods="${ids}_split"
scriptfile="$datadir/EM_cont.py"
echo "from wmem import merge_labels" > "$scriptfile"
echo "merge_labels.fill_connected_labels(\
image_in='$datadir/$dataset_ds$ipf.h5/$ids', \
outputpath='$datadir/$dataset_ds$opf.h5/$ods', \
data_in='$datadir/$dataset_ds.h5/data', \
maskMM_in='$datadir/${dataset_ds}_masks_maskMM.h5/maskMM_PP', \
check_split=True, checkonly=False, between=False, to_border=False)" >> "$scriptfile"
python -W ignore "$scriptfile"

h52nii '' "$dataset_ds" "${opf}" "${ods}" '' '' '-i zyx -o xyz -d uint16'

### making sure labels are contiguous  # OKAY
ipf="$core3D"
ids='labelMA_core3D_proofread_NoR_steps/labels_tv'
opf="$ipf"
ods="${ids}_split"
scriptfile="$datadir/EM_cont.py"
echo "from wmem import merge_labels" > "$scriptfile"
echo "merge_labels.fill_connected_labels(\
image_in='$datadir/$dataset_ds$ipf.h5/$ids', \
outputpath='$datadir/$dataset_ds$opf.h5/$ods', \
data_in='$datadir/$dataset_ds.h5/data', \
maskMM_in='$datadir/${dataset_ds}_masks_maskMM.h5/maskMM_PP', \
check_split=True, checkonly=True, between=False, to_border=False)" >> "$scriptfile"
python -W ignore "$scriptfile"

h52nii '' "$dataset_ds" "${opf}" "${ods}" '' '' '-i zyx -o xyz -d uint16'


srz=10 iter=0 toborder=1
# srz=20 iter=1 previter=10
# srz=40 iter=1 previter=10
# srz=80 iter=2 previter=40  # mostly finds the right connection, but doesn't result in a clean fill # the cylinder centre fill over to long a linear distance leads to errors
# srz=81 iter=3 previter=80  # too many things going wrong here;


shscript0="$datadir/EM_WSiter${iter}_script.sh"  && > "$shscript0"
pyscript1="$datadir/EM_WSiter${iter}_pyscript1.py" && > "$pyscript1"
pyscript2="$datadir/EM_WSiter${iter}_pyscript2.py" && > "$pyscript2"

### create WSmask_iter? ###
echo "python -W ignore \"$pyscript1\"" >> "$shscript0"
mpf='_masks_maskWS'
mds="maskWS_iter$((iter-1))"
if [ "$iter" == "0" ];
then
    ipf="$core3D"
    ids='labelMA_core3D_proofread_NoR_steps/labels_tv'
    mask_in="$datadir/${dataset_ds}_masks_maskMM.h5/maskMM_PP"
    mask_ds="$datadir/${dataset_ds}_masks_maskDS.h5/maskDS"
else
    ipf='_labels_labelMA_WS'
    ids="labelMA_ws${previter}_split_NoR_steps/labels_tv"
    mask_in="$datadir/$dataset_ds$mpf.h5/$mds"
    mask_ds=''
fi
image_in="$datadir/$dataset_ds$ipf.h5/$ids"
opf="$mpf"
ods="maskWS_iter$iter"
outputpath="$datadir/$dataset_ds$opf.h5/$ods"
echo "from wmem import merge_labels" > "$pyscript1"
echo "merge_labels.create_mask('$outputpath', '$image_in', '$mask_in', '$mask_ds')" >> "$pyscript1"
echo "h52nii '' "$dataset_ds" "${opf}" "${ods}" '' '' '-i zyx -o xyz -d uint16'" >> "$shscript0"

### find connection candidates
dpf=''
dds='data'
mpf='_masks_maskWS'
mds="maskWS_iter${iter}"
ods="labelMA_ws$srz"
if [ "$iter" == "0" ];
then
    ipf="$comb"
    ids="labelMA_nt_split"
    opf='_labels_labelMA_WS'
else
    ipf='_labels_labelMA_WS'
    ids="labelMA_ws${previter}_split_NoR_steps/labels_nt"
    opf="$ipf"
fi

if [ "$iter" == "3" ];
then
    # args="$datadir/${dataset_ds}${ipf}.h5/${ids} $datadir/${dataset_ds}${opf}.h5/${ods} --data $datadir/${dataset_ds}${dpf}.h5/${dds} -m 'watershed' -r $srz 10 10 -M"
    args="$datadir/${dataset_ds}${ipf}.h5/${ids} $datadir/${dataset_ds}${opf}.h5/${ods} --data $datadir/${dataset_ds}${dpf}.h5/${dds} -m 'watershed' -r $srz 10 10"
else
    # args="$datadir/${dataset_ds}${ipf}.h5/${ids} $datadir/${dataset_ds}${opf}.h5/${ods} --data $datadir/${dataset_ds}${dpf}.h5/${dds} --maskDS $datadir/${dataset_ds}${mpf}.h5/${mds} -m 'watershed' -r $srz 10 10 -M"
    args="$datadir/${dataset_ds}${ipf}.h5/${ids} $datadir/${dataset_ds}${opf}.h5/${ods} --data $datadir/${dataset_ds}${dpf}.h5/${dds} --maskDS $datadir/${dataset_ds}${mpf}.h5/${mds} -m 'watershed' -r $srz 10 10"
fi
# echo "conda activate h5para" >> "$shscript0"
# echo "mpiexec -n 14 python -W ignore $scriptdir/wmem/merge_labels.py -S $in $out $args" >> "$shscript0"
echo "python -W ignore $scriptdir/wmem/merge_labels.py -S $in $out $args" >> "$shscript0"
echo "h52nii '' "$dataset_ds" "${opf}" "${ods}" '' '' '-i zyx -o xyz -d uint16'" >> "$shscript0"

### filling between merged labels
echo "python -W ignore \"$pyscript2\"" >> "$shscript0"
echo "from wmem import merge_labels" > "$pyscript2"
mpf='_masks_maskMM'
mds="maskMM_PP"
ipf='_labels_labelMA_WS'
ids="labelMA_ws$srz"
opf="$ipf"
ods="${ids}_between"
echo "merge_labels.fill_connected_labels(\
'$datadir/${dataset_ds}${ipf}.h5/${ids}', \
'$datadir/${dataset_ds}.h5/data', \
'$datadir/${dataset_ds}${mpf}.h5/${mds}', \
searchradius=[$srz, 10, 10], between=True, \
outputpath='$datadir/${dataset_ds}${opf}.h5/${ods}')" >> "$pyscript2"
echo "h52nii '' "$dataset_ds" "${opf}" "${ods}" '' '' '-i zyx -o xyz -d uint16'" >> "$shscript0"
if [ "$iter" == "3" ];
then
    ipf='_labels_labelMA_WS'
    ids="labelMA_ws${srz}_between"
    opf="$ipf"
    ods="${ids}_toborder"
    echo "merge_labels.fill_connected_labels('$datadir/${dataset_ds}${ipf}.h5/${ids}', \
    '$datadir/${dataset_ds}.h5/data', '$datadir/${dataset_ds}${mpf}.h5/${mds}', \
    searchradius=[40, 30, 30], between=True, \
    outputpath='$datadir/${dataset_ds}${opf}.h5/${ods}')" >> "$pyscript2"
    echo "h52nii '' "$dataset_ds" "${opf}" "${ods}" '' '' '-i zyx -o xyz -d uint16'" >> "$shscript0"
    ipf='_labels_labelMA_WS'
    ids="labelMA_ws${srz}_between_toborder"
else
    ipf='_labels_labelMA_WS'
    ids="labelMA_ws${srz}_between"
fi
opf='_labels_labelMA_WS'
ods="labelMA_ws${srz}_split"
echo "merge_labels.fill_connected_labels(\
'$datadir/${dataset_ds}${ipf}.h5/${ids}', \
check_split=True, \
outputpath='$datadir/${dataset_ds}${opf}.h5/${ods}')" >> "$pyscript2"
echo "h52nii '' "$dataset_ds" "${opf}" "${ods}" '' '' '-i zyx -o xyz -d uint16'" >> "$shscript0"

### separating finished labels and segments
ipf='_labels_labelMA_WS'
ids="labelMA_ws${srz}_split"
opf=$ipf
ods="${ids}_NoR"
echo "python -W ignore $scriptdir/wmem/nodes_of_ranvier.py -S \
$datadir/${dataset_ds}${ipf}.h5/${ids} \
$datadir/${dataset_ds}${opf}.h5/${ods} \
--boundarymask $datadir/${dataset_ds}_masks_maskDS.h5/maskDS -s 500 -R" >> "$shscript0"
echo "h52nii '' "${dataset_ds}" "${opf}" "${ods}_steps/labels_tv" '' '' '-i zyx -o xyz -d uint16'" >> "$shscript0"
echo "h52nii '' "${dataset_ds}" "${opf}" "${ods}_steps/labels_nt" '' '' '-i zyx -o xyz -d uint16'" >> "$shscript0"

. "$shscript0"







python -W ignore "/Users/mkleinnijenhuis/oxdata/P01/EM/Myrf_01/SET-B/B-NT-S10-2f_ROI_00/EM_WSiter0_pyscript1.py"
h52nii '' B-NT-S10-2f_ROI_00ds7 _masks_maskWS maskWS_iter0 '' '' '-i zyx -o xyz -d uint16'
# conda activate h5para
# mpiexec -n 14
python -W ignore /Users/mkleinnijenhuis/workspace/EM/wmem/merge_labels.py -S   /Users/mkleinnijenhuis/oxdata/P01/EM/Myrf_01/SET-B/B-NT-S10-2f_ROI_00/B-NT-S10-2f_ROI_00ds7_labels_labelMA_comb_test.h5/labelMA_nt_split /Users/mkleinnijenhuis/oxdata/P01/EM/Myrf_01/SET-B/B-NT-S10-2f_ROI_00/B-NT-S10-2f_ROI_00ds7_labels_labelMA_WS.h5/labelMA_ws10 --data /Users/mkleinnijenhuis/oxdata/P01/EM/Myrf_01/SET-B/B-NT-S10-2f_ROI_00/B-NT-S10-2f_ROI_00ds7.h5/data --maskDS /Users/mkleinnijenhuis/oxdata/P01/EM/Myrf_01/SET-B/B-NT-S10-2f_ROI_00/B-NT-S10-2f_ROI_00ds7_masks_maskWS.h5/maskWS_iter0 -m 'watershed' -r 10 10 10
 # -M
h52nii '' B-NT-S10-2f_ROI_00ds7 _labels_labelMA_WS labelMA_ws10 '' '' '-i zyx -o xyz -d uint16'

python -W ignore /Users/mkleinnijenhuis/oxdata/P01/EM/Myrf_01/SET-B/B-NT-S10-2f_ROI_00/EM_WSiter0_pyscript2.py
h52nii '' B-NT-S10-2f_ROI_00ds7 _labels_labelMA_WS labelMA_ws10_between '' '' '-i zyx -o xyz -d uint16'
h52nii '' B-NT-S10-2f_ROI_00ds7 _labels_labelMA_WS labelMA_ws10_split '' '' '-i zyx -o xyz -d uint16'

python -W ignore /Users/mkleinnijenhuis/workspace/EM/wmem/nodes_of_ranvier.py -S /Users/mkleinnijenhuis/oxdata/P01/EM/Myrf_01/SET-B/B-NT-S10-2f_ROI_00/B-NT-S10-2f_ROI_00ds7_labels_labelMA_WS.h5/labelMA_ws10_split /Users/mkleinnijenhuis/oxdata/P01/EM/Myrf_01/SET-B/B-NT-S10-2f_ROI_00/B-NT-S10-2f_ROI_00ds7_labels_labelMA_WS.h5/labelMA_ws10_split_NoR --boundarymask /Users/mkleinnijenhuis/oxdata/P01/EM/Myrf_01/SET-B/B-NT-S10-2f_ROI_00/B-NT-S10-2f_ROI_00ds7_masks_maskDS.h5/maskDS -s 500 -R
h52nii '' B-NT-S10-2f_ROI_00ds7 _labels_labelMA_WS labelMA_ws10_split_NoR_steps/labels_tv '' '' '-i zyx -o xyz -d uint16'
h52nii '' B-NT-S10-2f_ROI_00ds7 _labels_labelMA_WS labelMA_ws10_split_NoR_steps/labels_nt '' '' '-i zyx -o xyz -d uint16'



# TODO: force to border
# TODO: test borderconnect after first iteration





























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
