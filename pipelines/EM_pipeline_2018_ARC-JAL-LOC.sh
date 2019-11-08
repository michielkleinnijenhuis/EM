conda activate EM

###=========================================================================###
### prepare environment
###=========================================================================###
scriptdir="$HOME/workspace/EM"
. "$scriptdir/pipelines/datasets.sh"
. "$scriptdir/pipelines/functions.sh"
. "$scriptdir/pipelines/submission.sh"
export PYTHONPATH="$PYTHONPATH:$HOME/workspace/pyDM3reader"  # FIXME
export PYTHONPATH="$PYTHONPATH:$HOME/workspace/maskSLIC"  # FIXME

# compute_env='JAL'
# compute_env='ARC'
# compute_env='ARCB'
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

#bs='0500' && prep_dataset "$dataset" "$bs" && echo "${#datastems[@]}"
bs='0700' && prep_dataset $dataset $bs && echo ${#datastems[@]}
xs=700 ys=700 zs=184  # blocksizes
xm=35 ym=35 zm=0  # margins
datastems_blocks && echo ${#datastems[@]}
blockdir=$datadir/blocks_$bs

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


# NOTE: do a NoR nt/tv filtering before continuing with watershed stage?



###=========================================================================###
### watershed merge of label segments
###=========================================================================###

### making sure labels are contiguous ###
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

### making sure labels are contiguous: already OKAY ###
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


### iterative watershed merge with progressively larger extent in z ###

# srz=10 iter=0 toborder=1
# srz=40 iter=1 previter=10
# srz=80 iter=2 previter=40  # mostly finds the right connection, but doesn't result in a clean fill # the cylinder centre fill over to long a linear distance leads to errors
srz=81 iter=3 previter=80  # too many things going wrong here;

shscript0="$datadir/EM_WSiter${iter}_script.sh"  && > "$shscript0"
pyscript1="$datadir/EM_WSiter${iter}_pyscript1.py" && > "$pyscript1"
pyscript2="$datadir/EM_WSiter${iter}_pyscript2.py" && > "$pyscript2"

### create WSmask_iter?
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

# find connection candidates
dpf=''
dds='data'
mpf='_masks_maskWS'
mds="maskWS_iter${iter}"
ods="labelMA_ws$srz"
# h5copy -i -o -s -d
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
# FIXME: not necessary anymore: changed to direct write of filled labels
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
    searchradius=[40, 30, 30], between=True, to_border=True, \
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

# TODO: force to border
# TODO: test borderconnect after first iteration
# FIXME: this section needs work: continue with half-baked MA compartment for now


###=========================================================================###
### add up all MA labels
###=========================================================================###
opf='_labels_labelMA_2D3D' ods='labelMA_WS'
ipf1='_labels_labelMA_WS' ids1='labelMA_ws10_split_NoR_steps/labels_tv'
ipf2='_labels_labelMA_WS' ids2='labelMA_ws40_split_NoR_steps/labels_tv'
python $scriptdir/wmem/combine_labels.py \
    "${datadir}/${dataset_ds}${ipf1}.h5/${ids1}" \
    "${datadir}/${dataset_ds}${ipf2}.h5/${ids2}" \
    "${datadir}/${dataset_ds}${opf}.h5/${ods}" -m 'add'

ipf1='_labels_labelMA_2D3D' ids1='labelMA_WS'
ipf2='_labels_labelMA_WS' ids2='labelMA_ws80_split_NoR_steps/labels_tv'
python $scriptdir/wmem/combine_labels.py \
    "${datadir}/${dataset_ds}${ipf1}.h5/${ids1}" \
    "${datadir}/${dataset_ds}${ipf2}.h5/${ids2}" \
    "${datadir}/${dataset_ds}${opf}.h5/${ods}" -m 'add'

ipf1='_labels_labelMA_2D3D' ids1='labelMA_WS'
ipf2='_labels_labelMA_WS' ids2='labelMA_ws81_split_NoR_steps/labels_tv'
python $scriptdir/wmem/combine_labels.py \
    "${datadir}/${dataset_ds}${ipf1}.h5/${ids1}" \
    "${datadir}/${dataset_ds}${ipf2}.h5/${ids2}" \
    "${datadir}/${dataset_ds}${opf}.h5/${ods}" -m 'add'

ipf1='_labels_labelMA_2D3D' ids1='labelMA_WS'
ipf2='_labels_labelMA_WS' ids2='labelMA_ws81_split_NoR_steps/labels_nt'
python $scriptdir/wmem/combine_labels.py \
    "${datadir}/${dataset_ds}${ipf1}.h5/${ids1}" \
    "${datadir}/${dataset_ds}${ipf2}.h5/${ids2}" \
    "${datadir}/${dataset_ds}${opf}.h5/${ods}" -m 'add'

opf='_labels_labelMA_2D3D' ods='labelMA'
ipf1='_labels_labelMA_2D3D' ids1='labelMA_WS'
ipf2='_labels_labelMA_core3D_test' ids2='labelMA_core3D_proofread_NoR_steps/labels_tv'
python $scriptdir/wmem/combine_labels.py \
    "${datadir}/${dataset_ds}${ipf1}.h5/${ids1}" \
    "${datadir}/${dataset_ds}${ipf2}.h5/${ids2}" \
    "${datadir}/${dataset_ds}${opf}.h5/${ods}" -m 'add'

h52nii '' B-NT-S10-2f_ROI_00ds7 _labels_labelMA_2D3D labelMA '' '' '-i zyx -o xyz -d uint16'
h52nii '' B-NT-S10-2f_ROI_00ds7 _labels_labelMA_2D3D labelMA_WS '' '' '-i zyx -o xyz -d uint16'


###=========================================================================###
### NOTE: there is mismatch between maskMM_PP and the labelMA (holes at mito's: bottom only)
### this should fix it
### TODO: identify where this went wrong
###=========================================================================###
ipf='_labels_labelMA_2D3D' ids='labelMA'
opf='_labels_labelMA_2D3D' ods='labelMA_filledm3'
methods='3'
python $scriptdir/wmem/fill_holes.py -S \
    $datadir/${dataset_ds}${ipf}.h5/${ids} \
    $datadir/${dataset_ds}${opf}.h5/${ods} \
    -m ${methods}
h52nii '' "${dataset_ds}" "${opf}" "${ods}" '' '' '-i zyx -o xyz -d uint16'
### update maskMM
ipf1='_masks_maskMM' ids1='maskMM_PP'
ipf2='_labels_labelMA_2D3D' ids2='labelMA_filledm3'
opf='_masks_maskMM' ods='maskMM_PP_filledm3'
python $scriptdir/wmem/combine_labels.py \
    "${datadir}/${dataset_ds}${ipf1}.h5/${ids1}" \
    "${datadir}/${dataset_ds}${ipf2}.h5/${ids2}" \
    "${datadir}/${dataset_ds}${opf}.h5/${ods}" -m 'mask'
h52nii '' "${dataset_ds}" "${opf}" "${ods}" '' '' '-i zyx -o xyz -d uint8'


###=========================================================================###
### detect nodes of Ranvier
###=========================================================================###
from wmem import nodes_of_ranvier
nodes_of_ranvier.detect_NoR(
    image_in='/Users/mkleinnijenhuis/oxdata/P01/EM/Myrf_01/SET-B/B-NT-S10-2f_ROI_00/B-NT-S10-2f_ROI_00ds7_labels_labelMA_2D3D.h5/labelMA_filledm3',
    maskMM='/Users/mkleinnijenhuis/oxdata/P01/EM/Myrf_01/SET-B/B-NT-S10-2f_ROI_00/B-NT-S10-2f_ROI_00ds7_masks_maskMM.h5/maskMM_PP_filledm3',
    encapsulate_threshold=0.8,
    outputpath='/Users/mkleinnijenhuis/oxdata/P01/EM/Myrf_01/SET-B/B-NT-S10-2f_ROI_00/B-NT-S10-2f_ROI_00ds7_labels_labelMA_2D3D.h5/labelMA_filledm3_nodes_thr0.8')
opf='_labels_labelMA_2D3D' ods='labelMA_filledm3_nodes_thr0.8'
h52nii '' "${dataset_ds}" "${opf}" "${ods}" '' '' '-i zyx -o xyz -d uint16'
opf='_labels_labelMA_2D3D' ods='labelMA_filledm3_nodes_thr0.8_steps/seg'
h52nii '' "${dataset_ds}" "${opf}" "${ods}" '' '' '-i zyx -o xyz -d uint8'
opf='_labels_labelMA_2D3D' ods='labelMA_filledm3_nodes_thr0.8_steps/rim'
h52nii '' "${dataset_ds}" "${opf}" "${ods}" '' '' '-i zyx -o xyz -d uint16'
opf='_labels_labelMA_2D3D' ods='labelMA_filledm3_nodes_thr0.8_steps/nonodes'
h52nii '' "${dataset_ds}" "${opf}" "${ods}" '' '' '-i zyx -o xyz -d uint16'

### get a wsmask /sheaths (TODO?: test distance threshold)
spf='_labels_labelMA_2D3D' sds='labelMA_filledm3_nodes_thr0.8_steps/nonodes'
mpf='_masks_maskMM' mds='maskMM_PP_filledm3'
opf='_labels_labelMM' ods="labelMM_nonodes"
python $scriptdir/wmem/separate_sheaths.py \
"$datadir/${dataset_ds}${spf}.h5/${sds}" \
"$datadir/${dataset_ds}${opf}.h5/${ods}" \
--maskMM "$datadir/${dataset_ds}${mpf}.h5/${mds}" -S
h52nii '' "${dataset_ds}" "${opf}" "${ods}_steps/sheaths" '' '' '-i zyx -o xyz -d uint16'
h52nii '' "${dataset_ds}" "${opf}" "${ods}_steps/wsmask" '' '' '-i zyx -o xyz'
h52nii '' "${dataset_ds}" "${opf}" "${ods}_steps/distance" '' '' '-i zyx -o xyz'

### get medwidths
weight=10.0; margin=20;
spf='_labels_labelMA_2D3D' sds='labelMA_filledm3_nodes_thr0.8_steps/nonodes'
mpf='_masks_maskMM' mds='maskMM_PP_filledm3'  # previous wsmask? it seems too restrictive
# wpf='_labels_labelMM' wds='labelMM_nonodes_steps/wsmask'
lpf='_labels_labelMM' lds="labelMM_nonodes_steps/sheaths"
opf='_labels_labelMM' ods="labelMM_nonodes_sigmoid_iter1"
python $scriptdir/wmem/separate_sheaths.py \
"$datadir/${dataset_ds}${spf}.h5/${sds}" \
"$datadir/${dataset_ds}${opf}.h5/${ods}" \
--maskMM "$datadir/${dataset_ds}${mpf}.h5/${mds}" \
--labelMM "$datadir/${dataset_ds}${lpf}.h5/${lds}" \
-S -w $weight -m $margin
h52nii '' "${dataset_ds}" "${opf}" "${ods}_steps/sheaths_sigmoid_10.0" '' '' '-i zyx -o xyz -d uint16'
h52nii '' "${dataset_ds}" "${opf}" "${ods}_steps/distance_sigmoid_10.0" '' '' '-i zyx -o xyz'
h52nii '' "${dataset_ds}" "${opf}" "${ods}_steps/wsmask_sigmoid_10.0" '' '' '-i zyx -o xyz'  # 1.5 * medwidth

# TODO: iterate

### add MA and MM to create MF
ipf1='_labels_labelMA_2D3D' ids1='labelMA_filledm3'
ipf2='_labels_labelMM' ids2="labelMM_nonodes_sigmoid_iter1_steps/sheaths_sigmoid_10.0"
opf='_labels_labelMF' ods="labelMF_nonodes"
python $scriptdir/wmem/combine_labels.py \
"${datadir}/${dataset_ds}${ipf1}.h5/${ids1}" \
"${datadir}/${dataset_ds}${ipf2}.h5/${ids2}" \
"${datadir}/${dataset_ds}${opf}.h5/${ods}" -m 'add'
h52nii '' "${dataset_ds}" "${opf}" "${ods}" '' '' '-i zyx -o xyz -d uint16'

### fill holes in MF
ipf='_labels_labelMF' ids="labelMF_nonodes"
opf='_labels_labelMF' ods="labelMF_nonodes_filledm3"
methods='3'
python $scriptdir/wmem/fill_holes.py -S \
$datadir/${dataset_ds}${ipf}.h5/${ids} \
$datadir/${dataset_ds}${opf}.h5/${ods} \
-m ${methods}
h52nii '' "${dataset_ds}" "${opf}" "${ods}" '' '' '-i zyx -o xyz -d uint16'











### WIP: mito

### get mitochondria
from wmem import utils, Image, LabelImage, MaskImage, nodes_of_ranvier
import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.measure import regionprops
from scipy.ndimage.filters import gaussian_filter
import pickle

def get_coords(prop, margin, dims):

    if len(prop.bbox) > 4:
        z, y, x, Z, Y, X = tuple(prop.bbox)
        z = max(0, z - margin)
        Z = min(dims[0], Z + margin)
    else:
        y, x, Y, X = tuple(prop.bbox)
        z = 0
        Z = 1

    y = max(0, y - margin)
    x = max(0, x - margin)
    Y = min(dims[1], Y + margin)
    X = min(dims[2], X + margin)

    return x, X, y, Y, z, Z

# FIXME: probably want to use the MF version with NoR removed as it is for extracting the sheaths
image_in = '/Users/mkleinnijenhuis/oxdata/P01/EM/Myrf_01/SET-B/B-NT-S10-2f_ROI_00/B-NT-S10-2f_ROI_00ds7_labels_labelMF.h5/labelMF_nonodes_filledm3'
mf = utils.get_image(image_in, imtype='Label')
# image_in = outputpath = '/Users/mkleinnijenhuis/oxdata/P01/EM/Myrf_01/SET-B/B-NT-S10-2f_ROI_00/B-NT-S10-2f_ROI_00ds7_labels_labelMA_2D3D.h5/labelMA_filledm2m3'
# ma = utils.get_image(image_in, imtype='Label')
# data_smoothed = gaussian_filter(data, sigma)

# ma = np.copy(mf.ds[:])
mask = np.zeros_like(mf.ds[:], dtype='bool')

do_distsum = False
if do_distsum:
    distsum = np.zeros_like(mf.ds[:], dtype='float')
else:
    image_in = '/Users/mkleinnijenhuis/oxdata/P01/EM/Myrf_01/SET-B/B-NT-S10-2f_ROI_00/B-NT-S10-2f_ROI_00ds7_labels_labelMF.h5/distsum'
    distsum = utils.get_image(image_in)
    distsum = distsum.ds[:]
    # load median sheath widths if provided
    medwidth_file = '/Users/mkleinnijenhuis/oxdata/P01/EM/Myrf_01/SET-B/B-NT-S10-2f_ROI_00/B-NT-S10-2f_ROI_00ds7_labels_labelMM_labelMM_nonodes_sigmoid_iter1.pickle'
    with open(medwidth_file, 'rb') as f:
        medwidths = pickle.load(f)

rp = regionprops(mf.ds[:])
margin = 10
elsize = np.absolute(mf.elsize)
dist_thr = 0.25
for prop in rp:
    print(prop.label)
    # get data cutout labels with margin
    x, X, y, Y, z, Z = get_coords(prop, margin, mf.ds.shape)
    MF_region = mf.ds[z:Z, y:Y, x:X]
    distsum_region = distsum[z:Z, y:Y, x:X]
    mask_region = mask[z:Z, y:Y, x:X]

    # get label distance map
    maskMF = MF_region == prop.label
    if do_distsum:
        dist = distance_transform_edt(maskMF, sampling=elsize)
        distsum_region = np.maximum(distsum_region, dist)
        distsum[z:Z, y:Y, x:X] = distsum_region
    else:
        # update the mask
        # dist_thr = 0.25
        if prop.label in medwidths.keys():
            dist_thr = medwidths[prop.label] * 0.5  # + 0.05  # * 1.5 # + 0.1
        else:
            dist_thr = 0  # TODO: check if this is the desired default
        dist_mask = np.logical_and(maskMF, distsum_region > dist_thr)
        np.logical_or(mask_region, dist_mask, mask_region)
        mask[z:Z, y:Y, x:X] = mask_region


if do_distsum:
    outputpath = '/Users/mkleinnijenhuis/oxdata/P01/EM/Myrf_01/SET-B/B-NT-S10-2f_ROI_00/B-NT-S10-2f_ROI_00ds7_labels_labelMF.h5/distsum'
    mo = Image(outputpath, **mf.get_props(dtype=distsum.dtype))
    mo.create()
    mo.write(distsum)
    mo.close()
else:
    outputpath = '/Users/mkleinnijenhuis/oxdata/P01/EM/Myrf_01/SET-B/B-NT-S10-2f_ROI_00/B-NT-S10-2f_ROI_00ds7_labels_labelMF.h5/maskMA'
    outputpath = '/Users/mkleinnijenhuis/oxdata/P01/EM/Myrf_01/SET-B/B-NT-S10-2f_ROI_00/B-NT-S10-2f_ROI_00ds7_labels_labelMF.h5/maskMA-0.05'
    outputpath = '/Users/mkleinnijenhuis/oxdata/P01/EM/Myrf_01/SET-B/B-NT-S10-2f_ROI_00/B-NT-S10-2f_ROI_00ds7_labels_labelMF.h5/maskMA-0.1'
    outputpath = '/Users/mkleinnijenhuis/oxdata/P01/EM/Myrf_01/SET-B/B-NT-S10-2f_ROI_00/B-NT-S10-2f_ROI_00ds7_labels_labelMF.h5/maskMAx1.5'
    outputpath = '/Users/mkleinnijenhuis/oxdata/P01/EM/Myrf_01/SET-B/B-NT-S10-2f_ROI_00/B-NT-S10-2f_ROI_00ds7_labels_labelMF.h5/maskMAx0.5'
    mo = MaskImage(outputpath, **mf.get_props(dtype='uint8'))
    mo.create()
    mo.write(mask)
    mo.close()

    ma = np.copy(mf.ds[:])
    ma[~mask] = 0
    outputpath = '/Users/mkleinnijenhuis/oxdata/P01/EM/Myrf_01/SET-B/B-NT-S10-2f_ROI_00/B-NT-S10-2f_ROI_00ds7_labels_labelMF.h5/labelMA'
    mo = Image(outputpath, **mf.get_props())
    mo.create()
    mo.write(ma)
    mo.close()

    mm = np.copy(mf.ds[:])
    mm[mask] = 0
    outputpath = '/Users/mkleinnijenhuis/oxdata/P01/EM/Myrf_01/SET-B/B-NT-S10-2f_ROI_00/B-NT-S10-2f_ROI_00ds7_labels_labelMF.h5/labelMM'
    mo = Image(outputpath, **mf.get_props())
    mo.create()
    mo.write(mm)
    mo.close()

    maskUA = ~mf.ds[:].astype('bool')
    outputpath = '/Users/mkleinnijenhuis/oxdata/P01/EM/Myrf_01/SET-B/B-NT-S10-2f_ROI_00/B-NT-S10-2f_ROI_00ds7_labels_labelMF.h5/maskUA'
    mo = MaskImage(outputpath, **mf.get_props(dtype='uint8'))
    mo.create()
    mo.write(maskUA)
    mo.close()

ipf=_labels_labelMF; ids=distsum
python -W ignore $scriptdir/wmem/stack2stack.py $datadir/${dataset_ds}${ipf}.h5/$ids $datadir/${dataset_ds}${ipf}_$ids.nii.gz -i zyx -o xyz
ipf=_labels_labelMF; ids=maskMA
python -W ignore $scriptdir/wmem/stack2stack.py $datadir/${dataset_ds}${ipf}.h5/$ids $datadir/${dataset_ds}${ipf}_$ids.nii.gz -i zyx -o xyz
ipf=_labels_labelMF; ids=maskMA-0.05
python -W ignore $scriptdir/wmem/stack2stack.py $datadir/${dataset_ds}${ipf}.h5/$ids $datadir/${dataset_ds}${ipf}_$ids.nii.gz -i zyx -o xyz
ipf=_labels_labelMF; ids=maskMA-0.1
python -W ignore $scriptdir/wmem/stack2stack.py $datadir/${dataset_ds}${ipf}.h5/$ids $datadir/${dataset_ds}${ipf}_$ids.nii.gz -i zyx -o xyz
ipf=_labels_labelMF; ids=maskMAx1.5
python -W ignore $scriptdir/wmem/stack2stack.py $datadir/${dataset_ds}${ipf}.h5/$ids $datadir/${dataset_ds}${ipf}_$ids.nii.gz -i zyx -o xyz
ipf=_labels_labelMF; ids=maskMAx0.5
python -W ignore $scriptdir/wmem/stack2stack.py $datadir/${dataset_ds}${ipf}.h5/$ids $datadir/${dataset_ds}${ipf}_$ids.nii.gz -i zyx -o xyz
sipf=_labels_labelMF; ids=labelMA
python -W ignore $scriptdir/wmem/stack2stack.py $datadir/${dataset_ds}${ipf}.h5/$ids $datadir/${dataset_ds}${ipf}_$ids.nii.gz -i zyx -o xyz
ipf=_labels_labelMF; ids=labelMM
python -W ignore $scriptdir/wmem/stack2stack.py $datadir/${dataset_ds}${ipf}.h5/$ids $datadir/${dataset_ds}${ipf}_$ids.nii.gz -i zyx -o xyz
ipf=_labels_labelMF; ids=maskUA
python -W ignore $scriptdir/wmem/stack2stack.py $datadir/${dataset_ds}${ipf}.h5/$ids $datadir/${dataset_ds}${ipf}_$ids.nii.gz -i zyx -o xyz









###=========================================================================###
### FROM HERE IS WIP / SCRATCH / OLD
###=========================================================================###

### image gradient  # TODO: check if sigma/2 actually does anything (ggm_sigma1 without anisotropic looks a lot better)
from wmem import utils, Image
import numpy as np
from scipy.ndimage import gaussian_gradient_magnitude
from scipy.ndimage.filters import gaussian_filter

image_in = '/Users/mkleinnijenhuis/oxdata/P01/EM/Myrf_01/SET-B/B-NT-S10-2f_ROI_00/B-NT-S10-2f_ROI_00ds7.h5/data'
im = utils.get_image(image_in)
data = im.ds[:].astype('float32')

for sigma in [1, 3, 5]:
    ggm = gaussian_gradient_magnitude(data, sigma=[sigma/2, sigma, sigma])
    outputpath = '/Users/mkleinnijenhuis/oxdata/P01/EM/Myrf_01/SET-B/B-NT-S10-2f_ROI_00/B-NT-S10-2f_ROI_00ds7.h5/ggm_sigma{}'.format(sigma)
    mo = Image(outputpath, **im.get_props(dtype='float32'))
    mo.create()
    mo.write(ggm)
    mo.close()
    im.close()

for sigma in [1, 3, 5]:
    data_smoothed = gaussian_filter(data, sigma=[sigma/2, sigma, sigma])
    outputpath = '/Users/mkleinnijenhuis/oxdata/P01/EM/Myrf_01/SET-B/B-NT-S10-2f_ROI_00/B-NT-S10-2f_ROI_00ds7.h5/data_sigma{}'.format(sigma)
    mo = Image(outputpath, **im.get_props(dtype='float32'))
    mo.create()
    mo.write(data_smoothed)
    mo.close()

for sigma in [0, 1]:
    sigma1 = sigma
    for sigma2 in [sigma + 1, sigma + 2]:
        if sigma1 == 0:
            dog = data - gaussian_filter(data, [sigma2/2, sigma2, sigma2])
        else:
            dog = gaussian_filter(data, [sigma1/2, sigma1, sigma1]) - gaussian_filter(data, [sigma2/2, sigma2, sigma2])
        outputpath = '/Users/mkleinnijenhuis/oxdata/P01/EM/Myrf_01/SET-B/B-NT-S10-2f_ROI_00/B-NT-S10-2f_ROI_00ds7.h5/DoG_{}-{}'.format(sigma1, sigma2)
        mo = Image(outputpath, **im.get_props(dtype='float32'))
        mo.create()
        mo.write(dog)
        mo.close()

im.close()

ipf=
for ids in ggm_sigma1 ggm_sigma3 ggm_sigma5 data_sigma1 data_sigma3 data_sigma5 DoG_0-1 DoG_0-2 DoG_1-2 DoG_1-3; do
    python -W ignore $scriptdir/wmem/stack2stack.py $datadir/${dataset_ds}${ipf}.h5/$ids $datadir/${dataset_ds}${ipf}_$ids.nii.gz -i zyx -o xyz
done



### detect mitochondria
from wmem import utils, Image, LabelImage, MaskImage, nodes_of_ranvier
import numpy as np
from scipy import ndimage as ndi
from scipy.ndimage import distance_transform_edt
from scipy.ndimage.filters import gaussian_filter
from skimage.measure import regionprops
from skimage.morphology import watershed, remove_small_objects
from wmem import utils, Image, LabelImage, MaskImage
from skimage.morphology import binary_dilation, binary_erosion, cube
import numpy as np

def find_local_maxima(img, min_dist=1, threshold=0.05):

    size = 2 * min_dist + 1
    if threshold == -float('Inf'):
        threshold = img.min()

    image_max = ndi.maximum_filter(img, size=size, mode='constant')
    mask = img == image_max
    mask &= img > threshold
    coordinates = np.column_stack(np.nonzero(mask))[::-1]
    # coordinates = peak_local_max(dog, min_distance=1, indices=False)
    out = np.zeros_like(img, dtype=np.bool)
    out[tuple(coordinates.T)] = True

    return out


image_in = '/Users/mkleinnijenhuis/oxdata/P01/EM/Myrf_01/SET-B/B-NT-S10-2f_ROI_00/B-NT-S10-2f_ROI_00ds7.h5/data_sigma1'
im = utils.get_image(image_in)
data = im.ds[:]
data_inv = np.absolute(data - np.amax(data))
outputpath = '/Users/mkleinnijenhuis/oxdata/P01/EM/Myrf_01/SET-B/B-NT-S10-2f_ROI_00/B-NT-S10-2f_ROI_00ds7_labels_labelMT.h5/data_inv'
mo = Image(outputpath, **im.get_props())
mo.create()
mo.write(data_inv)
mo.close()

image_in = '/Users/mkleinnijenhuis/oxdata/P01/EM/Myrf_01/SET-B/B-NT-S10-2f_ROI_00/B-NT-S10-2f_ROI_00ds7_labels_labelMF.h5/maskMAx1.5'
mask01 = utils.get_image(image_in)
mask = mask01.ds[:].astype('bool')
data_inv[~mask] = 0
out = find_local_maxima(data_inv, min_dist=2, threshold=1000)
outputpath = '/Users/mkleinnijenhuis/oxdata/P01/EM/Myrf_01/SET-B/B-NT-S10-2f_ROI_00/B-NT-S10-2f_ROI_00ds7_labels_labelMT.h5/peaks'
mo = MaskImage(outputpath, **im.get_props(dtype='uint8'))
mo.create()
mo.write(out)
mo.close()

data_inv = np.absolute(data - np.amax(data))
markers = ndi.label(out)[0]
mask_ws = np.logical_and(mask, data_inv > 1000)
labels = watershed(data_inv, markers, mask=mask_ws)
# remove single voxels (or clusters smaller than ~3)
remove_small_objects(labels, min_size=3, connectivity=1, in_place=True)  # 3 may already be too much
outputpath = '/Users/mkleinnijenhuis/oxdata/P01/EM/Myrf_01/SET-B/B-NT-S10-2f_ROI_00/B-NT-S10-2f_ROI_00ds7_labels_labelMT.h5/mito'
mo = Image(outputpath, **im.get_props(dtype=labels.dtype))
mo.create()
mo.write(labels)
mo.close()

ipf=_labels_labelMT; ids=data_inv
python -W ignore $scriptdir/wmem/stack2stack.py $datadir/${dataset_ds}${ipf}.h5/$ids $datadir/${dataset_ds}${ipf}_$ids.nii.gz -i zyx -o xyz
ipf=_labels_labelMT; ids=peaks
python -W ignore $scriptdir/wmem/stack2stack.py $datadir/${dataset_ds}${ipf}.h5/$ids $datadir/${dataset_ds}${ipf}_$ids.nii.gz -i zyx -o xyz
ipf=_labels_labelMT; ids=mito
python -W ignore $scriptdir/wmem/stack2stack.py $datadir/${dataset_ds}${ipf}.h5/$ids $datadir/${dataset_ds}${ipf}_$ids.nii.gz -i zyx -o xyz -d uint16






### snap to gradient magnitude
## TODO: put these functions somewhere

def find_max_ggm(axon, ggm):

    ggm_rim = [0]
    patch = np.copy(axon)
    for ii in range(0, 6):
        newpatch = binary_erosion(patch)
        rim = np.logical_xor(newpatch, patch)
        ggm_rim.append(np.mean(ggm[rim]))
        patch = newpatch

    neros = np.argmax(ggm_rim)

    patch = np.copy(axon)
    for ii in range(0, neros):
        newpatch = binary_erosion(patch)
        patch = newpatch

    print(ii, ggm_rim, neros)

    return ggm_rim, patch


def snap_to_max_gradient(image_in, ggm, go2D=False, outputpath=''):

    # Read inputs
    axons = utils.get_image(image_in, imtype='Label')
    ggm = utils.get_image(ggm, imtype='Mask')

    # Create outputs
    props = axons.get_props(protective=False)
    outpaths = {'out': outputpath}
    outpaths = utils.gen_steps(outpaths, save_steps=True)
    mo = LabelImage(outpaths['out'], **props)
    mo.create()
#     mo.ds[:] = axons.ds[:]

    for prop in regionprops(axons.ds):
#         if prop.label not in [730]:
#             continue
        print(prop.label)

        # Slice the axon region.
        slices = get_region_slices_around(axons, prop, searchradius=[1, 1, 1])[0]
        axons.slices = ggm.slices = mo.slices = slices
        axons_slcd = axons.slice_dataset(squeeze=False) == prop.label
        ggm_slcd = ggm.slice_dataset(squeeze=False)
        mo_slcd = mo.slice_dataset(squeeze=False)

        if go2D:
            iter_imgs = zip(axons_slcd, ggm_slcd, mo_slcd)
            for i, (slc, slc_ggm, slc_mo) in enumerate(iter_imgs):
                ggm_rim, patch = find_max_ggm(slc, slc_ggm)
                slc_mo[patch] = prop.label
        else:
            ggm_rim, patch = find_max_ggm(axons_slcd, ggm_slcd)
            mo_slcd[patch] = prop.label

        mo.write(mo_slcd)

    # Close images.
    ggm.close()
    axons.close()
    mo.close()

    return ggm_rim



from wmem import nodes_of_ranvier
nodes_of_ranvier.snap_to_max_gradient(
    image_in='/Users/mkleinnijenhuis/oxdata/P01/EM/Myrf_01/SET-B/B-NT-S10-2f_ROI_00/B-NT-S10-2f_ROI_00ds7_labels_labelMF.h5/labelMF_nonodes_filledm3',
    ggm='/Users/mkleinnijenhuis/oxdata/P01/EM/Myrf_01/SET-B/B-NT-S10-2f_ROI_00/B-NT-S10-2f_ROI_00ds7.h5/ggm_sigma1',
    go2D=True,
    outputpath='/Users/mkleinnijenhuis/oxdata/P01/EM/Myrf_01/SET-B/B-NT-S10-2f_ROI_00/B-NT-S10-2f_ROI_00ds7_labels_labelMF.h5/labelMF_nonodes_filledm3_ero-ggm2D')

opf='_labels_labelMF' ods='labelMF_nonodes_filledm3_ero-ggm'
h52nii '' "${dataset_ds}" "${opf}" "${ods}" '' '' '-i zyx -o xyz -d uint16'



# from wmem import utils, Image, LabelImage, MaskImage, nodes_of_ranvier
# import numpy as np
# from skimage.segmentation import inverse_gaussian_gradient, morphological_geodesic_active_contour
# image_in = '/Users/mkleinnijenhuis/oxdata/P01/EM/Myrf_01/SET-B/B-NT-S10-2f_ROI_00/B-NT-S10-2f_ROI_00ds7.h5/data'
# im = utils.get_image(image_in)
# data = im.ds[:]
# igg = inverse_gaussian_gradient(data, alpha=20.0, sigma=1.0)
# outputpath = '/Users/mkleinnijenhuis/oxdata/P01/EM/Myrf_01/SET-B/B-NT-S10-2f_ROI_00/B-NT-S10-2f_ROI_00ds7.h5/igg'
# mo = Image(outputpath, **im.get_props(dtype='float32'))
# mo.create()
# mo.write(igg.astype('float32'))
# mo.close()
#
# ipf=; ids=igg
# python -W ignore $scriptdir/wmem/stack2stack.py $datadir/${dataset_ds}${ipf}.h5/$ids $datadir/${dataset_ds}${ipf}_$ids.nii.gz -i zyx -o xyz
#
# image_in = '/Users/mkleinnijenhuis/oxdata/P01/EM/Myrf_01/SET-B/B-NT-S10-2f_ROI_00/B-NT-S10-2f_ROI_00ds7.h5/igg'
# im = utils.get_image(image_in)
# igg = im.ds[:]
# morphological_geodesic_active_contour(igg, iterations, init_level_set='circle', smoothing=1, threshold='auto', balloon=0)



### update labelMA by subtracting sheaths from MF_filled
ipf1='_labels_labelMF' ids1='labelMF_filledm3'
ipf2='_labels_labelMM' ids2="labelMM_steps/sheaths"
opf='_labels_labelMA_2D3D' ods='labelMA_filledm2m3'
python $scriptdir/wmem/combine_labels.py \
"${datadir}/${dataset_ds}${ipf1}.h5/${ids1}" \
"${datadir}/${dataset_ds}${ipf2}.h5/${ids2}" \
"${datadir}/${dataset_ds}${opf}.h5/${ods}" -m 'mask'
h52nii '' "${dataset_ds}" "${opf}" "${ods}" '' '' '-i zyx -o xyz -d uint16'



### making sure labels in 'labelMA_filledm2m3' are contiguous
from wmem import merge_labels
merge_labels.fill_connected_labels('/Users/mkleinnijenhuis/oxdata/P01/EM/Myrf_01/SET-B/B-NT-S10-2f_ROI_00/B-NT-S10-2f_ROI_00ds7_labels_labelMA_2D3D.h5/labelMA_filledm2m3', check_split=True, outputpath='/Users/mkleinnijenhuis/oxdata/P01/EM/Myrf_01/SET-B/B-NT-S10-2f_ROI_00/B-NT-S10-2f_ROI_00ds7_labels_labelMA_2D3D.h5/labelMA_filledm2m3_split')

opf='_labels_labelMA_2D3D' ods='labelMA_filledm2m3_split'
h52nii '' "${dataset_ds}" "${opf}" "${ods}" '' '' '-i zyx -o xyz -d uint16'

### making sure labels in 'labelMF_filledm3' are contiguous
from wmem import merge_labels
merge_labels.fill_connected_labels('/Users/mkleinnijenhuis/oxdata/P01/EM/Myrf_01/SET-B/B-NT-S10-2f_ROI_00/B-NT-S10-2f_ROI_00ds7_labels_labelMF.h5/labelMF_filledm3', check_split=True, outputpath='/Users/mkleinnijenhuis/oxdata/P01/EM/Myrf_01/SET-B/B-NT-S10-2f_ROI_00/B-NT-S10-2f_ROI_00ds7_labels_labelMF.h5/labelMF_filledm3_split')

opf='_labels_labelMF' ods='labelMF_filledm3_split'
h52nii '' "${dataset_ds}" "${opf}" "${ods}" '' '' '-i zyx -o xyz -d uint16'




### TODO: implement check for very thick myelin sheath outliers




# ### fill mitochondria
# ipf='_labels_labelMA_2D3D' ids='labelMA'
# opf='_labels_labelMA_2D3D' ods='labelMA_filledm2'
# methods='2'
# python $scriptdir/wmem/fill_holes.py -S \
# $datadir/${dataset_ds}${ipf}.h5/${ids} \
# $datadir/${dataset_ds}${opf}.h5/${ods} \
# -m ${methods} -s 9 9 9
# h52nii '' "${dataset_ds}" "${opf}" "${ods}" '' '' '-i zyx -o xyz -d uint16'
#
# ### update maskMM
# ipf1='_masks_maskMM' ids1='maskMM_PP'
# ipf2='_labels_labelMA_2D3D' ids2='labelMA_filledm2'
# opf='_masks_maskMM' ods='maskMM_PP_filledm2'
# python $scriptdir/wmem/combine_labels.py \
# "${datadir}/${dataset_ds}${ipf1}.h5/${ids1}" \
# "${datadir}/${dataset_ds}${ipf2}.h5/${ids2}" \
# "${datadir}/${dataset_ds}${opf}.h5/${ods}" -m 'mask'
# h52nii '' "${dataset_ds}" "${opf}" "${ods}" '' '' '-i zyx -o xyz -d uint8'
#
#
#
# ### get a wsmask
# # FIXME: take out nodes first for proper labelMM??
# spf='_labels_labelMA_2D3D' sds='labelMA_filledm2'
# mpf='_masks_maskMM' mds='maskMM_PP_filledm2'
# opf='_labels_labelMM' ods="labelMM"
# python $scriptdir/wmem/separate_sheaths.py \
# "$datadir/${dataset_ds}${spf}.h5/${sds}" \
# "$datadir/${dataset_ds}${opf}.h5/${ods}" \
# --maskMM "$datadir/${dataset_ds}${mpf}.h5/${mds}" -S
# h52nii '' "${dataset_ds}" "${opf}" "${ods}_steps/sheaths" '' '' '-i zyx -o xyz -d uint16'
# h52nii '' "${dataset_ds}" "${opf}" "${ods}_steps/wsmask" '' '' '-i zyx -o xyz -d uint16'
#
# ### add MA and MM to create MF
# ipf1='_labels_labelMA_2D3D' ids1='labelMA_filledm2'
# ipf2='_labels_labelMM' ids2="labelMM_steps/sheaths"
# opf='_labels_labelMF' ods="labelMF"
# python $scriptdir/wmem/combine_labels.py \
# "${datadir}/${dataset_ds}${ipf1}.h5/${ids1}" \
# "${datadir}/${dataset_ds}${ipf2}.h5/${ids2}" \
# "${datadir}/${dataset_ds}${opf}.h5/${ods}" -m 'add'
# h52nii '' "${dataset_ds}" "${opf}" "${ods}" '' '' '-i zyx -o xyz -d uint16'
#
# ### fill holes in MF (TODO: borders)
# ipf='_labels_labelMF' ids="labelMF"
# opf='_labels_labelMF' ods="labelMF_filledm3"
# methods='3' # scipy.ndimage.morphology.binary_fill_holes
# python $scriptdir/wmem/fill_holes.py -S \
# $datadir/${dataset_ds}${ipf}.h5/${ids} \
# $datadir/${dataset_ds}${opf}.h5/${ods} \
# -m ${methods}
# h52nii '' "${dataset_ds}" "${opf}" "${ods}" '' '' '-i zyx -o xyz -d uint16'
#
# ### update labelMA by subtracting sheaths from MF_filled
# ipf1='_labels_labelMF' ids1='labelMF_filledm3'
# ipf2='_labels_labelMM' ids2="labelMM_steps/sheaths"
# opf='_labels_labelMA_2D3D' ods='labelMA_filledm2m3'
# python $scriptdir/wmem/combine_labels.py \
# "${datadir}/${dataset_ds}${ipf1}.h5/${ids1}" \
# "${datadir}/${dataset_ds}${ipf2}.h5/${ids2}" \
# "${datadir}/${dataset_ds}${opf}.h5/${ods}" -m 'mask'
# h52nii '' "${dataset_ds}" "${opf}" "${ods}" '' '' '-i zyx -o xyz -d uint16'
#
# ### making sure labels in 'labelMA_filledm2m3' are contiguous
# from wmem import merge_labels
# merge_labels.fill_connected_labels('/Users/mkleinnijenhuis/oxdata/P01/EM/Myrf_01/SET-B/B-NT-S10-2f_ROI_00/B-NT-S10-2f_ROI_00ds7_labels_labelMA_2D3D.h5/labelMA_filledm2m3', check_split=True, outputpath='/Users/mkleinnijenhuis/oxdata/P01/EM/Myrf_01/SET-B/B-NT-S10-2f_ROI_00/B-NT-S10-2f_ROI_00ds7_labels_labelMA_2D3D.h5/labelMA_filledm2m3_split')
#
# opf='_labels_labelMA_2D3D' ods='labelMA_filledm2m3_split'
# h52nii '' "${dataset_ds}" "${opf}" "${ods}" '' '' '-i zyx -o xyz -d uint16'
#
# ### making sure labels in 'labelMF_filledm3' are contiguous
# from wmem import merge_labels
# merge_labels.fill_connected_labels('/Users/mkleinnijenhuis/oxdata/P01/EM/Myrf_01/SET-B/B-NT-S10-2f_ROI_00/B-NT-S10-2f_ROI_00ds7_labels_labelMF.h5/labelMF_filledm3', check_split=True, outputpath='/Users/mkleinnijenhuis/oxdata/P01/EM/Myrf_01/SET-B/B-NT-S10-2f_ROI_00/B-NT-S10-2f_ROI_00ds7_labels_labelMF.h5/labelMF_filledm3_split')
#
# opf='_labels_labelMF' ods='labelMF_filledm3_split'
# h52nii '' "${dataset_ds}" "${opf}" "${ods}" '' '' '-i zyx -o xyz -d uint16'




















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
