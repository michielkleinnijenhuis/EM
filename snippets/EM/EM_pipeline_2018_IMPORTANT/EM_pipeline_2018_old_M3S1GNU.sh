###=========================================================================###
### prepare environment
###=========================================================================###
scriptdir=$HOME/workspace/EM
source $scriptdir/pipelines/datasets.sh
source $scriptdir/pipelines/functions.sh
source $scriptdir/pipelines/submission.sh

compute_env='JAL'
prep_environment $scriptdir $compute_env

dataset='M3S1GNU'
# dataset='B-NT-S9-2a'
# dataset='B-NT-S10-2d_ROI_00'
# dataset='B-NT-S10-2d_ROI_02'
# dataset='B-NT-S10-2f_ROI_00'
# dataset='B-NT-S10-2f_ROI_01'
# dataset='B-NT-S10-2f_ROI_02'

bs='0500'  # bs='2000'
prep_dataset $dataset $bs
echo ${#datastems[@]}

echo -n -e "\033]0;$dataset\007"

###=========================================================================###
### EED sum0247
###=========================================================================###
# declare ipf='_probs' ids='sum0247' opf='_probs_eed' ods='sum0247_eed' wtime='03:10:00'
declare ipf='_probs_sums' ids='sum0247' opf='_probs_eed' ods='sum0247_eed' wtime='03:10:00'
eed 'h' 'a' $ipf $ids $opf $ods $wtime
declare ipf='_probs_eed' ids='sum0247_eed' opf='_probs_eed_sum0247' ods='sum0247_eed' args='-d float16'
mergeblocks 'h' '' '' $ipf $ids $opf $ods "$args"
declare ipf='_probs_eed_sum0247' ids='sum0247_eed' opf='_probs_eed_sum0247' ods='sum0247_eed' \
    brfun='np.mean' brvol='' slab=12 memcpu=60000 wtime='02:00:00' vol_slice=''
blockreduce 'h' $ipf $ids $opf $ods $brfun '' $slab $memcpu $wtime "$vol_slice"


###=========================================================================###
### EED sum16  # FIXME?: M3S1GNU_05480-06020_04480-05020_00000-00430
###=========================================================================###
# declare ipf='_probs' ids='sum016' opf='_probs_eed' ods='sum16_eed' wtime='03:10:00'
declare ipf='_probs_sums' ids='sum16' opf='_probs_eed' ods='sum16_eed' wtime='03:10:00'
eed 'h' 'a' $ipf $ids $opf $ods $wtime
declare ipf='_probs_eed' ids='sum16_eed' opf='_probs_eed_sum16' ods='sum16_eed' args='-d float16'
mergeblocks 'h' '' '' $ipf $ids $opf $ods "$args"
declare ipf='_probs_eed_sum16' ids='sum16_eed' opf='_probs_eed_sum16' ods='sum16_eed' \
    brfun='np.mean' brvol='' slab=12 memcpu=60000 wtime='02:00:00' vol_slice=''
blockreduce 'h' $ipf $ids $opf $ods $brfun '' $slab $memcpu $wtime "$vol_slice"


###=========================================================================###
### extract MA prediction
###=========================================================================###
declare ipf='_probs' ids='volume/predictions' opf='_probs1' ods='probMA' vols='1'
sum_volumes 'h' 'a' $ipf $ids $opf $ods "$vols"
declare ipf='_probs1' ids='probMA' opf='_probs1' ods='probMA' args='-d float16'
mergeblocks 'h' '' '' $ipf $ids $opf $ods "$args"
declare ipf='_probs1' ids='probMA' opf='_probs1' ods='probMA' \
    brfun='np.mean' brvol='' slab=12 memcpu=60000 wtime='02:00:00' vol_slice=''
blockreduce 'h' $ipf $ids $opf $ods $brfun '' $slab $memcpu $wtime "$vol_slice"

declare ipf='_probs1' ids='probMA' opf='_probs_eed' ods='probMA' wtime='03:10:00'
eed 'h' 'a' $ipf $ids $opf $ods $wtime
declare ipf='_probs_eed' ids='probMA' opf='_probs_eed_probMA' ods='probMA_eed' args='-d float16'
mergeblocks 'h' '' '' $ipf $ids $opf $ods "$args"
declare ipf='_probs_eed_probMA' ids='probMA_eed' opf='_probs_eed_probMA' ods='probMA_eed' \
    brfun='np.mean' brvol='' slab=12 memcpu=60000 wtime='02:00:00' vol_slice=''
blockreduce 'h' $ipf $ids $opf $ods $brfun '' $slab $memcpu $wtime "$vol_slice"

# # TODO: TMP rename probMA ...
# ipf=_probs_eed
# for datastem in "${datastems[@]}"; do
#     h5copy -p -i $datastem$ipf.h5 -s $dset -o M3S1GNU_masks.h5 -d $dset
# done


###=========================================================================###
### maskDS to 430 slices TODO: get the top 30 slices
###=========================================================================###
ipf=_masks; ids=maskDS; opf=_masks_maskDS; ods=maskDS; args='-D 30 0 1 0 0 1 0 0 1'
echo "python $scriptdir/wmem/stack2stack.py \
$datadir/$dataset$ipf.h5/$ids \
$datadir/$dataset$opf.h5/$ods $args" > EM_jp_maskDS_subset.sh
chmod +x EM_jp_maskDS_subset.sh
fsl_sub -q bigmem.q ./EM_jp_maskDS_subset.sh

declare ipf='_masks' ids='maskDS' opf='_masks' ods='maskDS' \
    brfun='np.amax' brvol='' slab=28 memcpu=60000 wtime='00:10:00' vol_slice=''
blockreduce 'd' $ipf $ids $opf $ods $brfun '' $slab $memcpu $wtime "$vol_slice"


###=========================================================================###
### prob2mask
###=========================================================================###
declare ipf='' ids='data' opf='_masks' ods='maskDS' arg='-g -l 0 -u 10000000'
prob2mask_datastems 'h' 'a' '' $ids $opf $ods "$arg"
declare ipf='_masks_maskDS' ids='maskDS' opf='_masks_maskDS' ods='maskDS'
mergeblocks 'h' '' '' $ipf $ids $opf $ods ''
declare ipf='_masks_maskDS' ids='maskDS' opf='_masks_maskDS' ods='maskDS' \
    brfun='np.amax' brvol='' slab=28 memcpu=60000 wtime='00:10:00' vol_slice=''
blockreduce 'h' $ipf $ids $opf $ods $brfun '' $slab $memcpu $wtime "$vol_slice"

declare ipf='_probs_eed' ids='sum0247_eed' opf='_masks_maskMM' ods='maskMM' arg='-g -l 0.5 -s 2000 -d 1 -S'
prob2mask_datastems 'h' 'a' $ipf $ids $opf $ods "$arg"
declare ipf='_masks_maskMM' ids='maskMM_steps/raw' opf='_masks_maskMM' ods='maskMM_raw'
mergeblocks 'h' '' '' $ipf $ids $opf $ods ''
declare ipf='_masks_maskMM' ids='maskMM_raw' opf='_masks_maskMM' ods='maskMM' \
    slab=12 arg='-g -l 0 -u 0 -s 2000 -d 1'
prob2mask 'h' $ipf $ids $opf $ods $slab "$arg"
declare ipf='_masks_maskMM' ids='maskMM' opf='_masks_maskMM' ods='maskMM' \
    brfun='np.amax' brvol='' slab=27 memcpu=60000 wtime='00:10:00' vol_slice=''
blockreduce 'h' $ipf $ids $opf $ods $brfun '' $slab $memcpu $wtime "$vol_slice"

declare ipf='_probs_eed' ids='sum16_eed' opf='_masks_maskICS' ods='maskICS' arg='-g -l 0.2'
prob2mask_datastems 'h' 'a' $ipf $ids $opf $ods "$arg"
declare ipf='_masks_maskICS' ids='maskICS' opf='_masks_maskICS' ods='maskICS'
mergeblocks 'h' '' '' $ipf $ids $opf $ods ''
declare ipf='_masks_maskICS' ids='maskICS' opf='_masks_maskICS' ods='maskICS' \
    brfun='np.amax' brvol='' slab=27 memcpu=60000 wtime='00:10:00' vol_slice=''
blockreduce 'h' $ipf $ids $opf $ods $brfun '' $slab $memcpu $wtime "$vol_slice"

# declare ipf='_probs1_eed' ids='probMA_eed' opf='_masks' ods='maskMA' arg='-g -l 0.2'  # BS102fROI00
declare ipf='_probs_eed' ids='probMA' opf='_masks_maskMA' ods='maskMA' arg='-g -l 0.2'  # FIXME: change to probMA_eed
prob2mask_datastems 'h' 'a' $ipf $ids $opf $ods "$arg"
declare ipf='_masks_maskMA' ids='maskMA' opf='_masks_maskMA' ods='maskMA'
mergeblocks 'h' '' '' $ipf $ids $opf $ods ''
declare ipf='_masks_maskMA' ids='maskMA' opf='_masks_maskMA' ods='maskMA' \
    brfun='np.amax' brvol='' slab=27 memcpu=60000 wtime='00:10:00' vol_slice=''
blockreduce 'h' $ipf $ids $opf $ods $brfun '' $slab $memcpu $wtime "$vol_slice"


for dset in 'maskDS' 'maskMM' 'maskICS' 'maskMA'; do
    h5copy -p -i ${dataset}_masks_$dset.h5 -s $dset -o ${dataset}_masks.h5 -d $dset
    h5copy -p -i ${dataset_ds}_masks_$dset.h5 -s $dset -o ${dataset_ds}_masks.h5 -d $dset
done

###=========================================================================###
### myelinated axon compartment
###=========================================================================###
declare ipf='_masks' ids='maskMM' opf='_labels' ods='labelMA_core2D' meanint='dummy'
conncomp '' '2D' $dataset $ipf $ids $opf $ods $meanint
conncomp '' '2D' $dataset_ds $ipf $ids $opf $ods $meanint
declare ipf='_labels' ids='labelMA_core2D' opf='_labels_mapall' ods='dummy' meanint='_probs_eed_sum16.h5/sum16_eed'
conncomp '' '2Dfilter' $dataset $ipf $ids $opf $ods $meanint
conncomp '' '2Dfilter' $dataset_ds $ipf $ids $opf $ods $meanint
declare ipf='_labels' ids='labelMA_core2D' opf='_labels_mapall' ods='dummy' meanint='dummy'
# conncomp '' '2Dprops' $dataset  $ipf $ids $opf $ods $meanint  # TOO LARGE
conncomp '' '2Dprops' $dataset_ds $ipf $ids $opf $ods $meanint

###=========================================================================###
# TODO: include scikit-learn classifier ipython notebook here
# jupyter nbconvert --to python 2Dprops_classification.ipynb
###=========================================================================###
python $scriptdir/wmem/connected_components_clf.py \
'/Users/michielk/oxdata/P01/EM/M3/M3S1GNU/M3S1GNUds7_clf.pkl' \
'/Users/michielk/oxdata/P01/EM/M3/M3S1GNU/M3S1GNUds7_scaler.pkl' \
-m 'test' -b $datadir/$dataset_ds$opf \
-i $datadir/$dataset_ds$ipf.h5/$ids \
-o $datadir/$dataset_ds$opf.h5/$ods \
-p ${props[@]} -A 3000 -I 0.8

declare ipf='_labels' ids='labelMA_core2D_prediction' opf='_labels_3DTEST' ods='labelMA_3Dlabeled' meanint='dummy'
# conncomp '' '2Dto3D' $dataset $ipf $ids $opf $ods $meanint
# conncomp '' '2Dto3D' $dataset_ds $ipf $ids $opf $ods $meanint
python $scriptdir/wmem/connected_components.py \
$datadir/$dataset_ds$ipf.h5/$ids \
$datadir/$dataset_ds$opf.h5/$ods \
-m '2Dto3D' -d 0












###=========================================================================###
### TODO: MA proofreading
###=========================================================================###

mpiexec -n 6 python $scriptdir/wmem/merge_slicelabels.py \
$datadir/${dataset_ds}_labels_3DTEST.h5/labelMA_3Dlabeled \
$datadir/${dataset_ds}_labels_3DTEST.h5/labelMA_3Diter1 \
-M -m 'MAstitch' -d 0 \
-q 3 -o 0.90
python $scriptdir/wmem/merge_slicelabels.py \
$datadir/${dataset_ds}_labels_3DTEST.h5/labelMA_3Dlabeled \
$datadir/${dataset_ds}_labels_3DTEST.h5/labelMA_3Diter1 \
--maskMM $datadir/${dataset_ds}_masks.h5/maskMM \
-m 'MAfwmap' -d 0 -r
# -l 6 1 1 -s 200


# fill holes
methods='2'; close=9;
python $scriptdir/wmem/fill_holes.py \
$datadir/${dataset_ds}_labels_3DTEST.h5/labelMA_3Diter1 \
$datadir/${dataset_ds}_labels_3DTEST.h5/labelMA_3Diter1_filled \
-m ${methods} \
--maskDS $datadir/${dataset_ds}_masks.h5/maskDS \
--maskMM $datadir/${dataset_ds}_masks.h5/maskMM \
--maskMX $datadir/${dataset_ds}_masks.h5/maskMM
#  \  #'_maskMM-0.02' 'stack' \
# --maskMA $datadir/${dataset_ds}_masks.h5/maskMA


python $scriptdir/wmem/nodes_of_ranvier.py \
$datadir/${dataset_ds}_labels_3DTEST.h5/labelMA_3Diter1 \
$datadir/${dataset_ds}_labels_3DTEST.h5/labelMA_3Diter1_NoR \
--boundarymask $datadir/${dataset_ds}_masks.h5/maskDS -S -R -s 200 \
-m 'watershed' \
--data $datadir/$dataset_ds.h5/data \
--maskMM $datadir/${dataset_ds}_masks.h5/maskMM


-l ${volws} 'stack' -s 5000 \
-o "${volws}${NoRpf}_iter${iter}" 'stack' \
-S '_maskDS_invdil' 'stack'

python $scriptdir/wmem/filter_NoR.py \
$datadir/${dataset_ds}_labels_3DTEST.h5/labelMA_3Diter1 \
$datadir/${dataset_ds}_labels_3DTEST.h5/labelMA_3Diter1_NoR \
--input2D


###=========================================================================###
### TODO: refine maskMM
###=========================================================================###


###=========================================================================###
### watershed
###=========================================================================###
xs=2000 ys=2000 zs=430  # blocksizes
xm=50 ym=50 zm=0  # margins
bs='2000'
blockdir=$datadir/blocks_$bs &&
    mkdir -p $blockdir &&
        datastems_blocks
declare ipf='_probs_eed_sum16' ids='sum16_eed' opf='_probs_eed_sum16' ods='sum16_eed'
split_blocks 'h' 'a' $ipf $ids $opf $ods
declare ipf='_probs_eed_sum16' ids='sum16_eed' opf='_ws' ods='l0.99_u1.00_s010' l=0.99 u=1.00 s=010
watershed 'h' '' $ipf $ids $opf $ods $l $u $s
sbatch --dependency=after:1635298 EM_sb_ws_l0.99_u1.00_s010_000.sh
fsl_sub -q bigmem.q -j 7709390 -t ./EM_jp_ws_l0.99_u1.00_s010_000.sh




datastem='B-NT-S10-2f_ROI_01_00480-01020_00480-01020_00000-00184' # datastem_index 18

splitblocks 'h' 'a' '' 'data' '' 'data' '' '-r 18 19'
fpat=${datastem}.h5
rsync -Pazv $host_arc:$datadir_arc/blocks_0500/$fpat $datadir_loc/blocks_0500/
h52nii '' "blocks_0500/${datastem}" '' 'data' '' '' '-i zyx -o xyz'

splitblocks 'h' 'a' '_probs' 'volume/predictions' '_probs' 'volume/predictions' '-r 18 19'
datastems=( B-NT-S10-2f_ROI_01_00480-01020_00480-01020_00000-00184 )
sum_volumes 'h' '' '_probs' 'volume/predictions' '_probs_sum16' 'sum16' '1 6'
fpat=${datastem}_probs.h5
rsync -Pazv $host_arc:$datadir_arc/blocks_0500/$fpat $datadir_loc/blocks_0500/
h52nii '' "blocks_0500/${datastem}" '_probs' 'volume/predictions' '' '' '-u -i zyxc -o xyzc'
fpat=${datastem}_probs_sum16.h5
rsync -Pazv $host_arc:$datadir_arc/blocks_0500/$fpat $datadir_loc/blocks_0500/
h52nii '' "blocks_0500/${datastem}" '_probs_sum16' 'sum16' '' '' '-i zyx -o xyz'

splitblocks 'h' 'a' '' 'data' '' 'data' '' '-r 18 19'
fpat=${datastem}_probs.h5
rsync -Pazv $host_arc:$datadir_arc/blocks_0500/$fpat $datadir_loc/blocks_0500/
h52nii '' "blocks_0500/${datastem}" '_probs' 'volume/predictions' '' '' '-u -i zyxc -o xyzc'

fpat=${datastem}_probs_eed_sum16.h5
rsync -Pazv $host_jal:$datadir_jal/blocks_0500/$fpat $datadir_loc/blocks_0500/
h52nii '' "blocks_0500/${datastem}" '_probs_eed_sum16' 'sum16_eed' '' '' '-i zyx -o xyz'
fpat=${datastem}_masks_maskMM.h5
rsync -Pazv $host_jal:$datadir_jal/blocks_0500/$fpat $datadir_loc/blocks_0500/
h52nii '' "blocks_0500/${datastem}" '_masks_maskMM' 'maskMM' '' '' '-i zyx -o xyz'

declare ipf='_probs_eed_sum16' ids='sum16_eed' opf='_ws'
l=0.99 u=1.00 s=010
ods="l${l}_u${u}_s${s}_maskMM"
python -W ignore $scriptdir/wmem/watershed_ics.py \
    $datadir/blocks_$bs/${datastems[18]}$ipf.h5/$ids \
    $datadir/blocks_$bs/${datastems[18]}$opf.h5/$ods \
    --masks $datadir/blocks_$bs/${datastem}_masks_maskMM.h5/maskMM \
    -l $l -u $u -s $s -i -S
h52nii '' "blocks_0500/${datastem}" '_ws' $ods '' '' '-i zyx -o xyz -d uint16'
h52nii '' "blocks_0500/${datastem}" '_ws' "${ods}_steps/mask" '' '' '-i zyx -o xyz'
h52nii '' "blocks_0500/${datastem}" '_ws' "${ods}_steps/seeds" '' '' '-i zyx -o xyz'

# declare ipf='_probs_eed_sum16' ids='sum16_eed' opf='_slic' ods='slic'
# python -W ignore $scriptdir/wmem/slicvoxels.py $datadir/blocks_$bs/${datastems[18]}$ipf.h5/$ids $datadir/blocks_$bs/${datastems[18]}$opf.h5/$ods -l 500 -c 0.2 -s 0 -e
# h52nii '' "blocks_0500/${datastems[18]}" '_slic' "slic" '' '' '-i zyx -o xyz -d uint32'
#
declare l=500 c=0.02 s=0.02
declare l=500 c=0.02 s=0.05 # best yet
declare l=0500 c=0.02 s=0.03
declare l=1000 c=0.02 s=0.05
declare l=1000 c=0.01 s=0.03
declare l=1000 c=0.03 s=0.03
declare l=3000 c=0.02 s=0.01 # not good
declare l=3000 c=2.00 s=0.03 # too compact
declare l=3000 c=0.20 s=0.03
declare ipf='' ids='data' opf='_slic' ods="slic_data_${l}_${c}_${s}"
python -W ignore $scriptdir/wmem/slicvoxels.py $datadir/blocks_$bs/${datastems[18]}$ipf.h5/$ids $datadir/blocks_$bs/${datastems[18]}$opf.h5/$ods -l $l -c $c -s $s -e
h52nii '' "blocks_0500/${datastems[18]}" $opf $ods '' '' '-i zyx -o xyz -d uint32'


declare l=0500 c=0.02 s=0.05  # best yet
declare l=0500 c=0.02 s=0.03
declare ipf='_probs_sum16' ids='sum16' opf='_slic' ods="slic_sum16_${l}_${c}_${s}"
python -W ignore $scriptdir/wmem/slicvoxels.py $datadir/blocks_$bs/${datastems[18]}$ipf.h5/$ids $datadir/blocks_$bs/${datastems[18]}$opf.h5/$ods -l $l -c $c -s $s -e
h52nii '' "blocks_0500/${datastems[18]}" $opf $ods '' '' '-i zyx -o xyz -d uint32'

declare l=1000 c=2.00 s=0.00
declare l=0500 c=0.02 s=0.05
declare l=1000 c=0.02 s=0.05
declare l=1000 c=0.02 s=0.03
declare l=1000 c=0.01 s=0.03
declare l=1000 c=0.04 s=0.03
declare l=1000 c=0.20 s=0.03
declare l=9000 c=0.20 s=0.03  # this looks very good!
declare l=9000 c=0.20 s=0.005
declare ipf='_probs' ids='volume/predictions' opf="_slic_slic4D_${l}_${c}_${s}" ods="slic4D_${l}_${c}_${s}"
python -W ignore $scriptdir/wmem/slicvoxels.py $datadir/blocks_$bs/${datastems[18]}$ipf.h5/$ids $datadir/blocks_$bs/${datastems[18]}$opf.h5/$ods -l $l -c $c -s $s -e
h52nii '' "blocks_0500/${datastems[18]}" $opf $ods '' '' '-i zyx -o xyz -d uint32'

###=========================================================================###
### agglomerate watershedMA ()
###=========================================================================###
export template='array' additions='conda' CONDA_ENV="scikit-image-devel_0.13"
export njobs=3 nodes=1 tasks=10 memcpu=6000 wtime="00:10:00" q="d"

l=0.99; u=1.00; s=005;
svoxpf="_ws_l${l}_u${u}_s${s}"
export jobname="agglo${svoxpf}"
export cmd="python $scriptdir/supervoxels/agglo_from_labelmask.py \
$datadir datastem \
-l '_labelMA_core_manedit' 'stack' -s ${svoxpf} 'stack' \
-o '_labelMA' -m '_maskMA'"
source $scriptdir/pipelines/template_job_$template.sh



###=========================================================================###
###
###=========================================================================###

# h52nii 'd' $dataset '' 'data' '' '' '-u -i zyx -o xyz -D 50 53 1 0 0 1 0 0 1'

h52nii 'd' $dataset '_probs_eed_sum0247' 'sum0247_eed' '' '' '-u -i zyx -o xyz -D 50 53 1 0 0 1 0 0 1'
h52nii 'd' $dataset '_probs_eed_sum16' 'sum16_eed' '' '' '-u -i zyx -o xyz -D 50 53 1 0 0 1 0 0 1'
h52nii 'd' $dataset '_masks_maskMM_raw' 'maskMM_raw' '' '' '-i zyx -o xyz -D 50 53 1 0 0 1 0 0 1'
h52nii 'd' $dataset '_masks_maskMM' 'maskMM' '' '' '-i zyx -o xyz -D 50 53 1 0 0 1 0 0 1'
h52nii 'd' $dataset '_masks_maskMM' 'maskMM' '' 'y' '-i zyx -o xyz -D 0 0 1 5050 5053 1 0 0 1'
h52nii 'd' $dataset '_masks_maskICS' 'maskICS' '' '' '-i zyx -o xyz -D 50 53 1 0 0 1 0 0 1'

h52nii 'd' $dataset '_masks' 'maskDS' '' 'y' '-i zyx -o xyz -D 0 0 1 5050 5053 1 0 0 1'


h52nii '' $dataset_ds '_probs_eed' 'sum0247_eed' '' '' '-u -i zyx -o xyz -d float'
h52nii '' $dataset_ds '_probs_eed' 'sum0247_eed' '' '' '-i zyx -o xyz -d float'




mergeblocks '' '_probs_eed' 'probs_eed' '_probs_eed' 'probs_eed' '' '' '-d float16 -M'

prob2mask_datastems 'h' '' 3 1
EM_sb_p2m_dstems_maskMA_002.sh
EM_sb_p2m_dstems_maskMA_005.sh
EM_sb_p2m_dstems_maskMA_006.sh
EM_sb_p2m_dstems_maskMA_007.sh
EM_sb_p2m_dstems_maskMA_008.sh
EM_sb_p2m_dstems_maskMA_010.sh
EM_sb_p2m_dstems_maskMA_014.sh
EM_sb_p2m_dstems_maskMA_016.sh
EM_sb_p2m_dstems_maskMA_017.sh

prob2mask_datastems 'h' 'm' 3 1

 # (dm3lib) (ij) (loci)
