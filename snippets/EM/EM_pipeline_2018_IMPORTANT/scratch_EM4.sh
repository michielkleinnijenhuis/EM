- M3S1GNU no maskMM_raw

declare ipf='_masks' ids='maskMM_steps/raw' opf='_masks_maskMM' ods='maskMM_raw'
scriptfile=$( mergeblocks 'h' '' '' $ipf $ids $opf $ods $args )
[[ -z $jid ]] && dep='' || dep="-j $jid"
jid=$( fsl_sub -q veryshort.q $dep $scriptfile )


h5copy -p -i ${dataset}_masks_maskMM.h5 -s maskMM_raw -o ${dataset}_masks_maskMM_raw.h5 -d maskMM_raw
mv ${dataset}_masks_maskMM.h5 ${dataset}_masks_maskMM.h5tmp
h5copy -p -i ${dataset}_masks_maskMM.h5tmp -s maskMM -o ${dataset}_masks_maskMM.h5 -d maskMM
rm ${dataset}_masks_maskMM.h5tmp


### data ISSUES after preprocessing
- poor ilastik prediction of 2dROI00
- intensity gradient in 2fROI00 2fROI01 2fROI02
- TODO: maskMM(_raw) includes the boundaries
- TODO: maskMM is too greedy

### processing ISSUES after preprocessing
# - M3S1GNU ds7_masks_maskDS shifted (_masks_maskDS is ok) => rerun downsampling maskDS
# - 2dROI02 maskMA not okay => rerun downsampling maskMA
# - 2dROI02 nii data does not open in itksnap => rename _data.nii.gz
# - 2fROI00 maskMA not okay => rerun maskMA pipeline
# - 2fROI01 and 2fROI02 _probs_eed_sum0247 _probs_eed_sum16 switched some data
# -- it's not in the masks, so it's probably in mergeblocks or eed-blockfiles have switched filenames, but it hasnt affected the labeling much => masks are okay thus eed is okay
# -- => partial rerun of mergeblocks and rerun of downsample for both
# -- => rerun from conncomp 2Dfilter onwards
# -- => copy files


- TODO: homogenize and check M3S1GNU





- get reduced maskMM-final (deal with MITO)
- perform watershed on ICS outside maskMM-final
- aggregate svox overlapping with MA to labelMA-final
- separate sheaths
- Neuroproof on ICS-UA




### shuffle labels
import os
from wmem import utils
import numpy as np

datadir = "/Users/michielk/oxdata/P01/EM/Myrf_01/SET-B/B-NT-S10-2f_ROI_01/blocks_0500"
fname = 'B-NT-S10-2f_ROI_01_00480-01020_00480-01020_00000-00184_slic4D'
dset = 'slic_9000_0.20_0.03'
dset = 'slic4D_9000_0.70_0.03'
dset = 'slic4D_9000_0.20_0.005'
fname = 'B-NT-S10-2f_ROI_01_00480-01020_00480-01020_00000-00184_slic_' + dset

h5path_in = os.path.join(datadir, '{}.h5/{}'.format(fname, dset))
h5file_in, ds_in, elsize, axlab = utils.h5_load(h5path_in)
fw = utils.shuffle_labels(ds_in)

fw.dtype = ds_in.dtype
niipath_out = os.path.join(datadir, '{}_{}_shuffled.nii.gz'.format(fname, dset))
#utils.write_to_nifti(niipath_out, fw[ds_in].astype('uint32'), elsize)  # np.flipud(elsize)
utils.write_to_nifti(niipath_out, np.transpose(fw[ds_in]).astype('uint32'), np.flipud(elsize))
h5file_in.close()











python -W ignore /home/ndcn-fmrib-water-brain/ndcn0180/workspace/EM/wmem/slicvoxels.py /data/ndcn-fmrib-water-brain/ndcn0180/EM/Myrf_01/SET-B/B-NT-S10-2f_ROI_01/blocks_0500/B-NT-S10-2f_ROI_01_00000-00520_00000-00520_00000-00184_probs.h5/volume/predictions /data/ndcn-fmrib-water-brain/ndcn0180/EM/Myrf_01/SET-B/B-NT-S10-2f_ROI_01/blocks_0500/B-NT-S10-2f_ROI_01_00000-00520_00000-00520_00000-00184_slic_slic4D_9000_0.20_0.03.h5/slic4D_9000_0.20_0.03 -l 9000 -c 0.20 -s 0.03 -e &

python -W ignore /home/ndcn-fmrib-water-brain/ndcn0180/workspace/EM/wmem/splitblocks.py /data/ndcn-fmrib-water-brain/ndcn0180/EM/Myrf_01/SET-B/B-NT-S10-2f_ROI_01/B-NT-S10-2f_ROI_01_probs.h5/volume/predictions B-NT-S10-2f_ROI_01 -p 2000 2000 2000 -q 0 20 20

for i in `seq 1644068 1644073`; do
scontrol update jobid=$i partition=devel MinMemoryCPU=60000 TimeLimit=00:10:00
done

for i in `seq 1644068 1644073`; do
scontrol update jobid=$i partition=compute MinMemoryCPU=125000 TimeLimit=05:10:00
done

for i in `seq 1644074 1644079`; do
scontrol update jobid=$i partition=compute MinMemoryCPU=6000 TimeLimit=01:10:00
done



step=64
for br in `seq 0 $step ${#datastems[@]}`; do
    declare ipf='_probs' ids='volume/predictions' opf='_probs' ods='volume/predictions' blockrange="-r $br $((br+step))"
    scriptfile=$( splitblocks 'h' 'a' $ipf $ids $opf $ods $blockrange )
    [[ $br == 0 ]] && dep='' || dep="--dependency=after:$jid"
    jid=$( sbatch -p devel $dep $scriptfile ) && jid=${jid##* }
done





h52nii '' $dataset_ds '_probs_eed_probMA' 'probMA_eed' '' 'probMA_eed_float' '-i zyx -o xyz'

declare ipf='_probs_eed_probMA' ids='probMA_eed' opf='_masks_maskMA_ds-0.99' ods='maskMA_ds-0.99' arg='-g -l 0.99'
python -W ignore $scriptdir/wmem/prob2mask.py \
        $datadir/$dataset_ds$ipf.h5/$ids \
        $datadir/$dataset_ds$opf.h5/$ods \
        -g -l 0.99 -u 1.00

h52nii '' $dataset_ds $opf $ods '' '' '-i zyx -o xyz'

step=64
for br in `seq 0 $step ${#datastems[@]}`; do
    declare ipf='_probs' ids='volume/predictions' opf='_probs' ods='volume/predictions' args="-r $br $((br+step)) -D 30 0 1 0 0 1 0 0 1 0 0 1"
    scriptfile=$( splitblocks 'h' 'a' $ipf $ids $opf $ods $args )
done


declare ipf='_probs' ids='volume/predictions' opf='_probstmp' ods='volume/predictions'

function get_cmd_s2s {
    # Get the command for converting h5.
    echo python -W ignore $scriptdir/wmem/stack2stack.py \
        $datadir/$datastem$ipf.h5/$ids \
        $datadir/$datastem$opf.h5/$ods $args
}

set_datastems $stemsmode
n=${#datastems[@]}

jobname="s2s"
additions='array'
CONDA_ENV=''
memcpu=6000
nodes=1
wtime='01:10:00'
tasks=16
njobs=$(( (n + tasks - 1) / tasks ))
args='-D 30 0 1 0 0 1 0 0 1 0 0 1'

fun=get_cmd_s2s
get_command_array_datastems $fun

unset JOBS && declare -a JOBS
array_job $njobs $tasks





# ###=========================================================================###
# ### threshold and downsample
# ###=========================================================================###
# # prob2mask_datastems 'd' 'm' 0 1
# # mergeblocks 'h' '_masks' 'maskDS' '_masks' 'maskDS' '' '' ''
# # blockreduce 'h' 5 1
#
# # prob2mask_datastems 'd' 'a' 1 1
# # mergeblocks 'h' '_masks' 'maskMM_steps/raw' '_masks' 'maskMM_raw'  '' '' ''
# # NOTE: use raw... mergeblocks 'h' '_masks' 'maskMM' '_masks' 'maskMM'
# # NOTE: and overwrite with filtered version =>
# # prob2mask 'h' 4 1
# # blockreduce 'h' 6 1
#
# # prob2mask_datastems 'd' 'a' 2 1
# # mergeblocks 'h' '_masks' 'maskICS' '_masks' 'maskICS' '' '' ''
# # blockreduce 'h' 8 1
#
# # prob2mask_datastems 'd' 'm' 3 1
# mergeblocks '' '_masks' 'maskMA' '_masks' 'maskMA' '' '' '' ''
# blockreduce 'h' 7 1
#
# # # h5dump -pH -d $ids $datadir/${dataset}${ipf}.h5 | grep CHUNKED
# # prob2mask h  # prob2mask h 1 1
# # blockreduce 'h'  # blockreduce h 1 1
#
# VARSETS[5]="ipf=_masks; ids=maskDS; \
#     opf=_masks; ods=maskDS; \
#     brfun='np.amax'; vol_br=; \
#     blocksize=$zmax; vol_slice=; \
#     memcpu=60000; wtime=00:10:00; q='d';"
# VARSETS[6]="ipf=_masks; ids=maskMM;
#     opf=_masks; ods=maskMM; \
#     brfun='np.amax'; vol_br=; \
#     blocksize=$zmax; vol_slice=; \
#     memcpu=60000; wtime=00:10:00; q='d';"
# VARSETS[7]="ipf=_masks; ids=maskMA; \
#     opf=_masks; ods=maskMA; \
#     brfun='np.amax'; vol_br=; \
#     blocksize=$zmax; vol_slice=; \
#     memcpu=60000; wtime=00:10:00; q='d';"
# VARSETS[8]="ipf=_masks; ids=maskICS; \
#     opf=_masks; ods=maskICS; \
#     brfun='np.amax'; vol_br=; \
#     blocksize=$zmax; vol_slice=; \
#     memcpu=60000; wtime=00:10:00; q='d';"
