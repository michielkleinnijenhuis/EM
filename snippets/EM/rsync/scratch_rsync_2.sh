host_jal=michielk@jalapeno.fmrib.ox.ac.uk
host_arc=ndcn0180@arcus-b.arc.ox.ac.uk
scriptdir_loc='/Users/michielk/workspace/EM'
scriptdir_jal='/home/fs0/michielk/workspace/EM'
scriptdir_arc='/home/ndcn-fmrib-water-brain/ndcn0180/workspace/EM'

dataset='M3S1GNU'
datadir_loc="/Users/michielk/oxdata/P01/EM/M3/$dataset"
datadir_jal="/vols/Data/km/michielk/oxdata/P01/EM/M3/$dataset"
datadir_arc="/data/ndcn-fmrib-water-brain/ndcn0180/EM/M3/$dataset"

dataset='B-NT-S10-2a'
dataset='B-NT-S10-2d_ROI_00'
dataset='B-NT-S10-2d_ROI_02'
dataset='B-NT-S10-2f_ROI_00'
dataset='B-NT-S10-2f_ROI_01'
dataset='B-NT-S10-2f_ROI_02'
datadir_loc="/Users/michielk/oxdata/P01/EM/Myrf_01/SET-B/$dataset"
datadir_jal="/vols/Data/km/michielk/oxdata/P01/EM/Myrf_01/SET-B/$dataset"
datadir_arc="/data/ndcn-fmrib-water-brain/ndcn0180/EM/Myrf_01/SET-B/$dataset"


rsync -Pazv $scriptdir_loc/pipelines/datasets.sh \
    $scriptdir_loc/pipelines/functions.sh \
    $scriptdir_loc/pipelines/submission.sh \
    $host_jal:$scriptdir_jal/pipelines/

rsync -Pazv $scriptdir_loc/pipelines/datasets.sh \
    $scriptdir_loc/pipelines/functions.sh \
    $scriptdir_loc/pipelines/submission.sh \
    $host_arc:$scriptdir_arc/pipelines/

rsync -Pazv $scriptdir_loc/wmem/splitblocks.py \
    $host_arc:$scriptdir_arc/wmem/


fpat="${dataset_ds}*.pkl"
rsync -Pazv $datadir_loc/$fpat $host_jal:$datadir_jal

fpat="*"
rsync -Pazvn $datadir_jal/$fpat $host_arc:$datadir_arc

fpat="*"
rsync -Pazvn $host_jal:$datadir_jal/$fpat $datadir_loc

fpat="$dataset_ds.h5"
rsync -Pazv $host_arc:$datadir_arc/$fpat $datadir_loc
fpat="${dataset}*_masks_mask*.h5"
rsync -Pazv $host_arc:$datadir_arc/$fpat $datadir_loc
fpat="${dataset_ds}*"
rsync -Pazv $host_jal:$datadir_jal/$fpat $datadir_loc
fpat="${dataset}*_labels*"
rsync -Pazv $host_jal:$datadir_jal/$fpat $datadir_loc

fpat="${dataset_ds}_probs_eed_sum*.h5"
rsync -Pazv $host_arc:$datadir_arc/$fpat $datadir_loc
fpat="${dataset}*_probs_eed_sum*.h5"
rsync -Pazv $host_arc:$datadir_arc/$fpat $datadir_jal
# fpat="${dataset_ds}_masks.h5"
# rsync -Pazv $host_arc:$datadir_arc/$fpat $datadir_loc

fpat="${dataset_ds}.h5"
rsync -Pazv $host_arc:$datadir_arc/$fpat $datadir_loc

fpat="${dataset}.h5"
rsync -Pazv $host_arc:$datadir_arc/$fpat $datadir_jal
fpat="${dataset}_probs.h5"
rsync -Pazv $host_arc:$datadir_arc/$fpat $datadir_jal

fpat="${dataset}_probs_eed_*.h5"
fpat="${dataset}_probs_eed_sum0247.h5"
fpat="${dataset}_probs_eed_sum16.h5"
fpat="${dataset}_probs_eed_probMA.h5"
fpat="${dataset}_masks_maskMM_raw.h5"
fpat="${dataset}_masks_maskMA.h5"
rsync -Pazv $host_arc:$datadir_arc/$fpat $datadir_jal

fpat=${dataset}_00000-00520_00000-00520_00000-00135_probs.h5
fpat=${dataset}_06980-07520_05980-06520_00000-00135_probs.h5
rsync -Pazv $host_arc:$datadir_arc/blocks_0500/$fpat $datadir_loc/blocks_0500/


fpat="${dataset}*.nii.gz"
rsync -Pazv $host_arc:$datadir_arc/$fpat $datadir_loc

fpat="${dataset}*_masks_maskMA.h5"
rsync -Pazv $host_arc:$datadir_arc/$fpat $datadir_loc
rsync -Pazv $host_arc:$datadir_arc/$fpat $datadir_jal

fpat="${dataset}*_masks_maskDS.h5"
rsync -Pazv $datadir_loc/$fpat $host_jal:$datadir_jal
rsync -Pazv $datadir_loc/$fpat $host_arc:$datadir_arc











# LOCAL-TO-JALAPENO
host=michielk@jalapeno.fmrib.ox.ac.uk
scriptdir="$HOME/workspace/EM"
scriptdir_rem='/home/fs0/michielk/workspace/EM'
fname=datasets.sh
fname=functions.sh
fname=submission.sh
rsync -Pazv $scriptdir/pipelines/$fname $host:$scriptdir_rem/pipelines/
rsync -Pazv $HOME/oxscripts/matlab/toolboxes/coherencefilter_version5b $host:/home/fs0/michielk/matlab/toolboxes/
rsync -Pazv $HOME/workspace/pyDM3reader $host:/home/fs0/michielk/workspace/
fname=mergeblocks.py
fname=utils.py
rsync -Pazv $scriptdir/wmem/$fname $host:$scriptdir_rem/wmem/

dataset='M3S1GNU'
dataset='B-NT-S10-2f_ROI_00'
dataset='B-NT-S10-2f_ROI_01'
dataset='B-NT-S10-2f_ROI_02'
dataset_ds="${dataset}ds7"
datadir="/Users/michielk/oxdata/P01/EM/M3/$dataset"
datadir_rem="/vols/Data/km/michielk/oxdata/P01/EM/M3/$dataset"
fname=M3S1GNUds7_probs_eed_sum0247.h5
rsync -Pazv $host:$datadir_rem/$fname $datadir
rsync -Pazv $host:$datadir_rem/blocks_2000/*ws.h5 $datadir/blocks_2000

dataset='B-NT-S10-2f_ROI_01'
dataset='B-NT-S10-2f_ROI_02'
dataset_ds="${dataset}ds7"
datadir="/Users/michielk/oxdata/P01/EM/Myrf_01/SET-B/$dataset"
datadir_rem="/vols/Data/km/michielk/oxdata/P01/EM/Myrf_01/SET-B/$dataset"
rsync -Pazv $host:$datadir_rem/*labels* $datadir
rsync -Pazv $host:$datadir_rem/*masks* $datadir


# LOCAL-TO-ARC
host=ndcn0180@arcus-b.arc.ox.ac.uk

scriptdir="$HOME/workspace/EM"
scriptdir_rem='/home/ndcn-fmrib-water-brain/ndcn0180/workspace/EM'
fname=datasets.sh
fname=functions.sh
fname=submission.sh
rsync -Pazv $scriptdir/pipelines/$fname $host:$scriptdir_rem/pipelines/
# fname=mergeblocks.py
# fname=utils.py
# fname=connected_components.py
# rsync -Pazv $scriptdir/wmem/$fname $host:$scriptdir_rem/wmem/

dataset='M3S1GNU'
dataset_ds="${dataset}ds7"
datadir="/Users/michielk/oxdata/P01/EM/M3/$dataset"
datadir_rem="/data/ndcn-fmrib-water-brain/ndcn0180/EM/M3/$dataset"
fname=M3S1GNUds7.h5
rsync -Pazv $host:$datadir_rem/$fname $datadir


dataset='B-NT-S10-2f_ROI_00'
dataset_ds="${dataset}ds7"
datadir="/Users/michielk/oxdata/P01/EM/Myrf_01/SET-B/$dataset"
datadir_rem="/data/ndcn-fmrib-water-brain/ndcn0180/EM/Myrf_01/SET-B/$dataset"
rsync -Pazv $host:$datadir_rem/$fname $datadir

dataset='B-NT-S10-2d_ROI_02'
dataset_ds="${dataset}ds7"
datadir="/Users/michielk/oxdata/P01/EM/Myrf_01/SET-B/$dataset"
datadir_rem="/data/ndcn-fmrib-water-brain/ndcn0180/EM/Myrf_01/SET-B/$dataset"
fname=pixclass_8class.ilp
rsync -Pazv $datadir/$fname $host:$datadir_rem/

# ARC-TO-LOCAL
ipf='_masks'
rsync -Pazv $host:$datadir_rem/$dataset$ipf.h5 $datadir
rsync -Pazv $host:$datadir_rem/$dataset_ds$ipf.h5 $datadir
ipf='_probs_eed'
rsync -Pazv $host:$datadir_rem/$dataset_ds$ipf.h5 $datadir
ipf='_probs_eed_sum0247'
rsync -Pazv $host:$datadir_rem/$dataset$ipf.h5 $datadir

datastem=B-NT-S10-2f_ROI_00_00480-01020_00480-01020_00000-00184
datastem=B-NT-S10-2f_ROI_00_00000-00520_00000-00520_00000-00184
datastem=B-NT-S10-2f_ROI_00_00000-00520_00480-01020_00000-00184
rsync -Pazv $host:$datadir_rem/blocks_0500/$datastem* $datadir/blocks_0500/

fname=blocks_0500/B-NT-S10-2f_ROI_00_06480-07020_01480-02020_00000-00184_probs_eed.h5
fname=blocks_0500/B-NT-S10-2f_ROI_00_06480-07020_01980-02520_00000-00184_probs_eed.h5
rsync -Pazv $host:$datadir_rem/$fname $datadir/blocks_0500/


fname=blocks/B-NT-S10-2f_ROI_00_03000-03500_03000-03500_00000-00184.h5
fname=B-NT-S10-2f_ROI_00ds7.h5
fname=B-NT-S10-2f_ROI_00ds7_masks.h5
fname=B-NT-S10-2f_ROI_00ds7_probs_eed.h5
fname=old/B-NT-S10-2f_ROI_00ds7_probs.h5
fname=B-NT-S10-2f_ROI_00.h5
fname=B-NT-S10-2f_ROI_00_masks.h5
rsync -Pazv $datadir/$fname $host:$datadir_rem/



# JALAPENO-TO-ARC  # ssh jalapeno.fmrib.ox.ac.uk
host='ndcn0180@arcus-b.arc.ox.ac.uk'
scriptdir="$HOME/workspace/EM"
scriptdir_rem='/home/ndcn-fmrib-water-brain/ndcn0180/workspace/EM'
rsync -Pazv $host:/home/ndcn-fmrib-water-brain/ndcn0180/workspace/ilastik-1.2.2post1-Linux.tar.bz2 $scriptdir
rsync -Pazv ~/workspace/hdf5-1.10.1.tar.gz $host:/home/ndcn-fmrib-water-brain/ndcn0180/workspace/
rsync -Pazv /vols/Data/km/michielk/workspace/*.c $host:/data/ndcn-fmrib-water-brain/ndcn0180/workspace/


dataset='B-NT-S10-2d_ROI_00'
dataset='B-NT-S10-2d_ROI_02'
dataset='B-NT-S10-2f_ROI_00'
dataset='B-NT-S10-2f_ROI_01'
dataset='B-NT-S10-2f_ROI_02'
dataset_ds="${dataset}ds7"
datadir="/vols/Data/km/michielk/oxdata/P01/EM/Myrf_01/SET-B/$dataset"
datadir_rem="/data/ndcn-fmrib-water-brain/ndcn0180/EM/Myrf_01/SET-B/$dataset"
rsync -Pazv $host:$datadir_rem/*.h5 $datadir
rsync -Pazv $datadir/blocks_0500/*.h5 $host:$datadir_rem/blocks_0500/JAL
rsync -Pazv $host:$datadir_rem/blocks_0500/*_sum*.h5 $datadir/blocks_0500/
rsync -Pazv $host:$datadir_rem/blocks_0500/*_probMA.h5 $datadir/blocks_0500/
rsync -Pazv $host:$datadir_rem/blocks_0500/*_probs1.h5 $datadir/blocks_0500/
rsync -Pazv $host:$datadir_rem/*_masks_maskDS.h5 $datadir
# rsync -Pazv $datadir/*ds7_masks_maskDS.h5 $host:$datadir_rem
rsync -Pazv $host:$datadir_rem/blocks_2000/*ws.h5 $datadir/blocks_2000

dataset='M3S1GNU'
dataset_ds="${dataset}ds7"
datadir="/vols/Data/km/michielk/oxdata/P01/EM/M3/$dataset"
datadir_rem="/data/ndcn-fmrib-water-brain/ndcn0180/EM/M3/$dataset"
rsync -Pazv $host:$datadir_rem/*_probs.h5 $datadir
rsync -Pazv $host:$datadir_rem/*_masks.h5 $datadir
rsync -Pazv $host:$datadir_rem/blocks_0500/*_probs.h5 $datadir/blocks_0500/
rsync -Pazv $datadir/M3S1GNU_masks*.h5 $host:$datadir_rem
