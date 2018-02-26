scriptdir="$HOME/workspace/EM"
export PYTHONPATH=$scriptdir
DATA="$HOME/oxdata/P01"
host=ndcn0180@arcus-b.arc.ox.ac.uk
basedir_rem='/data/ndcn-fmrib-water-brain/ndcn0180/EM/M3'
basedir="${DATA}/EM/Myrf_01/SET-B/B-NT-S10-2f_ROI_00"
dataset='M3S1GNU'

datadir_rem=$basedir_rem/${dataset}
dspf='ds'; ds=7;
dataset_ds=$dataset$dspf$ds

# source activate scikit-image-devel_0.13  # for mpi
source activate zwatershed

datadir=$basedir/${dataset_ds} && mkdir -p $datadir && cd $datadir

rsync -Pazv $host:$datadir_rem/M3S1GNUds7_probs1_eed2.h5 $datadir

python $scriptdir/wmem/stack2stack.py \
$datadir/${dataset_ds}_probs1_eed2.h5/probs_eed \
$datadir/${dataset_ds}_probs1_eed2.nii.gz -i zyx -o xyz

# TODO: regenerate 2Dlabels from maskMM instead
python $scriptdir/wmem/stack2stack.py \
$datadir/${dataset_ds}_probs1_eed2.h5/probs_eed \
$datadir/${dataset_ds}_probs1_eed2_cut.h5/probs_eed -X 1311 -Y 1255

props=('label' 'area' 'eccentricity' 'mean_intensity' 'solidity' 'extent' 'euler_number')

# map all properties of all labels in labelMA_core2D (i.e. all criteria set to None)
python $scriptdir/wmem/connected_components.py \
$datadir/${dataset_ds}_labelMA_core2D_fw_nf_label.h5/stack \
$datadir/${dataset_ds}_labels_mapall.h5 \
-m '2Dfilter' -d 0 \
--maskMB $datadir/${dataset_ds}_probs1_eed2_cut.h5/probs_eed \
-p ${props[@]}
