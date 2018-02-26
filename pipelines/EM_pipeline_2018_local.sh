# rsync -Pazv ndcn0180@arcus-b.arc.ox.ac.uk:/data/ndcn-fmrib-water-brain/ndcn0180/EM/Myrf_01/SET-B/B-NT-S10-2f_ROI_00/B-NT-S10-2f_ROI_00ds7_labels*  /Users/michielk/oxdata/P01/EM/Myrf_01/SET-B/B-NT-S10-2f_ROI_00

scriptdir="$HOME/workspace/EM"
export PYTHONPATH=$scriptdir
DATA="$HOME/oxdata/P01"
host=ndcn0180@arcus-b.arc.ox.ac.uk
basedir_rem='/data/ndcn-fmrib-water-brain/ndcn0180/EM/Myrf_01/SET-B'
basedir="${DATA}/EM/Myrf_01/SET-B"
dataset='B-NT-S10-2f_ROI_00'
datadir=$basedir/${dataset} && mkdir -p $datadir && cd $datadir
datadir_rem=$basedir_rem/${dataset}
#source datastems_blocks.sh
dspf='ds'; ds=7;
dataset_ds=$dataset$dspf$ds
source activate scikit-image-devel_0.13  # for mpi
source activate zwatershed  # for jupyter conda

###=========================================================================###
### create downsampled nifti base data
###=========================================================================###
rsync -Pazv $host:$datadir_rem/reg_ds7 $datadir

'''
import os
import numpy as np
from glob import glob
from skimage.io import imread, imsave
datadir = "/Users/michielk/oxdata/P01/EM/Myrf_01/SET-B/B-NT-S10-2f_ROI_00"
files = glob(os.path.join(datadir, 'reg_ds7', '*.tif'))
f = files[0]
data = imread(f)
row = np.zeros([data.shape[0], 1], dtype='uint8')
for f in files:
    data = imread(f)
    data = np.append(data, row, 1)
    imsave(f, data)
'''

python $scriptdir/wmem/series2stack.py \
$datadir/reg_ds7 $datadir/$dataset_ds.h5/data \
-r '*.tif' -O '.h5' -e 0.1 0.049 0.049
python $scriptdir/wmem/stack2stack.py \
$datadir/$dataset_ds.h5/data \
$datadir/${dataset_ds}_data.nii.gz

# B-NT-S10-2f_ROI_00_05480-06020_05480-06020_00000-00184_probs1_eed2.h5
# python $scriptdir/wmem/downsample_slices.py \
# $datadir/blocks_500/B-NT-S10-2f_ROI_00_05480-06020_05480-06020_00000-00184_probs1_eed2.h5/probs_eed \
# $datadir/blocks_500/B-NT-S10-2f_ROI_00_05480-06020_05480-06020_00000-00184ds7_probs1_eed2.h5/probs_eed \
# -f 7

# python $scriptdir/wmem/downsample_blockwise.py \
# $datadir/blocks_500/B-NT-S10-2f_ROI_00_05480-06020_05480-06020_00000-00184_probs1_eed2.h5/probs_eed \
# $datadir/blocks_500/B-NT-S10-2f_ROI_00_05480-06020_05480-06020_00000-00184ds7_probs1_eed2.h5/probs_eed \
# -B 1 ${ds} ${ds} -f 'np.mean'
# python $scriptdir/wmem/downsample_blockwise.py \
# $datadir/blocks_500/B-NT-S10-2f_ROI_00_05480-06020_05480-06020_00000-00184_probs1_eed2.h5/probs_eed \
# $datadir/blocks_500/B-NT-S10-2f_ROI_00_05480-06020_05480-06020_00000-00184ds7_probs1_eed2.h5/probs_eed \
# -B 1 ${ds} ${ds} -f 'np.mean' -D 0 100 1 0 0 1 0 0 1
# python $scriptdir/wmem/downsample_blockwise.py \
# $datadir/blocks_500/B-NT-S10-2f_ROI_00_05480-06020_05480-06020_00000-00184_probs1_eed2.h5/probs_eed \
# $datadir/blocks_500/B-NT-S10-2f_ROI_00_05480-06020_05480-06020_00000-00184ds7_probs1_eed2.h5/probs_eed \
# -B 1 ${ds} ${ds} -f 'np.mean' -D 100 0 1 0 0 1 0 0 1
# python $scriptdir/wmem/stack2stack.py \
# $datadir/blocks_500/B-NT-S10-2f_ROI_00_05480-06020_05480-06020_00000-00184ds7_probs1_eed2.h5/probs_eed \
# $datadir/blocks_500/B-NT-S10-2f_ROI_00_05480-06020_05480-06020_00000-00184ds7_probs1_eed2_probs_eed.nii.gz

# python $scriptdir/wmem/downsample_blockwise.py \
# $datadir/blocks_500/B-NT-S10-2f_ROI_00_05480-06020_05480-06020_00000-00184_probs1_eed2.h5/probs_eed \
# $datadir/blocks_500/B-NT-S10-2f_ROI_00_05480-06020_05480-06020_00000-00184ds7_probs1_eed2.h5/probs_eed \
# -B 1 ${ds} ${ds} -f 'np.mean'
#
# python $scriptdir/wmem/downsample_blockwise.py \
# $datadir/blocks_500/B-NT-S10-2f_ROI_00_05480-06020_05480-06020_00000-00184_probs.h5/volume/predictions \
# $datadir/blocks_500/B-NT-S10-2f_ROI_00_05480-06020_05480-06020_00000-00184ds7_probs.h5/volume/predictions \
# -B 1 ${ds} ${ds} 1 -f 'np.mean' -D 0 100 1 0 0 1 0 0 1 0 0 1
# python $scriptdir/wmem/downsample_blockwise.py \
# $datadir/blocks_500/B-NT-S10-2f_ROI_00_05480-06020_05480-06020_00000-00184_probs.h5/volume/predictions \
# $datadir/blocks_500/B-NT-S10-2f_ROI_00_05480-06020_05480-06020_00000-00184ds7_probs.h5/volume/predictions \
# -B 1 ${ds} ${ds} 1 -f 'np.mean' -D 100 184 1 0 0 1 0 0 1 0 0 1
# python $scriptdir/wmem/stack2stack.py \
# $datadir/blocks_500/B-NT-S10-2f_ROI_00_05480-06020_05480-06020_00000-00184ds7_probs.h5/volume/predictions \
# $datadir/blocks_500/B-NT-S10-2f_ROI_00_05480-06020_05480-06020_00000-00184ds7_probs.nii.gz -u
#
# python $scriptdir/wmem/downsample_blockwise.py \
# $datadir/blocks_500/B-NT-S10-2f_ROI_00_05480-06020_05480-06020_00000-00184ds7_probs.h5/volume/predictions \
# $datadir/blocks_500/B-NT-S10-2f_ROI_00_05480-06020_05480-06020_00000-00184test_probs.h5/volume/predictions \
# -B 1 ${ds} ${ds} 1 -f 'expand' -D 0 0 1 0 0 1 0 0 1 0 0 1
# python $scriptdir/wmem/stack2stack.py \
# $datadir/blocks_500/B-NT-S10-2f_ROI_00_05480-06020_05480-06020_00000-00184test_probs.h5/volume/predictions \
# $datadir/blocks_500/B-NT-S10-2f_ROI_00_05480-06020_05480-06020_00000-00184test_probs.nii.gz -u


rsync -Pazv $host:$datadir_rem/$dataset_ds.h5 $datadir
python $scriptdir/wmem/stack2stack.py \
$datadir/$dataset_ds.h5/data \
$datadir/${dataset_ds}_data.nii.gz -u

pf='_probs?_eed2'
rsync -Pazv $host:$datadir_rem/$dataset_ds$pf.h5 $datadir
for pf in '_probs0_eed2' '_probs1_eed2' '_probs2_eed2'; do
    python $scriptdir/wmem/stack2stack.py \
    $datadir/$dataset_ds$pf.h5/probs_eed \
    $datadir/$dataset_ds${pf}.nii.gz -u
done


###=========================================================================###
### create nifti maskMM
###=========================================================================###
pf='_masks'
rsync -Pazv $host:$datadir_rem/$dataset_ds$pf.h5 $datadir
pf='maskMM'
python $scriptdir/wmem/stack2stack.py \
$datadir/${dataset_ds}_masks.h5/$pf \
$datadir/${dataset_ds}_masks_$pf.nii.gz

pf='maskMA'
python $scriptdir/wmem/prob2mask.py \
${datadir}/${dataset_ds}_probs1_eed2.h5/probs_eed \
${datadir}/${dataset_ds}_masks.h5/$pf -l 0.2
python $scriptdir/wmem/stack2stack.py \
$datadir/${dataset_ds}_masks.h5/$pf \
$datadir/${dataset_ds}_masks_$pf.nii.gz
python $scriptdir/wmem/connected_components.py \
$datadir/${dataset_ds}_masks.h5/maskMA \
$datadir/${dataset_ds}_labels.h5/labelMA_core3D \
-m '3D'  # -q 10000
# --maskDS $datadir/${dataset_ds}_masks.h5/maskDS \
python $scriptdir/wmem/stack2stack.py \
$datadir/${dataset_ds}_labels.h5/labelMA_core3D \
$datadir/${dataset_ds}_labels_labelMA_core3D.nii.gz


# nodes of ranvier detection
python $scriptdir/wmem/nodes_of_ranvier.py \
$datadir/${dataset_ds}_labels.h5/labelMA_core3D \
$datadir/${dataset_ds}_labels.h5/labelMA_core3D_NoR \
--boundarymask $datadir/${dataset_ds}_masks.h5/maskDS \
-s 5000
python $scriptdir/wmem/stack2stack.py \
$datadir/${dataset_ds}_labels.h5/labelMA_core3D_NoR \
$datadir/${dataset_ds}_labels_labelMA_core3D_NoR.nii.gz

# iter=2
# python $scriptdir/supervoxels/nodes_of_ranvier.py \
# $datadir $datastem -m 'nomerge' \
# -l "${volws}${NoRpf}_iter$((iter-1))_automerged" 'stack' \
# -o "${volws}${NoRpf}_iter${iter}" 'stack' \
# -S '_maskDS_invdil' 'stack'


###=========================================================================###
### create nifti labels
###=========================================================================###
pf='_labels'
rsync -Pazv $host:$datadir_rem/$dataset_ds$pf.h5 $datadir

props=('label' 'area' 'eccentricity' 'mean_intensity' 'solidity' 'extent' 'euler_number')

# map all properties of all labels in labelMA_core2D (i.e. all criteria set to None)
python $scriptdir/wmem/connected_components.py \
$datadir/${dataset_ds}_labels.h5/labelMA_core2D \
$datadir/${dataset_ds}_labels_mapall.h5 \
-m '2Dfilter' -d 0 \
--maskMB $datadir/${dataset_ds}_probs1_eed2.h5/probs_eed \  # FIXME: enter mask?
-p ${props[@]}
# generate the forward mappings
python $scriptdir/wmem/connected_components.py \
$datadir/${dataset_ds}_labels.h5/labelMA_core2D \
$datadir/${dataset_ds}_labels_mapall.h5 \
-m '2Dprops' -d 0 \
-b $datadir/${dataset_ds}_labels_mapall \
-p ${props[@]}
# convert maps to nifti
for pf in ${props[@]}; do
python $scriptdir/wmem/stack2stack.py \
$datadir/${dataset_ds}_labels_mapall.h5/$pf \
$datadir/${dataset_ds}_labels_mapall_$pf.nii.gz
done
# 'labelMA_core2D' 'labelMA_2Dlabeled'


# python $scriptdir/wmem/connected_components.py \
# $datadir/${dataset}${dspf}${ds}_labels.h5/labelMA_core2D \
# $datadir/${dataset}${dspf}${ds}_labels.h5 \
# -m '2Dfilter' -d 0 \
# -p 'label' 'area' 'eccentricity' 'solidity' 'extent' 'euler_number' \
# -a 10 -A 1500 -E 1 -e 0 -s 0.50 -n 0.3
# source activate scikit-image-devel_0.13
# mpiexec -n 6 python $scriptdir/wmem/connected_components.py \
# $datadir/${dataset}${dspf}${ds}_labels.h5/labelMA_core2D \
# $datadir/${dataset}${dspf}${ds}_labels.h5 \
# -M -m '2Dfilter' -d 0 \
# -p 'label' 'area' 'eccentricity' 'solidity' 'extent' 'euler_number' \
# -a 10 -A 1500 -e 0 -E 1 -s 0.50

#
# -a min area: 10
# -A max area: 2000
# -E max eccentricity: 0.8
# -e min euler_number: -1
# -n min extent: 0.3
# -I min intensity: 0.2
# -s min solidity: 0.5?


### filter labels
import os
from wmem import utils
import numpy as np

datadir = "/Users/michielk/oxdata/P01/EM/Myrf_01/SET-B/B-NT-S10-2f_ROI_00"
fname = 'B-NT-S10-2f_ROI_00ds7_labels'
dset0 = 'labelMA_core2D'

h5path_in = os.path.join(datadir, '{}.h5/{}'.format(fname, dset0))
h5file_in, ds_in, elsize, axlab = utils.h5_load(h5path_in)
h5path_out = os.path.join(datadir, '{}.h5/{}_filtered'.format(fname, dset0))
h5file_out, ds_out = utils.h5_write(None, ds_in.shape, ds_in.dtype,
                                    h5path_out,
                                    element_size_um=elsize,
                                    axislabels=axlab)
ds_out[:, :, :] = ds_in
h5file_out.close()
h5file_in.close()

h5path_in = os.path.join(datadir, '{}.h5/{}_filtered'.format(fname, dset0))
h5file_in, ds_in, elsize, axlab = utils.h5_load(h5path_in)

basename = os.path.join(datadir, fname + '_mapall')
propname = 'label'
nppath = '{}_{}.npy'.format(basename, propname)
fw = np.load(nppath)
fw_orig = np.load(nppath)

mask = np.zeros([6, fw.shape[0]])

propname = 'area'
nppath = '{}_{}.npy'.format(basename, propname)
fwf = np.load(nppath)
mask[0, ...] = np.logical_and(fwf>10, fwf<2000)
# fw[~mask] = 0

propname = 'solidity'
nppath = '{}_{}.npy'.format(basename, propname)
fwf = np.load(nppath)
mask[1, ...] = fwf>0.5
# fw[~mask] = 0

propname = 'mean_intensity'
nppath = '{}_{}.npy'.format(basename, propname)
fwf = np.load(nppath)
mask[2, ...] = fwf>0.2
# fw[~mask] = 0

propname = 'eccentricity'
nppath = '{}_{}.npy'.format(basename, propname)
fwf = np.load(nppath)
mask[3, ...] = fwf<0.8
# fw[~mask] = 0

propname = 'extent'
nppath = '{}_{}.npy'.format(basename, propname)
fwf = np.load(nppath)
mask[4, ...] = fwf>0.3
# fw[~mask] = 0

propname = 'euler_number'
nppath = '{}_{}.npy'.format(basename, propname)
fwf = np.load(nppath)
mask[5, ...] = fwf>-1
# fw[~mask] = 0

masksum = np.sum(mask, 0)
maskbin = masksum > 3
fw[~maskbin] = 0

propname = 'area'
nppath = '{}_{}.npy'.format(basename, propname)
fwf = np.load(nppath)
m = fwf>3000  # always exclude
fw[m] = 0

propname = 'mean_intensity'
nppath = '{}_{}.npy'.format(basename, propname)
fwf = np.load(nppath)
m = fwf>0.5  # always include
fw[m] = fw_orig[m]

propname = 'solidity'
nppath = '{}_{}.npy'.format(basename, propname)
fwf = np.load(nppath)
m = fwf>0.8  # always include
fw[m] = fw_orig[m]

propname = 'solidity'
nppath = '{}_{}.npy'.format(basename, propname)
fwf = np.load(nppath)
m = fwf<0.8  # always exclude
fw[m] = 0

out = fw[ds_in[:,:,:]]
ds_in[:,:,:] = out

h5file_in.close()

# import matplotlib.pyplot as plt
# n_bins = 20
# fig, ax = plt.subplots(tight_layout=True)
# fwf[fwf>10000] = 0
# ax.hist(fwf, bins=n_bins)
# plt.show()



dset='labelMA_core2D'
python $scriptdir/wmem/stack2stack.py \
$datadir/${dataset_ds}_labels.h5/${dset}_filtered \
$datadir/${dataset_ds}_labels_${dset}_filtered.nii.gz
fslmaths $datadir/${dataset_ds}_labels_${dset}_filtered.nii.gz \
-bin $datadir/${dataset_ds}_labels_${dset}_filtered_mask.nii.gz




###
dset='labelMA_core2D'
pf='_pred'  # '_filtered'
python $scriptdir/wmem/connected_components.py \
$datadir/${dataset_ds}_labels.h5/${dset}${pf} \
$datadir/${dataset_ds}_labels.h5/labelMA_3Dlabeled \
-m '2Dto3D' -d 0
python $scriptdir/wmem/stack2stack.py \
$datadir/${dataset_ds}_labels.h5/labelMA_3Dlabeled \
$datadir/${dataset_ds}_labels_labelMA_3Dlabeled.nii.gz


mpiexec -n 6 python $scriptdir/wmem/merge_slicelabels.py \
$datadir/${dataset_ds}_labels.h5/labelMA_3Dlabeled \
$datadir/${dataset_ds}_labels.h5/labelMA_3Diter1 \
-M -m 'MAstitch' -d 0 \
-q 2 -o 0.50
python $scriptdir/wmem/merge_slicelabels.py \
$datadir/${dataset_ds}_labels.h5/labelMA_3Dlabeled \
$datadir/${dataset_ds}_labels.h5/labelMA_3Diter0 \
--maskMM $datadir/${dataset_ds}_masks.h5/maskMM \
-m 'MAfwmap' -d 0
python $scriptdir/wmem/stack2stack.py \
$datadir/${dataset_ds}_labels.h5/labelMA_3Diter0 \
$datadir/${dataset_ds}_labels_labelMA_3Diter0.nii.gz
python $scriptdir/wmem/merge_slicelabels.py \
$datadir/${dataset_ds}_labels.h5/labelMA_3Dlabeled \
$datadir/${dataset_ds}_labels.h5/labelMA_3Diter1 \
--maskMM $datadir/${dataset_ds}_masks.h5/maskMM \
-m 'MAfwmap' -d 0 \
-l 6 1 1 -s 200 -r
python $scriptdir/wmem/stack2stack.py \
$datadir/${dataset_ds}_labels.h5/labelMA_3Diter1 \
$datadir/${dataset_ds}_labels_labelMA_3Diter1.nii.gz
python $scriptdir/wmem/stack2stack.py \
$datadir/${dataset_ds}_labels.h5/labelMA_3Diter1_shuffled \
$datadir/${dataset_ds}_labels_labelMA_3Diter1_shuffled.nii.gz




mpiexec -n 6 python $scriptdir/wmem/merge_slicelabels.py \
$datadir/${dataset_ds}_labels.h5/labelMA_3Diter1 \
$datadir/${dataset_ds}_labels.h5/labelMA_3Diter2 \
-M -m 'MAstitch' -d 0 \
-q 2 -o 0.50
python $scriptdir/wmem/merge_slicelabels.py \
$datadir/${dataset_ds}_labels.h5/labelMA_3Diter1 \
$datadir/${dataset_ds}_labels.h5/labelMA_3Diter2 \
--maskMM $datadir/${dataset_ds}_masks.h5/maskMM \
-m 'MAfwmap' -d 0 \
-l 6 1 1
python $scriptdir/wmem/stack2stack.py \
$datadir/${dataset_ds}_labels.h5/labelMA_3Diter2 \
$datadir/${dataset_ds}_labels_labelMA_3Diter2.nii.gz






### shuffle labels
import os
from wmem import utils
import numpy as np

datadir = "/Users/michielk/oxdata/P01/EM/Myrf_01/SET-B/B-NT-S10-2f_ROI_00"
fname = 'B-NT-S10-2f_ROI_00ds7_labels'
dset = 'labelMA_core2D'  #'labelMA_3Dlabeled'  # 'labelMA_3Diter1'  # 'labelMA_core3D'  # 'label' 'labelMA_2Dlabeled' 'labelMA_core2D' 'labelMA_3Diter1'

h5path_in = os.path.join(datadir, '{}.h5/{}'.format(fname, dset))
h5file_in, ds_in, elsize, axlab = utils.h5_load(h5path_in)
fw = utils.shuffle_labels(ds_in)
fw.dtype = ds_in.dtype
h5path_out = os.path.join(datadir, '{}.h5/{}_shuffled'.format(fname, dset))
h5file_out, ds_out = utils.h5_write(None, ds_in.shape, ds_in.dtype,
                                    h5path_out,
                                    element_size_um=elsize,
                                    axislabels=axlab)

ds_out[:, :, :] = fw[ds_in]
h5file_out.close()
# niipath_out = os.path.join(datadir, '{}_{}_shuffled.nii.gz'.format(fname, dset))
# utils.write_to_nifti(niipath_out, np.transpose(fw[ds_in]), elsize)  # np.flipud(elsize)
h5file_in.close()

dset='labelMA_core2D'
python $scriptdir/wmem/stack2stack.py \
${datadir}/${dataset_ds}_labels.h5/${dset}_shuffled \
${datadir}/${dataset_ds}_labels_${dset}_shuffled.nii.gz


### mask labels
import os
from wmem import utils

datadir = "/Users/michielk/oxdata/P01/EM/Myrf_01/SET-B/B-NT-S10-2f_ROI_00"
h5path_in = os.path.join(datadir, 'B-NT-S10-2f_ROI_00ds7_labels.h5/label')
h5path_out = os.path.join(datadir, 'B-NT-S10-2f_ROI_00ds7_masks.h5/maskMA')

h5file_in, ds_in, elsize, axlab = utils.h5_load(h5path_in)
h5file_out, ds_out = utils.h5_write(None, ds_in.shape, 'bool',
                                    h5path_out,
                                    element_size_um=elsize,
                                    axislabels=axlab)

ds_out[:, :, :] = ds_in > 0

h5file_in.close()
h5file_out.close()

python $scriptdir/wmem/stack2stack.py \
${basepath}${dataset}${dspf}${ds}_masks.h5/maskMA \
${basepath}${dataset}${dspf}${ds}_masks_maskMA.nii.gz

# fslmaths B-NT-S10-2f_ROI_00ds7_labels_label.nii.gz -bin B-NT-S10-2f_ROI_00ds7_masks_maskMA.nii.gz
# fslmaths B-NT-S10-2f_ROI_00ds7_masks_maskMA.nii.gz -mul 2 -add B-NT-S10-2f_ROI_00ds7_masks_maskMM.nii.gz B-NT-S10-2f_ROI_00ds7_masks_maskMM-MA.nii.gz


fslmaths B-NT-S10-2f_ROI_00ds7_masks_maskMM.nii.gz -mul 2 -add B-NT-S10-2f_ROI_00ds7_masks_maskDS.nii.gz B-NT-S10-2f_ROI_00ds7_masks_maskMM-DS.nii.gz


# rsync -Pazv ndcn0180@arcus-b.arc.ox.ac.uk:/data/ndcn-fmrib-water-brain/ndcn0180/EM/Myrf_01/SET-B/B-NT-S10-2f_ROI_00/B-NT-S10-2f_ROI_00ds7_probs.h5  /Users/michielk/oxdata/P01/EM/Myrf_01/SET-B/B-NT-S10-2f_ROI_00
pf='_probs'
rsync -Pazv $host:$datadir_rem/$dataset_ds$pf.h5 $datadir


###=========================================================================###
### myelin test
###=========================================================================###
pf='_00480-01020_00480-01020_00000-00184'
pf='_00480-01020_00480-01020_00000-00184_probs'
pf='_00480-01020_00480-01020_00000-00184_masks'
rsync -Pazv $host:$datadir_rem/blocks_0500/$dataset$pf.h5 $datadir

# #vol1 + vol6
# import os
# from wmem import utils, stack2stack
# datadir = "/Users/michielk/oxdata/P01/EM/Myrf_01/SET-B/B-NT-S10-2f_ROI_00"
# dataset = 'B-NT-S10-2f_ROI_00ds7'
# dataset = 'B-NT-S10-2f_ROI_00_00480-01020_00480-01020_00000-00184'
# pf = '_probs'
# # dpf='_vol00+vol04+vol07'
# dpf='_vol00+vol02+vol04+vol07'
# # dpf='_vol00+vol02+vol04+vol05+vol07'
# h5path_in = os.path.join(datadir, '{}{}.h5'.format(dataset, pf), 'volume/predictions')
# h5path_out = os.path.join(datadir, '{}{}{}.h5'.format(dataset, pf, dpf), 'data')
# niipath_out = os.path.join(datadir, '{}{}{}{}.nii.gz'.format(dataset, pf, dpf, 'data'))
# h5file_in, ds_in, elsize, axlab = utils.h5_load(h5path_in)
# h5file_out, ds_out = utils.h5_write(None, ds_in.shape[:3], ds_in.dtype,
#                                     h5path_out,
#                                     element_size_um=elsize[:3],
#                                     axislabels=axlab[:3])
# ds_out[:,:,:] = ds_in[:,:,:,0] + ds_in[:,:,:,2] + ds_in[:,:,:,4] + ds_in[:,:,:,5] + ds_in[:,:,:,7]
# # mask = ds_out[:,:,:] < 0
# # ds_out[mask] = 0
# h5file_in.close()
# h5file_out.close()
# _ = stack2stack.stack2stack(h5path_out, niipath_out)

dpf='_vol00+vol02+vol04+vol07'
python $scriptdir/wmem/combine_vols.py \
$datadir/${dataset_ds}_probs.h5/volume/predictions \
$datadir/${dataset_ds}_probs${dpf}.h5/data \
-i 0 2 4 7

test_EED.m

# import os
# import numpy as np
# import h5py
# datadir = "/Users/michielk/oxdata/P01/EM/Myrf_01/SET-B/B-NT-S10-2f_ROI_00"
# filestem = 'B-NT-S10-2f_ROI_00_00480-01020_00480-01020_00000-00184'
# dset0 = 'data'
# h5path = os.path.join(datadir, filestem + '.h5')
# h5file0 = h5py.File(h5path, 'a')
# ds0 = h5file0[dset0]
# filestem = 'B-NT-S10-2f_ROI_00_00480-01020_00480-01020_00000-00184_probs_vol00+vol02+vol04+vol07_eed2'
# h5path = os.path.join(datadir, filestem + '.h5')
# h5file1 = h5py.File(h5path, 'a')
# ds = h5file1['probs_eed']
# ds.attrs['element_size_um'] = np.append(ds0.attrs['element_size_um'], 1)
# h5file1.close()

pf='_probs_vol00+vol02+vol04+vol07_eed2'
spf='probs_eed'
python $scriptdir/wmem/stack2stack.py \
$datadir/${dataset_ds}${pf}.h5/${spf} \
$datadir/${dataset_ds}${pf}_${spf}.nii.gz

###=========================================================================###
### new masks test
###=========================================================================###
pf='_probs_vol00+vol02+vol04+vol07_eed2'
spf='probs_eed'
python $scriptdir/wmem/prob2mask.py \
$datadir/${dataset_ds}${pf}.h5/${spf} \
${datadir}/${dataset_ds}${pf}_masks.h5/maskMM -l 0.5 -s 2000 -d 1 -S

python $scriptdir/wmem/stack2stack.py \
${datadir}/${dataset_ds}${pf}_masks.h5/maskMM \
${datadir}/${dataset_ds}${pf}_masks_maskMM.nii.gz
python $scriptdir/wmem/stack2stack.py \
${datadir}/${dataset_ds}${pf}_masks.h5/maskMM_steps/raw \
${datadir}/${dataset_ds}${pf}_masks_maskMM_steps_raw.nii.gz
python $scriptdir/wmem/stack2stack.py \
${datadir}/${dataset_ds}${pf}_masks.h5/maskMM_steps/mito \
${datadir}/${dataset_ds}${pf}_masks_maskMM_steps_mito.nii.gz
python $scriptdir/wmem/stack2stack.py \
${datadir}/${dataset_ds}${pf}_masks.h5/maskMM_steps/dil \
${datadir}/${dataset_ds}${pf}_masks_maskMM_steps_dil.nii.gz

###=========================================================================###
### watershed test
###=========================================================================###
pf='_00480-01020_00480-01020_00000-00184'
pf='_00480-01020_00480-01020_00000-00184_probs'
pf='_00480-01020_00480-01020_00000-00184_masks'
rsync -Pazv $host:$datadir_rem/blocks_0500/$dataset$pf.h5 $datadir

#vol1 + vol6
import os
from wmem import utils, stack2stack
datadir = "/Users/michielk/oxdata/P01/EM/Myrf_01/SET-B/B-NT-S10-2f_ROI_00"
dataset = 'B-NT-S10-2f_ROI_00ds7'
dataset = 'B-NT-S10-2f_ROI_00_00480-01020_00480-01020_00000-00184'
pf = '_probs'
dpf='_vol01+vol06-vol03'
h5path_in = os.path.join(datadir, '{}{}.h5'.format(dataset, pf), 'volume/predictions')
h5path_out = os.path.join(datadir, '{}{}{}.h5'.format(dataset, pf, dpf), 'data')
niipath_out = os.path.join(datadir, '{}{}{}{}.nii.gz'.format(dataset, pf, dpf, 'data'))
h5file_in, ds_in, elsize, axlab = utils.h5_load(h5path_in)
h5file_out, ds_out = utils.h5_write(None, ds_in.shape[:3], ds_in.dtype,
                                    h5path_out,
                                    element_size_um=elsize[:3],
                                    axislabels=axlab[:3])
ds_out[:,:,:] = ds_in[:,:,:,1] + ds_in[:,:,:,6] - ds_in[:,:,:,3]
# mask = ds_out[:,:,:] < 0
# ds_out[mask] = 0
h5file_in.close()
h5file_out.close()
_ = stack2stack.stack2stack(h5path_out, niipath_out)

# watershed
dataset_ds='B-NT-S10-2f_ROI_00_00480-01020_00480-01020_00000-00184'
l=0.90; u=1.00; s=010;
pf=_ws
spf=l${l}-u${u}-s${s}
dpf=_vol01+vol06-vol03
python $scriptdir/wmem/watershed_ics.py \
$datadir/${dataset_ds}_probs${dpf}.h5/data \
$datadir/${dataset_ds}${pf}.h5/${spf} \
--masks NOT $datadir/${dataset_ds}_masks.h5/maskMM \
XOR $datadir/${dataset_ds}_masks.h5/maskDS \
-l $l -u $u -s $s -S
# --seedimage $datadir/${dataset_ds}${pf}.h5/l0.99-u1.00-s010_steps/seeds \

# nifti
python $scriptdir/wmem/stack2stack.py \
$datadir/${dataset_ds}${pf}.h5/${spf}_steps/mask \
$datadir/${dataset_ds}${pf}_${spf}_steps_mask.nii.gz
python $scriptdir/wmem/stack2stack.py \
$datadir/${dataset_ds}${pf}.h5/${spf}_steps/seeds \
$datadir/${dataset_ds}${pf}_${spf}_steps_seeds.nii.gz
python $scriptdir/wmem/stack2stack.py \
$datadir/${dataset_ds}${pf}.h5/${spf} \
$datadir/${dataset_ds}${pf}_${spf}.nii.gz

pf=
spf='data'
python $scriptdir/wmem/stack2stack.py \
$datadir/${dataset_ds}${pf}.h5/${spf} \
$datadir/${dataset_ds}${pf}_${spf}.nii.gz


###=========================================================================###
### zwatershed
###=========================================================================###
# from zwatershed import zwatershed_and_metrics_h5
# from zwatershed import zwatershed_and_metrics
# (segs, rand) = zwatershed_and_metrics(segTrue, aff_graph, eval_thresh_list, seg_save_thresh_list)


# python $scriptdir/wmem/stack2stack.py \
# $datadir/${dataset_ds}_probs1_eed2.h5/probs_eed \
# $datadir/${dataset_ds}_probs1_eed2_main.h5/main

python $scriptdir/wmem/stack2stack.py \
$datadir/${dataset_ds}_probs.h5/volume/predictions \
$datadir/${dataset_ds}_probs_main.h5/main -o 'czyx'

python $scriptdir/wmem/stack2stack.py \
$datadir/${dataset_ds}_probs.h5/volume/predictions \
$datadir/${dataset_ds}_probs_main_vol00.h5/main -o 'czyx' -c 0 -C 1

python $scriptdir/wmem/stack2stack.py \
$datadir/${dataset_ds}_probs.h5/volume/predictions \
$datadir/${dataset_ds}_probs_main_vol01.h5/main -o 'czyx' -c 1 -C 2

# # %matplotlib nbagg
# import os
# from zwatershed import (partition_subvols,
#                         eval_with_par_map,
#                         stitch_and_save,
#                         merge_by_thresh)
# datadir = "/Users/michielk/oxdata/P01/EM/Myrf_01/SET-B/B-NT-S10-2f_ROI_00"
# # pred_file = os.path.join(datadir, 'B-NT-S10-2f_ROI_00ds7_probs1_eed2_main.h5')  # dataset: 'main'
# pred_file = os.path.join(datadir, 'B-NT-S10-2f_ROI_00ds7_probs_main_vol01.h5')  # dataset: 'main'
# out_folder = os.path.join(datadir, 'zws_vol01')
# max_len = 300
# partition_data = partition_subvols(pred_file,out_folder,max_len)
#
# NUM_WORKERS=5
# # eval_with_spark(partition_data[0])
# eval_with_par_map(partition_data[0], NUM_WORKERS)
#
# outname = os.path.join(datadir, 'B-NT-S10-2f_ROI_00ds7_probs_zws.h5')  # dataset: 'main'
# stitch_and_save(partition_data, outname)
#
#
# num,thresh = 0,2000
#
# '''
# plt.subplot(1,2,1)
# basic_file = h5py.File('/nobackup/turaga/singhc/par_zwshed/0_0_0_voll/'+'basic.h5','r')
# seg_init = np.array(basic_file['seg'])
# rg_init = np.array(basic_file['rg'])
# keeps = rg_init[:,0]<rg_init[:,1]
# rg_init = rg_init[keeps,:]
#
# seg_sizes_init = np.array(basic_file['counts'])
# basic_file.close()
# plt.imshow(seg_init[V,:,:], cmap=cmap)
# plt.title('seg_init')
# '''

# plt.subplot(1,2,2)
# f = h5py.File(outname, 'a')
# s,e = f['starts'][num],f['ends'][num]
# seg = f['seg'][s[0]:e[0]-3,s[1]:e[1]-3,s[2]:e[2]-3]
# seg_sizes = np.array(f['seg_sizes'])
# rg = np.array(f['rg_'+str(num)])
# f.close()
# plt.imshow(seg[V,:,:], cmap=cmap)
# plt.title('seg_after_stitching')
# plt.show()
#
# print "num_segs",len(np.unique(seg_init)),len(np.unique(seg))
# print "rg lens",len(rg_init),len(rg)
#
#
# # seg_init_merged = merge_by_thresh(seg_init,seg_sizes_init,rg_init,thresh)
# seg_merged = merge_by_thresh(seg,seg_sizes,rg,thresh)
#
# plt.subplot(1,2,1)
# plt.imshow(seg_init_merged[V,:,:], cmap=cmap)
# plt.title('merged init')
# plt.subplot(1,2,2)
# plt.imshow(seg_merged[V,:,:], cmap=cmap)
# plt.title('merged')
# plt.show()
#
# print "num_segs",len(np.unique(seg_init)),len(np.unique(seg))
# print "rg lens",len(rg_init),len(rg)
#
# # f = h5py.File(outname, 'a')
# # thresh = 0.5
# # seg_merged = merge_by_thresh(f['seg'], f['seg_sizes'], f['rg_i'], thresh)
# # f.close()


python $scriptdir/wmem/stack2stack.py \
$datadir/${dataset_ds}_probs_zws.h5/seg \
$datadir/${dataset_ds}_probs_zws_seg.nii.gz -d 'uint32'

python $scriptdir/wmem/stack2stack.py \
$datadir/zws/${dataset_ds}_probs_main_vol00_grad.h5/main \
$datadir/zws/${dataset_ds}_probs_main_vol00_grad.nii.gz

h5ls -v $datadir/zws/zws_vol01_0_0_0_vol/basic.h5
