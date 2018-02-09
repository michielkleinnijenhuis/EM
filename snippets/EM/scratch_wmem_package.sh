scriptdir="$HOME/workspace/EM"
# scriptdir="$HOME/workspace/EM/wmem"
datadir="$HOME/oxdata/P01/EM/scratch_wmem_package"
dataset='wmem'
dspf='ds'
ds=7
xe=0.0073; ye=0.0073; ze=0.05
export PYTHONPATH=$PYTHONPATH:$scriptdir

basepath=$datadir/${dataset}

python $scriptdir/wmem/downsample_slices.py \
$datadir/slices $datadir/slices_ds \
-f 4 -D 0 0 1 2500 0 1 2500 5000 1
python $scriptdir/wmem/series2stack.py \
$datadir/slices $basepath.h5/data \
-e ${ze} ${ye} ${xe} -o 'zyx' -s 5 20 20


python $scriptdir/wmem/downsample_slices.py $datadir/slices $datadir/slices_ds -d 4 -x 2500 -X 5000 -y 2500 -Y 3500 -z 4 -Z 7
python $scriptdir/wmem/series2stack.py $datadir/slices $basepath.h5/data -e ${ze} ${ye} ${xe} -a 'zyx' -c 5 20 20

python $scriptdir/wmem/stack2stack.py $basepath.h5/stack $basepath.nii.gz
python $scriptdir/wmem/stack2stack.py $basepath.h5/stack $basepath.h5/cutout -x 250 -X 750 -y 250 -Y 900 -a '.tif'
python $scriptdir/wmem/stack2stack.py $basepath.h5/stack $basepath.h5/transpose -o 'xyz'
python $scriptdir/wmem/stack2stack.py $basepath.h5/stack $basepath.h5/chunked -s 5 50 50
python $scriptdir/wmem/prob2mask.py $basepath.h5/stack $basepath.h5/mask -u 80
python $scriptdir/wmem/stack2stack.py $basepath.h5/mask ${basepath}_mask.nii.gz -a '.tif' -s 10 50 50
# python $scriptdir/wmem/prob2mask.py $basepath.h5/stack $basepath.h5/maskDS -l -1 -u 0
# python $scriptdir/wmem/stack2stack.py $basepath.h5/maskDS ${basepath}_maskDS.nii.gz -a '.tif' -s 10 50 50
python $scriptdir/wmem/prob2mask.py $basepath.h5/stack $basepath.h5/maskMB -u 50
python $scriptdir/wmem/stack2stack.py $basepath.h5/maskMB ${basepath}_maskMB.nii.gz -a '.tif' -s 10 50 50
python $scriptdir/wmem/reduceblocks.py $basepath.h5/mask $basepath.h5/mask_reduced
python $scriptdir/wmem/connected_components.py $basepath.h5/mask $basepath.h5/cc3d -M '3D'
python $scriptdir/wmem/connected_components.py $basepath.h5/mask $basepath.h5/cc2d -M '2D'
python $scriptdir/wmem/stack2stack.py $basepath.h5/cc2d ${basepath}_cc2d.nii.gz -a '.tif' -s 10 50 50
python $scriptdir/wmem/connected_components.py $basepath.h5/cc2d ${basepath}_fw.npy -M '2Dfilter' \
-d 0 --maskMB $basepath.h5/maskMB \
-a 10 -A 1500 -e 0 -x 0.30 -s 0.50 \
-p 'label' 'area' 'mean_intensity' 'eccentricity' 'solidity' 'extent' 'euler_number'
python $scriptdir/wmem/connected_components.py $basepath.h5/cc2d $basepath.h5/cc2dprops -b ${basepath}_fw -M '2Dprops' \
-p 'label' 'area' 'mean_intensity' 'eccentricity' 'solidity' 'extent' 'euler_number'
python $scriptdir/wmem/stack2stack.py $basepath.h5/cc2dprops/euler_number ${basepath}_cc2dprops_euler_number.nii.gz
python $scriptdir/wmem/stack2stack.py $basepath.h5/cc2dprops/label ${basepath}_cc2dprops_label.nii.gz

python $scriptdir/wmem/connected_components.py $basepath.h5/cc2dprops/label $basepath.h5/cc2dto3d -M '2Dto3D'

python $scriptdir/wmem/merge_slicelabels.py $basepath.h5/cc2dto3d $basepath.h5/3diter1 \
-M 'MAstitch' -d 0 -q 2 -t 0.50 -m
python $scriptdir/wmem/merge_slicelabels.py $basepath.h5/cc2dto3d $basepath.h5/3diter1 \
-M 'MAfwmap' -d 0 --maskMM $basepath.h5/mask -c 6 1 1 -s 200 -r

python $scriptdir/wmem/fill_holes.py $basepath.h5/cc2dto3d $basepath.h5/cc2dto3d_filled -M '1' -s 7 7 7 \
--outputholes $basepath.h5/cc2dto3d_holes


python $scriptdir/wmem/downsample_slices.py $datadir/slices $datadir/slices_${dspf}${ds} -d ${ds} -m
python $scriptdir/wmem/series2stack.py $datadir/slices_${dspf}${ds} $datadir/${dataset}_${dspf}${ds}.h5/stack -e 0.05 0.0511 0.0511 -a 'zyx' -c 5 20 20
python $scriptdir/wmem/stack2stack.py $datadir/${dataset}_${dspf}${ds}.h5/stack $datadir/${dataset}_${dspf}${ds}.nii.gz
python $scriptdir/wmem/stack2stack.py $datadir/${dataset}_${dspf}${ds}.h5/stack $datadir/${dataset}_${dspf}${ds}.nii.gz -a '.png'

python $scriptdir/wmem/prob2mask.py $datadir/${dataset}_${dspf}${ds}.h5/stack $datadir/${dataset}_${dspf}${ds}.h5/mask -l 80 -u 150
python $scriptdir/wmem/stack2stack.py $datadir/${dataset}_${dspf}${ds}.h5/mask $datadir/${dataset}_${dspf}${ds}_mask.nii.gz
# h5py+mpi4py not working locally
python $scriptdir/wmem/reduceblocks.py $datadir/${dataset}_${dspf}${ds}.h5/mask $datadir/${dataset}_${dspf}${ds}.h5/mask_reduced
python $scriptdir/wmem/reduceblocks.py $datadir/${dataset}_${dspf}${ds}.h5/mask_reduced $datadir/${dataset}_${dspf}${ds}.h5/mask_restored -f 'expand'

python $scriptdir/wmem/series2stack.py $datadir/slices_${dspf}${ds} $datadir/${dataset}_${dspf}${ds}.h5 'stack' -e 0.05 0.0511 0.0511 -a 'zyx' -c 5 20 20
python $scriptdir/wmem/prob2mask.py $datadir/${dataset}_${dspf}${ds}.h5/stack $datadir/${dataset}_${dspf}${ds}.h5/mask -l 80 -u 150
python $scriptdir/wmem/reduceblocks.py $datadir/${dataset}_${dspf}${ds}.h5/mask $datadir/${dataset}_${dspf}${ds}.h5/mask_reduced






scriptdir="$HOME/workspace/EM"
export PYTHONPATH=$PYTHONPATH:$scriptdir
datadir="$HOME/oxdata/P01/EM/scratch_wmem_package/ds7_arc"
dataset='M3S1GNUds7'
xe=0.0511; ye=0.0511; ze=0.05;
basepath=$datadir/${dataset}

# get cutout
python $scriptdir/wmem/stack2stack.py \
${basepath}.h5/stack ${basepath}_cutout.h5/data \
-x 250 -X 500 -y 250 -Y 500 -z 100 -Z 300
for dset in maskDS maskMM maskMM_preproc labelMM labelMA; do
python $scriptdir/wmem/stack2stack.py \
${basepath}_${pf}.h5/stack ${basepath}_cutout.h5/${pf} \
-x 250 -X 500 -y 250 -Y 500 -z 100 -Z 300
done
dataset='M3S1GNUds7_cutout'
basepath=$datadir/${dataset}

# seperate sheaths
# python $scriptdir/wmem/separate_sheaths.py \
# $basepath.h5/labelMA $basepath.h5/sheaths -s \
# --maskDS $basepath.h5/maskDS --maskMM $basepath.h5/maskMM -d 5 -w
python $scriptdir/wmem/separate_sheaths.py \
$basepath.h5/labelMA $basepath.h5/sheaths -s \
--maskDS $basepath.h5/maskDS --maskMM $basepath.h5/maskMM_preproc -w

# convert to nifti
for dset in data labelMA labelMM; do
python $scriptdir/wmem/stack2stack.py \
$basepath.h5/${dset} ${basepath}_${dset}.nii.gz
done
for dset in maskDS maskMM maskMM_preproc; do
python $scriptdir/wmem/stack2stack.py \
$basepath.h5/${dset} ${basepath}_${dset}.nii.gz -d 'uint8'
done
for dset in sheaths_simple distance_simple distance_sigmod sheaths; do
python $scriptdir/wmem/stack2stack.py \
$basepath.h5/${dset} ${basepath}_${dset}.nii.gz
done
for dset in wsmask madil01 madil02 madil05; do
python $scriptdir/wmem/stack2stack.py \
$basepath.h5/${dset} ${basepath}_${dset}.nii.gz -d 'uint8'
done


# fixing sw
python $scriptdir/wmem/separate_sheaths.py \
$basepath.h5/labelMA $basepath.h5/sheaths_simple -s \
--maskDS $basepath.h5/maskDS --maskMM $basepath.h5/maskMM -d 5


# python $scriptdir/wmem/separate_sheaths.py \
# $basepath.h5/labelMA $basepath.h5/sheaths_simple -s \
# --maskDS $basepath.h5/maskDS -d 5

python $scriptdir/wmem/separate_sheaths.py \
$basepath.h5/labelMA $basepath.h5/sheaths -s \
--maskWS $basepath.h5/wsmask \
-w --labelMM $basepath.h5/sheaths_simple

dset='distance_sigmod'; python $scriptdir/wmem/stack2stack.py $basepath.h5/$dset ${basepath}_${dset}.nii.gz
dset='sheaths'; python $scriptdir/wmem/stack2stack.py $basepath.h5/$dset ${basepath}_${dset}.nii.gz





python $scriptdir/wmem/separate_sheaths.py \
${basepath}_labelMA_final.h5/stack ${basepath}_MM.h5/sheaths -s

python $scriptdir/wmem/separate_sheaths.py \
${basepath}_labelMA_final.h5/stack ${basepath}_MM.h5/sheaths2 -s \
--maskDS ${basepath}_maskDS.h5/stack --maskMM ${basepath}_maskMM_final.h5/stack -d 5 \
--distance ${basepath}_MM.h5/distance

python $scriptdir/wmem/separate_sheaths.py \
${basepath}_labelMA_final.h5/stack ${basepath}_MM.h5/sheaths2 -s \
--maskWS ${basepath}_MM.h5/wsmask --distance ${basepath}_MM.h5/distance

python $scriptdir/wmem/separate_sheaths.py \
${basepath}_labelMA_final.h5/stack ${basepath}_MM.h5/sheaths_sw -s -w \
--maskWS ${basepath}_MM.h5/wsmask --labelMM ${basepath}_labelMM_final.h5/stack


python $scriptdir/wmem/stack2stack.py ${basepath}_MM.h5/madil05 ${basepath}_MM_madil05.nii.gz -d 'uint8'
python $scriptdir/wmem/stack2stack.py ${basepath}_MM.h5/wsmask ${basepath}_MM_wsmask.nii.gz -d 'uint8'
python $scriptdir/wmem/stack2stack.py ${basepath}_MM.h5/distance ${basepath}_MM_distance.nii.gz -x 250 -X 500 -y 250 -Y 500 -z 100 -Z 300
python $scriptdir/wmem/stack2stack.py ${basepath}_MM.h5/distance_sigmod ${basepath}_MM_distance_sigmod.nii.gz -x 250 -X 500 -y 250 -Y 500 -z 100 -Z 300
python $scriptdir/wmem/stack2stack.py ${basepath}_MM.h5/sheaths ${basepath}_MM_sheaths.nii.gz
python $scriptdir/wmem/stack2stack.py ${basepath}_MM.h5/sheaths2 ${basepath}_MM_sheaths2.nii.gz






python $scriptdir/wmem/watershed_ics.py \
$basepath.h5/data $basepath.h5/ws_ics -S -l 200 -u 300 \
--masks NOT $basepath.h5/maskMM AND $basepath.h5/maskDS

python $scriptdir/wmem/stack2stack.py ${basepath}.h5/seeds ${basepath}_seeds.nii.gz
python $scriptdir/wmem/stack2stack.py ${basepath}.h5/ws_ics ${basepath}_ws_ics.nii.gz


import os
from wmem.watershed_ics import watershed_ics
basepath = '/Users/michielk/oxdata/P01/EM/scratch_wmem_package/ds7_arc/M3S1GNUds7_cutout'
h5path_in = os.path.join(basepath + '.h5', 'data')
h5path_mds = os.path.join(basepath + '.h5', 'maskDS')
h5path_mmm = os.path.join(basepath + '.h5', 'maskMM')

h5path_out = os.path.join(basepath + '.h5', 'ws_ics')
watershed_ics(
    h5path_in, h5path_mds, h5path_mmm, h5path_seeds='',
    lower_threshold=200, upper_threshold=300, seed_size=64,
    min_labelsize=None,
    h5path_out=h5path_out, save_steps=True, protective=False
    )

ds_out, ds_sds = watershed_ics(
    h5path_in, h5path_mds, h5path_mmm, h5path_seeds='',
    lower_threshold=200, upper_threshold=300, seed_size=64,
    min_labelsize=None,
    h5path_out='', save_steps=True, protective=False
    )

# TODO: mergeblocks.py
# TODO: merge_slicelabels.py documentation
# TODO: merge agglo_from_ modules



# testing groups hdf5
import os
import h5py
import numpy as np
from wmem import utils
basepath = '/Users/michielk/oxdata/P01/EM/scratch_wmem_package/ds7_arc/M3S1GNUds7_cutout'
h5path_in = os.path.join(basepath + '.h5', 'data')
h5file_in, ds_in, elsize, axlab = utils.h5_load(h5path_in)
grp = h5file_in.create_group("bar")
subgrp = grp.create_group("baz")
grp = h5file_in.create_group("data/bar")
dset = h5file_in.create_dataset("data/bar", (100,))
grp2 = f.create_group("/some/long/path")


import os
import h5py
import numpy as np
from skimage.util import view_as_blocks
from scipy.stats import mode as scipy_mode
from wmem import utils
from wmem import reduceblocks

datadir = '/Users/michielk/oxdata/P01/EM/scratch_wmem_package'
inputfile = os.path.join(datadir, 'wmem_ds7.h5', 'mask')
outputfile = os.path.join(datadir, 'wmem_ds7.h5', 'reduced2')
data = reduceblocks.reduceblocks(inputfile, [1,7,7], 'np.amax', h5path_out='')
data = reduceblocks.reduceblocks(inputfile, [1,7,7], 'np.amax', h5path_out=outputfile, protective=True)
data_restored = reduceblocks.reduceblocks(outputfile, [1,7,7], 'expand', h5path_out='', protective=False)

h5path = '/Users/michielk/oxdata/P01/EM/scratch_wmem_package/wmem.h5'
h5file = h5py.File(h5path, 'r')


# python
import os
import sys
sys.path.append(r'/Users/michielk/workspace/EM/wmem')
from wmem import downsample_slices

# uhome = os.path.expanduser('~')
# inputdir = os.path.join(uhome, "/oxdata/P01/EM/scratch_wmem_package/slices")
# outputdir = os.path.join(uhome, "/oxdata/P01/EM/scratch_wmem_package/slices_ds7")
#
inputdir = "/Users/michielk/oxdata/P01/EM/scratch_wmem_package/slices"
outputdir = "/Users/michielk/oxdata/P01/EM/scratch_wmem_package/slices_ds7"
downsample_slices.downsample_slices(inputdir, outputdir, regex='*.tif', ds_factor=4, xyz=(0, 0, 0, 0, 0, 0), use_mpi=False)




from wmem import utils
basepath = '/Users/michielk/oxdata/P01/EM/scratch_wmem_package/ds7_arc/M3S1GNUds7'

h5path_in = os.path.join(basepath + '.h5', 'stack')

import nibabel as nib
dspath = basepath + '_MM_sheaths.nii.gz'
file_in = nib.load(dspath)



prob, es, al = utils.h5_load(h5path_probs,
                             channels=[2],
                             load_data=True)

