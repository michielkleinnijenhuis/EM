scriptdir="${HOME}/workspace/EM"
datadir="/Users/michielk/M3_S1_GNU_NP/test"
dset_name="m000"
x=2000;X=3000;y=2000;Y=3000;z=30;Z=460;

python $scriptdir/mesh/EM_separate_sheaths.py \
$datadir $dset_name \
--supervoxels '_prediction_NPminimal_ws_l0.95_u1.00_s064_PA_str2_iter5_parallelh5_thr0.1_alg1' 'stack' \
-x ${x} -X ${X} -y ${y} -Y ${Y} -z ${z} -Z ${Z}

python $scriptdir/mesh/EM_separate_sheaths.py \
$datadir $dset_name \
--supervoxels '_prediction_NPminimal_ws_l0.95_u1.00_s064_PA_str2_iter5_parallelh5_thr0.1_alg1' 'stack' \
-x ${x} -X ${X} -y ${y} -Y ${Y} -z ${z} -Z ${Z} -w -d

# goes up to 20+GB




import os
import sys
from argparse import ArgumentParser

import h5py
import numpy as np
from skimage.morphology import watershed
from scipy.ndimage.morphology import grey_dilation, binary_erosion
from scipy.special import expit
from scipy.ndimage import distance_transform_edt
from skimage.measure import label
from scipy.ndimage.morphology import binary_closing, binary_fill_holes, binary_dilation, grey_dilation
from skimage.measure import regionprops
from skimage.segmentation import relabel_sequential
from skimage.morphology import watershed, remove_small_objects

datadir="/Users/michielk/M3_S1_GNU_NP/test"
dset_name='m000_02000-03000_02000-03000_00030-00460'
elsize = [0.05, 0.0073, 0.0073]

filename = os.path.join(datadir, dset_name + '_maskMA.h5')
f = h5py.File(filename, 'r')
distance = distance_transform_edt(~f['stack'][:,:,:], sampling=np.absolute(elsize))
# 20GB
maskMA = f['stack'][:,:,:]
distance = distance_transform_edt(~maskMA, sampling=np.absolute(elsize))
f.close()

pf = '_ws_l0.95_u1.00_s064_labelMA'
pf = '_ws_l0.99_u1.00_s005_labelMA'
filename = os.path.join(datadir, dset_name + pf + '.h5')
f = h5py.File(filename, 'r')
filename = os.path.join(datadir, dset_name + '_maskMA.h5')
m = h5py.File(filename, 'r')
maskMA = np.array(m['stack'][:,:,:], dtype='bool')
filename = os.path.join(datadir, dset_name + pf + 'only.h5')
g = h5py.File(filename, 'w')
outds = g.create_dataset('stack', f['stack'].shape,
                         chunks=f['stack'].chunks,
                         dtype=f['stack'].dtype,
                         compression="gzip")
maskedMA = np.zeros_like(f['stack'][:,:,:])
maskedMA[maskMA] = f['stack'][:,:,:][maskMA]
g['stack'][:,:,:] = maskedMA
f.close()
m.close()
g.close()




pf = '_ws_l0.99_u1.00_s005_labelMAonly'
filename = os.path.join(datadir, dset_name + pf + '.h5')
f = h5py.File(filename, 'r')

binim = f['stack'][:,:,:] != 0
# does this bridge seperate MA's? YES, and eats from boundary
# binim = binary_closing(binim, iterations=10)
holes = label(~binim)

filename = os.path.join(datadir, dset_name + '_binim.h5')
g = h5py.File(filename, 'w')
outds = g.create_dataset('stack', f['stack'].shape,
                         chunks=f['stack'].chunks,
                         dtype=f['stack'].dtype,
                         compression="gzip")
g['stack'][:,:,:] = binim
g.close()

labelCount = np.bincount(holes.ravel())
background = np.argmax(labelCount)
holes[holes == background] = 0

labels_dil = grey_dilation(f['stack'][:,:,:], size=(3,3,3))

rp = regionprops(holes, labels_dil, cache=True)
mi = {prop.label: prop.max_intensity for prop in rp}
# don't understand why I need the +1 here!
fw = [mi[key] + 1 if key in mi.keys() else 0 for key in range(0, np.amax(holes) + 1)]
fw = np.array(fw)

holes_remapped = fw[holes]

filename = os.path.join(datadir, dset_name + '_test2.h5')
g = h5py.File(filename, 'w')
outds = g.create_dataset('stack', f['stack'].shape,
                         chunks=f['stack'].chunks,
                         dtype=f['stack'].dtype,
                         compression="gzip")
g['stack'][:,:,:] = holes_remapped
g.close()




binim = labels==0
binim = binary_closing(binim, iterations=10)
MA[binim] = l





### seeds to seed_size
filename = os.path.join(datadir, dset_name + '_maskICS_l0.95.h5')
f = h5py.File(filename, 'r')
seeds = label(f['stack'][:,:,:])
remove_small_objects(seeds, min_size=10000, in_place=True)
seeds = relabel_sequential(seeds)[0]
rp = regionprops(seeds, cache=True)
mi = {prop.label: prop.area for prop in rp}
fw = [mi[key] if key in mi.keys() else 0 for key in range(0, np.amax(seeds) + 1)]
fw = np.array(fw)
seeds_remapped = fw[seeds]

filename = os.path.join(datadir, dset_name + '_test3.h5')
g = h5py.File(filename, 'w')
outds = g.create_dataset('stack', f['stack'].shape,
                         chunks=f['stack'].chunks,
                         compression="gzip")
g['stack'][:,:,:] = seeds_remapped
g.close()

f.close()

### label slice-wise
filename = os.path.join(datadir, dset_name + '_maskMM.h5')
f = h5py.File(filename, 'r')
MM = f['stack'][:,:,:]
maxlabel = 0
seeds = np.zeros_like(MM, dtype='uint16')
for i in range(0, MM.shape[0]):
    seeds[i,:,:], num = label(~MM[i,:,:], return_num=True)
    seeds[i,:,:] += maxlabel
    maxlabel += num
    print(i)

seeds[MM == 1] = 0
filename = os.path.join(datadir, dset_name + '_test1.h5')
g = h5py.File(filename, 'w')
outds = g.create_dataset('stack', f['stack'].shape,
                         chunks=f['stack'].chunks,
                         compression="gzip")
g['stack'][:,:,:] = seeds
g.close()
rp = regionprops(seeds, cache=True)
mi = {prop.label: prop.area for prop in rp}
# remap labels to area size
fw = [mi[key] if key in mi.keys() else 0 for key in range(0, np.amax(seeds) + 1)]
fw = np.array(fw)
seeds_remapped = fw[seeds]
filename = os.path.join(datadir, dset_name + '_test2.h5')
g = h5py.File(filename, 'w')
outds = g.create_dataset('stack', f['stack'].shape,
                         chunks=f['stack'].chunks,
                         compression="gzip")
g['stack'][:,:,:] = seeds_remapped
g.close()
# filter small and large labels
fw = np.zeros(np.amax(seeds) + 1, dtype='int32')
for k, v in mi.iteritems():
    if ((mi[k] > 200) & (mi[k] < 20000)):
        fw[k] = k
seeds_filtered = fw[seeds]
filename = os.path.join(datadir, dset_name + '_test3.h5')
g = h5py.File(filename, 'w')
outds = g.create_dataset('stack', f['stack'].shape,
                         chunks=f['stack'].chunks,
                         compression="gzip")
g['stack'][:,:,:] = seeds_filtered
g.close()
# label filtered mask
labels = label(seeds_filtered != 0)
filename = os.path.join(datadir, dset_name + '_test4.h5')
g = h5py.File(filename, 'w')
outds = g.create_dataset('stack', f['stack'].shape,
                         chunks=f['stack'].chunks,
                         compression="gzip")
g['stack'][:,:,:] = labels
g.close()
# select labels with ...
eccentricity
equivalent_diameter
euler_number





source activate scikit-image-devel_0.13
scriptdir="${HOME}/workspace/EM"
datadir="/Users/michielk/M3_S1_GNU_NP/test"
dset_name="m000_02000-03000_02000-03000_00030-00460"
datastem="m000_02000-03000_02000-03000_00030-00460"

python $scriptdir/convert/prob2mask.py \
$datadir $datastem -p "" 'stack' -l 0 -u 10000000 -o '_maskDS'  # 2GB
python $scriptdir/convert/prob2mask.py \
$datadir $datastem -p '_probs0_eed2' 'stack' -l 0.2 -s 100000 -d 1 -o '_maskMM'  #6GB
python $scriptdir/convert/prob2mask.py \
$datadir $datastem -p '_probs0_eed2' 'stack' -l 0.02 -o '_maskMM-0.02' # 4GB
python $scriptdir/convert/prob2mask.py \
$datadir $datastem -p '_probs' 'volume/predictions' -c 3 -l 0.3 -o '_maskMB'  # 3GB

python $scriptdir/supervoxels/conn_comp.py $datadir $datastem --maskMM '_maskMM-0.02' 'stack'  # 10GB

pf='_labelMA'
python $scriptdir/convert/EM_stack2stack.py \
$datadir/${datastem}${pf}.h5 $datadir/${datastem}${pf}.nii.gz \
-e 0.0073 0.0073 0.05 -i 'zyx' -l 'xyz'  # 2GB

python $scriptdir/supervoxels/delete_labels.py \
$datadir ${datastem} -o '_manedit' -d `grep ${datastem} $editsfile | awk '{$1 = ""; print $0;}'`  # 2GB

l=0.95; u=1.00; s=064;
python $scriptdir/supervoxels/EM_watershed.py \
${datadir} ${datastem} -c 1 -l ${l} -u ${u} -s ${s}  # 20GB

l=0.95; u=1.00; s=064;
svoxpf="_ws_l${l}_u${u}_s${s}"
python $scriptdir/supervoxels/agglo_from_labelmask.py \
${datadir} $datastem \
-l '_labelMAmanedit' 'stack' -s ${svoxpf} 'stack' -m '_maskMA'  # 6GB

# python $scriptdir/supervoxels/fill_holes.py \
# $datadir $dset_name \
# -l '_ws_l0.99_u1.00_s005_labelMA' 'stack' \
# -m '_maskMA' 'stack' \
# --maskMM '_maskMM' 'stack' \
# --maskMA '_maskMA' 'stack' \
# -o '_filled1' -p '_holes1' -w 1  # -w 2: 4GB; -w 1 10GB
# ## TODO: need to look at this function! outputs are wrong

l=0.99; u=1.00; s=005;
svoxpf="_ws_l${l}_u${u}_s${s}"
python $scriptdir/mesh/EM_separate_sheaths.py\
 $datadir $datastem\
 -l "_labelMA${svoxpf}_manedit" 'stack'\
 --maskMM '_maskMM' 'stack' --maskDS '_maskDS' 'stack'\
 -w -d  # 20GB / 50GB





l=0.99; u=1.00; s=005;
svoxpf="_ws_l${l}_u${u}_s${s}"
python $scriptdir/mesh/EM_split_sheaths.py\
 $datadir $datastem\
 -l "_labelMA${svoxpf}_manedit" 'stack'\
 --maskMM '_maskMM' 'stack' --maskDS '_maskDS' 'stack'

l=0.99; u=1.00; s=005;
svoxpf="_ws_l${l}_u${u}_s${s}"
python $scriptdir/mesh/EM_split_sheaths.py\
 $datadir $datastem\
 -l "_distance" 'stack'\
 --maskMM '_maskMM' 'stack' --maskDS '_maskDS' 'stack'


python $scriptdir/convert/prob2mask.py\
 $datadir $datastem -p '_probs' 'volume/predictions' -c 3 -l 0.9 -o '_maskMB_l0.9'  # 3GB

python $scriptdir/convert/prob2mask.py\
 $datadir $datastem -p '_probs1_eed2' 'stack' -l 0.9 -o '_maskICS_l0.90'  # 3GB
python $scriptdir/convert/prob2mask.py\
 $datadir $datastem -p '_probs1_eed2' 'stack' -l 0.95 -o '_maskICS_l0.95'  # 3GB
python $scriptdir/convert/prob2mask.py\
 $datadir $datastem -p '_probs1_eed2' 'stack' -l 0.99 -o '_maskICS_l0.99'  # 3GB

export template='array' additions='conda' CONDA_ENV="scikit-image-devel_0.13"
export njobs=3 nodes=1 tasks=10 memcpu=6000 wtime="00:10:00" q="d"
export jobname="maskICS090"
export cmd="python $scriptdir/convert/prob2mask.py\
 $datadir datastem\
 -p '_probs1_eed2' 'stack' -l 0.90 -o '_maskICS-0.90'"
source $scriptdir/pipelines/template_job_$template.sh


python $scriptdir/supervoxels/conn_comp.py $datadir $datastem --maskMM '_maskMM-0.02' 'stack' -d -o '_testing'


# test agglo_from_label reverse
l=0.99; u=1.00; s=005;
svoxpf="_ws_l${l}_u${u}_s${s}"
python $scriptdir/supervoxels/agglo_from_svox.py\
 $datadir $datastem\
 -l '_labelMA_core2D_labeled' 'stack' -s ${svoxpf} 'stack'\
 -o '_labelMA_core2D_agglo'

l=0.99; u=1.00; s=005;
svoxpf="_ws_l${l}_u${u}_s${s}"
python $scriptdir/supervoxels/agglo_from_svox.py\
 $datadir $datastem\
 -l '_labelMA_core2D_labeled' 'stack' -s ${svoxpf} 'stack'\
 -o '_labelMA_core2D_agglo_simple' -m

python $scriptdir/supervoxels/merge_slicelabels.py\
 $datadir $datastem\
 -l '_labelMA_core2D_labeled' 'stack' --maskMM '_maskMM' 'stack'\
 -m 4 -t 0.01 -s 10000

export deletefile="$datadir/m000_labelMA_core2D_delete.txt"
echo "m000_02000-03000_02000-03000_00030-00460: 343 29479 2982 8038 6973 6108 15008 422763 36743 36890" >> $deletefile
export mergefile="$datadir/m000_labelMA_core2D_merge.txt"
echo "m000_02000-03000_02000-03000_00030-00460: 242 20679" >> $mergefile
export splitfile="$datadir/m000_labelMA_core2D_split.txt"
echo "m000_02000-03000_02000-03000_00030-00460: 308 332 359 24774 273 318 4763 243 21043 24774 221 301" >> $splitfile
# 332 359 280 ssplit and delete 1
# 280 is very messy
# 16540 is complicated

# axons with internodes
332 31181
4763 23325 27768
329 35055
245 (out of block)
380 23341 21043 17876
5504 (out of block)
18151 (very weak myelination)
345 374
249 16540
280 12918
312 29071 34287 26893  # 308: first remove segment?
201 29477
23314 9789
export mergefile="$datadir/m000_labelMA_core2D_merge.txt"
echo "m000_02000-03000_02000-03000_00030-00460: 242 20679 332 31181 4763 23325 4763 27768 329 35055 380 23341 380 21043 380 17876 345 374 249 16540 280 12918 312 29071 312 34287 312 26893 201 29477 23314 9789" >> $mergefile

# a part of 308 needs to merge with 41150, another with (29071 34287 26893 312)
export mergefile="$datadir/m000_labelMA_core2D_merge.txt"
echo "m000_02000-03000_02000-03000_00030-00460: 41150 42279" >> $mergefile  # 42279 was 308

# method for gap-filling
# mark for watershed
41150 (308/42279)
3247
1115









python $scriptdir/supervoxels/delete_labels.py\
 $datadir $datastem\
 -L '_labelMA_core2D_merged_manedit' 'stack'\
 -D 343 29479 2982 8038 6973 6108 15008 422763 36743 36890\
 -M 242 20679 332 31181 4763 23325 4763 27768 329 35055 380 23341 380 21043 380 17876 345 374 249 16540 280 12918 312 29071 312 34287 312 26893 201 29477 23314 9789\
 -S 308 332 359 24774 273 318 4763 243 21043 24774 221 301\
 -o '_proofread' -n -N -i 'xyz' -l 'zyx' -s 10000

# with small components removed after applying the maskMM as well
python $scriptdir/supervoxels/merge_slicelabels.py\
 $datadir $datastem\
 -l '_labelMA_core2D_labeled' 'stack' --maskMM '_maskMM' 'stack'\
 -m 4 -t 0.01 -s 10000 -o '_labelMA_core2D_merged2'



l=0.99; u=1.00; s=005;
svoxpf="_ws_l${l}_u${u}_s${s}"
python $scriptdir/mesh/EM_separate_sheaths.py\
 $datadir $datastem\
 -l '_labelMA_core2D_merged' 'stack'\
 --maskMM '_maskMM' 'stack' --maskDS '_maskDS' 'stack'\
 -w -d


# python $scriptdir/mesh/label2stl.py\
#  $datadir $datastem -L '_labelMA_core2D_merged' -f 'stack' -e 0.05 0.0073 0.0073
python $scriptdir/mesh/label2stl.py\
 $datadir $datastem -L '_labelMA_core2D_merged' -f 'stack' -e 0.05 0.0073 0.0073 -d 1 7 7
blender -b -P $scriptdir/mesh/stl2blender.py -- \
 $datadir/m000_02000-03000_02000-03000_00030-00460/dmcsurf_1-8-8 ${comp} 'MA' -L 'MA'\
 -s 10 0.5 True True True -d 0.0174533 -e 0.01 -r  # -l 100 0.2 0.01 True True True




for comp in MM NN GP UA; do
blender -b -P $scriptdir/mesh/stl2blender.py -- \
$datadir/dmcsurf_PAenforceECS ${comp} -L ${comp} -s 0.5 10 -d 0.2  -e 0.01
done
for comp in ECS; do
blender -b -P $scriptdir/mesh/stl2blender.py -- \
$datadir/dmcsurf ${comp} -L ${comp} -d 0.02
done

    for ob in bpy.data.objects:
        ob.hide = True
