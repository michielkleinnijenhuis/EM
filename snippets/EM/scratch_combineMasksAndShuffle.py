bpy.context.object.active_material.type = 'SURFACE'
bpy.context.object.active_material.texture_slots[0].texture_coords = 'UV'
bpy.context.object.active_material.texture_slots[0].uv_layer = "UVMap"


bpy.context.object.active_material.type = 'VOLUME'
bpy.context.object.active_material.texture_slots[0].texture_coords = 'ORCO'

bpy.context.object.active_material.texture_slots[0].mapping_y = 'Y'
bpy.context.object.active_material.texture_slots[0].mapping_x = 'Z'
bpy.context.object.active_material.texture_slots[0].mapping_z = 'X'

bpy.context.object.active_material.texture_slots[0].scale[1] = 0.5
bpy.context.object.active_material.texture_slots[0].scale[1] = 0.5
bpy.context.object.active_material.texture_slots[0].scale[2] = 0.5





# fslmaths M3S1GNUds7_maskMM_final.nii.gz -mul 1 \
# -add M3S1GNUds7_maskMA_final.nii.gz -mul 2 \
# -add M3S1GNUds7_maskUA_final.nii.gz -mul 3 \
# -add M3S1GNUds7_maskGL_final.nii.gz -mul 4 \
# M3S1GNUds7_maskXX_final.nii.gz


# combine masks
import os
import h5py
import numpy as np
from skimage.measure import regionprops, label

datadir = "/Users/michielk/oxdata/P01/EM/M3/M3_S1_GNU/M3S1GNU/ds7_arc"
dset_name = 'M3S1GNUds7'

labelvolume = ['_maskMM_final', 'stack']
f = h5py.File(os.path.join(datadir, dset_name + labelvolume[0] + '.h5'), 'r')
MM = f[labelvolume[1]][:, :, :].astype('bool')
f.close()

labelvolume = ['_maskMA_final', 'stack']
f = h5py.File(os.path.join(datadir, dset_name + labelvolume[0] + '.h5'), 'r')
MA = f[labelvolume[1]][:, :, :].astype('bool')
f.close()

labelvolume = ['_maskUA_final', 'stack']
f = h5py.File(os.path.join(datadir, dset_name + labelvolume[0] + '.h5'), 'r')
UA = f[labelvolume[1]][:, :, :].astype('bool')
f.close()

labelvolume = ['_maskGL_final', 'stack']
f = h5py.File(os.path.join(datadir, dset_name + labelvolume[0] + '.h5'), 'r')
GL = f[labelvolume[1]][:, :, :].astype('bool')
f.close()

MA = MA * 2
UA = UA * 3
GL = GL * 4
XX = MM + MA + UA + GL

gname = dset_name + '_maskXX' + '.h5'
gpath = os.path.join(datadir, gname)
gfile = h5py.File(gpath, 'w')
outds = gfile.create_dataset('stack', XX.shape,
                             dtype='uint32', compression='gzip')

outds[:,:,:] = XX
gfile.close()


xe=0.0511; ye=0.0511; ze=0.05;
datastem = 'M3S1GNUds7'
pf=_maskXX
python $scriptdir/convert/EM_stack2stack.py \
$datadir/"${datastem}${pf}.h5" \
$datadir/"${datastem}${pf}.nii" \
-e $xe $ye $ze -i 'zyx' -l 'xyz' &



# shuffle labels
import os
import h5py
import numpy as np
from skimage.measure import regionprops, label
from random import shuffle
import glob

dset_name = 'M3S1GNUds7'

datadir = "/Users/michielk/oxdata/P01/EM/M3/M3_S1_GNU/M3S1GNU/ds7_arc"
pf = '_labelALL_final'

datadir = "/Users/michielk/oxdata/P01/EM/M3/M3_S1_GNU/M3S1GNU/ds7_arc/pred_new"
pf = '_labelUA'

datadir = "/Users/michielk/oxdata/P01/EM/M3/M3_S1_GNU/M3S1GNU/ds7_arc/viz"
pf = '_labelMA_t0.1_ws_l0.99_u1.00_s010'
pf = '_ws_l0.99_u1.00_s010'
pf = '_labelMM_final'
pf = '_labelUA_final'

datadir = '/Users/michielk/oxdox/papers/abstracts/ISMRM2017/Eposter/anims/m000'
dset_name = 'm000_01000-01500_01000-01500_00030-00460'
pf = '_ws'
pf = '_PA'
pf = '_ws_l0.99_u1.00_s010'

# datadir = "/Users/michielk/oxdata/P01/EM/M3/M3_S1_GNU/M3S1GNU/ds7_arc/preprocessing"
datadir = '/Users/michielk/oxdox/papers/abstracts/ISMRM2017/Eposter/anims/M3S1GNUvols/data'
pf = '_labelMA_core2D'
pf = '_labelMA_core2D_fw_3Dlabeled'
pf = '_labelMA_core2D_fw_3Diter3_closed'
pf = '_labelMA_final'

labelvolume = [pf, 'stack']
f = h5py.File(os.path.join(datadir, dset_name + labelvolume[0] + '.h5'), 'r')
MM = f[labelvolume[1]][:, :, :]
f.close()

# for m000_..._PA.h5
# from skimage.segmentation import relabel_sequential
# MM = relabel_sequential(MM)[0]

ulabels = np.unique(MM)
maxlabel = np.amax(ulabels)
fw = np.array([l if l in ulabels else 0 for l in range(0, maxlabel + 1)])
mask = fw>0
fw_nz = fw[mask]
shuffle(fw_nz)
fw[mask] = fw_nz

gname = dset_name + pf +'_shuffled' + '.h5'
gpath = os.path.join(datadir, gname)
gfile = h5py.File(gpath, 'w')
outds = gfile.create_dataset('stack', MM.shape,
                             dtype='uint32', compression='gzip')

outds[:,:,:] = fw[MM]
gfile.close()


## copy elsize and labels
elsize = [0.05, 0.0511, 0.0511]
axislabels = 'zyx'
field = 'stack'

infiles = glob.glob(os.path.join(datadir, "{}{}*.h5".format(dset_name, pf)))
for fname in infiles:
    try:
        f = h5py.File(fname, 'a')
        f[field].attrs['element_size_um'] = elsize
        for i, l in enumerate(axislabels):
            f[field].dims[i].label = l
        f.close()
        print("%s done" % fname)
    except:
        print(fname)


## convert to nii
xe=0.0511; ye=0.0511; ze=0.05;
datastem='M3S1GNUds7'
pf=_labelALL_shuffled
pf='_labelMA_t0.1_ws_l0.99_u1.00_s010_shuffled'
pf='_ws_l0.99_u1.00_s010_shuffled'
pf='_labelMM_final_shuffled'
pf='_labelUA_final_shuffled'
python $scriptdir/convert/EM_stack2stack.py \
$datadir/"${datastem}${pf}.h5" \
$datadir/"${datastem}${pf}.nii" \
-e $xe $ye $ze -i 'zyx' -l 'xyz' &

xe=0.0073; ye=0.0073; ze=0.05;
datastem='M3S1GNU_06950-08050_05950-07050_00030-00460'
pf='_probs0_eed2'
python $scriptdir/convert/EM_stack2stack.py \
$datadir/"${datastem}${pf}.h5" \
$datadir/"${datastem}${pf}.nii" \
-e $xe $ye $ze -i 'zyx' -l 'xyz' &

xe=0.0073; ye=0.0073; ze=0.05;
datastem='m000_01000-01500_01000-01500_00030-00460'
pf=''
python $scriptdir/convert/EM_stack2stack.py \
$datadir/"${datastem}${pf}.h5" \
$datadir/"${datastem}${pf}.nii" \
-e $xe $ye $ze -i 'zyx' -l 'xyz' -u &
pf='_probs0_eed2'
pf='_probs2_eed2'
pf='_ws_shuffled'
pf='_PA_shuffled'
pf='_ws_l0.99_u1.00_s010_shuffled'
python $scriptdir/convert/EM_stack2stack.py \
$datadir/"${datastem}${pf}.h5" \
$datadir/"${datastem}${pf}.nii" \
-e $xe $ye $ze -i 'zyx' -l 'xyz' &



## slicewise shuffle
from skimage.segmentation import relabel_sequential

datadir = "/Users/michielk/oxdata/P01/EM/M3/M3_S1_GNU/M3S1GNU/ds7_arc/preprocessing"
pf = '_labelMA_2D'

labelvolume = [pf, 'stack']
f = h5py.File(os.path.join(datadir, dset_name + labelvolume[0] + '.h5'), 'r')
MM = f[labelvolume[1]][:, :, :]
f.close()

MMnew = []
for i, slc in enumerate(MM):
    slc = relabel_sequential(slc)[0]
    ulabels = np.unique(slc)
    maxlabel = np.amax(ulabels)
    fw = np.array([l if l in ulabels else 0 for l in range(0, maxlabel + 1)])
    mask = fw>0
    fw_nz = fw[mask]
    shuffle(fw_nz)
    fw[mask] = fw_nz
    MMnew.append(fw[slc])

MMnew = np.array(MMnew)

gname = dset_name + pf +'_shuffled' + '.h5'
gpath = os.path.join(datadir, gname)
gfile = h5py.File(gpath, 'w')
outds = gfile.create_dataset('stack', MMnew.shape,
                             dtype='uint32', compression='gzip')

outds[:,:,:] = MMnew
gfile.close()
