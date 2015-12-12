# local:

scriptdir=/Users/michielk/workspace/EM_seg/src
oddir=/Users/michielk/oxdata/originaldata/P01/EM/M3/17Feb15/montage/Montage_
datadir=/Users/michielk/oxdata/P01/EM/M3/M3_S1_SPL

for montage in 000 001 002 003; do

mdir=$datadir/tifs_m$montage && mkdir -p $mdir

sed "s?INPUTDIR?$oddir$montage?;\
    s?OUTPUTDIR?$mdir?g" \
    $scriptdir/EM_convert2tif.py \
    > $datadir/EM_convert2tif_$montage.py
ImageJ --headless $datadir/EM_convert2tif_$montage.py

mpiexec -n 5 python $scriptdir/EM_convert2stack_blocks.py \
-i $mdir \
-o $datadir/test_data_m$montage.h5 \
-f 'stack' \
-x 0 -X 4000 -y 0 -Y 4000 -z 0 -Z 200

rm -rf $mdir

done








import os
datadir = '/Users/michielk/oxdata/P01/EM/M3/M3_S1_SPL'
os.chdir(datadir)

import h5py
fd = h5py.File('test_data_m000.h5','r')
# fg = h5py.File('training_data_m000.h5','w')
# fg.create_dataset('stack', (500,500,60), chunks=(20,20,20), dtype='uint16')
# fg['stack'][0:500,0:500,0:60] = fd['stack'][0:500,0:500,0:60]
# fg.close()
fd.close()

import nibabel as nib
import numpy as np
mat = np.array([[0.0072, 0, 0, 0],[0, 0.0072, 0, 0],[0, 0, 0.050, 0],[0, 0, 0, 1]])

out = nib.Nifti1Image(fd['stack'][0:500,0:500,0:60], mat)
output_image = 'training_data_m000.nii.gz'
out.to_filename(os.path.join(datadir, output_image))

out = nib.Nifti1Image(fd['stack'][0:1000,0:1000,0:200], mat)
output_image = 'tmp_m000.nii.gz'
out.to_filename(os.path.join(datadir, output_image))





mpiexec -n 5 python $scriptdir/EM_convert2stack_blocks.py \
-i $mdir \
-o $datadir/tmp_data_m$montage.h5 \
-f 'stack' \
-x 0 -X 4000 -y 0 -Y 4000 -z 0 -Z 200

import h5py
import nibabel as nib
import numpy as np
import os

datadir = '/Users/michielk/oxdata/P01/EM/M3/M3_S1_SPL'
os.chdir(datadir)

fd = h5py.File('tmp_data_m000.h5','r')
mat = np.array([[0.0072, 0, 0, 0],[0, 0.0072, 0, 0],[0, 0, 0.050, 0],[0, 0, 0, 1]])
out = nib.Nifti1Image(fd['stack'][0:1000,0:1000,0:200], mat)
output_image = 'tmp_data_m000.nii.gz'
out.to_filename(os.path.join(datadir, output_image))
fd.close()
