from PIL import Image
from os import path
import numpy as np
import nibabel as nib

datadir = '/Users/michielk/oxdata/P01/EM/M2/J/orig'
outputdir = '/Users/michielk/oxdata/P01/EM/M2/J/crop0100'
filebase = 'J 3Oct14_3VBSED_slice_'
filebase = ''

for imno in range(950, 1000):
    input_image = filebase + str(imno).zfill(4) + '.tif'
    original = Image.open(path.join(datadir, input_image))
    cropped = original.crop((1000, 1000, 1500, 1500))
    output_image = str(imno).zfill(4) + '.tif'
    cropped.save(path.join(outputdir, output_image))
    img = nib.Nifti1Image(np.array(cropped), np.eye(4))
    output_image = str(imno).zfill(4) + '.nii.gz'
    img.to_filename(path.join(outputdir, output_image))
