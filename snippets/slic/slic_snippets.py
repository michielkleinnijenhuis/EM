##################
###  SNIPPET 1 ###
##################

### slicvoxels to segmentation

import os
import numpy as np
import h5py
import nibabel as nib
from mpi4py import MPI

datadir = '/Users/michielk/oxdata/P01/EM/M2/I/'
os.chdir(datadir)
x_start=0
x_end=50
y_start=0
y_end=500
z_start=0
z_end=500

fp = h5py.File(os.path.join(datadir, 'training_data0_Probabilities.h5'),'r', driver='mpio', comm=MPI.COMM_WORLD)
fs = h5py.File(os.path.join(datadir, 'training_data0_slicvoxels002.h5'),'r', driver='mpio', comm=MPI.COMM_WORLD)

# create mask
p = np.ravel(fp['stack'][x_start:x_end,y_start:y_end,z_start:z_end,0]) == 1
# get supervoxel labels within mask
s = np.ravel(fs['stack'][x_start:x_end,y_start:y_end,z_start:z_end])
# get unique set of supervoxels overlapping with mask (mask s with p)
u = np.unique(s[p])
# set these supervoxels to 1 in output o
# o = np.zeros(s.size, dtype='uint8')
# for i,lab in enumerate(u):
#     idx = np.nonzero(s==lab)
#     o[idx] = 1
# set these supervoxels to 1 in output o if overlapping with more than a prob=1 voxels
a = 100
o = np.zeros(s.size, dtype='uint8')
for i,lab in enumerate(u):
    idx = np.nonzero(s==lab)
    if np.count_nonzero(np.logical_and(s==lab,p)) > a:
        o[idx] = 1

o = np.reshape(o, fs['stack'][:,:,:].shape)
mat = np.array([[0.01, 0, 0, 0],[0, 0.01, 0, 0],[0, 0, 0.05, 0],[0, 0, 0, 1]])
slicout = nib.Nifti1Image(o.transpose([2,1,0]), mat)
slicout.to_filename(os.path.join(datadir, 'training_data0_slicsegmentation002_a100.nii.gz'))


svoxlabels = np.unique(fs['stack'][x_start:x_end,
                                   y_start:y_end,
                                   z_start:z_end])
fg = h5py.File(os.path.join(datadir, 'test_data_slicsegmentation_svoxbool.h5'), 'w', driver='mpio', comm=MPI.COMM_WORLD)
fg.create_dataset('svoxvec', (fs['stack'][-1,-1,-1],), dtype='bool')



python /Users/michielk/workspace/EM_seg/src/EM_mpi_createh5.py \
-o '/Users/michielk/oxdata/P01/EM/M2/I/training_data0_slicsegmentation002.h5' \
-g 'stack' \
-d 'uint8' \
-x 0 -X 50 -y 0 -Y 500 -z 0 -Z 500

fv = h5py.File(os.path.join(datadir, 'training_data0_slicsegmentation002.h5'), 'r+', driver='mpio', comm=MPI.COMM_WORLD)

svoxvec_idx = np.nonzero(fg['svoxvec'])

for x in range(0,50):
    print(x)
    for y in range(0,500):
        for z in range(0,500):
            if fs['stack'][x,y,z] in svoxvec_idx[0]:
                fv['stack'][x,y,z] = 1









for svox in svoxlabels:
print(svox)
svox_idx = np.nonzero(fs['stack'][x_start:x_end,
                                  y_start:y_end,
                                  z_start:z_end]==svox)
block = fp['stack'][x_start+np.amin(svox_idx[0]):x_start+np.amax(svox_idx[0]) + 1,
                    y_start+np.amin(svox_idx[1]):y_start+np.amax(svox_idx[1]) + 1,
                    z_start+np.amin(svox_idx[2]):z_start+np.amax(svox_idx[2]) + 1,0]
if np.mean(block[(svox_idx[0]-np.amin(svox_idx[0]),
                  svox_idx[1]-np.amin(svox_idx[1]),
                  svox_idx[2]-np.amin(svox_idx[2]))]) > 0.5:
    fg['svoxvec'][svox] = True







##################
###  SNIPPET 2 ###
##################

### slicvoxels to segmentation

import os
import numpy as np
import h5py
import nibabel as nib
from mpi4py import MPI

datadir = '/Users/michielk/oxdata/P01/EM/M2/I/'
os.chdir(datadir)
x_start=0
x_end=4000
y_start=0
y_end=4000
z_start=0
z_end=448

fp = h5py.File(os.path.join(datadir, 'test_data_Probabilities.h5'),'r', driver='mpio', comm=MPI.COMM_WORLD)
fs = h5py.File(os.path.join(datadir, 'test_data_slicvoxels_c002_s1000_div50.h5'),'r', driver='mpio', comm=MPI.COMM_WORLD)

p = fp['stack'][x_start:x_end,y_start:y_end,z_start:z_end,0] == 1
s = fs['stack'][x_start:x_end,y_start:y_end,z_start:z_end]

a = 100
forward_map = np.zeros(np.max(s)+1, 'bool')
x = np.bincount(s[p]) >= a
forward_map[0:len(x)] = x
o = forward_map[s]

mat = np.array([[0.01, 0, 0, 0],[0, 0.01, 0, 0],[0, 0, 0.05, 0],[0, 0, 0, 1]])
slicout = nib.Nifti1Image(o.astype('uint8'), mat)
slicout.to_filename(os.path.join(datadir, 'test_data_slicsegmentation002_a100.nii.gz'))

fd = h5py.File(os.path.join(datadir, 'test_data.h5'),'r', driver='mpio', comm=MPI.COMM_WORLD)
mat = np.array([[0.01, 0, 0, 0],[0, 0.01, 0, 0],[0, 0, 0.05, 0],[0, 0, 0, 1]])
slicout = nib.Nifti1Image(fd['stack'][x_start:x_end,y_start:y_end,z_start:z_end].astype('uint16'), mat)
slicout.to_filename(os.path.join(datadir, 'test_data_slicsegmentation002_a100_data.nii.gz'))




##################
###  SNIPPET 3 ###
##################

### slicvoxels to segmentation

import os
import numpy as np
import h5py
import nibabel as nib
from mpi4py import MPI

datadir = '/Users/michielk/oxdata/P01/EM/M2/I/'
os.chdir(datadir)
x_start=0
x_end=50
y_start=0
y_end=500
z_start=0
z_end=500

fp = h5py.File(os.path.join(datadir, 'training_data0_Probabilities.h5'),'r', driver='mpio', comm=MPI.COMM_WORLD)
fs = h5py.File(os.path.join(datadir, 'training_data0_slicvoxels002.h5'),'r', driver='mpio', comm=MPI.COMM_WORLD)

p = fp['stack'][x_start:x_end,y_start:y_end,z_start:z_end,0] == 1
s = fs['stack'][x_start:x_end,y_start:y_end,z_start:z_end]

m = s.max()
labels = np.unique(s)
labels0 = labels[labels != 0]
forward_map = np.zeros(m + 1, int)
forward_map[labels0] = np.arange(1, 1 + len(labels0))
if not (labels == 0).any():
        labels = np.concatenate(([0], labels))
relabeled = forward_map[s]

forward_map = np.bincount(s[p]) >= 200
relabeled = forward_map[s]

mat = np.array([[0.01, 0, 0, 0],[0, 0.01, 0, 0],[0, 0, 0.05, 0],[0, 0, 0, 1]])
slicout = nib.Nifti1Image(relabeled.transpose([2,1,0]).astype('uint8'), mat)
slicout.to_filename(os.path.join(datadir, 'training_data0_slicsegmentation002_test.nii.gz'))



u = np.unique(s[p])
a = 100
o = np.zeros(s.shape, dtype='uint8')
for i,lab in enumerate(u):
    idx = np.nonzero(s==lab)
    if np.count_nonzero(np.logical_and(s==lab,p)) > a:
        o[idx] = 1

mat = np.array([[0.01, 0, 0, 0],[0, 0.01, 0, 0],[0, 0, 0.05, 0],[0, 0, 0, 1]])
slicout = nib.Nifti1Image(o.transpose([2,1,0]), mat)
slicout.to_filename(os.path.join(datadir, 'training_data0_slicsegmentation002_a100.nii.gz'))




##################
###  SNIPPET 4 ###
##################


# slicvoxels on caribou
ssh -Y ndcn0180@arcus.oerc.ox.ac.uk
cd /data/ndcn-fmrib-water-brain/ndcn0180/EM/M2/I
module load python/2.7
qsub EM_slicvoxels_caribou_submit.sh

import os
import sys
import getopt
sys.path.remove('/system/software/linux-x86_64/lib/python2.7/site-packages/scikit_image-0.9.3-py2.7-linux-x86_64.egg')
from skimage import segmentation, io
import numpy as np

inputdir='/data/ndcn-fmrib-water-brain/ndcn0180/EM/M2/I/tifs/'
outputdir='/data/ndcn-fmrib-water-brain/ndcn0180/EM/M2/I/slic/'
supervoxelsize=500
x_start = 0
x_end = 1000
y_start = 0
y_end = 1000
z_start = 0
z_end = 200

stack=[]
for imno in range(z_start, z_end):
    input_image = os.path.join(inputdir, str(imno).zfill(4) + '.tif')
    original = io.imread(input_image)
    stack.append(original[x_start:x_end,y_start:y_end])

stack = np.array(stack).swapaxes(0,2)

n_segm = stack.size / supervoxelsize
segments = segmentation.slic(stack, n_segments=n_segm, compactness=0.1, sigma=3, spacing=[1, 1, 5], multichannel=False, convert2lab=True, enforce_connectivity=True)
segments = segments + 1
segments = np.array(segments, dtype='uint64')

for slice in range(0, segments.shape[2]):
    output_image = os.path.join(outputdir, str(slice).zfill(4) + '.tif')
    io.imsave(output_image, segments[:,:,slice])

rsync -avz ndcn0180@arcus.oerc.ox.ac.uk:/data/ndcn-fmrib-water-brain/ndcn0180/EM/M2/I/slic/ /Users/michielk/oxdata/P01/EM/M2/I/slic/


#   4 sys.path.insert(1,'/system/software/linux-x86_64/lib/python2.7/site-packages/numpy')
#   5 sys.path.insert(1,'/system/software/linux-x86_64/lib/python2.7/site-packages/ply-3.4-py2.7.egg')
#   6 sys.path.insert(1,'/system/software/linux-x86_64/lib/python2.7/site-packages/distribute-0.6.28-py2.7.egg')
#   7 sys.path.insert(1,'/system/software/linux-x86_64/lib/python2.7/site-packages/matplotlib-1.3.1-py2.7-linux-x86_64.egg')
#   8 sys.path.insert(1,'/system/software/linux-x86_64/lib/python2.7/site-packages/pyparsing-2.0.1-py2.7.egg')
#   9 sys.path.insert(1,'/system/software/linux-x86_64/lib/python2.7/site-packages/tornado-3.2-py2.7-linux-x86_64.egg')
#  10 sys.path.insert(1,'/system/software/linux-x86_64/lib/python2.7/site-packages/backports.ssl_match_hostname-3.4.0.2-py2.7.egg')
#  11 sys.path.insert(1,'/system/software/linux-x86_64/lib/python2.7/site-packages/six-1.5.2-py2.7.egg')
#  12 sys.path.insert(1,'/system/software/linux-x86_64/lib/python2.7/site-packages/pyfits-3.2.2-py2.7-linux-x86_64.egg')
#  13 sys.path.insert(1,'/system/software/linux-x86_64/lib/python2.7/site-packages/HTSeq-0.6.1p1-py2.7-linux-x86_64.egg')
#  14 sys.path.insert(1,'/system/software/linux-x86_64/lib/python2.7/site-packages/pyqi-0.3.2-py2.7.egg')
#  15 sys.path.insert(1,'/system/software/linux-x86_64/lib/python2.7/site-packages/click-3.3-py2.7.egg')
#  16 sys.path.insert(1,'/system/software/linux-x86_64/lib/python2.7/site-packages/scikit_bio-0.2.0-py2.7-linux-x86_64.egg')
#  17 sys.path.insert(1,'/system/software/linux-x86_64/lib/python2.7/site-packages/future-0.13.0-py2.7.egg')
#  18 sys.path.insert(1,'/system/software/linux-x86_64/lib/python2.7/site-packages/tax2tree-1.0_dev-py2.7.egg')
#  19 sys.path.insert(1,'/system/software/linux-x86_64/lib/python2.7/site-packages/pytz-2014.7-py2.7.egg')
#  20 sys.path.insert(1,'/system/software/linux-x86_64/lib/python2.7/site-packages/numexpr-2.1-py2.7-linux-x86_64.egg')
#  21 sys.path.insert(1,'/system/software/linux-x86_64/lib/python2.7/site-packages/Cython-0.21-py2.7-linux-x86_64.egg')
#  22 sys.path.insert(1,'/system/software/linux-x86_64/lib/python2.7/site-packages/openpyxl-2.0.3-py2.7.egg')
#  23 sys.path.insert(1,'/system/software/linux-x86_64/lib/python2.7/site-packages/jdcal-1.0-py2.7.egg')
#  24 sys.path.insert(1,'/system/software/linux-x86_64/lib/python2.7/site-packages/python_gflags-2.0-py2.7.egg')
#  25 sys.path.insert(1,'/system/software/linux-x86_64/lib/python2.7/site-packages/pandas-0.14.1-py2.7-linux-x86_64.egg')
#  26 sys.path.insert(1,'/system/software/linux-x86_64/lib/python2.7/site-packages/python_dateutil-2.2-py2.7.egg')
#  27 sys.path.insert(1,'/system/software/linux-x86_64/lib/python2.7/site-packages/nose-1.3.4-py2.7.egg')
#  28 sys.path.insert(1,'/system/software/linux-x86_64/lib/python2.7/site-packages/patsy-0.3.0-py2.7.egg')
#  29 sys.path.insert(1,'/system/software/linux-x86_64/lib/python2.7/site-packages/statsmodels-0.5.0-py2.7-linux-x86_64.egg')
#  30 sys.path.insert(1,'/system/software/linux-x86_64/lib/python2.7/site-packages/networkx-1.9.1-py2.7.egg')
#  31 sys.path.insert(1,'/system/software/linux-x86_64/lib/python2.7/site-packages/GDAL-1.11.1-py2.7-linux-x86_64.egg')
#  32 sys.path.insert(1,'/system/software/linux-x86_64/lib/python2.7/site-packages/Amara-1.2.0.2-py2.7.egg')
#  33 sys.path.insert(1,'/system/software/linux-x86_64/lib/python2.7/site-packages/4Suite_XML-1.0.2-py2.7-linux-x86_64.egg')
#  34 sys.path.insert(1,'/system/software/linux-x86_64/lib/python2.7/site-packages/rdflib-3.2.0-py2.7.egg')
#  35 sys.path.insert(1,'/system/software/linux-x86_64/lib/python2.7/site-packages/isodate-0.5.0-py2.7.egg')
#  36 sys.path.insert(1,'/system/software/linux-x86_64/lib/python2.7/site-packages/python_igraph-0.7-py2.7-linux-x86_64.egg')
#  37 #sys.path.remove('/system/software/linux-x86_64/lib/python2.7/site-packages/scikit_image-0.9.3-py2.7-linux-x86_64.egg')
#
# sys.path.insert()
# sys.path.insert('/system/software/linux-x86_64/lib/python2.7/site-packages/ply-3.4-py2.7.egg')
# sys.path.insert('/system/software/linux-x86_64/lib/python2.7/site-packages/distribute-0.6.28-py2.7.egg')
# sys.path.insert('/system/software/linux-x86_64/lib/python2.7/site-packages/matplotlib-1.3.1-py2.7-linux-x86_64.egg')
# sys.path.insert('/system/software/linux-x86_64/lib/python2.7/site-packages/pyparsing-2.0.1-py2.7.egg')
# sys.path.insert('/system/software/linux-x86_64/lib/python2.7/site-packages/tornado-3.2-py2.7-linux-x86_64.egg')
# sys.path.insert('/system/software/linux-x86_64/lib/python2.7/site-packages/backports.ssl_match_hostname-3.4.0.2-py2.7.egg')
# sys.path.insert('/system/software/linux-x86_64/lib/python2.7/site-packages/six-1.5.2-py2.7.egg')
# sys.path.insert('/system/software/linux-x86_64/lib/python2.7/site-packages/pyfits-3.2.2-py2.7-linux-x86_64.egg')
# sys.path.insert('/system/software/linux-x86_64/lib/python2.7/site-packages/HTSeq-0.6.1p1-py2.7-linux-x86_64.egg')
# sys.path.insert('/system/software/linux-x86_64/lib/python2.7/site-packages/pyqi-0.3.2-py2.7.egg')
# sys.path.insert('/system/software/linux-x86_64/lib/python2.7/site-packages/click-3.3-py2.7.egg')
# sys.path.insert('/system/software/linux-x86_64/lib/python2.7/site-packages/scikit_bio-0.2.0-py2.7-linux-x86_64.egg')
# sys.path.insert('/system/software/linux-x86_64/lib/python2.7/site-packages/future-0.13.0-py2.7.egg')
# sys.path.insert('/system/software/linux-x86_64/lib/python2.7/site-packages/tax2tree-1.0_dev-py2.7.egg')
# sys.path.insert('/system/software/linux-x86_64/lib/python2.7/site-packages/pytz-2014.7-py2.7.egg')
# sys.path.insert('/system/software/linux-x86_64/lib/python2.7/site-packages/numexpr-2.1-py2.7-linux-x86_64.egg')
# sys.path.insert('/system/software/linux-x86_64/lib/python2.7/site-packages/Cython-0.21-py2.7-linux-x86_64.egg')
# sys.path.insert('/system/software/linux-x86_64/lib/python2.7/site-packages/openpyxl-2.0.3-py2.7.egg')
# sys.path.insert('/system/software/linux-x86_64/lib/python2.7/site-packages/jdcal-1.0-py2.7.egg')
# sys.path.insert('/system/software/linux-x86_64/lib/python2.7/site-packages/python_gflags-2.0-py2.7.egg')
# sys.path.insert('/system/software/linux-x86_64/lib/python2.7/site-packages/pandas-0.14.1-py2.7-linux-x86_64.egg')
# sys.path.insert('/system/software/linux-x86_64/lib/python2.7/site-packages/python_dateutil-2.2-py2.7.egg')
# sys.path.insert('/system/software/linux-x86_64/lib/python2.7/site-packages/nose-1.3.4-py2.7.egg')
# sys.path.insert('/system/software/linux-x86_64/lib/python2.7/site-packages/patsy-0.3.0-py2.7.egg')
# sys.path.insert('/system/software/linux-x86_64/lib/python2.7/site-packages/statsmodels-0.5.0-py2.7-linux-x86_64.egg')
# sys.path.insert('/system/software/linux-x86_64/lib/python2.7/site-packages/networkx-1.9.1-py2.7.egg')
# sys.path.insert('/system/software/linux-x86_64/lib/python2.7/site-packages/GDAL-1.11.1-py2.7-linux-x86_64.egg')
# sys.path.insert('/system/software/linux-x86_64/lib/python2.7/site-packages/Amara-1.2.0.2-py2.7.egg')
# sys.path.insert('/system/software/linux-x86_64/lib/python2.7/site-packages/4Suite_XML-1.0.2-py2.7-linux-x86_64.egg')
# sys.path.insert('/system/software/linux-x86_64/lib/python2.7/site-packages/rdflib-3.2.0-py2.7.egg')
# sys.path.insert('/system/software/linux-x86_64/lib/python2.7/site-packages/isodate-0.5.0-py2.7.egg')
# sys.path.insert('/system/software/linux-x86_64/lib/python2.7/site-packages/python_igraph-0.7-py2.7-linux-x86_64.egg')
#
#
# /system/software/linux-x86_64/python/2.7.8/lib/python2.7/site-packages
# /panfs/pan01/vol035/data/ndcn-fmrib-water-brain/ndcn0180/EM/M2/I
# /home/ndcn-fmrib-water-brain/ndcn0180/.local/lib/python2.7/site-packages/gala-0.2dev-py2.7-linux-x86_64.egg
# /home/ndcn-fmrib-water-brain/ndcn0180/.local/lib/python2.7/site-packages/Pillow-2.6.0-py2.7-linux-x86_64.egg
# /home/ndcn-fmrib-water-brain/ndcn0180/.local/lib/python2.7/site-packages/scikit_image-0.11dev-py2.7-linux-x86_64.egg
# /home/ndcn-fmrib-water-brain/ndcn0180/.local/lib/python2.7/site-packages/h5py-2.4.0b1-py2.7-linux-x86_64.egg
# /home/ndcn-fmrib-water-brain/ndcn0180/.local/lib/python2.7/site-packages/networkx-2.0.dev_20141127130525-py2.7.egg
# /panfs/pan01/system/software/linux-x86_64/python/2.7.8/lib/python2.7/site-packages/decorator-3.4.0-py2.7.egg
# /home/ndcn-fmrib-water-brain/ndcn0180/.local/lib/python2.7/site-packages/viridis-0.3_dev-py2.7.egg
# /home/ndcn-fmrib-water-brain/ndcn0180/.local/lib/python2.7/site-packages/ilastik-1.1.3-py2.7.egg
# /home/ndcn-fmrib-water-brain/ndcn0180/.local/lib/python2.7/site-packages/lazyflow-0.0.0-py2.7.egg
# /home/ndcn-fmrib-water-brain/ndcn0180/.local/lib/python2.7/site-packages/blist-1.3.6-py2.7-linux-x86_64.egg
# /home/ndcn-fmrib-water-brain/ndcn0180/.local/lib/python2.7/site-packages/psutil-2.1.3-py2.7-linux-x86_64.egg
# /home/ndcn-fmrib-water-brain/ndcn0180/.local/lib/python2.7/site-packages/Cython-0.22pre-py2.7-linux-x86_64.egg
# /system/software/linux-x86_64/python/2.7.8/lib/python2.7/site-packages/pycuda-2014.1-py2.7-linux-x86_64.egg
# /system/software/linux-x86_64/python/2.7.8/lib/python2.7/site-packages/decorator-3.4.0-py2.7.egg
# /system/software/linux-x86_64/python/2.7.8/lib/python2.7/site-packages/pytest-2.6.3-py2.7.egg
# /system/software/linux-x86_64/python/2.7.8/lib/python2.7/site-packages/pytools-2014.3.1-py2.7.egg
# /system/software/linux-x86_64/python/2.7.8/lib/python2.7/site-packages/py-1.4.25-py2.7.egg
# /system/software/linux-x86_64/python/2.7.8/lib/python2.7/site-packages/GDAL-1.11.1-py2.7-linux-x86_64.egg
# /home/ndcn-fmrib-water-brain/ndcn0180/workspace/FlyEM/gala
# /system/software/linux-x86_64/python/2.7.8/lib/python27.zip
# /system/software/linux-x86_64/python/2.7.8/lib/python2.7
# /system/software/linux-x86_64/python/2.7.8/lib/python2.7/plat-linux2
# /system/software/linux-x86_64/python/2.7.8/lib/python2.7/lib-tk
# /system/software/linux-x86_64/python/2.7.8/lib/python2.7/lib-old
# /system/software/linux-x86_64/python/2.7.8/lib/python2.7/lib-dynload
# /home/ndcn-fmrib-water-brain/ndcn0180/.local/lib/python2.7/site-packages
# /home/ndcn-fmrib-water-brain/ndcn0180/.local/lib/python2.7/site-packages/PIL
# /system/software/linux-x86_64/python/2.7.8/lib/python2.7/site-packages



##################
###  SNIPPET 5 ###
##################

### DEPRECATED!!!### (use EM_slicsegmentation.py instead)

# slicvoxels to segmentation
import os
import numpy as np
import h5py
import nibabel as nib
from scipy.ndimage.measurements import label
from skimage.morphology import remove_small_objects

datadir = "/Users/michielk/oxdata/P01/EM/M3/M3_S1_GNU/pipeline_test"
pixprob_trainingset = "pixprob_training"
dataset = "segclass_training"

os.chdir(datadir)

fsegm = h5py.File(os.path.join(datadir, dataset + '_probs.h5'),'r')
layer = 0
segm = np.ravel(fsegm['volume/predictions'][:,:,:,layer])
segm = segm > 0.7

fslic = h5py.File(os.path.join(datadir, dataset + '_probs_slicvoxels.h5'),'r')
slic = np.ravel(fslic['stack'][:,:,:])

slicsegm = np.zeros_like(slic, dtype='bool')

svoxlabels = np.unique(slic)
for svox in svoxlabels:
    svox_idx = np.nonzero(slic==svox)
    if np.count_nonzero(segm[svox_idx]) > svox_idx[0].size / 4:
        slicsegm[svox_idx] = True

svoxlabels = np.unique(slic[segm])

for svox in svoxlabels:
    slicsegm[slic==svox] = True

# for svox in svoxlabels:
#     slicsegm = slicsegm & slic==svox
#
# it = np.nditer(slic, flags=['f_index'])
# while not it.finished:
#     slicsegm[it.index] = it[0] in svoxlabels
#     it.iternext()

slicsegm = np.reshape(slicsegm, fslic['stack'][:,:,:].shape)
slse = h5py.File(os.path.join(datadir, 'pixprob_training_probs_slicsegmentation1-4.h5'), 'w')
slse.create_dataset('stack', data=slicsegm)
label_im, nb_labels = label(slicsegm)
# remove_small_objects(label_im, min_size=1000, in_place=True)
slse = h5py.File(os.path.join(datadir, 'pixprob_training_probs_slicsegmentation1-4_labels.h5'), 'w')
slse.create_dataset('stack', data=label_im)

slse.close()
fsegm.close()
fslic.close()

python $scriptdir/EM_stack2stack.py \
"$datadir/${pixprob_trainingset}_probs_slicsegmentation1-4.h5" \
"$datadir/${pixprob_trainingset}_probs_slicsegmentation1-4.nii.gz" \
-e 0.05 0.0073 0.0073 \
-d 'int32' -f 'stack' -g 'stack'
python $scriptdir/EM_stack2stack.py \
"$datadir/${pixprob_trainingset}_probs_slicsegmentation1-4_labels.h5" \
"$datadir/${pixprob_trainingset}_probs_slicsegmentation1-4_labels.nii.gz" \
-e 0.05 0.0073 0.0073 \
-d 'int32' -f 'stack' -g 'stack'
