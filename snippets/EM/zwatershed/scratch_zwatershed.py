conda create -n zwatershed -c conda-forge zwatershed
source activate zwatershed
conda install matplotlib jupyter
jupyter notebook
# python2.7


import numpy as np
import sys
import time
import os
import h5py
import os.path as op
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from multiprocessing import Pool
from itertools import product

from zwatershed import *

from par_funcs import *

%matplotlib inline
sys.path.append('..')
cmap = matplotlib.colors.ListedColormap(np.vstack( ((0, 0, 0), np.random.rand(1e6, 3))) )
V = 20

# -------------------------------- parameters ---------------------------------------
pred_file = '/Users/michielk/oxdata/P01/EM/M3/M3_S1_GNU/M3S1GNU/test/M3S1GNU_06950-08050_05950-07050_00030-00460_probs1_eed2.h5'
out_folder = '/Users/michielk/oxdata/P01/EM/M3/M3_S1_GNU/M3S1GNU/test/'
outname = out_folder + 'out.h5'
NUM_WORKERS = 4
MAX_LEN = 100

partition_data = partition_subvols(pred_file,out_folder,max_len=MAX_LEN)



import os
import h5py
import numpy as np

datadir = '/Users/michielk/oxdata/P01/EM/M3/M3_S1_GNU/M3S1GNU/test'
dset_name = "M3S1GNU_06950-08050_05950-07050_00030-00460"
pf="_probs"
labelvolume = [pf, 'volume/predictions']

fname = dset_name + labelvolume[0] + '.h5'
fpath = os.path.join(datadir, fname)
f = h5py.File(fpath, 'r')

gname = dset_name + pf +'_permuted' + '.h5'
gpath = os.path.join(datadir, gname)
g = h5py.File(gpath, 'w')

dims = np.array(f[labelvolume[1]].shape)

outds = g.create_dataset('main', dims[::-1], compression='gzip')

outds[:, :, :, :] = np.transpose(f[labelvolume[1]])
g.close()
f.close()


scriptdir="${HOME}/workspace/EM"
datadir='/Users/michielk/oxdata/P01/EM/M3/M3_S1_GNU/M3S1GNU/test/out'
xe=0.0511; ye=0.0511; ze=0.05;
datastem='M3S1GNU'
python $scriptdir/convert/EM_stack2stack.py \
$datadir/out.h5 \
$datadir/out_seg.nii \
-f seg -e $xe $ye $ze -i 'zyx' -l 'xyz' -d uint32 &
