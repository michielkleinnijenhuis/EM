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

python $scriptdir/wmem/stack2stack.py \
$datadir/${dataset_ds}.h5/data \
$datadir/${dataset_ds}_tmp.nii.gz -a .tif

python $scriptdir/wmem/stack2stack.py \
$datadir/${dataset_ds}_labels.h5/labelMA_core2D_prediction \
$datadir/${dataset_ds}_labels_labelMA_core2D_prediction.nii.gz -a .tif -d uint16

cd ~/workspace/dojo
./dojo.py
# ./dojo.py /Users/michielk/oxdata/P01/EM/Myrf_01/SET-B/B-NT-S10-2f_ROI_00/dojo





datadir = '/Users/michielk/oxdata/P01/EM/Myrf_01/SET-B/B-NT-S10-2f_ROI_00';
invol = 'B-NT-S10-2f_ROI_00ds7_labels';
infield = '/labelMA_core2D';

stackinfo = h5info([datadir filesep invol '.h5'], infield)

imshow()





from __future__ import print_function

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
from wmem import utils

datadir = "/Users/michielk/oxdata/P01/EM/Myrf_01/SET-B/B-NT-S10-2f_ROI_00"

fname = 'B-NT-S10-2f_ROI_00ds7'
dset0 = 'data'
h5path_in = os.path.join(datadir, '{}.h5/{}'.format(fname, dset0))
h5file_in, ds_in, elsize, axlab = utils.h5_load(h5path_in)

fname = 'B-NT-S10-2f_ROI_00ds7_labels'
dset0 = 'labelMA_core2D_pred'
h5path_lb = os.path.join(datadir, '{}.h5/{}'.format(fname, dset0))
h5file_lb, ds_lb, elsize, axlab = utils.h5_load(h5path_lb)

V = 0
cmap = matplotlib.colors.ListedColormap(np.vstack( ((0, 0, 0), np.random.rand(1e6, 3))) )

fig, axs = plt.subplots(1, 2)
ax = axs[0]
ax.imshow(ds_in[V,:,:], cmap=plt.get_cmap('Greys'))
ax.set_title('orig')

ax = axs[1]
ax.imshow(ds_lb[V,:,:], cmap=cmap)
ax.set_title('labels')

co = plt.ginput(n=-1, timeout=0, show_clicks=True)

print("clicked", co)

plt.show()

h5file_in.close()
h5file_lb.close()
