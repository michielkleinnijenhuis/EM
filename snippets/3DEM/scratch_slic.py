### on anaconda3 env scikit-image-devel_p34
source ~/.bashrc
module load hdf5-parallel/1.8.14_mvapich2_intel
scriptdir="$HOME/workspace/EM"
datadir="$DATA/EM/M3/M3_S1_GNU" && cd $datadir
dataset='m000'
# slicvoxels
nvox=500; comp=2; smooth=0.05;  #nvox=500; comp=0.02; smooth=0.01;
qsubfile=$datadir/EM_slic_s${nvox}_c${comp}_o${smooth}.sh
echo '#!/bin/bash' > $qsubfile
echo "#SBATCH --nodes=1" >> $qsubfile
echo "#SBATCH --ntasks-per-node=4" >> $qsubfile
echo "#SBATCH --time=01:00:00" >> $qsubfile
echo "#SBATCH --job-name=EM_slic" >> $qsubfile
echo "export PATH=/home/ndcn-fmrib-water-brain/ndcn0180/anaconda3/bin:\$PATH" >> $qsubfile
echo "source activate scikit-image-devel_p34" >> $qsubfile
xs=100; ys=100; #xs=500; ys=500;
z=200; Z=300;  #z=30; Z=460;
for x in 1000 1500; do
for y in 1000 1500; do
# x=1000; y=1000;
[ $x == 5000 ] && X=5217 || X=$((x+xs))
[ $y == 4000 ] && Y=4460 || Y=$((y+ys))
datastem="${dataset}_`printf %05d ${x}`-`printf %05d ${X}`_`printf %05d ${y}`-`printf %05d ${Y}`_`printf %05d ${z}`-`printf %05d ${Z}`"
echo "python $scriptdir/supervoxels/EM_slicvoxels.py \
${datadir}/${datastem}.h5 \
${datadir}/${datastem}_slic_s`printf %05d ${nvox}`_c`printf %.3f ${comp}`_o`printf %.3f ${smooth}`.h5 \
-f 'stack' -g 'stack' -s ${nvox} -c ${comp} -o ${smooth} -u \
> EM_slic_${datastem} &" >> $qsubfile
done
done
echo "wait" >> $qsubfile
sbatch -p compute $qsubfile





### slicsegmentation local test
import os
import sys
import h5py
import numpy as np
probvoxelcount = 100
f = h5py.File("/Users/michielk/oxdata/P01/EM/M3/M3_S1_GNU/m000_01000-01500_01000-01500_00030-00460_slic_s00500_c0.020_o0.010.h5", 'r')
s = f['stack'][:,:,:]
f.close()
# f = h5py.File("/Users/michielk/oxdata/P01/EM/M3/M3_S1_GNU/m000_01000-01500_01000-01500_00030-00460_MA_sMAkn_sUAkn_ws_filled.h5", 'r')
f = h5py.File("/Users/michielk/oxdata/P01/EM/M3/M3_S1_GNU/m000_01000-01500_01000-01500_00030-00460_PA.h5", 'r')
# f = h5py.File("/Users/michielk/oxdata/P01/EM/M3/M3_S1_GNU/m000_01000-01500_01000-01500_00030-00460_MM_ws_sw.h5", 'r')
p = f['stack'][:,:,:]
f.close()
segmlabel = np.zeros_like(p)
for l in np.unique(p):
    print(l)
    forward_map = np.zeros(np.amax(s) + 1, 'bool')
    m = p == l
    x = np.bincount(s[m]) >= probvoxelcount
    forward_map[0:len(x)] = x
    segments = forward_map[s]
    segmlabel[segments] = l
# f = h5py.File("/Users/michielk/oxdata/P01/EM/M3/M3_S1_GNU/m000_01000-01500_01000-01500_00030-00460_MA_slicsegmentation.h5", 'w')
f = h5py.File("/Users/michielk/oxdata/P01/EM/M3/M3_S1_GNU/m000_01000-01500_01000-01500_00030-00460_PA_slicsegmentation.h5", 'w')
# segmlabel[segmlabel>0] = 1
# f = h5py.File("/Users/michielk/oxdata/P01/EM/M3/M3_S1_GNU/m000_01000-01500_01000-01500_00030-00460_MM1_slicsegmentation.h5", 'w')
dset = f.create_dataset('stack', segmlabel.shape, dtype='int16', compression='gzip')
dset[:] = segmlabel
f.close()





### import using ``mh`` abbreviation which is common:
import mahotas as mh
# Load one of the demo images
im = mh.demos.load('nuclear')
# Automatically compute a threshold
T_otsu = mh.thresholding.otsu(im)
# Label the thresholded image (thresholding is done with numpy operations
seeds,nr_regions = mh.label(im > T_otsu)
# Call seeded watershed to expand the threshold
labeled = mh.cwatershed(im.max() - im, seeds)
# 2Donly?!
mh.segmentation.slic(array, spacer=16, m=1.0, max_iters=128)





### watershed voxels local test
import os
import sys
from scipy.ndimage.filters import gaussian_filter
from skimage.morphology import watershed
import h5py
import numpy as np

datadir = "/Users/michielk/oxdata/P01/EM/M3/M3_S1_GNU/"
xy = [[1000,1500],[1000,1500]]
dset_name = "m000_" + str(xy[0][0]).zfill(5) + "-" + str(xy[0][1]).zfill(5) + "_" + str(xy[1][0]).zfill(5) + "-" + str(xy[1][1]).zfill(5) + "_00200-00300"
f = h5py.File(os.path.join(datadir, dset_name + ".h5"), 'r')
n_segm = f['stack'][:,:,:].size / 500
seeds =
data = gaussian_filter(f['stack'][:,:,:], gsigma)
segments = watershed(-data, seeds)
segments = segments + 1
print("Number of supervoxels: ", np.amax(segments))
sv = h5py.File(os.path.join(datadir, dset_name + "_slic_s00500_c0.020_o0.020.h5"), 'w')
dset = sv.create_dataset('stack', segments.shape, dtype='int64', compression='gzip')
dset[:] = segments
sv['stack'].attrs['element_size_um'] = [0.05,0.0073,0.0073]
sv.close()
