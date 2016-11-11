### get all the raw datafiles and convert to nifti's ###
#rsync -avz ndcn0180@arcus.arc.ox.ac.uk:/data/ndcn-fmrib-water-brain/ndcn0180/EM/M3/M3_S1_GNU/testblock/m000_?????-?????_?????-?????_?????-?????.h5 /Users/michielk/oxdata/P01/EM/M3/M3_S1_GNU/
for f in `ls m000_?????-?????_?????-?????_?????-??460.h5`; do
mv ${f} ${f/460.h5/430.h5}
done
for f in `ls m000_?????-?????_?????-?????_?????-?????.h5`; do
python $scriptdir/convert/EM_stack2stack.py ${f} ${f/.h5/.nii.gz} -i 'zyx' -l 'xyz' -e -0.0073 -0.0073 0.05 -u
done
### get the MA segmentation ###
pf='_probs_ws_MA'
rsync -avz ndcn0180@arcus.arc.ox.ac.uk:/data/ndcn-fmrib-water-brain/ndcn0180/EM/M3/M3_S1_GNU/testblock/m000_01000-02000_01000-02000_00000-00430${pf}.h5 /Users/michielk/oxdata/P01/EM/M3/M3_S1_GNU/
python $scriptdir/convert/EM_stack2stack.py \
"${datadir}/${dataset}${pf}.h5" \
"${datadir}/${dataset}${pf}.nii.gz" -i 'zyx' -l 'xyz' -e -0.0073 -0.0073 0.05

### convert nii.gz segmentation to .h5 ###
import os
import numpy as np
import nibabel as nib
import h5py

def loadh5(datadir, dname, fieldname='stack'):
    """"""
    f = h5py.File(os.path.join(datadir, dname), 'r')
    if len(f[fieldname].shape) == 2:
        stack = f[fieldname][:,:]
    if len(f[fieldname].shape) == 3:
        stack = f[fieldname][:,:,:]
    if len(f[fieldname].shape) == 4:
        stack = f[fieldname][:,:,:,:]
    if 'element_size_um' in f[fieldname].attrs.keys():
        element_size_um = f[fieldname].attrs['element_size_um']
    else:
        element_size_um = None
    f.close()
    return stack, element_size_um

def writeh5(stack, datadir, fp_out, fieldname='stack', dtype='uint16', element_size_um=None):
    """"""
    g = h5py.File(os.path.join(datadir, fp_out), 'w')
    g.create_dataset(fieldname, stack.shape, dtype=dtype, compression="gzip")
    g[fieldname][:,:,:] = stack
    if element_size_um.any():
        g[fieldname].attrs['element_size_um'] = element_size_um
    g.close()

datadir = '/Users/michielk/oxdata/P01/EM/M3/M3_S1_GNU'
dataset='m000_01000-02000_01000-02000_00030-00460'
pf = '_probs_ws_MA_probs_ws_MA_manseg'

_, elsize = loadh5(datadir, dataset + '.h5')
img = nib.load(os.path.join(datadir, dataset + pf + '.nii.gz'))
data = img.get_data()
writeh5(np.transpose(data), datadir, dataset + pf + '.h5', element_size_um=elsize)
#rsync -avz /Users/michielk/oxdata/P01/EM/M3/M3_S1_GNU/m000_01000-02000_01000-02000_00000-00430_segman.h5 ndcn0180@arcus.arc.ox.ac.uk:/data/ndcn-fmrib-water-brain/ndcn0180/EM/M3/M3_S1_GNU/testblock/
#rsync -avz /Users/michielk/oxdata/P01/EM/M3/M3_S1_GNU/m000_01000-02000_02000-03000_00030-00460_probs_ws_MA_probs_ws_MA_manseg.h5 ndcn0180@arcus.arc.ox.ac.uk:/data/ndcn-fmrib-water-brain/ndcn0180/EM/M3/M3_S1_GNU/

### load hdf5 and zip in hdf5 ###
data, elsize = loadh5(datadir, dataset + '_probs0_eed2.h5')
writeh5(data, datadir, dataset + '_zip.h5', dtype='float', element_size_um=elsize)


scriptdir="$HOME/workspace/EM"
datadir="$DATA/EM/M3/M3_S1_GNU/testblock" && cd $datadir
dataset='m000'
refsect='0250'

x=1000; X=2000; y=1000; Y=2000; z=0; Z=430;  # mem +- 20GB for MA
qsubfile=$datadir/EM_p2l_${x}-${X}.sh
echo '#!/bin/bash' > $qsubfile
echo "#SBATCH --nodes=1" >> $qsubfile
echo "#SBATCH --ntasks-per-node=1" >> $qsubfile
echo "#SBATCH --time=01:00:00" >> $qsubfile
#echo "#SBATCH --mem=50000" >> $qsubfile
echo "#SBATCH --job-name=EM_ws" >> $qsubfile
echo "python $scriptdir/mesh/prob2labels.py $datadir $dataset \
--SEfile '_segman.h5' --MMfile '_probs_ws_MMdistsum_distfilter.h5' --UAfile '_probs_ws_UA.h5' --PAfile '_probs_ws_PA.h5' \
-n 5 -o 220 235 491 -s 430 4460 5217 \
-x $x -X $X -y $y -Y $Y -z $z -Z $Z > $datadir/output_${x}-${X}_${y}-${Y} &" >> $qsubfile
echo "wait" >> $qsubfile
#sbatch -p compute $qsubfile
sbatch -p devel $qsubfile

--SEfile '_seg.h5' --MAfile '_probs_ws_MAfilled.h5' --MMfile '_probs_ws_MMdistsum_distfilter.h5' --UAfile '_probs_ws_UA.h5' --PAfile '_probs_ws_PA.h5'
