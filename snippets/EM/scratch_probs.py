scriptdir='/Users/michielk/workspace/EM';
datadir='/Users/michielk/oxdata/P01/EM/M3/M3_S1_GNU/M3S1GNU/ds7_arc/viz';
invol='m000_01000-01500_01000-01500_00030-00460_probs';

cd(datadir)
addpath(scriptdir)

EM_eed([datadir filesep snippets filesep eed], invol, '/volume/predictions', '/stack', 5)


import os
import h5py
from nibabel import Nifti1Image
import numpy as np

scriptdir='/Users/michielk/workspace/EM';
datadir='/Users/michielk/oxdata/P01/EM/M3/M3_S1_GNU/M3S1GNU/ds7_arc/viz';
invol='m000_01000-01500_01000-01500_00030-00460_probs';
element_size_um = [0.0073, 0.0073, 0.05, 1]

f = h5py.File(os.path.join(datadir, invol + '.h5'), 'r')
vol = f['volume/predictions'][:, :, :]
f.close()

vol = np.transpose(vol, [2, 1, 0, 3])
mat = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
if element_size_um is not None:
    mat[0][0] = element_size_um[0]
    mat[1][1] = element_size_um[1]
    mat[2][2] = element_size_um[2]

Nifti1Image(vol, mat).to_filename(os.path.join(datadir, invol + '.nii.gz'))




datadir='/Users/michielk/oxdata/P01/EM/M3/M3_S1_GNU/M3S1GNU/ds7_arc';
invol='M3S1GNUds7';
element_size_um = [0.0073, 0.0073, 0.05]

f = h5py.File(os.path.join(datadir, invol + '.h5'), 'r')

vol = f['volume/predictions'][:, :, :]
f.close()


# correct element_sizes
import os
import h5py
import glob

inputdir = '/Users/michielk/oxdata/P01/EM/M3/M3_S1_GNU/M3S1GNU/ds7_arc/viz'

# regexp = "*_probs.h5"
# field = 'volume/predictions'
# elsize = [0.05, 0.0073, 0.0073, 1]
# axislabels = 'zyxc'

regexp = "*_probs_eed2.h5"
elsize = [0.05, 0.0073, 0.0073, 1]
axislabels = 'zyxc'
field = 'stack'

regexp = "*_probs*_eed2.h5"
regexp = "*_shuffled.h5"
regexp = "*_maskMM.h5"
elsize = [0.05, 0.0073, 0.0073]
axislabels = 'zyx'
field = 'stack'

infiles = glob.glob(os.path.join(inputdir, regexp))
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



blender -b probs_v4.blend -a








####
def print_attrs(name, obj):
    print name
    for key, val in obj.attrs.iteritems():
        print "    %s: %s" % (key, val)

f = h5py.File('foo.hdf5','r')
f.visititems(print_attrs)


names = []
def h5_dataset_add(name, obj):
    if isinstance(obj.id, h5py.h5d.DatasetID):
        names.append(name)

def h5_dataset_add(name, obj):
    names.append(name)

f.visititems(h5_dataset_add)



import os
import h5py
from nibabel import Nifti1Image
import numpy as np

scriptdir='/Users/michielk/workspace/EM';
datadir='/Users/michielk/oxdata/P01/EM/M3/M3_S1_GNU/M3S1GNU/ds7_arc/viz';
datastem = 'm000_01000-01500_01000-01500_00030-00460'

invol = '_probs%d_eed2' % 0
f = h5py.File(os.path.join(datadir, datastem + invol + '.h5'), 'r')
gname = datastem + '_probs_eed2' + '.h5'
gpath = os.path.join(datadir, gname)
gfile = h5py.File(gpath, 'w')
outds = gfile.create_dataset('stack', f['stack'].shape + (6,),
                             dtype='float64', compression='gzip')

for i in range(0, 6):
    invol = '_probs%d_eed2' % i
    f = h5py.File(os.path.join(datadir, datastem + invol + '.h5'), 'r')
    outds[:, :, :, i] = f['stack'][:, :, :]
    f.close()

gfile.close()


# symlinks to
basedir=~/oxdox/papers/abstracts/ISMRM2017/Eposter/anims/m000
cd ${basedir}
for vol in voltex_m000_probs voltex_m000_probs_eed2; do
for sub in 000 001 002 003 004 005; do
lndir=${basedir}/${vol}.${sub}/IMAGE_SEQUENCE/vol0000
ordir=${basedir}/${vol}/IMAGE_SEQUENCE/vol0${sub}
mkdir -p ${lndir} && cd ${lndir}
cp ${basedir}/${vol}/*.npy ${basedir}/${vol}.${sub}/
for fname in `ls $ordir`; do
ln -s ../../../${vol}/IMAGE_SEQUENCE/vol0${sub}/${fname} ${fname}
done
done
done
