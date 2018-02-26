###=========================================================================###
### prepare environment
###=========================================================================###
export PATH=${DATA}/anaconda2/bin:$PATH
export CONDA_PATH="$(conda info --root)"
export scriptdir="${HOME}/workspace/EM"
export PYTHONPATH=$scriptdir
export PYTHONPATH=$PYTHONPATH:$HOME/workspace/pyDM3reader
imagej=/system/software/linux-x86_64/fiji/20140602/ImageJ-linux64  # 20170515
ilastik=$HOME/workspace/ilastik-1.2.2post1-Linux/run_ilastik.sh


###=========================================================================###
### dataset parameters
###=========================================================================###
basedir="${DATA}/EM/Myrf_01/SET-B"
dataset='B-NT-S10-2f_ROI_00'
datadir=$basedir/${dataset} && mkdir -p $datadir && cd $datadir
source datastems_blocks.sh
# Image Width: 8423 Image Length: 8316


###=========================================================================###
### downsample
###=========================================================================###
module load mpich2/1.5.3__gcc

export template='single' additions='mpi'
export njobs=1 nodes=1 tasks=16 memcpu=10000 wtime="00:40:00" q=""
export jobname="ds"
scriptfile=$datadir/EM_ds_script.sh
echo '#!/bin/bash' > $scriptfile
echo "PATH=$CONDA_PATH:\$PATH" >> $scriptfile
echo "source activate root" >> $scriptfile
echo "PYTHONPATH=$PYTHONPATH" >> $scriptfile
echo "python $scriptdir/wmem/downsample_slices.py \
$datadir/reg $datadir/reg_ds7 -r '*.tif' -f 7 -M" >> $scriptfile
export cmd="$scriptfile"
chmod +x $scriptfile
source $scriptdir/pipelines/template_job_$template.sh


###=========================================================================###
### correct matlab h5 after EED
###=========================================================================###

import os
import numpy as np
import h5py

datadir = "/data/ndcn-fmrib-water-brain/ndcn0180/EM/M3/M3S1GNU"
datadir = "/Users/michielk/oxdata/P01/EM/Myrf_01/SET-B/B-NT-S10-2f_ROI_00"
datadir = '/data/ndcn-fmrib-water-brain/ndcn0180/EM/Myrf_01/SET-B/B-NT-S10-2f_ROI_00'

filestem = "B-NT-S10-2f_ROI_00"
dset0 = 'data'
h5path = os.path.join(datadir, filestem + '.h5')
h5file0 = h5py.File(h5path, 'a')
ds0 = h5file0[dset0]
# ds0.attrs['element_size_um'] = np.array([0.1, 0.007, 0.007])

filestem = "B-NT-S10-2f_ROI_00_02500-03500_02500-03500_00100-00120"
filestem = "B-NT-S10-2f_ROI_00_03000-03500_03000-03500_00000-00184"
filestem = "B-NT-S10-2f_ROI_00_02500-03500_02500-03500_00100-00120_probs"
h5path = os.path.join(datadir, filestem + '.h5')
h5file1 = h5py.File(h5path, 'a')
ds = h5file1['data']
ds.attrs['element_size_um'] = ds0.attrs['element_size_um']
h5file1.close()

filestem = "B-NT-S10-2f_ROI_00_probs"
filestem = "B-NT-S10-2f_ROI_00_02500-03500_02500-03500_00100-00120_probs"
h5path = os.path.join(datadir, filestem + '.h5')
h5file1 = h5py.File(h5path, 'a')
ds = h5file1['volume/predictions']
ds.attrs['element_size_um'] = np.append(ds0.attrs['element_size_um'], 1)
h5file1.close()


filestems = ["B-NT-S10-2f_ROI_00_probs0_eed2", "B-NT-S10-2f_ROI_00_probs1_eed2", "B-NT-S10-2f_ROI_00_probs2_eed2"]
for filestem in filestems:
    h5path = os.path.join(datadir, filestem + '.h5')
    h5file1 = h5py.File(h5path, 'a')
    dsets = ['probs_eed']
    for dset in dsets:
        ds = h5file1[dset]
        ds.attrs['element_size_um'] = ds0.attrs['element_size_um']
        for i, label in enumerate(ds0.attrs['DIMENSION_LABELS']):
            ds.dims[i].label = label
    h5file1.close()


from glob import glob
files = glob(os.path.join(datadir, 'blocks', '*00184.h5'))
for f in files:
    h5file1 = h5py.File(f, 'a')
    dsets = ['data']
    for dset in dsets:
        ds = h5file1[dset]
        ds.attrs['element_size_um'] = ds0.attrs['element_size_um']
        for i, label in enumerate(ds0.attrs['DIMENSION_LABELS']):
            ds.dims[i].label = label
    h5file1.close()

from glob import glob
files = glob(os.path.join(datadir, 'blocks', '*00184_probs.h5'))
for f in files:
    h5file1 = h5py.File(f, 'a')
    dsets = ['volume/predictions']
    for dset in dsets:
        ds = h5file1[dset]
        ds.attrs['element_size_um'] = np.append(ds0.attrs['element_size_um'], 1)
        for i, label in enumerate(ds0.attrs['DIMENSION_LABELS']):
            ds.dims[i].label = label
        ds.dims[3].label = 'c'
    h5file1.close()

from glob import glob
files = glob(os.path.join(datadir, 'blocks_500', '*00184.h5'))
for f in files:
    h5file1 = h5py.File(f, 'a')
    dsets = ['data']
    for dset in dsets:
        ds = h5file1[dset]
        ds.attrs['element_size_um'] = ds0.attrs['element_size_um']
        for i, label in enumerate(ds0.attrs['DIMENSION_LABELS']):
            ds.dims[i].label = label
    h5file1.close()

from glob import glob
files = glob(os.path.join(datadir, 'blocks_500', '*00184_probs.h5'))
for f in files:
    h5file1 = h5py.File(f, 'a')
    dsets = ['volume/predictions']
    for dset in dsets:
        ds = h5file1[dset]
        ds.attrs['element_size_um'] = np.append(ds0.attrs['element_size_um'], 1)
        for i, label in enumerate(ds0.attrs['DIMENSION_LABELS']):
            ds.dims[i].label = label
        ds.dims[3].label = 'c'
    h5file1.close()

from glob import glob
files = glob(os.path.join(datadir, 'blocks_500', '*eed2.h5'))
for f in files:
    h5file1 = h5py.File(f, 'a')
    dsets = ['probs_eed']
    for dset in dsets:
        ds = h5file1[dset]
        ds.attrs['element_size_um'] = ds0.attrs['element_size_um']
        for i, label in enumerate(ds0.attrs['DIMENSION_LABELS']):
            ds.dims[i].label = label
    h5file1.close()

from glob import glob
files = glob(os.path.join(datadir, 'blocks_500', '*masks.h5'))
for f in files:
    h5file1 = h5py.File(f, 'a')
    dsets = ['maskDS', 'maskMM']
    for dset in dsets:
        ds = h5file1[dset]
        ds.attrs['element_size_um'] = ds0.attrs['element_size_um']
        for i, label in enumerate(ds0.attrs['DIMENSION_LABELS']):
            ds.dims[i].label = label
    h5file1.close()




filestem = "B-NT-S10-2f_ROI_00_probs"
h5path = os.path.join(datadir, filestem + '.h5')
h5file1 = h5py.File(h5path, 'a')
dsets = ['volume/predictions']

for dset in dsets:
    ds = h5file1[dset]
    ds.attrs['element_size_um'] = np.append(ds0.attrs['element_size_um'], 1)

filestem = "B-NT-S10-2f_ROI_00_05480-06020_05480-06020_00000-00184_probs1_eed2"
filestem = " B-NT-S10-2f_ROI_00_probs"
h5path = os.path.join(datadir, filestem + '.h5')
h5file1 = h5py.File(h5path, 'a')
dsets = ['probs_eed']


for dset in dsets:
    ds = h5file1[dset]
    ds.attrs['element_size_um'] = ds0.attrs['element_size_um']
    for i, label in enumerate(ds0.attrs['DIMENSION_LABELS']):
        ds.dims[i].label = label

h5file0.close()
h5file1.close()


###=========================================================================###
### patterns matching filenames
###=========================================================================###

pat = 'ROI'
# pat = '([0-9]{5}-[0-9]{5}){3}'
pat = '([0-9]{5}-[0-9]{5})'
str='B-NT-S10-2f_ROI_00_07980-08423_06480-07020_00000-00184_probs0_eed2.h5'
re.findall(pat, str)
result = re.search(pat, str)
print(result.start(), result.end())
