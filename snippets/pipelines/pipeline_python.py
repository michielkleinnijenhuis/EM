datadir = '/Users/michielk/oxdata/P01/EM/Myrf_01/SET-B/B-NT-S10-2f_ROI_00'
dataset = 'B-NT-S10-2f_ROI_00'
regname = 'ref'
xe=0.007; ye=0.007; ze=0.1;
ds = 7
dataset_ds = '{}ds{}'.format(dataset, ds)

data = series2stack(
    os.path.join(datadir, regname),
    element_size_um=[ze, ye, xe],
    chunksize=[3, 20, 20],
    outputpath=os.path.join(datadir, '{}.h5'.format(dataset), 'data'),
    save_steps=True)


import os

datadir = '/Users/michielk/oxdata/test'
dataset = 'test'
regname = 'dm3'
xe=0.007; ye=0.007; ze=0.1;
ds = 7
dataset_ds = '{}ds{}'.format(dataset, ds)

from wmem.series2stack import series2stack
outputpath=os.path.join(datadir, '{}.h5'.format(dataset), 'data'),
op = os.path.join(datadir, '{}_data.nii.gz'.format(dataset))
data = series2stack(
    os.path.join(datadir, regname),
    elsize=[ze, ye, xe],
    chunksize=[3, 20, 20],
    outputpath=op,
    save_steps=True)

from wmem.downsample_blockwise import downsample_blockwise
# op = os.path.join(datadir, '{}.h5'.format(dataset_ds), 'data')
ip = os.path.join(datadir, '{}_data.nii.gz'.format(dataset))
op = os.path.join(datadir, '{}_data.nii.gz'.format(dataset_ds))
data_ds = downsample_blockwise(
    ip,
    blockreduce=[1, ds, ds],
    func='np.mean',
    outputpath=op,
    save_steps=True)

ip = os.path.join(datadir, '{}_data.nii.gz'.format(dataset_ds))
from wmem.prob2mask import prob2mask
mask = prob2mask(
    ip,
    dataslices=None,
    lower_threshold=0,
    upper_threshold=1,
    size=0,
    dilation=0,
    go2D=False,
    outputpath='',
    save_steps=False,
    protective=False,
    )

stack2stack
