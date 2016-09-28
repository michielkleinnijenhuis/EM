#!/usr/bin/env python

### watershed train and test data
import os
import sys
from argparse import ArgumentParser

import h5py
import numpy as np
from scipy.ndimage import label
from scipy.ndimage.morphology import binary_dilation, binary_fill_holes
from scipy.ndimage.filters import gaussian_filter
# from skimage.measure import label
from skimage.segmentation import relabel_sequential
from skimage.morphology import watershed, remove_small_objects

def main(argv):

    parser = ArgumentParser(description='...')

    parser.add_argument('datadir', help='...')
    parser.add_argument('dset_name', help='...')
    parser.add_argument('-l', '--lower_threshold', type=float, default=0, help='...')
    parser.add_argument('-u', '--upper_threshold', type=float, default=1, help='...')
    parser.add_argument('-s', '--seed_size', type=int, default=64, help='...')
    parser.add_argument('-g', '--gsigma', nargs=3, type=float, default=[0.146, 1, 1], help='...')
    parser.add_argument('-o', '--outpf', default='_ws', help='...')

    args = parser.parse_args()

    datadir = args.datadir
    dset_name = args.dset_name

    gsigma = args.gsigma
    lower_threshold = args.lower_threshold
    upper_threshold = args.upper_threshold
    seed_size = args.seed_size
    outpf = args.outpf
    outpf = '%s_l%.2f_u%.2f_s%03d' % (outpf, lower_threshold, upper_threshold, seed_size)

    data, elsize = loadh5(datadir, dset_name)
    datamask = data != 0
    datamask = binary_dilation(binary_fill_holes(datamask))  # TODO
#     data = gaussian_filter(data, gsigma)

    prob_ics = loadh5(datadir, dset_name + '_probs', fieldname='volume/predictions')[0][:,:,:,1]
    prob_myel = loadh5(datadir, dset_name + '_probs0_eed2')[0]
    myelin = prob_myel > 0.2
    # prob_mito = loadh5(datadir, dataset + '_probs2_eed2.h5')[0]
    # mito = prob_mito > 0.3
#     seeds = label(np.logical_and(data>lower_threshold, data<=upper_threshold))[0]
    seeds = label(np.logical_and(prob_ics>lower_threshold, prob_ics<=upper_threshold))[0]
    remove_small_objects(seeds, min_size=seed_size, in_place=True)
    seeds = relabel_sequential(seeds)[0]
    writeh5(seeds, datadir, dset_name + outpf + "seeds", element_size_um=elsize)
    print(len(np.unique(seeds)))
    MA = watershed(-prob_ics, seeds, mask=np.logical_and(~myelin, datamask))
    writeh5(MA, datadir, dset_name + outpf, element_size_um=elsize)


def loadh5(datadir, dname, fieldname='stack'):
    """"""
    f = h5py.File(os.path.join(datadir, dname + '.h5'), 'r')
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
    g = h5py.File(os.path.join(datadir, fp_out + '.h5'), 'w')
    g.create_dataset(fieldname, stack.shape, dtype=dtype, compression="gzip")
    if len(stack.shape) == 2:
        g[fieldname][:,:] = stack
    elif len(stack.shape) == 3:
        g[fieldname][:,:,:] = stack
    elif len(stack.shape) == 4:
        g[fieldname][:,:,:,:] = stack
    if element_size_um is not None:
        g[fieldname].attrs['element_size_um'] = element_size_um
    g.close()


if __name__ == "__main__":
    main(sys.argv[1:])

