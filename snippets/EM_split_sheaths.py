#!/usr/bin/env python

import os
import sys
from argparse import ArgumentParser

import h5py
import numpy as np
from skimage.morphology import watershed
from scipy.ndimage.morphology import grey_dilation, binary_erosion
from scipy.special import expit
from scipy.ndimage import distance_transform_edt
from skimage.morphology import remove_small_objects, binary_dilation, ball

def main(argv):

    parser = ArgumentParser(description='...')

    parser.add_argument('datadir',
                        help='...')
    parser.add_argument('dset_name',
                        help='...')
    parser.add_argument('-l', '--labelvolume', default=['_labelMA', 'stack'],
                        nargs=2,
                        help='...')
    parser.add_argument('--maskDS', default=['_maskDS', '/stack'], nargs=2,
                        help='...')
    parser.add_argument('--maskMM', default=['_maskMM', '/stack'], nargs=2,
                        help='...')
    parser.add_argument('--maskMA', default=None, nargs=2,
                        help='...')

    args = parser.parse_args()

    datadir = args.datadir
    dset_name = args.dset_name
    labelvolume = args.labelvolume
    maskDS = args.maskDS
    maskMM = args.maskMM
    maskMA = args.maskMA

    elsize = loadh5(datadir, dset_name)[1]
    MA = loadh5(datadir, dset_name + labelvolume[0],
                fieldname=labelvolume[1])[0]
    maskDS = loadh5(datadir, dset_name + maskDS[0],
                    fieldname=maskDS[1], dtype='bool')[0]
    maskMM = loadh5(datadir, dset_name + maskMM[0],
                    fieldname=maskMM[1], dtype='bool')[0]
    if maskMA is not None:
        maskMA = loadh5(datadir, dset_name + maskMA[0],
                        fieldname=maskMA[1], dtype='bool')[0]
    else:
        maskMA = MA != 0

    seeds = label(np.logical_and(prob > lower_threshold,
                                 prob <= upper_threshold))[0]

#     thr = 0.2
#     distmask = np.ones_like(MA, dtype='bool')
#     distmask[MA > thr] = 0
#     distmask[MA == 0] = 0
#     writeh5(distmask, datadir, dset_name + '_distmask',
#             element_size_um=elsize, dtype='uint8')
# 
#     maskMM[maskMA == 1] = 0
#     maskMM[distmask == 1] = 0
#     writeh5(maskMM, datadir, dset_name + '_maskMM_dist',
#             element_size_um=elsize, dtype='uint8')

#     maskMA_dil = binary_dilation(maskMA, selem=ball(10))
#     writeh5(maskMA_dil, datadir, dset_name + '_maskMAdil',
#             element_size_um=elsize, dtype='uint8')
# 
#     maskMM_filt = np.logical_not(maskMA_dil, maskMM)
#     writeh5(maskMM_filt, datadir, dset_name + '_maskMMfilt',
#             element_size_um=elsize, dtype='uint8')

# ========================================================================== #
# function defs
# ========================================================================== #


def sigmoid_weighted_distance(MM, MA, elsize):
    """"""

    lmask = np.zeros((MM.shape[0], MM.shape[1], MM.shape[2], 
                      len(np.unique(MA)[1:])), dtype='bool')
    distsum = np.ones_like(MM, dtype='float')
    medwidth = {}
    for i,l in enumerate(np.unique(MA)[1:]):  # TODO: implement mpi?
        print(i,l)
        dist = distance_transform_edt(MA!=l, sampling=np.absolute(elsize))
        # get the median distance at the outer rim:
        MMfilled = MA + MM
        binim = MMfilled == l
        rim = np.logical_xor(binary_erosion(binim), binim)
        medwidth[l] = np.median(dist[rim])
        # labelmask for voxels further than nmed medians from the object (mem? write to disk?)
        nmed = 2  # TODO: make into argument
        maxdist = nmed * medwidth[l]
        lmask[:,:,:,i] = dist > maxdist
        # median width weighted sigmoid transform on distance function
        weighteddist = expit(dist/medwidth[l])  # TODO: create more pronounced transform
        distsum = np.minimum(distsum, weighteddist)

    return distsum, lmask


def loadh5(datadir, dname, fieldname='stack', dtype=None):
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

    if dtype is not None:
        stack = np.array(stack, dtype=dtype)

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
    main(sys.argv)
