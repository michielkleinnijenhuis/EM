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


def main(argv):
    
    parser = ArgumentParser(description='...')
    
    parser.add_argument('datadir', help='...')
    parser.add_argument('dataset', help='...')
    parser.add_argument('--maskDS', default=['_maskDS', '/stack'], nargs=2, help='...')
    parser.add_argument('--maskMM', default=['_maskMM', '/stack'], nargs=2, help='...')
    parser.add_argument('--maskMA', default=['_maskMA', '/stack'], nargs=2, help='...')
    parser.add_argument('--supervoxels', default=['_supervoxels', '/stack'], nargs=2, help='...')
    parser.add_argument('-w', '--sigmoidweighting_MM', action='store_true')
    parser.add_argument('-d', '--distancefilter_MM', action='store_true')
    parser.add_argument('-n', '--nzfills', type=int, default=5, 
                        help='number of characters for section ranges')
    parser.add_argument('-e', '--element_size_um', type=float, nargs='*', 
                        default=None, 
                        help='dataset element sizes (in order of outlayout)')
    parser.add_argument('-x', default=0, type=int, help='first x-index')
    parser.add_argument('-X', type=int, help='last x-index')
    parser.add_argument('-y', default=0, type=int, help='first y-index')
    parser.add_argument('-Y', type=int, help='last y-index')
    parser.add_argument('-z', default=0, type=int, help='first z-index')
    parser.add_argument('-Z', type=int, help='last z-index')

    args = parser.parse_args()
    
    datadir = args.datadir
    dataset = args.dataset
    maskDS = args.maskDS
    maskMM = args.maskMM
    maskMA = args.maskMA
    supervoxels = args.supervoxels
    sigmoidweighting_MM = args.sigmoidweighting_MM
    distancefilter_MM = args.distancefilter_MM
    nzfills = args.nzfills
    element_size_um = args.element_size_um
    x = args.x
    X = args.X
    y = args.y
    Y = args.Y
    z = args.z
    Z = args.Z
    
    dset_info = {'datadir': datadir, 'base': dataset, 
                 'nzfills': nzfills, 'postfix': '', 
                 'x': x, 'X': X, 'y': y, 'Y': Y, 'z': z, 'Z': Z}
    dset_name = dataset_name(dset_info)

    elsize = loadh5(datadir, dset_name)[1]
    maskDS = loadh5(datadir, dset_name + maskDS[0], 
                    fieldname=maskDS[1], dtype='bool')[0]
    maskMM = loadh5(datadir, dset_name + maskMM[0], 
                    fieldname=maskMM[1], dtype='bool')[0]
    maskMA = loadh5(datadir, dset_name + maskMA[0], 
                    fieldname=maskMA[1], dtype='bool')[0]
    svox = loadh5(datadir, dset_name + supervoxels[0], 
                  fieldname=supervoxels[1])[0]

    if element_size_um is not None:
        elsize = element_size_um
    dset_info['elsize'] = elsize

    ### watershed on myelin to separate individual sheaths
    MA = svox
    MA[~maskMA] = 0
    outpf = '_MM'

    # watershed on simple distance transform
    distance = distance_transform_edt(~maskMA, sampling=np.absolute(elsize))
    outpf = outpf + '_ws'
    MM = watershed(distance, grey_dilation(MA, size=(3,3,3)), 
                   mask=np.logical_and(maskMM, maskDS))
    writeh5(MM, datadir, dset_name + outpf,
            element_size_um=elsize, dtype='int32')

    # watershed on sigmoid-modulated distance transform
    if sigmoidweighting_MM:
        outpf = outpf + '_sw'
        distsum, lmask = sigmoid_weighted_distance(MM, MA, elsize)
        writeh5(distsum, datadir, dset_name + outpf + '_distsum', 
                element_size_um=elsize, dtype='float')
        MM = watershed(distsum, grey_dilation(MA, size=(3,3,3)), 
                       mask=np.logical_and(maskMM, maskDS))
        writeh5(MM, datadir, dset_name + outpf,
                element_size_um=elsize, dtype='int32')
    else:  # TODO simple distance th
        lmask = np.zeros((MM.shape[0], MM.shape[1], MM.shape[2], 
                  len(np.unique(MA)[1:])), dtype='bool')

    if distancefilter_MM:  # very mem-intensive
        outpf = outpf + '_df'
        for i,l in enumerate(np.unique(MA)[1:]):
            MM[np.logical_and(lmask[:,:,:,i], MM==l)] = 0
        writeh5(MM, datadir, dset_name + outpf,
                element_size_um=elsize, dtype='int32')


# ========================================================================== #
# function defs
# ========================================================================== #


def dataset_name(dname_info):
    nf = dname_info['nzfills']
    dname = dname_info['base'] + \
                '_' + str(dname_info['x']).zfill(nf) + \
                '-' + str(dname_info['X']).zfill(nf) + \
                '_' + str(dname_info['y']).zfill(nf) + \
                '-' + str(dname_info['Y']).zfill(nf) + \
                '_' + str(dname_info['z']).zfill(nf) + \
                '-' + str(dname_info['Z']).zfill(nf) + \
                dname_info['postfix']
    
    return dname

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
