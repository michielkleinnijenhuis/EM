#!/usr/bin/env python

"""
python prob2labels.py ...
"""

# TODO: apply mask for volume edges

import os
import sys
from argparse import ArgumentParser

import h5py
import numpy as np
from skimage.morphology import watershed, remove_small_objects, dilation, erosion, square
from skimage.segmentation import random_walker, relabel_sequential
from scipy.ndimage.measurements import label
from scipy.ndimage import distance_transform_edt
from scipy.ndimage.filters import gaussian_filter
from skimage.morphology import medial_axis
from scipy.ndimage import find_objects


def main(argv):
    
    parser = ArgumentParser(description='...')
    
    parser.add_argument('datadir', help='...')
    parser.add_argument('dataset', help='...')
    parser.add_argument('--SEfile', help='...')
    parser.add_argument('--MAfile', help='...')
    parser.add_argument('--MMfile', help='...')
    parser.add_argument('--UAfile', help='...')
    parser.add_argument('--PAfile', help='...')
#     parser.add_argument('-m', '--MAseeds', default=[0.6,100], nargs=2, help='...')
    parser.add_argument('-x', default=0, type=int, help='first x-index')
    parser.add_argument('-X', type=int, help='last x-index')
    parser.add_argument('-y', default=0, type=int, help='first y-index')
    parser.add_argument('-Y', type=int, help='last y-index')
    parser.add_argument('-z', default=0, type=int, help='first z-index')
    parser.add_argument('-Z', type=int, help='last z-index')
    
    args = parser.parse_args()
    
    datadir = args.datadir
    dataset = args.dataset
    SEfile = args.SEfile
    MAfile = args.MAfile
    MMfile = args.MMfile
    UAfile = args.UAfile
    PAfile = args.PAfile
#     MAseeds = args.MAseeds
    x = args.x
    X = args.X
    y = args.y
    Y = args.Y
    z = args.z
    Z = args.Z
#     datadir = '/Users/michielk/oxdata/P01/EM/M3/M3_S1_GNU'
#     dataset = 'm000'
#     dataset = 'm000_0-1000_0-1000_0-100'
#     dataset = 'm000_2000-3000_2000-3000_0-100'
    dataset = dataset + '_' + str(x) + '-' + str(X) + '_' + str(y) + '-' + str(Y) + '_' + str(z) + '-' + str(Z)
    
    segmOffset = [50,60,122]  #zyx for m000 (100 section full FOV)
    if SEfile:
        segm = loadh5(datadir, dataset + SEfile)
    else:
        segm = np.zeros((100, 4111, 4235), dtype='uint16')
        segmOrig = loadh5(datadir, '0250_m000_seg.h5')
        segm[segmOffset[0],
             segmOffset[1]:segmOffset[1]+segmOrig.shape[0],
             segmOffset[2]:segmOffset[2]+segmOrig.shape[1]] = segmOrig
        segm = segm[z:Z,y:Y,x:X]
        writeh5(segm, datadir, dataset + '_seg.h5')
    
    ### load the dataset
    if not MAfile:
        prob_myel = loadh5(datadir, dataset + '_probs0_eed2.h5')
        myelin = prob_myel > 0.2
    if not MMfile:
        prob_myel = loadh5(datadir, dataset + '_probs0_eed2.h5')
        myelin = prob_myel > 0.2
    if not UAfile:
        data = loadh5(datadir, dataset + '.h5')
        gaussian_filter(data, [0.146,1,1], output=data)  # TODO: get from argument or h5
    
    ### get the myelinated axons (MA)
    if MAfile:
        MA = loadh5(datadir, dataset + MAfile)
    else:
        #     seeds_MA = remove_small_objects(label(prob_axon>=MAseeds[0])[0], MAseeds[1])
        #     ws = watershed(prob_myel, seeds_MA, mask=~myelin)
        #     writeh5(ws, datadir, dataset + '_probs_ws_MAws.h5')
        #     MA = label(ws > 0)[0]  # TODO!
        seeds_MA = segm
        seeds_MA[np.logical_or(seeds_MA<1000,seeds_MA>2000)] = 0
        MA = watershed(prob_myel, seeds_MA, mask=~myelin)
        bc = np.bincount(np.ravel(MA))
        largest_label = bc[1:].argmax() + 1
        MA[MA==largest_label] = 0  # NOTE that the worst separated MA is lost this way
        writeh5(MA, datadir, dataset + '_probs_ws_MA.h5')
    
    ### watershed on the myelin to separate individual sheaths
    if MMfile:
        MM = loadh5(datadir, dataset + MMfile)
    else:
        distance = distance_transform_edt(MA==0, sampling=[0.05,0.0073,0.0073])  # TODO: get from h5 (also for writes)
        distance[~myelin] = 0
        MM = watershed(distance, dilation(MA), mask=myelin)
        writeh5(MM, datadir, dataset + '_probs_ws_MM.h5')
    
    ### watershed the unmyelinated axons
    if UAfile:
        UA = loadh5(datadir, dataset + UAfile)
    else:
        seeds_UA = segm
        seeds_UA[seeds_UA<2000] = 0
        seedslice = np.zeros(seeds_UA.shape[1:])
        for label in np.unique(seeds_UA)[1:]:
            binim = seeds_UA[segmOffset[0],:,:]==label
            mask = erosion(binim, square(5))
            seedslice[mask] = label
        seeds_UA[segmOffset[0],:,:] = seedslice
        UA = watershed(-data, seeds_UA, mask=~np.logical_or(MM,MA))
        writeh5(UA, datadir, dataset + '_probs_ws_UA.h5')
    
    ### combine the MM, MA and UA segmentations
    if not PAfile:
        writeh5(MA+MM+UA, datadir, dataset + '_probs_ws_PA.h5')


def loadh5(datadir, dname, fieldname='stack'):
    """"""
    f = h5py.File(os.path.join(datadir, dname), 'r')
    if len(f[fieldname].shape) == 2:
        stack = f[fieldname][:,:]
    if len(f[fieldname].shape) == 3:
        stack = f[fieldname][:,:,:]
    if len(f[fieldname].shape) == 4:
        stack = f[fieldname][:,:,:,:]
    f.close()
    return stack

def writeh5(stack, datadir, fp_out, fieldname='stack', dtype='uint16'):
    """"""
    g = h5py.File(os.path.join(datadir, fp_out), 'w')
    g.create_dataset(fieldname, stack.shape, dtype=dtype)
    g[fieldname][:,:,:] = stack
    g.close()


if __name__ == "__main__":
    main(sys.argv)
