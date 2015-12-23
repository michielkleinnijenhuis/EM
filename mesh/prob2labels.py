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
from skimage.morphology import watershed, remove_small_objects, erosion, square  #, dilation
from scipy.ndimage.morphology import grey_dilation, grey_erosion, binary_erosion, binary_fill_holes, binary_closing, generate_binary_structure
from scipy.special import expit
from skimage.segmentation import random_walker, relabel_sequential
from scipy.ndimage.measurements import label, labeled_comprehension
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
    parser.add_argument('-n', '--nzfills', type=int, default=5, 
                        help='number of characters for section ranges')
    parser.add_argument('-e', '--element_size_um', type=float, nargs='*', default=None, 
                        help='dataset element sizes (in order of outlayout)')
    parser.add_argument('-g', '--gsigma', nargs=3, type=float, default=[0.146, 1, 1], 
                        help='number of characters for section ranges')
    parser.add_argument('-o', '--segmOffset', nargs=3, type=int, default=[50, 60, 122], 
                        help='number of characters for section ranges')
    parser.add_argument('-s', '--fullshape', nargs=3, type=int, default=(430, 4460, 5217), 
                        help='number of characters for section ranges')
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
    nzfills = args.nzfills
    gsigma = args.gsigma
    segmOffset = [o for o in args.segmOffset]
    fullshape = args.fullshape
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
#     fullshape = (100, 4111, 4235)
#     fullshape = (430, 4460, 5217)
#     dataset = dataset + '_' + str(x) + '-' + str(X) + '_' + str(y) + '-' + str(Y) + '_' + str(z) + '-' + str(Z)
    dataset = dataset + '_' + str(x).zfill(nzfills) + '-' + str(X).zfill(nzfills) + \
                        '_' + str(y).zfill(nzfills) + '-' + str(Y).zfill(nzfills) + \
                        '_' + str(z).zfill(nzfills) + '-' + str(Z).zfill(nzfills)
    
    ### load the dataset
    data, elsize = loadh5(datadir, dataset + '.h5')
    
    if elsize.any():
        pass
    elif args.element_size_um.any():
        elsize = args.element_size_um
    
    datamask = data == 0
    if not MAfile:
        prob_myel = loadh5(datadir, dataset + '_probs0_eed2.h5')[0]
#         prob_mito = loadh5(datadir, dataset + '_probs2_eed2.h5')
        myelin = prob_myel > 0.2
#         myelin = np.logical_and(prob_myel > 0.2, prob_mito < 0.2)
    if not MMfile:
        prob_myel = loadh5(datadir, dataset + '_probs0_eed2.h5')[0]
        myelin = prob_myel > 0.2
    if not UAfile:
        gaussian_filter(data, gsigma, output=data)
    
#     segmOffset = [50,60,122]  #zyx for m000 (100 section full FOV)
#     segmOffset = [220,491,235]  #zyx for m000 (430 section full FOV)
    if SEfile:
        segm = loadh5(datadir, dataset + SEfile)
    else:
        segm = np.zeros(fullshape, dtype='uint16')
        segmOrig = loadh5(datadir, '0250_m000_seg.h5')[0]
        segm[segmOffset[0],
             segmOffset[1]:segmOffset[1]+segmOrig.shape[0],
             segmOffset[2]:segmOffset[2]+segmOrig.shape[1]] = segmOrig
        segm = segm[z:Z,y:Y,x:X]
        writeh5(segm, datadir, dataset + '_seg.h5', element_size_um=elsize)
    
    ### get the myelinated axons (MA)
    if MAfile:
        MA = loadh5(datadir, dataset + MAfile)[0]
    else:
        #     seeds_MA = remove_small_objects(label(prob_axon>=MAseeds[0])[0], MAseeds[1])
        #     ws = watershed(prob_myel, seeds_MA, mask=~myelin)
        #     writeh5(ws, datadir, dataset + '_probs_ws_MAws.h5')
        #     MA = label(ws > 0)[0]  # TODO!
        seeds_MA = np.copy(segm)
        seeds_MA[np.logical_or(seeds_MA<1000,seeds_MA>2000)] = 0
        MA = watershed(prob_myel, seeds_MA, mask=np.logical_and(~myelin, ~datamask))
        bc = np.bincount(np.ravel(MA))
        largest_label = bc[1:].argmax() + 1
        print("largest label {!s} was removed".format(largest_label))
        MA[MA==largest_label] = 0  # NOTE that the worst separated MA is lost this way
        writeh5(MA, datadir, dataset + '_probs_ws_MA.h5', element_size_um=elsize)
        # fill holes
        for l in np.unique(MA)[1:]:
            ### fill holes
            labels = label(MA!=l)[0]
            labelCount = np.bincount(labels.ravel())
            background = np.argmax(labelCount)
            MA[labels != background] = l
            ### closing
            binim = MA==l
            binim = binary_closing(binim, iterations=10)
            MA[binim] = l
            ### fill holes
            labels = label(MA!=l)[0]
            labelCount = np.bincount(labels.ravel())
            background = np.argmax(labelCount)
            MA[labels != background] = l
            ### update myelin mask
            myelin[MA != 0] = False
            print(l)
        writeh5(MA, datadir, dataset + '_probs_ws_MAfilled.h5', element_size_um=elsize)
    
    ### watershed on the myelin to separate individual sheaths
    if MMfile:
        MM = loadh5(datadir, dataset + MMfile)[0]
    else:
        distance = distance_transform_edt(MA==0, sampling=[0.05, 0.0073020253, 0.0073020253])  # TODO: get from h5 (also for writes)
        writeh5(distance, datadir, dataset + '_distance.h5', dtype='float', element_size_um=elsize)
        #MM = watershed(distance, dilation(MA), mask=myelin)  # when skimage >0.11.3 is available
        MM = watershed(distance, grey_dilation(MA, size=(3,3,3)), mask=np.logical_and(myelin, ~datamask))
        writeh5(MM, datadir, dataset + '_probs_ws_MM.h5', element_size_um=elsize)
        
        distsum = np.ones_like(MM, dtype='float')
        lmask = np.zeros((MM.shape[0],MM.shape[1],MM.shape[2],len(np.unique(MA)[1:])), dtype='bool')
        medwidth = {}
        for i,l in enumerate(np.unique(MA)[1:]):  # TODO: implement mpi
            dist = distance_transform_edt(MA!=l, sampling=[0.05, 0.0073, 0.0073])  # TODO: get from h5 (also for writes)
            # get the median distance at the outer rim:
            MMfilled = MA+MM
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
            print(l)
        writeh5(distsum, datadir, dataset + '_probs_ws_distsum.h5', dtype='float', element_size_um=elsize)
        
        MM = watershed(distsum, grey_dilation(MA, size=(3,3,3)), mask=np.logical_and(myelin, ~datamask))
        writeh5(MM, datadir, dataset + '_probs_ws_MMdistsum.h5', element_size_um=elsize)
        
        for i,l in enumerate(np.unique(MA)[1:]):
            MM[np.logical_and(lmask[:,:,:,i], MM==l)] = 0
        writeh5(MM, datadir, dataset + '_probs_ws_MMdistsum_distfilter.h5', element_size_um=elsize)
    
    ### watershed the unmyelinated axons
    if UAfile:
        UA = loadh5(datadir, dataset + UAfile)[0]
    else:
        seeds_UA = np.copy(segm)
        seedslice = seeds_UA[segmOffset[0],:,:]
        seedslice[seedslice<2000] = 0
        for l in np.unique(seeds_UA)[1:]:
            seedslice[erosion(seedslice==l, square(5))] = l
        seeds_UA[segmOffset[0],:,:] = seedslice
        UA = watershed(-data, seeds_UA, 
                       mask=np.logical_and(~datamask, ~np.logical_or(MM,MA)))
        writeh5(UA, datadir, dataset + '_probs_ws_UA.h5', element_size_um=elsize)
    
    ### combine the MM, MA and UA segmentations
    if not PAfile:
        writeh5(MA+MM+UA, datadir, dataset + '_probs_ws_PA.h5', element_size_um=elsize)


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


if __name__ == "__main__":
    main(sys.argv)
