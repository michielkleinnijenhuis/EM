import os
import sys
from argparse import ArgumentParser
# import copy

import h5py
import numpy as np
import xml.etree.ElementTree
from skimage.morphology import watershed, square
from scipy.ndimage.morphology import grey_dilation, \
    binary_erosion, binary_dilation, binary_closing, binary_fill_holes
from scipy.special import expit
from scipy.ndimage.measurements import label
from scipy.ndimage import distance_transform_edt
from scipy.ndimage.filters import gaussian_filter

datadir = '/Users/michielk/oxdata/P01/EM/M3/M3_S1_GNU'
dataset = 'm000'
x=1000; X=1500; y=1000; Y=1500; z=30; Z=460; nzfills = 5
dset_info = {'datadir': datadir, 'base': dataset,
             'nzfills': nzfills, 'postfix': '',
             'x': x, 'X': X, 'y': y, 'Y': Y, 'z': z, 'Z': Z}
# dset_name = dataset_name(dset_info)
dset_name = 'm000_01000-01500_01000-01500_00030-00460'

# probs, elsize = loadh5(datadir, dset_name + '_probs', 'volume/predictions')
# prob_mito = probs[:,:,:,2]
# writeh5(prob_mito, datadir, dset_name + "_probs2", element_size_um=elsize, dtype=float)
prob_mito, elsize = loadh5(datadir, dset_name + '_probs2_eed2')
dset_info['elsize'] = elsize
mito = prob_mito > 0.3
writeh5(mito, datadir, dset_name + '_mito', element_size_um=elsize)
mitos = label(mito)
writeh5(mitos, datadir, dset_name + '_mitos', element_size_um=elsize)

annotationfile = os.path.join(dset_info['datadir'], dset_name + '_knossos', 'annotation_MC.xml')
objs = get_knossos_controlpoints(annotationfile)
# seeds_MC = np.zeros_like(mito, dtype='uint16')
# seeds_MC = seeds_knossos(dset_info, seeds_MC, 'MC', '_sknMC', fill_edges=False)
# MCs = np.unique(seeds_MC)

mitolabs = []
for obj in objs:
    for _, kcoord in obj['nodedict'].iteritems():
        dcoord = knossoscoord2dataset(dset_info, kcoord)
        try:
            labval = mitos[dcoord[0], dcoord[1], dcoord[2]]
            print(labval)
        except:
            pass
        mitolabs.append(labval)

mitolabs = np.unique(mitolabs)
mitolabs = mitolabs[mitolabs>0]

mitosel = np.zeros_like(mito, dtype='uint16')
for ml in mitolabs:
    mitosel[mitos == ml] = ml
writeh5(mitosel, datadir, dset_name + '_mitosel', element_size_um=elsize)

# forward_map = np.zeros(np.max(mitos) + 1, 'bool')
# forward_map[0:len(mitolabs)] = mitolabs
# segments = forward_map[mitos]
# writeh5(segments, datadir, dset_name + '_mitosel', element_size_um=elsize)

# MM, ICS, MC, membrane, nMM, nMC
probs, elsize = loadh5(datadir, dset_name + '_probs')
mitoplus = probs[:,:,:,2] + probs[:,:,:,5]
writeh5(mitoplus, datadir, dset_name + '_mitoplus', element_size_um=elsize, dtype='float')
MMplus = probs[:,:,:,0] + probs[:,:,:,4]
writeh5(MMplus, datadir, dset_name + '_MMplus', element_size_um=elsize, dtype='float')
MMplusg = gaussian_filter(MMplus, [0.584, 4, 4])
writeh5(MMplusg, datadir, dset_name + '_MMplusg', element_size_um=elsize, dtype='float')
mitoMMplusg = probs[:,:,:,2] * (1-MMplusg)
writeh5(mitoMMplusg, datadir, dset_name + '_mitoMMplusg', element_size_um=elsize, dtype='float')
mitoplusMMplusg = mitoplus * (1-MMplusg)
writeh5(mitoplusMMplusg, datadir, dset_name + '_mitoplusMMplusg', element_size_um=elsize, dtype='float')


SEfile = '_seg.h5'
x=1000; X=1500; y=1000; Y=1500; z=30; Z=460;
fullshape = (430, 4460, 5217)
segmOffset = [50, 60, 122]
segm = np.zeros(fullshape, dtype='uint16')
segmOrig = loadh5(datadir, '0250_m000_seg')[0]
segm[segmOffset[0],
     segmOffset[1]:segmOffset[1]+segmOrig.shape[0],
     segmOffset[2]:segmOffset[2]+segmOrig.shape[1]] = segmOrig
segm = segm[z:Z,y:Y,x:X]
segsliceno = segmOffset[0] - z
# segm = loadh5(datadir, dset_name + SEfile)[0]

# UA = loadh5(datadir, dset_name + UAfile)[0]

outpf = '_UA'
# seedimage
seeds_UA = np.copy(segm)
seedslice = seeds_UA[segsliceno,:,:]
seedslice[seedslice<2000] = 0
final_seedslice = np.zeros_like(seedslice)
for l in np.unique(seedslice)[1:]:
    final_seedslice[binary_erosion(seedslice == l, square(5))] = l
seeds_UA[segsliceno,:,:] = final_seedslice
writeh5(seeds_UA, datadir, dset_name + '_seeds_UA', element_size_um=elsize)

outpf = 'tmp'
# if UAsegfile:
#     outpf = outpf + UAsegfile
#     seeds_UA = seeds_neighbours(dset_info, seeds_UA, UAsegfile)
if UAknossosfile:
    outpf = outpf + UAknossosfile
#             seeds_UA = seeds_knossos(dset_info, seeds_UA, 'UA', outpf)
# for a single uninterupted skeleton, it should be one connected component (8-conn?)
# it might be good to create an immutable 'core' tube first and ws from there (interpolate knossos control points?)
    seeds_UA = seeds_knossos(dset_info, seeds_UA, 'UA', outpf, fill_edges=True, dilate_seeds=True)
# watershed
outpf = outpf + '_ws'
UA = watershed(-data, seeds_UA,
               mask=np.logical_and(datamask, ~np.logical_or(MM,MA)))
