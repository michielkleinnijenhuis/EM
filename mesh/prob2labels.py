#!/usr/bin/env python

"""
python prob2labels.py ...
"""

# TODO: apply mask for volume edges

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


def main(argv):
    
    parser = ArgumentParser(description='...')
    
    parser.add_argument('datadir', help='...')
    parser.add_argument('dataset', help='...')
    parser.add_argument('--SEsection', default='0250_m000_seg', help='...')
    parser.add_argument('--SEfile', help='...')
    parser.add_argument('--maskfile', help='...')
    parser.add_argument('--MAfile', help='...')
    parser.add_argument('--MAsegfile', help='...')
    parser.add_argument('--MAknossosfile', help='...')
    parser.add_argument('--MMfile', help='...')
    parser.add_argument('--UAfile', help='...')
    parser.add_argument('--UAsegfile', help='...')
    parser.add_argument('--UAknossosfile', help='...')
    parser.add_argument('--PAfile', help='...')
    parser.add_argument('-f', '--fillholes_MA', action='store_true')
    parser.add_argument('-w', '--sigmoidweighting_MM', action='store_true')
    parser.add_argument('-d', '--distancefilter_MM', action='store_true')
    parser.add_argument('-n', '--nzfills', type=int, default=5, 
                        help='number of characters for section ranges')
    parser.add_argument('-e', '--element_size_um', type=float, nargs='*', 
                        default=None, 
                        help='dataset element sizes (in order of outlayout)')
    parser.add_argument('-g', '--gsigma', nargs=3, type=float, 
                        default=[0.146, 1, 1], 
                        help='number of characters for section ranges')
    parser.add_argument('-o', '--segmOffset', nargs=3, type=int, 
                        default=[50, 60, 122], 
                        help='number of characters for section ranges')
    parser.add_argument('-s', '--fullshape', nargs=3, type=int, 
                        default=(430, 4460, 5217), 
                        help='number of characters for section ranges')
    parser.add_argument('-x', default=0, type=int, help='first x-index')
    parser.add_argument('-X', type=int, help='last x-index')
    parser.add_argument('-y', default=0, type=int, help='first y-index')
    parser.add_argument('-Y', type=int, help='last y-index')
    parser.add_argument('-z', default=0, type=int, help='first z-index')
    parser.add_argument('-Z', type=int, help='last z-index')
    
    args = parser.parse_args()
    
    datadir = args.datadir
    dataset = args.dataset
    SEsection = args.SEsection
    SEfile = args.SEfile
    maskfile = args.maskfile
    MAfile = args.MAfile
    MAsegfile = args.MAsegfile
    MAknossosfile = args.MAknossosfile
    MMfile = args.MMfile
    UAfile = args.UAfile
    UAsegfile = args.UAsegfile
    UAknossosfile = args.UAknossosfile
    PAfile = args.PAfile
    fillholes_MA = args.fillholes_MA
    sigmoidweighting_MM = args.sigmoidweighting_MM
    distancefilter_MM = args.distancefilter_MM
    nzfills = args.nzfills
    element_size_um = args.element_size_um
    gsigma = args.gsigma
    segmOffset = [o for o in args.segmOffset]
    fullshape = args.fullshape
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
    
    
    ### load (and smooth) the dataset
    data, elsize = loadh5(datadir, dset_name)
    # TODO: handle negative elsize here?
    if element_size_um is not None:
        elsize = element_size_um
    dset_info['elsize'] = elsize
#     if not UAfile:
    data = gaussian_filter(data, gsigma)
    
    ### load the probabilities and extract the myelin
#     if (not MAfile or not MMfile):
    prob_myel = loadh5(datadir, dset_name + '_probs0_eed2')[0]
    myelin = prob_myel > 0.2
        # TODO: more intelligent way to select myelin 
        # (e.g. remove mito first)
#         prob_mito = loadh5(datadir, dataset + '_probs2_eed2.h5')
#         myelin = np.logical_and(prob_myel > 0.2, prob_mito < 0.2)
    
    ### get a mask for invalid data ranges
    if maskfile:
        datamask = loadh5(datadir, dset_name + maskfile)[0]
    else:
        datamask = data != 0
        datamask = binary_dilation(binary_fill_holes(datamask))  # TODO
    
    ### get the single-section segmentation
    if SEfile:
        try:
            segm = loadh5(datadir, dset_name + SEfile)[0]
        except:
            segm = np.zeros(fullshape, dtype='uint16')
            segmOrig = loadh5(datadir, SEsection)[0]
            segm[segmOffset[0],
                 segmOffset[1]:segmOffset[1]+segmOrig.shape[0],
                 segmOffset[2]:segmOffset[2]+segmOrig.shape[1]] = segmOrig
            segm = segm[z:Z,y:Y,x:X]
            writeh5(segm, datadir, dset_name + '_seg', 
                    element_size_um=elsize)
    else:
        segm = np.zeros_like(data, dtype='uint16')
    segsliceno = segmOffset[0] - z
    
    
    ### get the myelinated axons (MA)
    if MAfile:
        MA = loadh5(datadir, dset_name + MAfile)[0]
    else:
        outpf = '_MA'
        # seedimage
        seeds_MA = np.copy(segm)
        if SEfile:
            outpf = outpf + SEfile
            seeds_MA = seeds_section(dset_info, seeds_MA, outpf)
        if MAsegfile:
            outpf = outpf + MAsegfile
            seeds_MA = seeds_neighbours(dset_info, seeds_MA, MAsegfile)
        if MAknossosfile:
            outpf = outpf + MAknossosfile
            seeds_MA = seeds_knossos(dset_info, seeds_MA, 'MA', outpf)
        if UAknossosfile:
            outpf = outpf + UAknossosfile
            seeds_MA = seeds_knossos(dset_info, seeds_MA, 'UA', outpf, 1)
        # watershed
        outpf = outpf + '_ws'
        MA = watershed(-data, seeds_MA, 
                       mask=np.logical_and(~myelin, datamask))
        MA[MA==1] = 0  # MA = remove_largest(MA)
        writeh5(MA, datadir, dset_name + outpf, element_size_um=elsize)
        # fill holes
        if fillholes_MA:
            outpf = outpf + '_filled'
            MA = fill_holes(MA)
            writeh5(MA, datadir, dset_name + outpf, element_size_um=elsize)
    myelin[MA != 0] = False
    
    ### watershed on myelin to separate individual sheaths
    if MMfile:
        MM = loadh5(datadir, dset_name + MMfile)[0]
    else:
        outpf = '_MM'
        # watershed on simple distance transform
        distance = distance_transform_edt(MA==0, sampling=np.absolute(elsize))
        outpf = outpf + '_ws'
        MM = watershed(distance, grey_dilation(MA, size=(3,3,3)), 
                       mask=np.logical_and(myelin, datamask))
        writeh5(MM, datadir, dset_name + outpf, element_size_um=elsize)
        # watershed on sigmoid-modulated distance transform
        if sigmoidweighting_MM:
            outpf = outpf + '_sw'
            distsum, lmask = sigmoid_weighted_distance(MM, MA, elsize)
            writeh5(distsum, datadir, dset_name + outpf + '_distsum', 
                    element_size_um=elsize, dtype='float')
            MM = watershed(distsum, grey_dilation(MA, size=(3,3,3)), 
                           mask=np.logical_and(myelin, datamask))
            writeh5(MM, datadir, dset_name + outpf, element_size_um=elsize)
        else:  # TODO simple distance th
            lmask = np.zeros((MM.shape[0], MM.shape[1], MM.shape[2], 
                      len(np.unique(MA)[1:])), dtype='bool')
        if distancefilter_MM:  # very mem-intensive
            outpf = outpf + '_df'
            for i,l in enumerate(np.unique(MA)[1:]):
                MM[np.logical_and(lmask[:,:,:,i], MM==l)] = 0
            writeh5(MM, datadir, dset_name + outpf, element_size_um=elsize)
    
    ### watershed the unmyelinated axons (almost identical to MA)
    if UAfile:
        UA = loadh5(datadir, dset_name + UAfile)[0]
    else:
        outpf = '_UA'
        # seedimage
        seeds_UA = np.copy(segm)
        if SEfile:
            seedslice = seeds_UA[segsliceno,:,:]
            seedslice[seedslice<2000] = 0
            final_seedslice = np.zeros_like(seedslice)
            for l in np.unique(seedslice)[1:]:
                final_seedslice[binary_erosion(seedslice == l, square(5))] = l
            seeds_UA[segsliceno,:,:] = final_seedslice
            writeh5(seeds_UA, datadir, dset_name + '_seeds_UA', element_size_um=elsize)
        if UAsegfile:
            outpf = outpf + UAsegfile
            seeds_UA = seeds_neighbours(dset_info, seeds_UA, UAsegfile)
        if UAknossosfile:
            outpf = outpf + UAknossosfile
            seeds_UA = seeds_knossos(dset_info, seeds_UA, 'UA', outpf)
        # watershed
        outpf = outpf + '_ws'
        UA = watershed(-data, seeds_UA, 
                       mask=np.logical_and(datamask, ~np.logical_or(MM,MA)))
        # save results
        writeh5(UA, datadir, dset_name + outpf, element_size_um=elsize)
    
    ### combine the MM, MA and UA segmentations
    if not PAfile:
        outpf = '_PA'
        writeh5(MA+MM+UA, datadir, dset_name + outpf, element_size_um=elsize)

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

def seeds_section(dset_info, seeds, write_pf='_sMAse'):
    """"""
    dset_name = dataset_name(dset_info)
    
    # include a seed here for the ECS, i.e 
    # set all non-MA, non-MM to 1
    seeds[seeds<1000] = 0
    seeds[seeds>2000] = 1
    
    writeh5(seeds, dset_info['datadir'], dset_name + write_pf, 
            element_size_um=dset_info['elsize'])
    
    return seeds

def seeds_neighbours(dset_info, seeds, segfile, 
                     load_pf='_sMAse', 
                     write_pf='_sMAnb'):
    """"""
    dset_name = dataset_name(dset_info)
    
    seeds = loadh5(dset_info['datadir'], dset_name + load_pf)[0]
    
    neighbours = [[-1000,0,0], [1000,0,0], [0,-1000,0], [0,1000,0]]
    nb_name_info = dset_info.copy()
    
    for i, side in enumerate(neighbours):
        nb_name_info['x'] = dset_info['x'] + side[0]
        nb_name_info['X'] = dset_info['X'] + side[0]
        nb_name_info['y'] = dset_info['y'] + side[1]
        nb_name_info['Y'] = dset_info['Y'] + side[1]
        nb_name_info['z'] = dset_info['z'] + side[2]
        nb_name_info['Z'] = dset_info['Z'] + side[2]
        nb_name = dataset_name(nb_name_info)
        try:
            sidesection = loadh5(dset_info['datadir'], nb_name + segfile)[0]
            if i == 0:
                seeds[:,:,0] = sidesection[:,:,-1]  # 'ITKsnap-right'
            elif i == 1:
                seeds[:,:,-1] = sidesection[:,:,0]  # 'ITKsnap-left'
            elif i == 2:
                seeds[:,0,:] = sidesection[:,-1,:]  # 'ITKsnap-anterior'
            elif i == 3:
                seeds[:,-1,:] = sidesection[:,0,:]  # 'ITKsnap-posterior'
        except:
            pass
    
    writeh5(seeds, dset_info['datadir'], dset_name + write_pf, 
            element_size_um=dset_info['elsize'])
    
    return seeds

def seeds_knossos(dset_info, seeds, comp, write_pf, fixval=None):
    """"""
    dset_name = dataset_name(dset_info)
    
    annotationfile = os.path.join(dset_info['datadir'], 
                                  dset_name + '_knossos', 
                                  'annotation_' + comp + '.xml')
    objs = get_knossos_controlpoints(annotationfile)
    for obj in objs:
        print(obj['name'])
        if fixval:
            objval = fixval
        else:
            objval = int(obj['name'][3:7])
        for _, coords in obj['nodedict'].iteritems():
            # knossos-coords are in 1-based, yxz-order, 1000x1000x430 frame
            # (for m000_01000-02000_01000-02000_00030-00460.h5)q
            knossosoffset = [1000,1000,30]  # (yxz)  # FIXME!!!
            # index into the main dataset (5217x4460x460)
            coord_x = coords[1] + knossosoffset[1] - 1 # - dset_info['x']
            coord_y = coords[0] + knossosoffset[0] - 1 # - dset_info['y']
            coord_z = coords[2] + knossosoffset[2] - 1 # - dset_info['z']
            # index into loaded subset
            coord_x = coord_x - dset_info['x']
            coord_y = coord_y - dset_info['y']
            coord_z = coord_z - dset_info['z']  # - 100  # FIXME!!!
#             print(coords, [coord_z,coord_y,coord_x])
            try:
                seeds[coord_z,coord_y,coord_x] = objval
            except:
                pass
#                 print([coord_z,coord_y,coord_x], 'in ', str(objval), ' out of range')
    
    writeh5(seeds, dset_info['datadir'], dset_name + write_pf, 
            element_size_um=dset_info['elsize'])
    
    return seeds


def remove_largest(MA):
    """"""
    bc = np.bincount(np.ravel(MA))
    largest_label = bc[1:].argmax() + 1
    MA[MA==largest_label] = 0
    print("largest label {!s} was removed".format(largest_label))
    
    return MA

def fill_holes(MA):
    """"""
    for l in np.unique(MA)[1:]:
        # fill holes
        labels = label(MA!=l)[0]
        labelCount = np.bincount(labels.ravel())
        background = np.argmax(labelCount)
        MA[labels != background] = l
        # closing
        binim = MA==l
        binim = binary_closing(binim, iterations=10)
        MA[binim] = l
        # fill holes
        labels = label(MA!=l)[0]
        labelCount = np.bincount(labels.ravel())
        background = np.argmax(labelCount)
        MA[labels != background] = l
    
    return MA

def get_knossos_controlpoints(annotationfile):
    """"""
    things = xml.etree.ElementTree.parse(annotationfile).getroot()
    objs = []
    for thing in things.findall('thing'):
        obj = {}
        obj['name'] = thing.get('comment')
        # nodes
        nodedict = {}
        nodes = thing.findall('nodes')[0]
        for node in nodes.findall('node'):
            coord = [int(node.get(ax)) for ax in 'xyz']
            nodedict[node.get('id')] = coord
        obj['nodedict'] = nodedict
        # edges
        edgelist = []
        edges = thing.findall('edges')[0]
        for edge in edges.findall('edge'):
            edgelist.append([edge.get('source'),edge.get('target')])
        obj['edgelist'] = edgelist
        # polylines
        # add the nodes in nodedict to the seedpoints
        # add the voxels intersected by each edge to the seedpoints
        # TODO: handle splits and doubles
        # NOTE: x and y are swapped in knossos; knossos is in xyz coordframe
        # NOTE: knossos has a base-1 coordinate system
        # append
        objs.append(obj)
    
    return objs

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
    main(sys.argv)
