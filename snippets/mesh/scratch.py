import os
import sys
from argparse import ArgumentParser

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

def knossoscoord2dataset(dset_info, coords, knossosoffset=[1000,1000,30]):
    # knossos-coords are in 1-based, yxz-order, 1000x1000x430 frame
    # (for m000_01000-02000_01000-02000_00030-00460.h5)q
    # index into the main dataset (5217x4460x460)
    coord_x = coords[1] + knossosoffset[1] - 1 # - dset_info['x']
    coord_y = coords[0] + knossosoffset[0] - 1 # - dset_info['y']
    coord_z = coords[2] + knossosoffset[2] - 1 # - dset_info['z']
    # index into loaded subset
    coord_x = coord_x - dset_info['x']
    coord_y = coord_y - dset_info['y']
    coord_z = coord_z - dset_info['z']  # - 100  # FIXME!!!
#     print(coords, [coord_z,coord_y,coord_x])
    return (coord_z, coord_y, coord_x)


dset_info = {'datadir': '/Users/michielk/oxdata/P01/EM/M3/M3_S1_GNU', 
             'base': 'm000', 
             'nzfills': 5, 
             'postfix': '', 
             'x': 1000, 'X': 1500, 
             'y': 1000, 'Y': 1500, 
             'z': 200, 'Z': 300}
comp = 'MA'
dset_name = dataset_name(dset_info)
annotationfile = os.path.join(dset_info['datadir'], 
                              dset_name + '_knossos', 
                              'annotation_' + comp + '.xml')
objs = get_knossos_controlpoints(annotationfile)
edges = 1

obj = objs[1]
edge = obj['edgelist'][0]


points = []
for _, coords in obj['nodedict'].iteritems():
    points.append(knossoscoord2dataset(dset_info, coords))

for edge in obj['edgelist']:
    point0 = knossoscoord2dataset(dset_info, obj['nodedict'][edge[0]])
    point1 = knossoscoord2dataset(dset_info, obj['nodedict'][edge[1]])
    points_z = np.linspace(point0[0], point1[0], 100).astype(int)
    points_y = np.linspace(point0[1], point1[1], 100).astype(int)
    points_x = np.linspace(point0[2], point1[2], 100).astype(int)
    p = [[z,y,x] for z,y,x in zip(points_z, points_y, points_x)]
    points = points + p














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
        if edges:
            pass
        else:
            for _, coords in obj['nodedict'].iteritems():
                coords = knossoscoord2dataset(dset_info, coords, 
                                              knossosoffset=[1000,1000,30])
                try:
                    seeds[coords[0],coords[1],coords[2]] = objval
                except:
                    pass
    #                 print([coord_z,coord_y,coord_x], 'in ', str(objval), ' out of range')
    
    writeh5(seeds, dset_info['datadir'], dset_name + write_pf, 
            element_size_um=dset_info['elsize'])
    
    return seeds

