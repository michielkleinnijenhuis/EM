#!/usr/bin/env python

"""
python label2stl.py ...
"""

import os
import sys
from argparse import ArgumentParser
import errno
import h5py
import numpy as np
import vtk
from skimage.morphology import remove_small_objects
from skimage.measure import label
from scipy.ndimage.interpolation import shift
from scipy.ndimage.morphology import binary_fill_holes



def main(argv):
    
    parser = ArgumentParser(description='...')
    
    parser.add_argument('datadir', help='...')
    parser.add_argument('dataset', help='...')
    parser.add_argument('-f', '--fieldnamein', 
                        help='input hdf5 fieldname <stack>')
    parser.add_argument('-n', '--nzfills', type=int, default=5, 
                        help='number of characters for section ranges')
    parser.add_argument('-e', '--element_size_um', type=float, nargs='*', default=None, 
                        help='dataset element sizes')
    parser.add_argument('-E', '--enforceECS', action='store_true')
    parser.add_argument('-o', '--zyxOffset', nargs=3, type=int, default=[0,0,0], help='...')
    parser.add_argument('-L', '--labelimages', default=[], nargs='+', help='...')
    parser.add_argument('-r', '--rsfac', default=(1,1,1), type=float, nargs=3, help='...')
    parser.add_argument('-M', '--maskimages', default=None, nargs='+', help='...')
    parser.add_argument('-x', default=0, type=int, help='first x-index')
    parser.add_argument('-X', type=int, help='last x-index')
    parser.add_argument('-y', default=0, type=int, help='first y-index')
    parser.add_argument('-Y', type=int, help='last y-index')
    parser.add_argument('-z', default=0, type=int, help='first z-index')
    parser.add_argument('-Z', type=int, help='last z-index')
    
    args = parser.parse_args()
    
    datadir = args.datadir
    dataset = args.dataset
    fieldnamein = args.fieldnamein
    nzfills = args.nzfills
    zyxOffset = args.zyxOffset
    labelimages = args.labelimages
    maskimages = args.maskimages
    enforceECS = args.enforceECS
    
    x = args.x
    X = args.X
    y = args.y
    Y = args.Y
    z = args.z
    Z = args.Z
    dataset = dataset + '_' + str(x).zfill(nzfills) + '-' + str(X).zfill(nzfills) + \
                        '_' + str(y).zfill(nzfills) + '-' + str(Y).zfill(nzfills) + \
                        '_' + str(z).zfill(nzfills) + '-' + str(Z).zfill(nzfills)
    
    surfname = 'dmcsurf'
    if enforceECS:
        surfname = surfname + '_enforceECS'
    surfdir = os.path.join(datadir, dataset, surfname)
    mkdir_p(surfdir)
    
    ### load the mask
    if maskimages:
        mask = loadh5(datadir, dataset + maskimages[0], fieldnamein)[0]
        for m in maskimages[1:]:
            newmask = loadh5(datadir, dataset + m, fieldnamein)[0]
            mask = mask | np.array(newmask, dtype='bool')
    else:
        mask, _ = loadh5(datadir, dataset + labelimages[0], fieldnamein)
        mask = np.ones_like(mask, dtype='bool')
    
    ECSmask = np.zeros_like(mask, dtype='bool')
    ### process the labelimages
    for l in labelimages:
        compdict = {}
        labeldata, elsize = loadh5(datadir, dataset + l, fieldnamein)
        labeldata[~mask] = 0
#         labeldata, elsize = resample_volume(labeldata, True, elsize, res)
        if 'PA' in l:
            compdict['MM'] = np.unique(labeldata[np.logical_and(labeldata>1000, labeldata<2000)])
            compdict['UA'] = np.unique(labeldata[np.logical_and(labeldata>2000, labeldata<6000)])
            compdict['GB'] = np.unique(labeldata[np.logical_and(labeldata>6000, labeldata<7000)])
            compdict['GP'] = np.unique(labeldata[np.logical_and(labeldata>7000, labeldata<8000)])
        else:
            compdict['MA'] = np.unique(labeldata)
        labeldata = remove_small_objects(labeldata, 100)
        if enforceECS:
            labeldata = enforce_ECS(labeldata)
            writeh5(labeldata, datadir, dataset + l + '_enforceECS', 
                    element_size_um=elsize)
        labels2meshes_vtk(surfdir, compdict, np.transpose(labeldata), 
                          spacing=np.absolute(elsize)[::-1], offset=zyxOffset[::-1])
        ECSmask[labeldata>0] = True
    
    compdict['ECS'] = [1]
    binary_fill_holes(ECSmask, output=ECSmask)
    writeh5(~ECSmask, datadir, dataset + '_ECSmask', 
            element_size_um=elsize)
    
    labels2meshes_vtk(surfdir, compdict, np.transpose(~ECSmask), 
                      spacing=np.absolute(elsize)[::-1], 
                      offset=zyxOffset[::-1])


def remove_small_objects(labeldata, minvoxelcount):
    """"""
#     remove_small_objects(L[lc], min_size=minvoxelcount, connectivity=1, in_place=True)
    labeled, nlabels = label(labeldata, connectivity=1, return_num=True)
    x = np.bincount(np.ravel(labeled)) <= minvoxelcount
    forward_map = np.zeros(nlabels + 1, 'bool')
    forward_map[0:len(x)] = x
    tinysegments = forward_map[labeled]
    labeldata[tinysegments>0] = 0
    
    return labeldata

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc: # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else: raise

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

def enforce_ECS(labelimage, MA_labels=[]):
    """"""
    L = np.copy(labelimage)
    # for MA in MA_labels:
    #     L[labelimage==MA] = 0
    M = labelimage==0
    # connc = [[i,j,k] for i in [-1,0,1] for j in [-1,0,1] for k in [-1,0,1]]
    connc = [[-1,-1,-1],[ 0,-1,-1],[ 1,-1,-1],[-1, 0,-1],[ 0, 0,-1],[ 1, 0,-1],
             [-1, 1,-1],[ 0, 1,-1],[ 1, 1,-1],[-1,-1, 0],[ 0,-1, 0],[ 1,-1, 0]]
    # connc = connc[:12]  # TODO: select only the 'half'-cube
    for volshift in connc:
        K = shift(L, volshift, order=0)
        M = np.logical_or(M,
                          np.logical_and(L != K,
                                         np.logical_and(L != 0,
                                                        K != 0)))
    L[M] = 0
    # for MA in MA_labels:
    #     L[labelimage==MA] = MA
    return L

def labels2meshes_vtk(surfdir, compdict, labelimage, labels=[], spacing=[1,1,1], offset=[0,0,0], nvoxthr=0):
    """"""
    if not labels:
        labels = np.unique(labelimage)
        labels = np.delete(labels, 0)  # labels = np.unique(labelimage[labelimage>0])
    print('number of labels to process: ', len(labels))
    labelimage = np.lib.pad(labelimage.tolist(), ((1, 1), (1, 1), (1, 1)), 'constant')
    dims = labelimage.shape
    
    vol = vtk.vtkImageData()
    vol.SetDimensions(dims[0], dims[1], dims[2])
    vol.SetOrigin(offset[0]*spacing[0] + spacing[0],
                  offset[1]*spacing[1] + spacing[1],
                  offset[2]*spacing[2] + spacing[2])  # vol.SetOrigin(0, 0, 0)
    vol.SetSpacing(spacing[0], spacing[1], spacing[2])
    sc = vtk.vtkFloatArray()
    sc.SetNumberOfValues(labelimage.size)
    sc.SetNumberOfComponents(1)
    sc.SetName('tnf')
    for ii,val in enumerate(np.ravel(labelimage.swapaxes(0,2))):  # why swapaxes???
        sc.SetValue(ii,val)
    vol.GetPointData().SetScalars(sc)
    dmc = vtk.vtkDiscreteMarchingCubes()
    dmc.SetInput(vol)
    dmc.ComputeNormalsOn()
    
    for ii,label in enumerate(labels):
        for labelclass, labels in compdict.items():
            if label in labels:
                break
            else:
                labelclass = 'NN'
        ndepth = 1
        fpath = os.path.join(surfdir, labelclass + 
                             '.{:05d}.{:02d}.stl'.format(label, ndepth))
        print("Processing labelnr " + str(ii) + 
              " of class " + labelclass + 
              " with value: " + str(label))
#         print("Saving to " + fpath)
        
        dmc.SetValue(0, label)
        # dmc.GenerateValues(nb_labels, 0, nb_labels)
        dmc.Update()
        writer = vtk.vtkSTLWriter()
        writer.SetInputConnection(dmc.GetOutputPort())
        writer.SetFileName(fpath)
        writer.Write()


if __name__ == "__main__":
    main(sys.argv)
