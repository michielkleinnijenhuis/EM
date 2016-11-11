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
from skimage.measure import label, block_reduce
from scipy.ndimage.interpolation import shift
from scipy.ndimage.morphology import binary_fill_holes


def main(argv):

    parser = ArgumentParser(description='...')

    parser.add_argument('datadir', help='...')
    parser.add_argument('dset_name', help='...')
    parser.add_argument('-f', '--fieldnamein',
                        help='input hdf5 fieldname <stack>')
    parser.add_argument('-n', '--nzfills', type=int, default=5,
                        help='number of characters for section ranges')
    parser.add_argument('-e', '--element_size_um', type=float, nargs='*',
                        default=None,
                        help='dataset element sizes')
    parser.add_argument('-E', '--enforceECS', action='store_true')
    parser.add_argument('-o', '--zyxOffset', nargs=3, type=int, default=[],
                        help='...')
    parser.add_argument('-L', '--labelimages', default=[], nargs='+',
                        help='...')
    parser.add_argument('-r', '--rsfac', nargs=3, type=float,
                        default=(1, 1, 1),
                        help='...')
    parser.add_argument('-M', '--maskimages', default=None, nargs='+',
                        help='...')
    parser.add_argument('-x', default=0, type=int,
                        help='first x-index')
    parser.add_argument('-X', type=int,
                        help='last x-index')
    parser.add_argument('-y', default=0, type=int,
                        help='first y-index')
    parser.add_argument('-Y', type=int,
                        help='last y-index')
    parser.add_argument('-z', default=0, type=int,
                        help='first z-index')
    parser.add_argument('-Z', type=int,
                        help='last z-index')
    parser.add_argument('-b', '--blockoffset', nargs=3, type=int,
                        default=[0, 0, 0],
                        help='...')
    parser.add_argument('-d', '--blockreduce', nargs=3, type=int, default=[],
                        help='...')

    args = parser.parse_args()

    datadir = args.datadir
    dset_name = args.dset_name
    fieldnamein = args.fieldnamein
    nzfills = args.nzfills
    zyxOffset = args.zyxOffset
    labelimages = args.labelimages
    maskimages = args.maskimages
    enforceECS = args.enforceECS
    blockoffset = args.blockoffset
    blockreduce = args.blockreduce
    elsize = args.element_size_um

    dset_info, x, X, y, Y, z, Z = split_filename(dset_name, blockoffset)
    x = args.x
    X = args.X
    y = args.y
    Y = args.Y
    z = args.z
    Z = args.Z
    if not zyxOffset:
        zyxOffset = [z, y, x]

    surfname = 'dmcsurf'
    if blockreduce:
        surfname = surfname + '_%d-%d-%d' % (blockreduce[0],
                                             blockreduce[1],
                                             blockreduce[2])
    else:
        surfname = surfname + '_1-1-1'
    if enforceECS:
        surfname = surfname + '_enforceECS'
    surfdir = os.path.join(datadir, dset_name, surfname)
    mkdir_p(surfdir)

    # load the mask
    if maskimages:
        mask = loadh5(datadir, dset_name + maskimages[0], fieldnamein)[0]
        for m in maskimages[1:]:
            newmask = loadh5(datadir, dset_name + m, fieldnamein)[0]
            mask = mask | np.array(newmask, dtype='bool')
    else:
        mask = loadh5(datadir, dset_name + labelimages[0], fieldnamein)[0]
        mask = np.ones_like(mask, dtype='bool')
    if blockreduce:
        mask = block_reduce(mask, block_size=tuple(blockreduce), func=np.amax)

    ECSmask = np.zeros_like(mask, dtype='bool')
    # process the labelimages
    for l in labelimages:
        compdict = {}
        labeldata, elsize, al = loadh5(datadir, dset_name + l, fieldnamein)
        if blockreduce:
            labeldata = block_reduce(labeldata, block_size=tuple(blockreduce),
                                     func=np.amax)  # FIXME: reducefunc
            elsize = [e*b for e, b in zip(elsize, blockreduce)]
            zyxOffset = [o/b for o, b in zip(zyxOffset, blockreduce)]
        labeldata[~mask] = 0
#         labeldata, elsize = resample_volume(labeldata, True, elsize, res)
        # TODO: generalize
        if 'PA' in l:
            mask = np.logical_and(labeldata > 1000, labeldata < 2000)
            compdict['MM'] = np.unique(labeldata[mask])
            mask = np.logical_and(labeldata > 2000, labeldata < 6000)
            compdict['UA'] = np.unique(labeldata[mask])
            mask = np.logical_and(labeldata > 6000, labeldata < 7000)
            compdict['GB'] = np.unique(labeldata[mask])
            mask = np.logical_and(labeldata > 7000, labeldata < 8000)
            compdict['GP'] = np.unique(labeldata[mask])
        else:
            compdict['MA'] = np.unique(labeldata)
        labeldata = remove_small_objects(labeldata, 100)
        if enforceECS:
            labeldata = enforce_ECS(labeldata)
            writeh5(labeldata, datadir, dset_name + l + '_enforceECS',
                    element_size_um=elsize, axislabels=al)
        labels2meshes_vtk(surfdir, compdict, np.transpose(labeldata),
                          spacing=np.absolute(elsize)[::-1],
                          offset=zyxOffset[::-1])
        ECSmask[labeldata > 0] = True

    compdict['ECS'] = [1]
    binary_fill_holes(ECSmask, output=ECSmask)
    writeh5(~ECSmask, datadir, dset_name + '_ECSmask',
            element_size_um=elsize, axislabels=al)

    labels2meshes_vtk(surfdir, compdict, np.transpose(~ECSmask),
                      spacing=np.absolute(elsize)[::-1],
                      offset=zyxOffset[::-1])


# ========================================================================== #
# function defs
# ========================================================================== #


def reducefunc(block, axis=None):

    count = np.bincount(block.ravel())
    val = np.argmax(count, axis=axis)

    return np.array(val)


def dataset_name(dset_info):
    """Return the basename of the dataset."""

    nf = dset_info['nzfills']
    dname = dset_info['base'] + \
        '_' + str(dset_info['x']).zfill(nf) + \
        '-' + str(dset_info['X']).zfill(nf) + \
        '_' + str(dset_info['y']).zfill(nf) + \
        '-' + str(dset_info['Y']).zfill(nf) + \
        '_' + str(dset_info['z']).zfill(nf) + \
        '-' + str(dset_info['Z']).zfill(nf) + \
        dset_info['postfix']

    return dname


def split_filename(filename, blockoffset=[0, 0, 0]):
    """Extract the data indices from the filename."""

    datadir, tail = os.path.split(filename)
    fname = os.path.splitext(tail)[0]
    parts = fname.split("_")
    x = int(parts[1].split("-")[0]) - blockoffset[0]
    X = int(parts[1].split("-")[1]) - blockoffset[0]
    y = int(parts[2].split("-")[0]) - blockoffset[1]
    Y = int(parts[2].split("-")[1]) - blockoffset[1]
    z = int(parts[3].split("-")[0]) - blockoffset[2]
    Z = int(parts[3].split("-")[1]) - blockoffset[2]

    dset_info = {'datadir': datadir, 'base': parts[0],
                 'nzfills': len(parts[1].split("-")[0]),
                 'postfix': '_'.join(parts[4:]),
                 'x': x, 'X': X, 'y': y, 'Y': Y, 'z': z, 'Z': Z}

    return dset_info, x, X, y, Y, z, Z


def remove_small_objects(labeldata, minvoxelcount):
    """"""

    labeled, nlabels = label(labeldata, connectivity=1, return_num=True)
    x = np.bincount(np.ravel(labeled)) <= minvoxelcount
    forward_map = np.zeros(nlabels + 1, 'bool')
    forward_map[0:len(x)] = x
    tinysegments = forward_map[labeled]
    labeldata[tinysegments > 0] = 0

    return labeldata


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def loadh5(datadir, dname, fieldname='stack', dtype=None):
    """Load a h5 stack."""

    f = h5py.File(os.path.join(datadir, dname + '.h5'), 'r')

    if len(f[fieldname].shape) == 2:
        stack = f[fieldname][:, :]
    if len(f[fieldname].shape) == 3:
        stack = f[fieldname][:, :, :]
    if len(f[fieldname].shape) == 4:
        stack = f[fieldname][:, :, :, :]

    element_size_um, axislabels = get_h5_attributes(f[fieldname])

    f.close()

    if dtype is not None:
        stack = np.array(stack, dtype=dtype)

    return stack, element_size_um, axislabels


def writeh5(stack, datadir, fp_out, fieldname='stack',
            dtype='uint16', element_size_um=None, axislabels=None):
    """Write a h5 stack."""

    g = h5py.File(os.path.join(datadir, fp_out + '.h5'), 'w')
    g.create_dataset(fieldname, stack.shape, dtype=dtype, compression="gzip")

    if len(stack.shape) == 2:
        g[fieldname][:, :] = stack
    elif len(stack.shape) == 3:
        g[fieldname][:, :, :] = stack
    elif len(stack.shape) == 4:
        g[fieldname][:, :, :, :] = stack

    write_h5_attributes(g[fieldname], element_size_um, axislabels)

    g.close()


def get_h5_attributes(stack):
    """Get attributes from a stack."""

    element_size_um = axislabels = None

    if 'element_size_um' in stack.attrs.keys():
        element_size_um = stack.attrs['element_size_um']

    if 'DIMENSION_LABELS' in stack.attrs.keys():
        axislabels = stack.attrs['DIMENSION_LABELS']

    return element_size_um, axislabels


def write_h5_attributes(stack, element_size_um=None, axislabels=None):
    """Write attributes to a stack."""

    if element_size_um is not None:
        stack.attrs['element_size_um'] = element_size_um

    if axislabels is not None:
        for i, l in enumerate(axislabels):
            stack.dims[i].label = l


def enforce_ECS(labelimage, MA_labels=[]):
    """"""
    L = np.copy(labelimage)
    # for MA in MA_labels:
    #     L[labelimage==MA] = 0
    M = labelimage == 0
    # connc = [[i,j,k] for i in [-1,0,1] for j in [-1,0,1] for k in [-1,0,1]]
    connc = [[-1, -1, -1], [0, -1, -1], [1, -1, -1],
             [-1, 0, -1], [0, 0, -1], [1, 0, -1],
             [-1, 1, -1], [0, 1, -1], [1, 1, -1],
             [-1, -1, 0], [0, -1, 0], [1, -1, 0]]
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


def labels2meshes_vtk(surfdir, compdict, labelimage, labels=[],
                      spacing=[1, 1, 1], offset=[0, 0, 0], nvoxthr=0):
    """"""

    if not labels:
        labels = np.unique(labelimage)
        labels = np.delete(labels, 0)
        # labels = np.unique(labelimage[labelimage > 0])
    print('number of labels to process: ', len(labels))
    labelimage = np.lib.pad(labelimage.tolist(),
                            ((1, 1), (1, 1), (1, 1)), 'constant')
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
    # why swapaxes???
    for ii, val in enumerate(np.ravel(labelimage.swapaxes(0, 2))):
        sc.SetValue(ii, val)
    vol.GetPointData().SetScalars(sc)
    dmc = vtk.vtkDiscreteMarchingCubes()
    dmc.SetInput(vol)
    dmc.ComputeNormalsOn()

    for ii, label in enumerate(labels):
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
