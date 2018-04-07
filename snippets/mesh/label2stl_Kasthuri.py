#!/usr/bin/env python

"""
python label2stl.py ...
"""

import os
import sys
from argparse import ArgumentParser
import errno
import h5py
from mpi4py import MPI
import numpy as np
# import nibabel as nib
import vtk
import stl
from skimage.morphology import binary_closing, watershed, remove_small_objects
from skimage.measure import marching_cubes, correct_mesh_orientation, label
from skimage.transform import downscale_local_mean
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import shift, zoom
# from scipy.ndimage.morphology import generate_binary_structure
# from scipy.misc import imresize

# TODO:     # assign parent label to nested objects???
# NOTE: in catmaid x=0;y=0; is in the top-left corner; this checks out with writing to hdf5 and viewing in Fiji
# NOTE: for the nifti's viewed in itksnap, this is the rightbottom corner (LPI) => negate x and y to display as RAI
# NOTE: for the exported stl's via VTK, the axes should be numpy-transposed (i.e. reversed)


def main(argv):
    
    parser = ArgumentParser(description='...')
    
    parser.add_argument('datadir', help='...')
    parser.add_argument('-S', '--scale', default=3, help='...')
    parser.add_argument('-s', '--zyxSpacing', default=(1,1,1), type=float, nargs=3, help='...')
    parser.add_argument('-o', '--zyxOffset', default=(0,0,0), type=int, nargs=3, help='...')
    parser.add_argument('-l', '--zyxLength', default=(100,100,100), type=int, nargs=3, help='...')
    parser.add_argument('-r', '--rsfac', default=(1,1,1), type=float, nargs=3, help='...')
    parser.add_argument('-L', '--labelimages', default=['segments'], nargs='+', help='...')
    parser.add_argument('-M', '--maskimages', default=['mask'], nargs='+', help='...')
    parser.add_argument('-w', '--ws', action='store_true', 
                        help='use watershed to fill volume')
    parser.add_argument('-m', '--usempi', action='store_true', 
                        help='use mpi4py')
    
    args = parser.parse_args()
    
    datadir = args.datadir
    scale = args.scale
#     cutout = args.cutout
    zyxSpacingOrig = args.zyxSpacing
    zyxOffset = args.zyxOffset
    zyxLength = args.zyxLength
    rsfac = args.rsfac
    labelimages = args.labelimages
    maskimages = args.maskimages
    ws = args.ws
    usempi = args.usempi
    
    cu_scale = '-{0}-'.format(scale)
    cu = [(o,zyxLength[i]) for i,o in enumerate(zyxOffset)]
    cus = ['{0}_{1}-'.format(o, o + l) for o,l in cu]
    cutout = '-default-hdf5' + cu_scale + cus[2] + cus[1] + cus[0] + 'ocpcutout.h5'
    
    res = {}
    if all(np.array(rsfac)==1):
        res = {'op': 'orig', 'rs': rsfac}
    elif any(np.array(rsfac)<1):
        res = {'op': 'us', 'rs': (1/rsfac[0],1/rsfac[1],1/rsfac[1])}
    elif any(np.array(rsfac)>1):
        res = {'op': 'ds', 'rs': rsfac}
    
    
    
    ### load the data
    data = loadh5(datadir, 'kasthuri11cc' + cutout, '/default/CUTOUT')
    data, zyxSpacing = resample_volume(data, False, zyxSpacingOrig, res)
    writeh5(data, datadir, 'data_' + res['op'] + '.h5', dtype='uint8')
    data_smooth = gaussian_filter(data, [zyxSpacing[2]/zyxSpacing[0], 1, 1])
    writeh5(data_smooth, datadir, 'data_smooth_' + res['op'] + '.h5', dtype='uint8')
    
    ### load the mask
    mask = np.zeros_like(data, dtype='bool')
    for m in maskimages:
        newmask = loadh5(datadir, 'kat11' + m + cutout, '/default/CUTOUT')
        mask = mask | np.array(newmask, dtype='bool')
    mask, _ = resample_volume(mask, True, zyxSpacingOrig, res)
    
    ### process the labelimages
    for l in labelimages:
        labeldata = loadh5(datadir, 'kat11' + l + cutout, '/default/CUTOUT')
        labeldata[~mask] = 0
        labeldata, _ = resample_volume(labeldata, True, zyxSpacingOrig, res)
        writeh5(labeldata, datadir, l + '_' + res['op'] + '.h5', dtype='uint32')
        
        compdict = {}
        if l == 'segments':
            ### load the label-to-compartment mappings
            compdict['DD'] = np.loadtxt(os.path.join(datadir, os.pardir, 'dendrites.txt'), dtype='int')
            compdict['UA'] = np.loadtxt(os.path.join(datadir, os.pardir, 'axons.txt'), dtype='int')
            compdict['MA'] = np.loadtxt(os.path.join(datadir, os.pardir, 'MA.txt'), dtype='int')
            compdict['MM'] = np.loadtxt(os.path.join(datadir, os.pardir, 'MM.txt'), dtype='int')
            
            labeldata = remove_small_objects(labeldata, 100)
            
            L = {}
            ### separate the nested components and fill holes with parent label
            # define the object hierarchies  # TODO: automate detection of parent/child
            # NOTE: MA not fully encapsulated by MM; TODO: does this cause problems for DifSim?
            # NOTE: 6640 is myelinated but parent is not segmented; TODO: adapt segmentation
            # NOTE: still one myelinated axon missing (the small one running along the green central dendrite) (count is 7 in the videos)
            nested_labels_MM_MA = [{'parent': 2064, 'children': [4247]}, 
                                   {'parent': 4387, 'children': [4143]}, 
                                   {'parent': 5004, 'children': [4142]}, 
                                   {'parent': 5006, 'children': [3962]}, 
                                   {'parent': 5105, 'children': [4249]}]
            L['segments'], L['MA'] = dissolve_nesting(labeldata, nested_labels_MM_MA)  # first level
            # NOTE: it might be better to create seperate images for top level objects and nested objects 
            # and then infer a single object number from the overlap of the nested objects
            # top-level objects are: DD, UA, MM; nested objects are: mito, vesicles, MA; TODO: what are sysnapses???
            # TODO: check if nested objects are fully contained in a single object 
            # (and do not touch boundary; and do not extend into ECS) (e.g., not the case for synapses)
            
            ### watershed the segments to label every voxel in the volume
            if ws:
                L['segments'] = watershed(-data_smooth, L['segments'])
                writeh5(L['segments'], datadir, 'segments_ws_' + res['op'] + '.h5')
                L_ECS = enforce_ECS(L['segments'])  # TODO: create flag to control this
                writeh5(L_ECS, datadir, 'L_ECS_' + res['op'] + '.h5')
            
            # create the new labelclasses from 'segments' and remove the old one
            # NB better to keep them as the parent classes as this saves processing in vtk-meshing
            # NB not for distributed processing
            ### update the compartment class volume
    #         Lclass = np.zeros_like(data, dtype='S2')
    #         Lclass.fill('NN')
    #         for labelclass, labels in compdict.items():
    #             for label in labels:
    #                 Lclass[L['segments']==label] = labelclass
    #         newlabels = ['NN', 'DD', 'UA', 'MM']
    #         for nl in newlabels:
    #             L[nl] = np.copy(L['segments'])
    #             L[nl][Lclass!=nl] = 0
    #         L.pop("segments", None)
            for _, labeldata in L.items():
                labels2meshes_vtk(datadir, compdict, np.transpose(labeldata), 
                                  spacing=zyxSpacing[::-1], offset=zyxOffset[::-1])
        else:
            compdict[l] = np.unique(labeldata)
            labeldata = remove_small_objects(labeldata, 100)
            labels2meshes_vtk(datadir, compdict, np.transpose(labeldata), 
                              spacing=zyxSpacing[::-1], offset=zyxOffset[::-1])



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

def resample_volume(data, labelflag, zyxSpacing, resdict):
    ### preprocess images (i.e., choose a resolution)
    res = resdict['op']
    rs = resdict['rs']
    if res == 'orig':  # orig
        zyxSpac = zyxSpacing
        
#         data_smooth = gaussian_filter(data, sigma)
    elif res == 'ds':  # downscale x,y  # NOTE: vesicles have higher res segmentations???
        zyxSpac = [zyxSpacing[0]*rs[0],zyxSpacing[1]*rs[1],zyxSpacing[2]*rs[2]]
        sigma = [zyxSpac[2]/zyxSpac[0], 1, 1]
        if labelflag:
            data = data[::rs[0],::rs[1],::rs[2]]
        else:
            data = downscale_local_mean(data, rs)
#             data = gaussian_filter(data, sigma)
#         for l in labeldict.keys():
#             labeldict[l] = labeldict[l][::rs[0],::rs[1],::rs[2]]
    elif res == 'us':  # upscale z
        zyxSpac = [zyxSpacing[0]/rs[0],zyxSpacing[1]/rs[1],zyxSpacing[2]/rs[2]]
        sigma = [1,1,1]
        if labelflag:
            data = zoom(data, rs, order=0)
        else:
            data = zoom(data, rs)
#         for l in labeldict.keys():
#             labeldict[l] = zoom(labeldict[l], rs, order=0)
    
    return data, zyxSpac

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc: # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else: raise

def loadh5(datadir, dname, fieldname='stack'):
    """"""
    f = h5py.File(os.path.join(datadir, dname), 'r')
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

def dissolve_nesting(labelimage, nested_labels):
    """"""
    labelimage_parent = np.copy(labelimage)
    labelimage_child = np.zeros_like(labelimage, dtype='uint32')
    for n in nested_labels:
        for c in n['children']:
            labelimage_child[labelimage_parent==c] = c
            labelimage_parent[labelimage_parent==c] = n['parent']
    return labelimage_parent, labelimage_child

def labels2meshes_ski(labelimage, labels, spacing=[0,0,0], nvoxthr=0): # deprecated and underdeveloped: use vtk implementation instead
    """"""
    for label in labels:
        labelmask = labelimage == label
        if np.count_nonzero(labelmask) > nvoxthr:
            labelmask = binary_closing(labelmask)
            verts, faces = marching_cubes(labelmask, 0, spacing=spacing)
            faces = correct_mesh_orientation(labelmask, verts, faces, spacing=spacing, gradient_direction='descent')
            # Fancy indexing to define two vector arrays from triangle vertices
            actual_verts = verts[faces]
            a = actual_verts[:, 0, :] - actual_verts[:, 1, :]
            b = actual_verts[:, 0, :] - actual_verts[:, 2, :]
            # Find normal vectors for each face via cross product
            crosses = np.cross(a, b)
            normals = crosses / (np.sum(crosses ** 2, axis=1) ** (0.5))[:, np.newaxis]
            ob = stl.Solid(name=label)
            for ii, _ in enumerate(faces):
                ob.add_facet(normals[ii], actual_verts[ii])
            with open(str(label) + ".stl", 'w') as f:  #with open("allobjects.stl", 'a') as f:
                ob.write_ascii(f)  #ob.write_binary(f)
                f.write("\n");

def labels2meshes_vtk(datadir, compdict, labelimage, labels=[], spacing=[1,1,1], offset=[0,0,0], nvoxthr=0):
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
    
    surfdir = 'dmcsurf'
    mkdir_p(os.path.join(datadir, surfdir))
    
    for ii,label in enumerate(labels):
        for labelclass, labels in compdict.items():
            if label in labels:
                break
            else:
                labelclass = 'NN'
        ndepth = 1
        fpath = os.path.join(datadir, surfdir,
                             labelclass + 
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
