#!/usr/bin/env python

"""Vizualize results with NeuroBlender.

"""

import sys
import argparse
import os
from random import shuffle

import numpy as np

# from skimage.segmentation import relabel_sequential

from wmem import parse, utils, LabelImage

try:
    import bpy
except ImportError:
    print("bpy could not be loaded")


def main(argv):
    """Vizualize results with NeuroBlender."""

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
    parser = parse.parse_neuroblender(parser)
    parser = parse.parse_common(parser)
    args = parser.parse_args()

    neuroblender(
        args.inputfile,
        args.outputfile,
        args.save_steps,
        args.protective,
        )


def neuroblender(
        h5path_in,
        h5path_out='',
        save_steps=False,
        protective=False,
        ):
    """Vizualize results with NeuroBlender."""


def shuffle_labels(ulabels):
    """Shuffle labels to a random order."""

    fw = np.array([l if l in ulabels else 0
                   for l in range(0, max(ulabels) + 1)])
    mask = fw != 0
    fw_nz = fw[mask]
    shuffle(fw_nz)
    fw[mask] = fw_nz

    return fw


def get_shuffled_fwmap_for_volumeset(vols):

    imgs = []
    for vol in vols:
        imgs.append(vol2im(vol, load_data=True))
    ulabels = set([])

    for im in imgs:
        ulabels = ulabels | set(im.ulabels)
        im.close()

    fw = shuffle_labels(ulabels)

    # # TODO: handle possible missing/additional labels etc
    # # TODO: datarange has to be equal too
    # # TODO: check if fwmap is valid for the vol
    # # check if maxlabel is equal
    # if im.maxlabel > len(fw):
    #     print('warning: labelsets not equal (more: appending to forward map)')
    #     # TODO: append
    # elif im.maxlabel < len(fw):
    #     print('warning: labelsets not equal (less: going ahead with forward map)')
    # # use sets to compare labelsets
    # s = set(im.ulabels)
    # t = set(np.where(fw != 0)[0])
    # # setdiff = ulabels ^ fwlabels  # s - t  # t - s
    # print('warning: labelsets not equal')
    # print(s - t, t - s)

    return fw


def quickrigid(im, translations=[0, 0, 0]):
    affine = np.eye(4)
    affine[0][0] = im.elsize[2]
    affine[1][1] = im.elsize[1]
    affine[2][2] = im.elsize[0]
    affine[0][3] = translations[0]
    affine[1][3] = translations[1]
    affine[2][3] = translations[2]
    return affine


def normalize_data(data, datamin=None, datamax= None):
    """Normalize data between 0 and 1."""

    data = data.astype('float64')
    if datamin is None:
        datamin = np.amin(data)
    if datamax is None:
        datamax = np.amax(data)
    data -= datamin
    data *= 1/(datamax-datamin)

    return data, [datamin, datamax]


def vol2im(vol, load_data=False):

    fname = '{}{}.h5'.format(vol['dataset'], vol['ipf'])
    inputfile = os.path.join(vol['datadir'], '{}'.format(fname))
    inputpath = '{}/{}'.format(fname, vol['ids'])
    im = utils.get_image(inputpath, imtype=vol['imtype'], load_data=load_data)

    return im


def load_fw_from_vol(vol):

    im = vol2im(vol, load_data=False)
    comps = im.split_path()
    im.close()
    filename = '{}.h5'.format(comps['fname'])
    outputpath = os.path.join(comps['dir'], 'blender', filename, comps['int'][1:])
    fw = np.load(os.path.join(outputpath, 'fwmap.npy'))

    return fw


def save_affine(vol, translations=[0, 0, 0]):

    im = vol2im(vol)
    texdict = {'affine': quickrigid(im, translations=translations)}
    comps = im.split_path()
    im.close()
    filename = '{}.h5'.format(comps['fname'])
    outputpath = os.path.join(comps['dir'], 'blender', filename, comps['int'][1:])
    np.save(os.path.join(outputpath, 'affine'), np.array(texdict['affine']))


def get_datarange_from_vol(vol):

    im = vol2im(vol, load_data=True)
    datarange = [np.amin(im.ds[:]), np.amax(im.ds[:])]
    if vol['imtype'] == 'Label':
        ulabels = im.ulabels
    else:
        ulabels = None
    im.close()

    return datarange, ulabels


def create_neuroblender_texture(vol, fw=[], relabel=False,
                                translations=[0, 0, 0],
                                zslices=0):

    # read input
    im = vol2im(vol)

    # prep output
    comps = im.split_path()
    props = im.get_props(protective=False)
    fname = '{}.h5'.format(comps['fname'])
    outputpath = os.path.join(comps['dir'], 'blender', fname, comps['int'][1:])
    datapath = os.path.join(outputpath, 'IMAGE_SEQUENCE', 'vol0000')
    utils.mkdir_p(datapath)
    mo = LabelImage(datapath, **props)
    mo.format = '.tifs'
    mo.create()

    zslices = zslices or im.dims[0]

    for slc_min in range(0, im.dims[0], zslices):
        print(slc_min)

        slc_max = min(slc_min + zslices, im.dims[0])
        im.slices[0] = mo.slices[0] = slice(slc_min, slc_max, 1)
        data = im.slice_dataset()

        if vol['imtype'] == 'Label':
#             if relabel:
#                 data, fwmap, _ = relabel_sequential(data)
            if np.array(fw).size == 0:
                fw = shuffle_labels(im.ulabels)
            data = fw[data]

        if 'datarange' in vol.keys():
            datarange = vol['datarange']
        else:
            datarange = [None, None]
        data, datarange = normalize_data(data, datarange[0], datarange[1])

        data = np.flip(data, axis=1)

        mo.write(data)
        mo.close()

    texdict = {}
    if vol['imtype'] == 'Label':
        if 'ulabels' in vol.keys():
            texdict['labels'] = vol['ulabels']
        else:
            texdict['labels'] = im.ulabels
    else:
        texdict['labels'] = []
    texdict['fwmap'] = fw
    texdict['datarange'] = datarange
    texdict['dims'] = np.array(im.dims)[::-1]
    if 'translations' in vol.keys():
        translations = vol['translations']
    texdict['affine'] = quickrigid(im, translations)

    im.close()

    for pf in ('affine', 'dims', 'datarange', 'labels', 'fwmap'):
        np.save(os.path.join(outputpath, pf), np.array(texdict[pf]))

    return fw


def import_vvol(name, texdir, as_overlay=False, type=''):
    """"""

    nb = bpy.context.scene.nb

    # import data
    dir = os.path.join(texdir, 'IMAGE_SEQUENCE', 'vol0000')
    is_label = False
    if as_overlay:
        if type == 'Label':
            is_label = True
    bpy.ops.nb.import_voxelvolumes(
        name=name,
        directory=dir,
        files=[{"name":"0000.png"}],
        has_valid_texdir=True,
        texdir=texdir,
        is_overlay=as_overlay,
        is_label=is_label,
        )
    vvol = nb.voxelvolumes[-1]
    # set up carver
    vvol_path = 'nb.voxelvolumes["{}"]'.format(name)
    bpy.ops.nb.import_carvers(name="Carver", parentpath=vvol_path)
    carver = vvol.carvers[-1]
    carver_path = '{}.carvers["{}.Carver"]'.format(vvol_path, name)
    bpy.ops.nb.import_carveobjects(name="slice", carveobject_type_enum='slice', parentpath=carver_path)
    carveob = carver.carveobjects[-1]
    carveob.slicethickness = [0.8, 0.8, 0.8]
    return vvol, carver, carveob

def setup_material(vvol, type='data'):
    vvol.rendertype = 'SURFACE'
    mat = bpy.data.materials[vvol.name]
    tex = bpy.data.textures[vvol.name]
    tex.voxel_data.interpolation = 'NEREASTNEIGHBOR'
    if type == 'Mask':  # NOTE: can use 'Label' type for mask as well
        # tex.color_ramp.elements[1].position = 0.001
        # tex.color_ramp.elements[1].color = (1, 0, 0, 1)
        nb = bpy.context.scene.nb
        nb.index_voxelvolumes = 2
        vvol.colourmap_enum = 'label'
    elif type == 'Label':
        nb = bpy.context.scene.nb
        nb.index_voxelvolumes = 2
        vvol.colourmap_enum = 'label'
        # tex.color_ramp.elements[1].position = 0.001
        # tex.color_ramp.elements[1].color = (1, 0, 0, 1)
    else:
        mat.alpha = 1
        tex.contrast = 0.75
        tex.color_ramp.elements[0].position = 0.8


def label_mode(tex, mode='Label', colour=[1, 0, 0], transparency=1):
    cr = tex.color_ramp
    elements = cr.elements
    cr.color_mode = 'HSV'
    elements[0].position = 0
    elements[1].position = 0.001
    elements[2].position = 1
    if mode == 'Label':
        elements[0].color = (0, 0, 0, 0)
        elements[1].color = (1, 0.001, 0, 1)
        elements[2].color = (1, 0, 0, 1)
    if mode == 'Mask':
        elements[0].color = (0, 0, 0, 0)
        elements[1].color = colour + [transparency]
        elements[2].color = colour + [transparency]


def step_yoked_vvols(vvols, delta=[0, 0, 0], absvals=[]):
    for vi, vvol in enumerate(vvols):
        if vi == 0:
            if absvals:
                for i, d in zip([0, 1, 2], absvals):
                    vvol[2].slicethickness[i] = d
            for i, d in zip([0, 1, 2], delta):
                vvol[2].slicethickness[i] = vvol[2].slicethickness[i] + d
        else:
            delta = [0.0001, 0.0001, 0.0001]
            for i, d in zip([0, 1, 2], delta):
                vvol[2].slicethickness[i] = vvols[vi-1][2].slicethickness[i] + d


def scene_setup():

    nb = bpy.context.scene.nb

    # set up scene
    bpy.ops.nb.import_presets()
    preset = nb.presets[-1]
    cam = preset.cameras[-1]
    cam.cam_view_enum_LR = 'C'
    cam.cam_view_enum_IS = 'C'
    cam.cam_view_enum_AP = 'A'
    cam.cam_distance = 3
    camob = bpy.data.objects[cam.name]
    camob.data.type = 'ORTHO'
    camob.data.ortho_scale = 50


def load_vol(vol):

    h5_fname = '{}{}.h5'.format(vol['dataset'], vol['ipf'])
    texdir = os.path.join(vol['datadir'], 'blender', h5_fname, vol['ids'])

    vvol = import_vvol(vol['name'], texdir)
    print(vvol)

    setup_material(vvol[0], type=vol['imtype'])

    return vvol


def blend_masks(mask1, mask2):
    # FIXME: assuming carver here with default name
    bpy.data.objects['{}.Carver'.format(mask1)].hide = True
    bpy.data.objects['{}.Carver'.format(mask1)].hide_render = True
    # - add pred texture to filter texture in second texture slot
    mat = bpy.data.materials[mask2]
    mat.texture_slots.add()
    ts = mat.texture_slots[-1]
    ts.texture = bpy.data.textures[mask1]
    # - set mapping to 'Generated'
    ts.texture_coords = 'ORCO'
    # - tick Influence 'Aplha' and 'Emit'
    ts.use_map_alpha = True
    ts.use_map_emit = True
    # - select Blendmode 'Add' for second texture slot
    ts.blend_type = 'ADD'


def partial_volume_texture(ob, f, s, l, blocksize_z, offset_z):

    tex = bpy.data.textures[ob.name]
    tex.image_user.frame_duration = f * blocksize_z
    ob.scale[2] = s * blocksize_z
    tex.image_user.frame_offset = f * offset_z
    ob.location[2] = tex.image_user.frame_offset * s


def get_orig_vals(ob):

    tex = bpy.data.textures[ob.name]
    frames = tex.image_user.frame_duration
    scale_z = ob.scale[2]
    loc_z = ob.location[2]
    return frames, scale_z, loc_z



def get_vols(these_vols, datadir, dataset):

    vols = {
        # base
        'data': {
            'name': 'data',
            'datadir': datadir,
            'dataset': dataset,
            'ipf': '',
            'ids': 'data',
            'imtype': 'Data',
            },
        'data_ds7': {
            'name': 'data_ds7',
            'datadir': datadir,
            'dataset': dataset,
            'ipf': '',
            'ids': 'data',
            'imtype': 'Data',
            },
        'maskMM_ds7': {
            'name': 'maskMM_ds7',
            'datadir': datadir,
            'dataset': dataset,
            'ipf': '_masks_maskMM',
            'ids': 'maskMM_PP',
            'imtype': 'Mask',
            },
        # 3D labeling
        'core3D': {
            'name': 'core3D',
            'datadir': datadir,
            'dataset': dataset,
            'ipf': '_labels_labelMA_core3D',
            'ids': 'labelMA_core3D',
            'imtype': 'Label',
            },
        'core3D_tv': {
            'name': 'core3D_tv',
            'datadir': datadir,
            'dataset': dataset,
            'ipf': '_labels_labelMA_core3D',
            'ids': os.path.join('labelMA_core3D_proofread_NoR_steps', 'labels_tv'),
            'imtype': 'Label',
            },
        'core3D_nt': {
            'name': 'core3D_nt',
            'datadir': datadir,
            'dataset': dataset,
            'ipf': '_labels_labelMA_core3D',
            'ids': os.path.join('labelMA_core3D_proofread_NoR_steps', 'labels_nt'),
            'imtype': 'Label',
            },
        # 2D classification
        'core2D': {
            'name': 'core2D',
            'datadir': datadir,
            'dataset': dataset,
            'ipf': '_labels_labelMA_core2D',
            'ids': 'labelMA_core2D',
            'imtype': 'Label',
            },
        'filter': {
            'name': 'filter',
            'datadir': datadir,
            'dataset': dataset,
            'ipf': '_labels_labelMA_core2D',
            'ids': os.path.join('labelMA_filter', 'label'),
            'imtype': 'Label',
            },
        'pred': {
            'name': 'pred',
            'datadir': datadir,
            'dataset': dataset,
            'ipf': '_labels_labelMA',
            'ids': 'labelMA_pred',
            'imtype': 'Label',
            },
        # 2D aggregate and fill
        '2Dmerge': {
            'name': '2Dmerge',
            'datadir': datadir,
            'dataset': dataset,
            'ipf': '_labels_labelMA_core2D_test',
            'ids': 'labelMA_pred_nocore3D_proofread_2Dmerge_q2-o0.50',
            'imtype': 'Label',
            },
        '2Daggr': {
            'name': '2Daggr',
            'datadir': datadir,
            'dataset': dataset,
            'ipf': '_labels_labelMA_core2D_test',
            'ids': 'labelMA_pred_nocore3D_proofread_2Dmerge_q2-o0.50_closed',
            'imtype': 'Label',
            },
        # 2D watershed fill
        'labelMA_nt': {
            'name': 'labelMA_nt',
            'datadir': datadir,
            'dataset': dataset,
            'ipf': '_labels_labelMA_comb_test',
            'ids': 'labelMA_nt',
            'imtype': 'Label',
            },
        'labelWS': {
            'name': 'labelWS',
            'datadir': datadir,
            'dataset': dataset,
            'ipf': '_labels_labelMA_2D3D',
            'ids': 'labelMA_WS',
            'imtype': 'Label',
            },
        # final MA
        'labelMA_filledm3': {
            'name': 'labelMA_filledm3',
            'datadir': datadir,
            'dataset': dataset,
            'ipf': '_labels_labelMA_2D3D',
            'ids': 'labelMA_filledm3',
            'imtype': 'Label',
            },
        # Nodes of ranvier
        'labelMA_nonodes': {
            'name': 'labelMA_nonodes',
            'datadir': datadir,
            'dataset': dataset,
            'ipf': '_labels_labelMA_2D3D',
            'ids': os.path.join('labelMA_filledm3_nodes_thr0.8_steps', 'nonodes'),
            'imtype': 'Label',
            },
        # Mitochondria  # NOTE: should be replaced still
        'labelMA_filledm2m3': {
            'name': 'labelMA_filledm2m3',
            'datadir': datadir,
            'dataset': dataset,
            'ipf': '_labels_labelMA_2D3D',
            'ids': 'labelMA_filledm2m3',
            'imtype': 'Label',
            },
        # separated sheaths  # NOTE: should be replaced still
        'labelMA_agglo_slice': {
            'name': 'labelMA_agglo_slice',
            'datadir': datadir,
            'dataset': dataset,
            'ipf': 'us_labels_labelMA_agglotmp',
            'ids': 'labelMAx0.5_l27200_u00000_s00010_nonodes_slice',
            'imtype': 'Label',
            'translations': [0, 0, 100*0.1],
            },
        'labelMA_agglo_filledm5_slice': {
            'name': 'labelMA_agglo_filledm5_slice',
            'datadir': datadir,
            'dataset': dataset,
            'ipf': 'us_labels_labelMA_agglotmp',
            'ids': 'labelMAx0.5_l27200_u00000_s00010_filledm5_nonodes_slice',
            'imtype': 'Label',
            'translations': [0, 0, 100*0.1],
            },
        'sheaths_iter0_slice': {
            'name': 'sheaths_iter0_slice',
            'datadir': datadir,
            'dataset': dataset,
            'ipf': '_labels_labelMMtmp',
            'ids': os.path.join('labelMM_steps', 'sheaths_slice'),
            'imtype': 'Label',
            'translations': [0, 0, 100*0.1],
            },
        'sheaths_iter1_slice': {
            'name': 'sheaths_iter1_slice',
            'datadir': datadir,
            'dataset': dataset,
            'ipf': '_labels_labelMMtmp',
            'ids': os.path.join('labelMM_sigmoid_iter1_steps', 'sheaths_sigmoid_10.0_slice'),
            'imtype': 'Label',
            'translations': [0, 0, 100*0.1],
            },
        # separated sheaths (full)  # NOTE: should be replaced still
        'ws': {
            'name': 'ws',
            'datadir': datadir,
            'dataset': dataset,
            'ipf': '_ws',
            'ids': 'l27200_u00000_s00010',
            'imtype': 'Label',
            },
        'maskMM_TD': {
            'name': 'maskMM_TD',
            'datadir': datadir,
            'dataset': dataset,
            'ipf': '_masks_maskMM',
            'ids': 'maskMM_TD',
            'imtype': 'Mask',
            },
        'labelMA_agglo': {
            'name': 'labelMA_agglo',
            'datadir': datadir,
            'dataset': dataset,
            'ipf': 'us_labels_labelMA_agglo',
            'ids': 'labelMAx0.5_l27200_u00000_s00010',
            'imtype': 'Label',
            },
        'labelMA_agglo_noNoR': {
            'name': 'labelMA_agglo_noNoR',
            'datadir': datadir,
            'dataset': dataset,
            'ipf': 'us_labels_labelMA_agglo',
            'ids': 'labelMAx0.5_l27200_u00000_s00010_nonodes',
            'imtype': 'Label',
            },
        'labelMA_agglo_filledm5': {
            'name': 'labelMA_agglo_filledm5',
            'datadir': datadir,
            'dataset': dataset,
            'ipf': 'us_labels_labelMA_agglo',
            'ids': 'labelMAx0.5_l27200_u00000_s00010_filledm5',
            'imtype': 'Label',
            },
        'labelMA_agglo_filledm5_noNoR': {
            'name': 'labelMA_agglo_filledm5_noNoR',
            'datadir': datadir,
            'dataset': dataset,
            'ipf': 'us_labels_labelMA_agglo',
            'ids': 'labelMAx0.5_l27200_u00000_s00010_filledm5_nonodes',
            'imtype': 'Label',
            },
        'sheaths_iter0': {
            'name': 'sheaths_iter0',
            'datadir': datadir,
            'dataset': dataset,
            'ipf': '_labels_labelMM',
            'ids': os.path.join('labelMM_iter0_steps', 'sheaths'),
            'imtype': 'Label',
            },
        'sheaths_iter1': {
            'name': 'sheaths_iter1',
            'datadir': datadir,
            'dataset': dataset,
            'ipf': '_labels_labelMM',
            'ids': os.path.join('labelMM_iter1_steps', 'sheaths'),
            'imtype': 'Label',
            },
        'sheaths_iter2': {
            'name': 'sheaths_iter2',
            'datadir': datadir,
            'dataset': dataset,
            'ipf': '_labels_labelMM',
            'ids': os.path.join('labelMM_iter2_steps', 'sheaths'),
            'imtype': 'Label',
            },
        'sheaths_iter3': {
            'name': 'sheaths_iter3',
            'datadir': datadir,
            'dataset': dataset,
            'ipf': '_labels_labelMM',
            'ids': os.path.join('labelMM_iter3_steps', 'sheaths'),
            'imtype': 'Label',
            },
        'sheaths_iter4': {
            'name': 'sheaths_iter4',
            'datadir': datadir,
            'dataset': dataset,
            'ipf': '_labels_labelMM',
            'ids': os.path.join('labelMM_iter4_steps', 'sheaths'),
            'imtype': 'Label',
            },
        'dist_iter0': {
            'name': 'dist_iter0',
            'datadir': datadir,
            'dataset': dataset,
            'ipf': '_labels_labelMM',
            'ids': os.path.join('labelMM_iter0_steps', 'distance'),
            'imtype': 'Data',
            },
        'dist_iter4': {
            'name': 'dist_iter4',
            'datadir': datadir,
            'dataset': dataset,
            'ipf': '_labels_labelMM',
            'ids': os.path.join('labelMM_iter4_steps', 'distance'),
            'imtype': 'Label',
            },
    }

    those_vols = [vols[volname] for volname in these_vols]

    return those_vols


if __name__ == "__main__":
    main(sys.argv)
