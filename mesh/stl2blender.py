#!/usr/bin/env python

"""
blender -b -P stl2blender.py -- <stldir> <outfilename>
"""

import os
import sys
import argparse
from glob import glob
from random import random

from bpy import context as C
from bpy import data as D
from bpy import ops as O

# sys.path.append("/Users/michielk/workspace/cgal-swig-bindings/build/build-python")
# from CGAL import CGAL_Mesh_3

def main(argv):
    
    if "--" not in argv:
        argv = []  # as if no args are passed
    else:
        argv = argv[argv.index("--") + 1:]  # get all args after "--"
    
    parser = argparse.ArgumentParser(description='Combine stl objects into blender scene.')
    
    parser.add_argument('stldir', help='the input data directory')
    parser.add_argument('outfilename', help='the output file name')
    parser.add_argument('-L', '--labelimages', default=['UA'], nargs='+', help='...')
    parser.add_argument('-e', '--ecs_shrinkvalue', type=float, help='...')
    parser.add_argument('-s', '--smoothparams', type=float, nargs=2, help='...')
    parser.add_argument('-d', '--decimation', type=float, help='...')
    
    args = parser.parse_args(argv)
    
    stldir = args.stldir
    outfilename = args.outfilename
    compartments = args.labelimages
    ecs_shrinkvalue = args.ecs_shrinkvalue
    smoothparams = args.smoothparams
    decimation = args.decimation
    
#     compartments = ['NN', 'DD', 'MM', 'MA', 'UA', 'mito', 'vesicles', 'synapses']
    # TODO: default to all compartments in stldir
    
    for comp in compartments:
        fps = glob(os.path.join(stldir, comp + '*.stl'))
        nonmanifolds = {}
        for fp in fps:
            O.import_mesh.stl(filepath=fp)
            ob = C.scene.objects.active
            consistent_outward_normals(ob)
            if decimation:
                decimate_mesh(ob, decimation)
            if smoothparams:
                print(smoothparams[0], smoothparams[1])
                smooth_mesh(ob, smoothparams[0], smoothparams[1])
            if ecs_shrinkvalue:
                shrink_mesh(ob, ecs_shrinkvalue)
            colour = [random() for _ in range(0,3)]
            transparent_mat = make_material('mat', colour, 0.2)
            set_material(ob, transparent_mat)
            activate_viewport_transparency(ob)
            idxs = non_manifold_vertices(ob)
            if idxs:
                nonmanifolds[ob.name] = idxs
        print(nonmanifolds.keys())
        ob = connect_fibres(comp, comp + '*', ob.name)
    
    blendfile = os.path.join(stldir, outfilename + '.blend')
    O.wm.save_as_mainfile(filepath=blendfile)

def connect_fibres(name, pat, ref):
    """Connect all duplicated equivalent meshes in one object."""
    O.object.select_all(action='DESELECT')
    O.object.select_pattern(pattern=pat)
    C.scene.objects.active = D.objects[ref]
    O.object.join()
    C.scene.objects.active.name = name
    C.scene.objects.active.data.name = name + 'Mesh'
    O.object.select_all(action='DESELECT')
    ob = D.objects[name]
    return ob

def consistent_outward_normals(ob):
    """Make the normals of the object face outward"""
    O.object.select_all(action='DESELECT')
    ob.select = True
    O.object.mode_set(mode='EDIT')
    O.mesh.select_all(action='SELECT')
    O.mesh.normals_make_consistent(inside=False)
    O.object.mode_set(mode='OBJECT')

def make_material(name, colour=[0.8,0.8,0.8], alpha=1):
    """Return a material with transparency."""
    mat = D.materials.new(name)
    mat.diffuse_color = colour
    mat.alpha = alpha
    return mat

def set_material(ob, mat):
    """Append the material to the object."""
    me = ob.data
    me.materials.append(mat)

def activate_viewport_transparency(ob):
    """Activate transparency for the Blender 3D viewport."""
    ob.active_material.use_transparency = True
    ob.show_transparent = True

def non_manifold_vertices(ob):
    O.object.select_all(action='DESELECT')
    ob.select = True
    O.object.mode_set(mode='EDIT')
    O.mesh.select_all(action='DESELECT')
    C.tool_settings.mesh_select_mode = (True , False , False)
    O.object.mode_set(mode="OBJECT")
    O.object.mode_set(mode='EDIT')
    O.mesh.select_non_manifold()
    O.object.mode_set(mode = 'OBJECT')
    idxs = [i.index for i in ob.data.vertices if i.select]
    # [i.co for i in ob.data.vertices if i.select]
    return idxs

def decimate_mesh(ob, ratio):
    O.object.select_all(action='DESELECT')
    ob.select = True
    O.object.modifier_add(type='DECIMATE')
    C.object.modifiers["Decimate"].ratio = ratio
    O.object.modifier_apply(apply_as='DATA', modifier="Decimate")

def shrink_mesh(ob, value):
    O.object.select_all(action='DESELECT')
    ob.select = True
    O.object.mode_set(mode='EDIT')
    O.transform.shrink_fatten(value=value)
    O.object.mode_set(mode='OBJECT')

def smooth_mesh(ob, factor, iterations):
    O.object.select_all(action='DESELECT')
    ob.select = True
    O.object.modifier_add(type='SMOOTH')
    C.object.modifiers["Smooth"].factor = factor
    C.object.modifiers["Smooth"].iterations = iterations
    O.object.modifier_apply(apply_as='DATA', modifier="Smooth")

if __name__ == "__main__":
    main(sys.argv)
