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
    parser.add_argument('-S', '--stlfiles', default=[], nargs='+', help='...')
    parser.add_argument('-d', '--decimationparams', type=float, nargs='+', default=None, help='...')
    parser.add_argument('-s', '--smoothparams', nargs='+', default=None, help='...')
    parser.add_argument('-l', '--smoothlaplacian', nargs='+', default=None, help='...')
    parser.add_argument('-e', '--ecs_shrinkvalue', type=float, default=None, help='...')
    parser.add_argument('-c', '--connect', action='store_true',
                        help='connect fibres into one object')
    parser.add_argument('-r', '--randomcolour', action='store_true',
                        help='assign a random colour to each axon')
    
    args = parser.parse_args(argv)
    
    stldir = args.stldir
    outfilename = args.outfilename
    stlfiles = args.stlfiles
    compartments = args.labelimages
    ecs_shrinkvalue = args.ecs_shrinkvalue
    smoothparams = args.smoothparams
    smoothlaplacian = args.smoothlaplacian
    decimationparams = args.decimationparams
    connect = args.connect
    randomcolour = args.randomcolour
    
#     compartments = ['NN', 'DD', 'MM', 'MA', 'UA', 'mito', 'vesicles', 'synapses']
    # TODO: default to all compartments in stldir
    
    for comp in compartments:
        if not stlfiles:
            stlfiles = glob(os.path.join(stldir, comp + '*.stl'))
        nonmanifolds = {}
        colour = [random() for _ in range(0,3)]
        mat = make_material('mat', colour, 0.2)
        for fp in stlfiles:
            O.import_mesh.stl(filepath=fp)
            ob = C.scene.objects.active
            consistent_outward_normals(ob)
            if decimationparams is not None:
                decimate_mesh_planar(ob, angle_limit=decimationparams[0])
                triangulate_mesh(ob)
            if smoothparams is not None:
                smooth_mesh_default(ob,
                                    iterations=int(smoothparams[0]),
                                    factor=float(smoothparams[1]),
                                    use_x=bool(smoothparams[2]),
                                    use_y=bool(smoothparams[3]),
                                    use_z=bool(smoothparams[4]))
            if smoothlaplacian is not None:
                smooth_mesh_laplacian(ob,
                                      iterations=int(smoothlaplacian[0]),
                                      lambda_factor=float(smoothlaplacian[1]),
                                      lambda_border=float(smoothlaplacian[2]),
                                      use_x=bool(smoothlaplacian[3]),
                                      use_y=bool(smoothlaplacian[4]),
                                      use_z=bool(smoothlaplacian[5]))
            shrink_mesh(ob, ecs_shrinkvalue)
            if randomcolour:
                colour = [random() for _ in range(0,3)]
                mat = make_material('mat', colour, 0.2)
            set_material(ob, mat)
            activate_viewport_transparency(ob)
            idxs = non_manifold_vertices(ob)
            if idxs:
                nonmanifolds[ob.name] = idxs
        print(nonmanifolds.keys())
        if connect:
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
    """"""
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

def decimate_mesh_default(ob, ratio=0.1, vg=""):
    """"""
    mod = ob.modifiers.new("decimate", type='DECIMATE')
    mod.decimate_type = 'COLLAPSE'
    mod.ratio = ratio
    mod.vertex_group = vg
    mod.use_collapse_triangulate = True
    O.object.modifier_apply(modifier=mod.name)
    print('mesh ' + vg + 
          ' collapsed according to ratio: ' + 
          str(ratio))

def decimate_mesh_planar(ob, angle_limit=0.0174533, vg=""):
    """"""
    mod = ob.modifiers.new("decimate", type='DECIMATE')
    mod.decimate_type = 'DISSOLVE'
    mod.angle_limit = angle_limit
    mod.vertex_group = vg
    O.object.modifier_apply(modifier=mod.name)
    print('mesh ' + vg + 
          ' dissolved according to angle: ' + 
          str(angle_limit))

def shrink_mesh(ob, value):
    """"""
    O.object.select_all(action='DESELECT')
    ob.select = True
    O.object.mode_set(mode='EDIT')
    O.transform.shrink_fatten(value=value)  # (value=0.0, mirror=False, proportional='DISABLED', proportional_edit_falloff='SMOOTH', proportional_size=1.0, snap=False, snap_target='CLOSEST', snap_point=(0.0, 0.0, 0.0), snap_align=False, snap_normal=(0.0, 0.0, 0.0), release_confirm=False)
    O.object.mode_set(mode='OBJECT')

def smooth_mesh_default(ob, iterations=10, factor=0.5, use_x=True, use_y=True, use_z=True):
    """"""
    mod = ob.modifiers.new("smooth", type='SMOOTH')
    mod.iterations = iterations
    mod.factor = factor
    mod.use_x = use_x
    mod.use_y = use_y
    mod.use_z = use_z
    O.object.modifier_apply(modifier=mod.name)

def smooth_mesh_laplacian(ob, iterations=100, lambda_factor=0.2, lambda_border=0.01, use_x=True, use_y=True, use_z=True, vg=""):
    """"""
    mod = ob.modifiers.new("laplaciansmooth", type='LAPLACIANSMOOTH')
    mod.iterations = iterations
    mod.lambda_factor = lambda_factor
    mod.lambda_border = lambda_border
    mod.use_x = use_x
    mod.use_y = use_y
    mod.use_z = use_z
    mod.vertex_group = vg
    mod.use_volume_preserve = True
    mod.use_normalized = True
    O.object.modifier_apply(modifier=mod.name)

def triangulate_mesh(ob):
    """Triangulate the mesh."""
    C.scene.objects.active = ob
    C.tool_settings.mesh_select_mode=(True,False,False)
    O.object.mode_set(mode = 'EDIT')
    O.mesh.select_all(action='SELECT')
    O.mesh.quads_convert_to_tris()
    O.object.mode_set(mode='OBJECT')


if __name__ == "__main__":
    main(sys.argv)
