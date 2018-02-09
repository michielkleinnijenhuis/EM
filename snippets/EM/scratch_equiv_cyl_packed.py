import os
import numpy as np
import pickle

### parameters ###
datadir = "/Users/michielk/oxdata/P01/EM/M3/M3_S1_GNU/M3S1GNU/ds7_arc"
blenddir = os.path.join(datadir, "M3S1GNUds7/dmcsurf_1-1-1/blends/d0.02")

comp = 'MF'
with open(os.path.join(blenddir, "%s_info-ext.pickle" % comp), 'rb') as f:
    info_MF = pickle.load(f)

comp = 'MA'
with open(os.path.join(blenddir, "%s_info-ext.pickle" % comp), 'rb') as f:
    info_MA = pickle.load(f)

radii_MF = [inf['radius'] for inf in info_MF]
np.savetxt(os.path.join(blenddir, comp + '_radii.txt'), radii)
# do circle packing on MF radii
radii_MA = [inf['radius'] for inf in info_MA]

pos = np.loadtxt(os.path.join(blenddir, 'MF_pos.txt'))


for inf_MF, inf_MA, p in zip(info_MF, info_MA, pos):
    loc_z = 0
    theta = pi / nverts
    if inf_MA['length'] > inf_MF['length']:
        print(inf_MA['length'], inf_MF['length'])
    for inf in [inf_MF, inf_MA]:
        npoints = len(inf['centreline'])
        range_z = inf['centreline'][-1][2] - inf['centreline'][0][2]
        mid_idx = floor(npoints / 2)
        # for 32 vertices in circumference the volume mismatch (shapes/cyl) is 1.006454543135078
        # this should be exact:
        area_tri = inf['volume'] / (nverts * 2 * inf['length'])
        area_tri = inf['volume'] / (nverts * 2 * inf['length'])
        r = np.sqrt((2 * area_tri) / (sin(theta) * cos(theta)))
        bpy.ops.mesh.primitive_cylinder_add(vertices=nverts,
                                            radius=r,
                                            depth=inf['length'],
                                            location=[p[0], p[1], loc_z],
                                            rotation=[0, 0, 0])
        ob = C.scene.objects.active
        ob.name = inf['object_name']
        ob.data.name = inf['object_name'] + 'Mesh'
        triangulate_mesh(ob)
