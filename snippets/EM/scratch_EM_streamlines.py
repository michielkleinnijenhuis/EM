###=========================================================================###
### mesh component analysis and merge
###=========================================================================###

### imports ###
import os
import numpy as np
import pickle

### parameters ###
datadir = "/Users/michielk/oxdata/P01/EM/M3/M3_S1_GNU/M3S1GNU/ds7_arc"
datadir = '/Users/mkleinnijenhuis/oxdata/P01/EM/Myrf_01/SET-B/B-NT-S10-2f_ROI_00/blender';
# blenddir = os.path.join(datadir, "M3S1GNUds7/dmcsurf_1-1-1/blends/d0.02")
blenddir = '/Users/mkleinnijenhuis/oxdata/P01/EM/Myrf_01/SET-B/B-NT-S10-2f_ROI_00/blender';
comp = 'B-NT-S10-2f_ROI_00_labels_labelMF-labelMF'
filetemp = "_label%s_final_d0.02collapse_s100-0.1_%d.blend"
suffixes = range(1,7)
pat = ' label%s final' % comp

xs=8423; ys=8316; zs=184;
xo=0; yo=0; zo=0;
xe=0.007; ye=0.007; ze=0.1;
# B-NT-S10-2f_ROI_00_labels_labelMF-labelMF_slcws_centroid_x
centroid_x = os.path.join(datadir, 'stats', '%s_slcws_centroid_x.txt' % comp)
centroid_y = os.path.join(datadir, 'stats', '%s_slcws_centroid_y.txt' % comp)
jump_threshold = ze * 20

nverts = 32

### functions ###
def import_objects(comp, fpaths):
    for fpath in fpaths:
        with bpy.data.libraries.load(fpath) as (data_from, data_to):
            data_to.objects = data_from.objects
        for ob in data_to.objects:
            if ob is not None:
                bpy.context.scene.objects.link(ob)

def get_mesh_info(obs):
    obs[0].select = True
    bpy.context.scene.objects.active = obs[0]
    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.select_all(action='DESELECT')
    info = []
    for ob in obs:
        ob.select = True
        bpy.context.scene.objects.active = ob
        bpy.ops.mcell.meshalyzer()
        ob.select = False
        ma = bpy.context.scene.mcell.meshalyzer
        info.append({'object_name': ma.object_name,
                     'vertices': ma.vertices,
                     'edges': ma.edges,
                     'faces': ma.faces,
                     'area': ma.area,
                     'volume': ma.volume,
                     'watertight': ma.watertight,
                     'manifold': ma.manifold,
                     'normal_status': ma.normal_status})
    return info

def get_streamlines(centroidfiles, zyxElsize, zyxSize, zyxOffset):
    # get centroid coordinate for every slice
    streamlines_x = np.transpose(np.loadtxt(centroidfiles[0])) * zyxElsize[2]
    streamlines_y = np.transpose(np.loadtxt(centroidfiles[1])) * zyxElsize[1]
    sl_z = np.tile(np.array(range(0, zyxSize[0])), [streamlines_x.shape[0], 1])
    streamlines_z = sl_z * zyxElsize[0] + zyxOffset[0] * zyxElsize[0]
    streamlines_raw = np.dstack((streamlines_x, streamlines_y, streamlines_z))
    # remove nans
    streamlines = []
    for sl in streamlines_raw:
        streamline = []
        for point in sl:
            if not np.isnan(point).any():
                streamline.append(point)
        streamlines.append(streamline)
    return streamlines

def generate_streamlines(name, streamlines):
    curve = bpy.data.curves.new(name=name, type='CURVE')
    curve.dimensions = '3D'
    ob = bpy.data.objects.new(name, curve)
    bpy.context.scene.objects.link(ob)
    for streamline in streamlines:
        make_polyline_ob(curve, streamline)
    return ob

def make_polyline_ob(curvedata, cList):
    """Create a 3D curve from a list of points."""
    polyline = curvedata.splines.new('POLY')
    polyline.points.add(len(cList) - 1)
    for num in range(len(cList)):
        x, y, z = cList[num]
        polyline.points[num].co = (x, y, z, 1)
    polyline.order_u = len(polyline.points) - 1
    polyline.use_endpoint_u = True

def calculate_spline_lengths(curve, jump_threshold):
    ii = 0
    lengths = []
    hasNoRs = []
    for j, spline in enumerate(curve.splines):
        length = 0
        jumped = False
        for i, point in enumerate(spline.points[1:]):
            l = (spline.points[i].co-point.co).length
            length += l
            jumped = jumped or l > jump_threshold
        lengths.append(length)
        hasNoRs.append(jumped)
    return lengths, hasNoRs

def connect_fibres(name, pat, ref):
    """Connect all duplicated equivalent meshes in one object."""
    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.object.select_pattern(pattern=pat)
    bpy.context.scene.objects.active = bpy.data.objects[ref]
    bpy.ops.object.join()
    bpy.context.scene.objects.active.name = name
    bpy.context.scene.objects.active.data.name = name + 'Mesh'
    bpy.ops.object.select_all(action='DESELECT')
    ob = bpy.data.objects[name]
    return ob

def triangulate_mesh(ob):
    """Triangulate the mesh."""
    bpy.context.scene.objects.active = ob
    bpy.context.tool_settings.mesh_select_mode=(True,False,False)
    bpy.ops.object.mode_set(mode = 'EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.quads_convert_to_tris()
    bpy.ops.object.mode_set(mode='OBJECT')



### import objects (assumes stl2blender run with 6 processes #1-6)
fpaths = [os.path.join(blenddir, filetemp % (comp, i)) for i in suffixes]
import_objects(comp, fpaths)

### mesh analysis
obs = [ob for ob in bpy.data.objects
       if ob.name.startswith(pat)]
info = get_mesh_info(obs)
with open(os.path.join(blenddir, "%s_info.pickle" % comp), 'wb') as f:
    pickle.dump(info, f)


### centreline analysis
streamlines = get_streamlines([centroid_x, centroid_y],
                               [ze, ye, xe], [zs, ys, xs], [zo, yo, xo])
outfile = os.path.join(datadir, '%s_centrelines.npz' % comp)
np.savez_compressed(outfile, centrelines=streamlines)
medax = generate_streamlines('medax', streamlines)
lengths, hasNoRs = calculate_spline_lengths(medax.data, jump_threshold)

streamlines_jump = [streamline for streamline, hasNoR in zip(streamlines, hasNoRs) if hasNoR]
outfile = os.path.join(datadir, '%s_centrelines_jump.npz' %comp)
np.savez_compressed(outfile, centrelines=streamlines_jump)
jump = generate_streamlines('jump', streamlines_jump)

streamlines_nojump = [streamline for streamline, hasNoR in zip(streamlines, hasNoRs) if not hasNoR]
outfile = os.path.join(datadir, '%s_centrelines_nojump.npz' %comp)
np.savez_compressed(outfile, centrelines=streamlines_nojump)
nojump = generate_streamlines('nojump', streamlines_nojump)

volumes = [inf['volume'] for inf in info]
radii = np.sqrt(np.divide(volumes, np.pi*np.array(lengths)))
for inf, sl, NoR, length, radius in zip(info, streamlines, hasNoRs, lengths, radii):
    inf['centreline'] = sl
    inf['length'] = length
    inf['NoR'] = NoR
    inf['radius'] = radius

with open(os.path.join(blenddir, "%s_info-ext.pickle" % comp), 'wb') as f:
    pickle.dump(info, f)

### connect objects into compartment and save clean combined blend file
ob = connect_fibres(comp, '%s*' % pat, bpy.data.objects[0].name)
bpy.context.scene.objects.unlink(medax)
bpy.context.scene.objects.unlink(jump)
fpath = os.path.join(blenddir, "%s.blend" % comp)
bpy.ops.wm.save_as_mainfile(filepath=fpath)

bpy.context.scene.objects.unlink(ob)

###=========================================================================###
### generate equivalent cylinders
###=========================================================================###

with open(os.path.join(blenddir, "%s_info-ext.pickle" % comp), 'rb') as f:
    info = pickle.load(f)

for inf in info:  # [info[0]]
    npoints = len(inf['centreline'])
    range_z = inf['centreline'][-1][2] - inf['centreline'][0][2]
    loc_z = inf['centreline'][0][2] + range_z / 2
    mid_idx = floor(npoints / 2)
    if npoints % 2:
        loc_x = inf['centreline'][mid_idx][0]
        loc_y = inf['centreline'][mid_idx][1]
    else:
        loc_x = np.mean([inf['centreline'][mid_idx][0], inf['centreline'][mid_idx+1][0]])
        loc_y = np.mean([inf['centreline'][mid_idx][1], inf['centreline'][mid_idx+1][1]])
    # for 32 vertices in circumference the volume mismatch (shapes/cyl) is 1.006454543135078
    # this should be exact:
    theta = pi / nverts
    area_tri = inf['volume'] / (nverts * 2 * inf['length'])
    r = np.sqrt((2 * area_tri) / (sin(theta) * cos(theta)))
    bpy.ops.mesh.primitive_cylinder_add(vertices=nverts,
                                        radius=r,
                                        depth=inf['length'],
                                        location=[loc_x, loc_y, loc_z],
                                        rotation=[0, 0, 0])
    ob = C.scene.objects.active
    ob.name = inf['object_name']
    ob.data.name = inf['object_name'] + 'Mesh'
    triangulate_mesh(ob)


streamlines = []
for inf in info:
    streamlines.append(inf['centreline'])

medax = generate_streamlines('medax', [streamlines[0]])


# obs = [ob for ob in bpy.data.objects
#        if ob.name.startswith(pat)]
# info_cyl = get_mesh_info(obs)
#
# volumes_shapes = [inf['volume'] for inf in info[:10]]
# volumes_cyl = [inf['volume'] for inf in info_cyl]
#
# np.divide(volumes_shapes, volumes_cyl)

# calculate volume mismatch for every cylinder with meshalyzer
# then correct radii accordingly in creating cylinders
# then check again

for ob in bpy.data.objects:
    ob.hide = True

for ob in bpy.data.objects[10:]:
    bpy.data.objects.remove(ob)
