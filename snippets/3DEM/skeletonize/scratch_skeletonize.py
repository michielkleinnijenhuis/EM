for every element (MA,MM,UA) calculate
- orientation
- length
- volume

ma = C.scene.mcell.meshalyzer
ma.volume
ma.area


# distance = distance_transform_edt(MA==0, sampling=np.absolute(elsize))

###====================###
### slicewise centroid ###
###====================###
import os
import h5py
import numpy as np
from skimage.measure import regionprops
from scipy.ndimage import distance_transform_edt

datadir = '/Users/michielk/oxdata/P01/EM/M3/M3_S1_GNU'
dset_name = 'm000_01000-01500_01000-01500_00030-00460'
zyxOffset = np.array([30,1000,1000])
MAfile = '_MA_sMAkn_sUAkn_ws_filled'
MAfile = '_PA'
MA, elsize = loadh5(os.path.join(datadir, dset_name), dset_name + MAfile)
# label_image = MA==1128

# slcs = range(0,430,10)
slcs = np.r_[0:430, 429]
a = np.empty((len(slcs),3,))
a[:] = np.NAN
objs = np.unique(MA)[1:]
it = [(str(obj), a.copy()) for obj in objs]
d = {k: v for (k, v) in it}
for i,slc in enumerate(slcs):
    rps = regionprops(MA[slc,:,:])
    for rp in rps:
        if MA[slc,int(rp['centroid'][0]),int(rp['centroid'][1])] == rp['label']:
            d[str(rp['label'])][i,:] = np.append(slc, rp['centroid'])
        # elif x,y is completely off ...
        else:
            distance = distance_transform_edt(MA[slc,:,:]==rp['label'], sampling=np.absolute(elsize[-2:]))
            i_maxdist = np.unravel_index(np.argmax(distance), distance.shape)
            d[str(rp['label'])][i,:] = np.append(slc, list(i_maxdist))
            print('else found: ', slc, rp['centroid'], rp['label'], i_maxdist)

# write coordlist to knossos-xml
from xml.etree.ElementTree import Element, SubElement, ElementTree, parse
import xml.dom.minidom
from zipfile import ZipFile

infile = os.path.join(datadir, dset_name, dset_name + '_knossos', 'annotation_MA.xml')
outfile = os.path.join(datadir, dset_name, dset_name + '_knossos', 'annotation.xml')
outzip = os.path.join(datadir, dset_name, dset_name + '_knossos', 'annotation.zip')

things = Element("things")
pars = ElementTree(file=infile).getroot().find('parameters')
things.append(pars)
node_id = 1
for k,v in d.iteritems():
    # the tree
    thing = SubElement(things, "thing")
    thing.attrib['id'] = k
    thing.attrib['color.r'] = "-1."
    thing.attrib['color.g'] = "-1."
    thing.attrib['color.b'] = "-1."
    thing.attrib['color.a'] = "1."
    thing.attrib['comment'] = k
    # nodes and edges
    nodes = SubElement(thing, "nodes")
    edges = SubElement(thing, "edges")
    firstnode = True
    for i,coord in enumerate(v):
        if not any(np.isnan(coord)):
            node = SubElement(nodes, "node")
            node.attrib['id'] = str(node_id)
            node.attrib['radius'] = "1.5"
            node.attrib['x'] = str(int(coord[1]) + 1)  #knossos is 1-based
            node.attrib['y'] = str(int(coord[2]) + 1)
            node.attrib['z'] = str(int(coord[0]) + 1)
            node.attrib['inVp'] = "0"
            node.attrib['inMag'] = "1"
            node.attrib['time'] = "16680000"
            if not firstnode:
                edge = SubElement(edges, "edge")
                edge.attrib['source'] = str(node_id - 1)
                edge.attrib['target'] = str(node_id)
            firstnode = False
            node_id += 1
            # node_ids = node_ids.append(node_id)
            # edges
        else:
            firstnode = True

comments = SubElement(things, "comments")
branchpoints = SubElement(things, "branchpoints")
ElementTree(things).write(outfile)

xml = xml.dom.minidom.parse(outfile)
pretty_xml_as_string = xml.toprettyxml()
with open(outfile, "w") as t:
    t.write(pretty_xml_as_string)
    t.close()

os.chdir(os.path.join(datadir, dset_name, dset_name + '_knossos'))
with ZipFile('annotation.zip', 'w') as z:
    z.write('annotation.xml')

# remove the nan-rows
for label, dcoords in d.iteritems():
    d[label] = dcoords[~np.isnan(dcoords).any(axis=1)]

# get coordlist for use in e.g. blender (convert dsetspace to worldspace-um)
d2 = {}
for label, dcoords in d.iteritems():
    wcoords = np.copy(dcoords)
    wcoords += np.tile(zyxOffset, [dcoords.shape[0], 1])
    wcoords *= np.tile(elsize, [dcoords.shape[0], 1])
    kcoords = np.copy(dcoords)
    kcoords += np.tile(zyxOffset, [dcoords.shape[0], 1])
    d2[label] = (dcoords, wcoords, kcoords)

# write coordlist to simple txt file
os.chdir(os.path.join(datadir, dset_name, dset_name + '_skel'))
for k, v in d2.iteritems():
    np.save('skel' + k, v[1])



###====================###
### Blender processing ###
###====================###
import os
import numpy as np

def make_polyline_ob(curvedata, cList):
    """Create a 3D curve from a list of points."""
    polyline = curvedata.splines.new('POLY')
    polyline.points.add(len(cList)-1)
    for num in range(len(cList)):
        x, y, z = cList[num]
        polyline.points[num].co = (z, y, x, 1)
    polyline.order_u = len(polyline.points)-1
    polyline.use_endpoint_u = True

datadir = '/Users/michielk/oxdata/P01/EM/M3/M3_S1_GNU'
dset_name = 'm000_01000-01500_01000-01500_00030-00460'
streamline = np.load(os.path.join(datadir, dset_name, dset_name + '_skel', 'skel2197.npy'))
name = 'UA.02197.01_skel'
curve = bpy.data.curves.new(name=name, type='CURVE')
curve.dimensions = '3D'
ob = bpy.data.objects.new(name, curve)
bpy.context.scene.objects.link(ob)

make_polyline_ob(curve, streamline)


# get length of the skeleton
# get the mean orientation of the segments
# get the dispersion over the segments
# get a dispersion measurements from MA, MM and UA seperately
# label 2222 is a double label in _PA


###===============###
### Fiji skeleton ### fail (same as ML)
###===============###


###===================###
### MATLAB Skeleton3D ### # fail
###===================###
addpath ~/oxscripts/matlab/toolboxes/Skeleton3D
datadir = '/Users/michielk/oxdata/P01/EM/M3/M3_S1_GNU';
cd(datadir);
invol = 'm000_01000-01500_01000-01500_00030-00460_PA';
infield = '/stack';
stackinfo = h5info([datadir filesep invol '.h5'], infield);
data = h5read([datadir filesep invol '.h5'], infield);
bin = data == 1128;
u = uint8(Skeleton3D(bin));
fname = 'm000_01000-01500_01000-01500_00030-00460_PA_skel.h5';
outfield = '/stack';
h5create(fname, outfield, size(u), 'Deflate', 4, 'Chunksize', stackinfo.ChunkSize);
h5write(fname, outfield, u);

###======###
### TBSS ### fail
###======###
fslmaths m000_01000-01500_01000-01500_00030-00460_PA.nii.gz -thr 1128 -uthr 1128 -bin -s 5 m000_01000-01500_01000-01500_00030-00460_PA_MA1128.nii.gz
$FSLDIR/bin/tbss_skeleton -i m000_01000-01500_01000-01500_00030-00460_PA_MA1128.nii.gz -o m000_01000-01500_01000-01500_00030-00460_PA_MA1128_skeleton.nii.gz

###======###
### MASB ### # does not give me what I want
###======###
# python install fails
pip install git+git://github.com/tudelft3d/masbpy@master
# C++ install succeeds, but 'brew install clang-omp' for OpenMP fails
git clone https://github.com/tudelft3d/masbcpp.git
cd masbcpp
cmake .
make
# ./compute_ma --help
# ./compute_normals --help

import numpy as np
ob = bpy.data.objects['MA.01120.01']
me = ob.data
verts = np.empty(len(me.vertices)*3, dtype=np.float64)
me.vertices.foreach_get('co', verts)
verts.shape = (len(me.vertices), 3)
np.save('/Users/michielk/coords.npy', verts)
norms = np.empty(len(me.vertices)*3, dtype=np.float64)
me.vertices.foreach_get('normal', norms)
norms.shape = (len(me.vertices), 3)
np.save('/Users/michielk/normals.npy', norms)

./compute_normals ~ ~
./compute_ma ~ ~

ma_in = np.load('/Users/michielk/ma_coords_in.npy')
qi_in = np.load('/Users/michielk/ma_qidx_in.npy')
ma_out = np.load('/Users/michielk/ma_coords_out.npy')
qi_out = np.load('/Users/michielk/ma_qidx_out.npy')

make_polyline('MAT_test', 'mat_test', ma_in)

def make_polyline(objname, curvename, cList):
    """Create a 3D curve from a list of points."""
    curvedata = D.curves.new(name=curvename, type='CURVE')
    curvedata.dimensions = '3D'
    objectdata = D.objects.new(objname, curvedata)
    objectdata.location = (0,0,0)
    C.scene.objects.link(objectdata)
    polyline = curvedata.splines.new('POLY')
    polyline.points.add(len(cList)-1)
    for num in range(len(cList)):
        x, y, z = cList[num]
        polyline.points[num].co = (x, y, z, 1)
    return objectdata



###==============###
### SCIKIT-IMAGE ### fail
###==============###
conda create --name scikit-image-devel
source activate scikit-image-devel
conda install numpy
pip install git+git://github.com/scikit-image/scikit-image@master

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

import os
import sys
import h5py
import numpy as np
from skimage.morphology import skeletonize_3d
from scipy.ndimage import distance_transform_edt

datadir = "/Users/michielk/oxdata/P01/EM/M3/M3_S1_GNU"
x=1000; X=1500; y=1000; Y=1500; z=30; Z=460;
dataset = 'm000'
fieldnamein = 'stack'
nzfills = 5
zyxOffset = [30,1000,1000]
labelimages = ['_PA']
dataset = dataset + '_' + str(x).zfill(nzfills) + '-' + str(X).zfill(nzfills) + \
                    '_' + str(y).zfill(nzfills) + '-' + str(Y).zfill(nzfills) + \
                    '_' + str(z).zfill(nzfills) + '-' + str(Z).zfill(nzfills)

mask, elsize = loadh5(os.path.join(datadir, dataset), dataset + labelimages[0], fieldnamein)
MA1128 = mask==1128
MA1128_skel = skeletonize_3d(MA1128)

writeh5(MA1128_skel, datadir, os.path.join(datadir, dataset + '_MA1128_skeletonize_3d'), 'stack', 'uint8', elsize)

distance = distance_transform_edt(MA1128, sampling=np.absolute(elsize))
writeh5(distance, datadir, os.path.join(datadir, dataset + '_MA1128_edt'), 'stack', 'float', elsize)
