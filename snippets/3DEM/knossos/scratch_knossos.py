# knossos data prep
git clone https://github.com/knossos-project/knossos_python_tools.git

scriptdir="$HOME/workspace/EM"
DATA="$HOME/oxdata"
datadir="$DATA/P01/EM/M3/M3_S1_GNU"
dataset='m000'
z=30; Z=460;
x=1000; y=1000;
[ $x == 5000 ] && X=5217 || X=$((x+1000))
[ $y == 4000 ] && Y=4460 || Y=$((y+1000))
datastem=${dataset}_`printf %05d ${x}`-`printf %05d ${X}`_`printf %05d ${y}`-`printf %05d ${Y}`_`printf %05d ${z}`-`printf %05d ${Z}`
pf=
pf=_smooth
python $scriptdir/convert/EM_stack2stack.py \
${datadir}/${datastem}${pf}.h5 ${datadir}/${datastem}${pf}.tif \
-i 'zyx' -l 'xyz' -e -0.0073 -0.0073 0.05 -u -n 5

#python /Users/michielk/workspace/knossos_python_tools/knossos_cuber/knossos_cuber/knossos_cuber_gui.py
mkdir -p ${datadir}/${datastem}${pf}_knossos
cuberdir=/Users/michielk/workspace/knossos_python_tools/knossos_cuber/knossos_cuber/
cd $cuberdir
python knossos_cuber.py -f tif ${datadir}/${datastem}${pf} ${datadir}/${datastem}${pf}_knossos
# ERROR: config.ini



# 'knossos xml to python' snippet
def get_knossos_groundtruth(datadir, dataset):
    import os
    import xml.etree.ElementTree
    things = xml.etree.ElementTree.parse(os.path.join(datadir, dataset + '_knossos', 'annotation.xml')).getroot()
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
        # NOTE: knossos has a base-1 coordinate system
        # append
        objs.append(obj)
    return objs




# integrate knossos seeds into _seg
import os
import sys
from argparse import ArgumentParser
import h5py
import numpy as np
from skimage.morphology import watershed, remove_small_objects, erosion, square  #, dilation
from scipy.ndimage.morphology import grey_dilation, grey_erosion, binary_erosion, binary_fill_holes, binary_closing, generate_binary_structure
from scipy.special import expit
from skimage.segmentation import random_walker, relabel_sequential
from scipy.ndimage.measurements import label, labeled_comprehension
from scipy.ndimage import distance_transform_edt
from scipy.ndimage.filters import gaussian_filter
from skimage.morphology import medial_axis
from scipy.ndimage import find_objects

datadir = "/Users/michielk/oxdata/P01/EM/M3/M3_S1_GNU"
x=1000; X=2000; y=1000; Y=2000; z=30; Z=460;
nzfills = 5
gsigma = 1
dataset = 'm000'
SEfile = '_seg.h5'
MAfile = '_probs_ws_MA.h5'
MMfile = '_probs_ws_MMdistsum_distfilter.h5'
segmOffset = [250,235,491]
dataset = dataset + '_' + str(x).zfill(nzfills) + '-' + str(X).zfill(nzfills) + \
                    '_' + str(y).zfill(nzfills) + '-' + str(Y).zfill(nzfills) + \
                    '_' + str(z).zfill(nzfills) + '-' + str(Z).zfill(nzfills)
### load the dataset
data, elsize = loadh5(datadir, dataset + '.h5')
data = gaussian_filter(data, gsigma)
# load the section segmentation
segm = loadh5(datadir, dataset + SEfile)[0]
segsliceno = segmOffset[0] - z
# load the MA and MM segmentations
MA = loadh5(datadir, dataset + MAfile)[0]
MM = loadh5(datadir, dataset + MMfile)[0]

# prepare seedslice
seeds_UA = np.copy(segm)
seedslice = seeds_UA[segsliceno,:,:]
seedslice[seedslice<2000] = 0
final_seedslice = np.zeros_like(seedslice)
for l in np.unique(seedslice)[1:]:
    final_seedslice[binary_erosion(seedslice == l, square(5))] = l

seeds_UA[segsliceno,:,:] = final_seedslice

# add knossos seedpoints to seed image
objs = get_knossos_groundtruth(datadir, dataset)
import pprint; pp = pprint.PrettyPrinter(indent=4); pp.pprint(objs[0])
print(np.count_nonzero(seeds_UA))
for obj in objs:
    objval = int(obj['name'][3:7])
    print objval
    # TODO: handle splits and doubles
    for node, coords in obj['nodedict'].iteritems():
        # x and y are swapped in knossos; knossos is in xyz coordframe
        seeds_UA[coords[2]-1,coords[0]-1,coords[1]-1] = objval

print(np.count_nonzero(seeds_UA))
writeh5(seeds_UA, datadir, dataset + '_seeds_UA_knossos.h5', element_size_um=elsize)



# UA = watershed(-data, seeds_UA,
#                mask=np.logical_and(~datamask, ~np.logical_or(MM,MA)))
# writeh5(UA, datadir, dataset + '_probs_ws_UA.h5', element_size_um=elsize)



unzip annotation-160108T1640.058.k.zip
mv annotation.xml annotation_MA.xml


# SegEM
git clone https://github.com/mhlabCodingTeam/SegEM.git
initalSettings
