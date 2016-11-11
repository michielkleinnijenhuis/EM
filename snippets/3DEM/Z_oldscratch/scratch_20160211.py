- decimate planar 1 deg
- assigns sides/nonsides to vertex groups
- triangulate
- smooth in z (0.5, 10) on nonsides
- laplacian smooth (100, 0.2) (on nonsides?)


ob = C.scene.objects.active
decimate_mesh_planar(ob)
triangulate_mesh(ob)
smooth_mesh_default(ob, use_x=False, use_y=False)
shrink_mesh(ob, 0.0073)
smooth_mesh_laplacian(ob)



# decimate 0.1
laplacian smooth


# To get MM only: boolean modifier (diff) on (MM-MA)
# To get ECS only: boolean modifier (diff) on (ECSbox - compartmentUnion) (Perhaps for every neuron separately)

for comp in MA MM NN GP UA; do
blender -b -P $scriptdir/mesh/stl2blender.py -- \
$datadir/dmcsurf ${comp} -L ${comp} -s 0.5 10 -d 0.2  -e 0.01
done



scriptdir="$HOME/workspace/EM"
DATA="$HOME/oxdata"
datadir="$DATA/P01/EM/M3/M3_S1_GNU" && cd $datadir
dataset='m000'
x=1000; X=1500; y=1000; Y=1500; z=30; Z=460;
python $scriptdir/mesh/label2stl.py $datadir $dataset \
-L '_PA_enforceECS' -f '/stack' -n 5 -o $z $y $x \
-x $x -X $X -y $y -Y $Y -z $z -Z $Z
for comp in MM NN GP UA; do
blender -b -P $scriptdir/mesh/stl2blender.py -- \
$datadir/dmcsurf ${comp} -L ${comp}
done




scriptdir="$HOME/workspace/EM"
DATA="$HOME/oxdata"
datadir="$DATA/P01/EM/M3/M3_S1_GNU" && cd $datadir
dataset='m000'
x=1000; X=1500; y=1000; Y=1500; z=200; Z=300;
python $scriptdir/mesh/label2stl.py $datadir $dataset \
-L '_PA' -f '/stack' -n 5 -o $z $y $x \
-x $x -X $X -y $y -Y $Y -z $z -Z $Z






from glob import glob
fibrepaths = glob(path.join('/Users/michielk/oxdata/P01/EM/M3/M3_S1_GNU', 'dmcsurf_PA', 'UA' + '*.blend'))
for fibre in fibrepaths:
    with D.libraries.load(fibre) as (data_from, data_to):
        data_to.objects = data_from.objects
    for obj in data_to.objects:
        if obj is not None:
            C.scene.objects.link(obj)


import bpy.ops as O
for ob in D.objects:
    if ob.name.startswith('MA'):
        O.object.select_all(action='DESELECT')
        ob.select = True
        C.scene.objects.active = ob
        O.object.mode_set(mode='EDIT')
        O.mesh.select_all(action='SELECT')
        O.transform.shrink_fatten(value=-0.02)
        O.object.mode_set(mode='OBJECT')


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


comp = 'UA'
bpy.ops.object.select_pattern(pattern=comp+'*')
C.scene.objects.active = bpy.context.selected_objects[0]
O.object.join()

ob = C.scene.objects.active
ob = connect_fibres(comp, comp + '*', ob.name)
