# single compartment debug (issues for MM NN UA; MA and GP are OK)
comp=UA
source $HOME/workspace/DifSim/utilities/geoms/pipeline_geoms_3DEM.sh
build_geom M3_S1_GNU '' _${comp} 1 1 01:00:00 difsim_geom_3DEM.py ${comp}
# open_geom M3_S1_GNU '' _${comp} &
run_difsim M3_S1_GNU '' _${comp} 1 1 30:00:00 REFLECTIVE D7.5e-06_5comp N100_5comp P0.0e+00_5comp SE yzqc-003dir 100 '' ASCII 0 compute

/Users/michielk/oxdata/P01/difsim/SE/M3_S1_GNU_${comp}/N100_5comp_P0.0e+00_5comp_D7.5e-06_5comp_REFLECTIVE_SE_pst_yzqc-003dir_100us
~/workspace/mcell/debug/mcell /Users/michielk/oxdata/P01/difsim/geoms/M3_S1_GNU_${comp}_files/mcell/difsim.main_N100_5comp_P0.0e+00_5comp_D7.5e-06_5comp_REFLECTIVE_SE_pst_yzqc-003dir_100us.mdl


# single neuron debug
comp=MM
source $HOME/workspace/DifSim/utilities/geoms/pipeline_geoms_3DEM.sh

build_geom M3_S1_GNU '' _${comp} 1 1 01:00:00 difsim_geom_3DEM.py '3DEM' ${comp}




source $HOME/workspace/DifSim/utilities/geoms/pipeline_geoms_3DEM.sh
scriptdir=$HOME/workspace/DifSim/utilities
DATA=/Users/michielk/oxdata/P01
datadir=$DATA/difsim

comp=MM
for axon in `ls $DATA/EM/M3/M3_S1_GNU/dmcsurf_PA/${comp}*.stl`; do
geomname="${axon##*/}"_d0.2
[ -f $datadir/geoms/$geomname.blend ] || { echo $geomname;
mkdir -p $datadir/geoms/${geomname}_files/geom
cd $datadir/geoms/${geomname}_files
cp $axon $datadir/geoms/${geomname}_files/geom/
blender -b -P $scriptdir/geoms/difsim_geom_3DEM.py -- $datadir $geomname -m '3DEM' -c ${comp} > $datadir/"${axon##*/}"_blender.txt ; }
done

for axon in `ls $DATA/EM/M3/M3_S1_GNU/dmcsurf_PA/${comp}*.stl`; do
geomname="${axon##*/}"_d0.2
run_difsim $geomname '' '' 1 1 30:00:00 REFLECTIVE D7.5e-06_5comp N100_5comp P0.0e+00_5comp SE yzqc-003dir 100 '' ASCII 0 compute >> $datadir/${geomname}_difsim.txt
done

Fatal error: Y partitions closer than interaction diameter
  Y partition #2 at 7.21095
  Y partition #3 at 7.21879
  Interaction diameter 0.0112838
Fatal error: Y partitions closer than interaction diameter
  Y partition #2 at 10.401
  Y partition #3 at 10.412
  Interaction diameter 0.0112838
Fatal error: X partitions closer than interaction diameter
  X partition #2 at 9.9703
  X partition #3 at 9.9802
  Interaction diameter 0.0112838

for geonmame in MM.01135.01.stl MM.01167.01.stl MM.01502.01.stl; do
run_difsim $geomname '' '' 1 1 30:00:00 REFLECTIVE D7.5e-06_5comp N100_5comp P0.0e+00_5comp SE yzqc-003dir 100 '' ASCII 0 compute >> $datadir/"${axon##*/}"_difsim.txt
done



blender -b -P $scriptdir/mesh/stl2blender.py -- $datadir/dmcsurf ${comp} -L ${comp} -s 0.5 10 -d 0.2  -e 0.01







1+5
2+4
3+6

verts = np.loadtxt('/Users/michielk/walls.txt')
verts = np.reshape(verts, [verts.shape[0]/3,-1,3])
v = [(vert[0], vert[1], vert[2]) for vert in verts]
e = []

# ECS overlapped at EPS_C (not at 1e-6)
v1 = (7.20571184158, 7.20539617538, 1.47832834721)
v2 = (7.20571184158, 7.20539617538, 23.16508865356)
v3 = (7.20571184158, 11.06627464294, 23.16508865356)
v1 = (7.20571184158, 11.06627464294, 1.47832834721)
v2 = (7.20571184158, 7.20539617538, 1.47832834721)
v3 = (7.20571184158, 11.06627464294, 23.16508865356)

# bare edge MM?
v1 = (10.87895679474, 8.87973499298, 14.69262790680)
v2 = (10.87892436981, 8.87971782684, 14.69274330139)
v3 = (10.87894153595, 8.87975215912, 14.69285583496)

# walls overlapped MM (identicals)
v1 = (7.324373722, 8.061169624, 10.492948532)
v2 = (7.324340343, 8.061152458, 10.493062019)
v3 = (7.324357033, 8.061186790, 10.493175507)

v = [v1, v2, v3]
e = [[0,1],[1,2],[2,0]]

me = D.meshes.new('tmp' + "Mesh")
ob = D.objects.new('tmp' , me)
C.scene.objects.link(ob)

me = ob.data
me.from_pydata(v, e, [])
C.scene.objects.active = ob
me.update()

ob.select = True
