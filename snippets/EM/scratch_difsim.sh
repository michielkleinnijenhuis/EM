source $HOME/workspace/DifSim/utilities/geoms/pipeline_geoms_3DEM.sh

for c in '3' '4' '5' '6'; do
build_geom M3_S1_GNU $c '' 1 1 01:00:00 difsim_geom_3DEM.py BLEND MF MA
run_difsim M3_S1_GNU $c '' 1 1 30:00:00 REFLECTIVE D7.5e-06_3comp N100_3comp P0.0e+00_3comp SE yzqc-003dir 100 '' ASCII 0 compute
done

build_geom M3_S1_GNU '0' '' 1 1 01:00:00 difsim_geom_3DEM.py BLEND MF MA
run_difsim M3_S1_GNU '0' '' 1 1 30:00:00 REFLECTIVE D7.5e-06_3comp N100_3comp P0.0e+00_3comp SE yzqc-003dir 100 '' ASCII 0 compute

build_geom M3_S1_GNU '0' '' 1 1 01:00:00 difsim_geom_3DEM.py BLEND MF
run_difsim M3_S1_GNU '0' '' 1 1 30:00:00 REFLECTIVE D7.5e-06_2comp N100_2comp P0.0e+00_2comp SE yzqc-003dir 100 '' ASCII 0 compute

build_geom M3_S1_GNU 'd0.02' '' 1 1 01:00:00 difsim_geom_3DEM.py BLEND MF MA

run_difsim M3_S1_GNU 'd0.02' '' 2 16 00:10:00 REFLECTIVE D7.5e-06_3comp N100_3comp P0.0e+00_3comp SE yzqc-003dir 100 '' ASCII 0 devel

run_difsim M3S1GNU 'cu' '' 1 1 30:00:00 REFLECTIVE D7.5e-06_5comp N1000_5comp P0.0e+00_5comp SE yzqc-003dir 100 '' ASCII 0 compute


~/workspace/mcell/build/mcell difsim.main_N100_3comp_P0.0e+00_3comp_D7.5e-06_3comp_REFLECTIVE_SE_pst_yzqc-003dir_100us.mdl


host="ndcn0180@arcus-b.arc.ox.ac.uk"
remdir="/data/ndcn-fmrib-water-brain/ndcn0180/difsim/geoms"
localdir="/Users/michielk/oxdata/P01/difsim/geoms"
f=M3_S1_GNUd0.02
rsync -avz $localdir/${f}* $host:$remdir

run_difsim M3_S1_GNU 'd0.02' '' 1 1 00:10:00 REFLECTIVE D7.5e-06_3comp N100000_3comp P0.0e+00_3comp SE yzqc-003dir 100 '' OFF 0 devel
