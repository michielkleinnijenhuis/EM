###=========================================================================###
### cutout
###=========================================================================###
scriptdir="$HOME/workspace/EM"
DATA="$HOME/oxdata"
datadir="$DATA/P01/EM/M3/M3_S1_GNU/M3S1GNU/ds7_arc" && cd $datadir
outdir="/Users/michielk/oxdox/papers/abstracts/ISMRM2017/Eposter/anims/M3S1GNUvols/cutout"
dataset='M3S1GNUds7'

for pf in '' _maskGL_final _maskMA_final _maskMF_final _maskMM_final _maskUA_final _maskXX \
_labelGL_final _labelMA_final _labelMF_final _labelMM_final _labelUA_final _labelALL_final; do
# pf=_labelMA_t0.1_final_ws_l0.99_u1.00_s010
python $scriptdir/convert/EM_stack2stack.py \
${datadir}/${dataset}${pf}.h5 ${outdir}/${dataset}cu${pf}.h5 \
-i 'zyx' -l 'zyx' -e -0.0511 -0.0511 0.05 \
-x 300 -X 500 -y 300 -Y 500 -z 0 -Z 430
done

python $scriptdir/convert/EM_stack2stack.py \
${outdir}/${dataset}cu.h5 ${outdir}/${dataset}cu.nii.gz \
-i 'zyx' -l 'xyz' -e -0.0511 -0.0511 0.05 -u
for pf in '' _maskGL_final _maskMA_final _maskMF_final _maskMM_final _maskUA_final _maskXX \
_labelGL_final _labelMA_final _labelMF_final _labelMM_final _labelUA_final _labelALL_final; do
pf=_labelMA_t0.1_final_ws_l0.99_u1.00_s010
python $scriptdir/convert/EM_stack2stack.py \
${outdir}/${dataset}cu${pf}.h5 ${outdir}/${dataset}cu${pf}.nii.gz \
-i 'zyx' -l 'xyz' -e -0.0511 -0.0511 0.05
done

###=========================================================================###
### label2stl
###=========================================================================###
### on local machine
scriptdir="$HOME/workspace/EM"
datadir="/Users/michielk/oxdox/papers/abstracts/ISMRM2017/Eposter/anims/M3S1GNUvols/cutout"
datastem='M3S1GNUds7cu'

for pf in 'GL' 'MA' 'MF' 'UA'; do
for pf in 'MA' 'MF'; do
python $scriptdir/mesh/label2stl.py $datadir $datastem \
-L "_label${pf}_final" -f 'stack' -o 0 0 0
done
# MArejects@1000vox: set([1569, 326, 488, 1962, 3051, 1804, 1903, 467, 212, 2486, 378])
# MFrejects@1000vox: set([355, 229, 488, 489, 1962, 1903, 467, 2486, 186, 495])

cd $datadir/$datastem/dmcsurf-1-1-1
for f in `ls`; do
mv ${f} ${f/_label/label}
done

for pf in 'GL' 'MA' 'MF' 'UA'; do
comp="label${pf}_final"
blender -b -P $scriptdir/mesh/stl2blender.py -- \
"$datadir/$datastem/dmcsurf_1-1-1" ${comp} -L ${comp} \
-s 100 0.1 True True True -d 0.1 -e 0 -c
done




source $HOME/workspace/DifSim/utilities/geoms/pipeline_geoms_3DEM.sh
build_geom M3S1GNUcu '' '' 1 1 01:00:00 difsim_geom_3DEM.py BLEND MF MA GL UA
run_difsim M3S1GNUcu '' '' 1 1 30:00:00 REFLECTIVE D7.5e-06_5comp N100_5comp P0.0e+00_5comp SE yzqc-003dir 100 '' ASCII 0 compute
