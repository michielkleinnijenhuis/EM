
host="ndcn0180@arcus-b.arc.ox.ac.uk"
remdir="/data/ndcn-fmrib-water-brain/ndcn0180/EM/M3/M3_S1_GNU_old"
localdir="/Users/michielk/oxdata/P01/EM/M3/M3_S1_GNU/M3S1GNU/ds7_arc/pred_new/test"

f="M3S1GNU_00000-01050_*-*_00030-00460_prediction_NPminimal_ws_l0.99_u1.00_s010_PA_str2_iter5_parallelh5_thr0.5_alg1.h5"
f="M3S1GNU_00950-02050_*-*_00030-00460_prediction_NPminimal_ws_l0.99_u1.00_s010_PA_str2_iter5_parallelh5_thr0.5_alg1.h5"
f="M3S1GNU_06950-08050_05950-07050_00030-00460_probs1_eed2_main.h5"
rsync -avz "$host:$remdir/${f}" $localdir

## testing EM_mergeblocks.py
scriptdir="${HOME}/workspace/EM"
source activate scikit-image-devel_0.13

datadir="/Users/michielk/oxdata/P01/EM/M3/M3_S1_GNU/M3S1GNU/ds7_arc/pred_new/test"
python $scriptdir/convert/EM_mergeblocks.py $datadir \
M3S1GNU_00000-01050_00000-01050_00030-00460 M3S1GNU_00000-01050_00950-02050_00030-00460 \
M3S1GNU_00000-01050_01950-03050_00030-00460 M3S1GNU_00000-01050_02950-04050_00030-00460 \
M3S1GNU_00000-01050_03950-05050_00030-00460 M3S1GNU_00000-01050_04950-06050_00030-00460 \
M3S1GNU_00000-01050_05950-07050_00030-00460 M3S1GNU_00000-01050_06950-08050_00030-00460 \
M3S1GNU_00000-01050_07950-08786_00030-00460 \
-i "_prediction_NPminimal_ws_l0.99_u1.00_s010_PA_str2_iter5_parallelh5_thr0.5_alg1" 'stack' \
-o '_amaxX4' 'stack' -b 30 0 0 -p 430 1000 1000 \
-q 0 50 50 -s 460 8786 9179 -l -r -F -B 1 7 7 -f 'np.amax' -n

datastem='M3S1GNU'
comp='_amaxX4'
python $scriptdir/mesh/label2stl.py $datadir $datastem \
-L ${comp} -f 'stack' -o 30 0 0
blender -b -P $scriptdir/mesh/stl2blender.py -- \
"$datadir/$datastem/dmcsurf_1-1-1" ${comp} -L ${comp} \
-s 100 0.1 True True True -d 0.02 -e 0



datadir="/Users/michielk/oxdata/P01/EM/M3/M3_S1_GNU/M3S1GNU/ds7_arc/pred_new/test"
python $scriptdir/convert/EM_mergeblocks.py $datadir \
M3S1GNU_00000-01050_00000-01050_00030-00460 M3S1GNU_00000-01050_00950-02050_00030-00460 \
M3S1GNU_00950-02050_00000-01050_00030-00460 M3S1GNU_00950-02050_00950-02050_00030-00460 \
-i "_prediction_NPminimal_ws_l0.99_u1.00_s010_PA_str2_iter5_parallelh5_thr0.5_alg1" 'stack' \
-o '_amaxX7' 'stack' -b 30 0 0 -p 430 1000 1000 \
-q 0 50 50 -s 460 2050 2050 -l -r -F -B 1 7 7 -f 'np.amax' -n

datastem='M3S1GNU'
comp='_amaxX6'
python $scriptdir/mesh/label2stl.py $datadir $datastem \
-L ${comp} -f 'stack' -o 30 0 0
blender -b -P $scriptdir/mesh/stl2blender.py -- \
"$datadir/$datastem/dmcsurf_1-1-1" ${comp} -L ${comp} \
-s 100 0.1 True True True -d 0.02 -e 0



xe=0.0511; ye=0.0511; ze=0.05;
datastem='M3S1GNU'
pf='_amaxX4'
python $scriptdir/convert/EM_stack2stack.py \
$datadir/"${datastem}${pf}.h5" \
$datadir/"${datastem}${pf}.nii" \
-e $xe $ye $ze -i 'zyx' -l 'xyz' &


xe=0.0073; ye=0.0073; ze=0.05;
datastem='M3S1GNU_00000-01050_00000-01050_00030-00460'
datastem='M3S1GNU_00000-01050_00950-02050_00030-00460'
pf='_prediction_NPminimal_ws_l0.99_u1.00_s010_PA_str2_iter5_parallelh5_thr0.5_alg1'
python $scriptdir/convert/EM_stack2stack.py \
$datadir/"${datastem}${pf}.h5" \
$datadir/"${datastem}${pf}.nii" \
-e $xe $ye $ze -i 'zyx' -l 'xyz' &
