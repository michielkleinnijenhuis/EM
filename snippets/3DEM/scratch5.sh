datastem=M3S1GNU_06950-08050_05950-07050_00030-00460
datastem=M3S1GNU_00950-02050_01950-03050_00030-00460
pf=_ws_l0.99_u1.00_s010_svoxsetsMAdel

pf=_labelMA_core2D_fw_3Diter3_closed_proofread2
pf=${pvol}_proofread
pf=_labelMA_core2D_fw_3Diter3_closed_proofread_proofread_split
pf=_labelMF
pf=_labelMM_ws
pf=_labelMA_t0.1_final_ws_l0.99_u1.00_s010
pf=_06950-08050_05950-07050_00030-00460_ws_l0.99_u1.00_s010_svoxsets_MAdel
pf=_labelALL
python $scriptdir/convert/EM_stack2stack.py \
$datadir/"${datastem}${pf}.h5" \
$datadir/"${datastem}${pf}.nii" \
-e $xe $ye $ze -i 'zyx' -l 'xyz' &


pf="${volws}"
python $scriptdir/convert/EM_stack2stack.py \
$datadir/${datastem}${pf}.h5 $datadir/${datastem}${pf}.nii \
-e $xe $ye $ze -i 'zyx' -l 'xyz' &

pf="${volws}${NoRpf}_iter${iter}"
pf="${volws}${NoRpf}_iter${iter}_automerged"
python $scriptdir/convert/EM_stack2stack.py \
$datadir/${datastem}${pf}.h5 $datadir/${datastem}${pf}.nii \
-e $xe $ye $ze -i 'zyx' -l 'xyz' &




pf='_prediction_NPminimal_ws_l0.99_u1.00_s010_PA_str2_iter5_parallelh5_thr0.5_alg1'
python $scriptdir/convert/EM_mergeblocks.py \
$datadir/pred_new ${datastems[*]} \
-i ${pf} 'stack' -o "${pf}_amax_nb" 'stack' \
-b $zo $yo $xo -p $zs $ys $xs -q $zm $ym $xm -s $zmax $ymax $xmax \
-l -n -F -B 1 7 7 -f 'np.amax' &



pf='_prediction_NPminimal_ws_l0.99_u1.00_s010_PA_str2_iter5_parallelh5_thr0.5_alg1_amax_nb'
python $scriptdir/convert/EM_stack2stack.py \
$datadir/${dataset}${pf}.h5 $datadir/${datastem}${pf}.h5 \
-e $ze $(echo $ye*$ds | bc) $(echo $xe*$ds | bc) -i 'zyx' -l 'zyx' \
-X 1311 -Y 1255 -m
python $scriptdir/convert/EM_stack2stack.py \
$datadir/${datastem}${pf}.h5 \
$datadir/${datastem}${pf}.nii \
-e $xe $ye $ze -i 'xyz' -l 'xyz'

pf=_maskUA
pf=_maskALL
pf=_labelALL
pf=_labelMM_ws
pf='_maskALLlabeled'
pf='_maskALLlabeled_large'

for pf in '_maskALLlabeled_small' '_maskALLlabeled_large' '_maskALLlabeled_large'; do
python $scriptdir/convert/EM_stack2stack.py \
$datadir/"${datastem}${pf}.h5" \
$datadir/"${datastem}${pf}.nii" \
-e $xe $ye $ze -i 'zyx' -l 'xyz' &
done

pf=_maskALL
python $scriptdir/convert/EM_stack2stack.py \
$datadir/"${datastem}${pf}.h5" \
$datadir/"${datastem}${pf}.nii" \
-e $xe $ye $ze -i 'zyx' -l 'xyz' &



pf='_prediction_NPminimal_ws_l0.99_u1.00_s010_PA_str2_iter5_parallelh5_thr0.5_alg1'
for datastem in ${datastems[@]}; do
ln -s ../$datastem${pf}.h5 $datastem${pf}.h5
done









datastem=M3S1GNUds7
pf='_prediction_NPminimal_ws_l0.99_u1.00_s010_PA_str2_iter5_parallelh5_thr0.5_alg1_mode'
pf='_prediction_NPminimal_ws_l0.99_u1.00_s010_PA_str2_iter5_parallelh5_thr0.5_alg1_mode_nb'

mv M3S1GNU${pf}.h5 ${datastem}${pf}.h5
h5ls -v ${datastem}${pf}.h5

python $scriptdir/convert/EM_stack2stack.py \
$datadir/${datastem}${pf}.h5 $datadir/${datastem}${pf}.nii \
-e $xe $ye $ze -i 'xyz' -l 'xyz'





source datastems_09blocks.sh
for pf in "_maskDS" "_maskMM" "_labelMM_ws" "_labelMA_2D_proofread"; do
    for datastem in ${datastems[@]}; do
        python $scriptdir/convert/EM_stack2stack.py \
        $datadir/${dataset}${dspf}${ds}${pf}.h5 $datadir/${datastem}${pf}.h5 \
        -e $xe $ye $ze -i 'zyx' -l 'zyx' -p $datastem -b $xo $yo $zo &
    done
    for datastem in ${datastems[@]}; do
        python $scriptdir/convert/EM_stack2stack.py \
        $datadir/${dataset}${dspf}${ds}${pf}.h5 $datadir/${datastem}${pf}.nii \
        -e $xe $ye $ze -i 'zyx' -l 'xyz' -p $datastem -b $xo $yo $zo &
    done
done

datastem=M3S1GNUds7_00000-00438_00000-00419_00030-00460
python $scriptdir/mesh/EM_separate_sheaths.py \
$datadir $datastem \
-l "_labelMA_2D_proofread" 'stack' \
-l "_labelMM_ws" 'stack' \
--maskMM '_maskMM' 'stack' --maskDS '_maskDS' 'stack' \
-o '_labelMM_remote' 'stack' --MAdilation 3 -w

for pf in '_labelMM_local_MAdilation' '_labelMM_local_MM_wsmask' '_labelMM_local_dist' '_labelMM_local_ws' ; do
python $scriptdir/convert/EM_stack2stack.py \
$datadir/"${datastem}${pf}.h5" \
$datadir/"${datastem}${pf}.nii" \
-e $xe $ye $ze -i 'zyx' -l 'xyz' &
done


source datastems_09blocks.sh
for pf in "_maskDS" "_maskMM" "_labelMM_ws" "_labelMA_2D_proofread"; do
    python $scriptdir/convert/EM_stack2stack.py \
    $datadir/${dataset}${dspf}${ds}${pf}.h5 $datadir/${datastem}${pf}.h5 \
    -e $xe $ye $ze -i 'zyx' -l 'zyx' -p $datastem -b $xo $yo $zo &
    python $scriptdir/convert/EM_stack2stack.py \
    $datadir/${dataset}${dspf}${ds}${pf}.h5 $datadir/${datastem}${pf}.nii \
    -e $xe $ye $ze -i 'zyx' -l 'xyz' -p $datastem -b $xo $yo $zo &
done

for pf in '_labelMM_remote_MAdilation' '_labelMM_remote_MM_wsmask' '_labelMM_remote_dist' '_labelMM_remote_ws' ; do
for pf in '_labelMM_remote_distsum' '_labelMM_remote_sw'; do
python $scriptdir/convert/EM_stack2stack.py \
$datadir/"${datastem}${pf}.h5" \
$datadir/"${datastem}${pf}.nii" \
-e $xe $ye $ze -i 'zyx' -l 'xyz' &
done



for pf in "_maskALLlabeled_large"; do
    for datastem in ${datastems[@]}; do
        python $scriptdir/convert/EM_stack2stack.py \
        $datadir/${dataset}${dspf}${ds}${pf}.h5 $datadir/${datastem}${pf}.nii \
        -e $xe $ye $ze -i 'zyx' -l 'xyz' -p $datastem -b $xo $yo $zo -d uint32 &
    done
done
