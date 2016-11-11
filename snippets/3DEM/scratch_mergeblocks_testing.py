source activate scikit-image-devel_0.13
scriptdir="${HOME}/workspace/EM"
datadir="/Users/michielk/oxdata/P01/EM/M3/M3_S1_GNU/M3S1GNU/test" && cd $datadir
datastem='M3S1GNUds7'
CC2Dstem='_labelMA_core2D'

source datastems_90blocks.sh

# with mpi
dspf='ds'; ds=7;
pf='_ws_l0.99_u1.00_s005'
pf='_ws_l0.99_u1.00_s010'
mpirun -np 2 python $scriptdir/convert/EM_mergeblocks.py \
$datadir ${datastems[*]} \
-i ${pf} 'stack' \
-b $zo $yo $xo -p $zs $ys $xs -q $zm $ym $xm -s $zmax $ymax $xmax \
-l -r -F -B 1 ${ds} ${ds} -f 'np.amax' -m

# without mpi
dspf='ds'; ds=7;
pf='_ws_l0.99_u1.00_s005'
pf='_ws_l0.99_u1.00_s010'
python $scriptdir/convert/EM_mergeblocks.py \
$datadir ${datastems[*]} \
-i ${pf} 'stack' \
-b $zo $yo $xo -p $zs $ys $xs -q $zm $ym $xm -s $zmax $ymax $xmax \
-l -r -F -B 1 ${ds} ${ds} -f 'mode'

python $scriptdir/convert/EM_stack2stack.py \
$datadir/M3S1GNUds7${pf}.h5 $datadir/M3S1GNUds7${pf}.nii \
-e $xe $ye $ze -i 'zyx' -l 'xyz' -d uint32 &


for pf in "_ws_l0.99_u1.00_s005" "_ws_l0.99_u1.00_s010"; do
    for datastem in ${datastems[@]}; do
        python $scriptdir/convert/EM_stack2stack.py \
        $datadir/${dataset}${pf}.h5 $datadir/${datastem}${pf}.h5 \
        -e $xe $ye $ze -i 'zyx' -l 'xyz' -p $datastem -b $xo $yo $zo -d uint32 &
        python $scriptdir/convert/EM_stack2stack.py \
        $datadir/${dataset}${pf}.h5 $datadir/${datastem}${pf}.nii \
        -e $xe $ye $ze -i 'zyx' -l 'xyz' -p $datastem -b $xo $yo $zo -d uint32 &
    done
done
for datastem in ${datastems[@]}; do
    mv ${datastem}_ws_l0.99_u1.nii.gz ${datastem}_ws_l0.99_u1.00_s005.nii.gz
    # mv ${datastem}_maskMA_ws_l0.99_u1.nii.gz ${datastem}_maskMA_ws_l0.99_u1.00_s005.nii.gz
    # mv ${datastem}_labelMA_ws_l0.99_u1.nii.gz ${datastem}_labelMA_ws_l0.99_u1.00_s005.nii.gz
done



#### testing mergeblocks with mpi
source activate scikit-image-devel_0.13
scriptdir="${HOME}/workspace/EM"
datadir="${HOME}/oxdata/P01/EM/M3/M3_S1_GNU/M3S1GNU/ds7_arc/mpitest" && cd $datadir
datastem='M3S1GNUds7'

source datastems_09blocks.sh
for pf in '' '_maskMM' '_labelMA'; do
    for datastem in ${datastems[@]}; do
        python $scriptdir/convert/EM_stack2stack.py \
        $datadir/${dataset}${pf}.h5 $datadir/${datastem}${pf}.h5 \
        -e $ze $ye $xe -i 'zyx' -l 'zyx' -p $datastem -b $xo $yo $zo -s 20 20 20
    done
done

pf='_labelMA'
mpirun -np 3 python $scriptdir/convert/EM_mergeblocks.py \
$datadir ${datastems[*]} \
-i ${pf} 'stack' \
-b $xo $yo $zo -p $xs $ys $zs -q $xm $ym $zm -s $xmax $ymax $zmax \
-l -m -F -B 1 4 4 -f 'squeezed_mode'

pf='_labelMA'
python $scriptdir/convert/EM_mergeblocks.py \
$datadir ${datastems[*]} \
-i ${pf} 'stack' \
-b $xo $yo $zo -p $xs $ys $zs -q $xm $ym $zm -s $xmax $ymax $zmax \
-l -r
