

xe=0.0511; ye=0.0511; ze=0.05;
datastem='M3S1GNUds7'
pf='_segmented_large'
python $scriptdir/convert/EM_stack2stack.py \
$datadir/"${datastem}${pf}.h5" \
$datadir/"${datastem}${pf}.nii" \
-e $xe $ye $ze -i 'zyxc' -l 'xyzc' &
