source activate scikit-image-devel_0.13
scriptdir="${HOME}/workspace/EM"
datadir="/Users/michielk/oxdata/P01/EM/M3/M3_S1_GNU/M3S1GNU/ds7_arc" && cd $datadir
dataset='M3S1GNU'
dspf='ds'; ds=7;
datastem=${dataset}${dspf}${ds}
CC2Dstem='_labelMA_core2D'

python $scriptdir/supervoxels/conn_comp.py \
$datadir $datastem -M '3D' \
--maskDS '_maskDS' 'stack' --maskMM '_maskMM' 'stack' \
-a 200 -q 2000 -o '_labelMA_core3D' 'stack' &

mpirun -np 6 python $scriptdir/supervoxels/conn_comp.py \
$datadir $datastem -M '2D' \
-d 0 -o ${CC2Dstem} 'stack' -m

# mpi not working yet
python $scriptdir/supervoxels/conn_comp.py \
$datadir $datastem -M '2Dfilter' \
-d 0 --maskMB '_maskMB' 'stack' \
-i ${CC2Dstem} 'stack' -o "${CC2Dstem}_fw" 'stack' \
-a 10 -A 1500 -e 0 -x 0.30 -s 0.50 \
-p 'label' 'area' 'mean_intensity' 'eccentricity' \
'solidity' 'extent' 'euler_number'

# mpi not working yet
python $scriptdir/supervoxels/conn_comp.py \
$datadir $datastem -M '2Dprops' \
-i ${CC2Dstem} 'stack' -o "${CC2Dstem}_fw" 'stack' \
-p 'label' 'area' 'mean_intensity' 'eccentricity' \
'solidity' 'extent' 'euler_number'

python $scriptdir/supervoxels/conn_comp.py \
$datadir $datastem -M '2Dto3Dlabel' \
-i "${CC2Dstem}_fw_label" 'stack' -o "${CC2Dstem}_fw_3Dlabeled" 'stack'

mpirun -np 6 python $scriptdir/supervoxels/merge_slicelabels.py \
$datadir $datastem -M 'MAstitch' \
-l "${CC2Dstem}_fw_3Dlabeled" 'stack' -o "${CC2Dstem}_fw_3Diter1" 'stack' \
-d 0 -r 4 -t 0.50 -m
python $scriptdir/supervoxels/merge_slicelabels.py \
$datadir $datastem -M 'MAfwmap' -d 0 \
-l "${CC2Dstem}_fw_3Dlabeled" 'stack' -o "${CC2Dstem}_fw_3Diter1" 'stack' \
--maskMM '_maskMM' 'stack' -c 1 -s 200
mpirun -np 6 python $scriptdir/supervoxels/merge_slicelabels.py \
$datadir $datastem -M 'MAstitch' \
-l "${CC2Dstem}_fw_3Diter1" 'stack' -o "${CC2Dstem}_fw_3Dmerged" 'stack' \
-d 0 -r 4 -t 0.50 -m;
python $scriptdir/supervoxels/merge_slicelabels.py \
$datadir $datastem -M 'MAfwmap' -d 0 \
-l "${CC2Dstem}_fw_3Diter1" 'stack' -o "${CC2Dstem}_fw_3Dmerged" 'stack' \
--maskMM '_maskMM' 'stack' -c 1 -s 200

### inspect properties
python $scriptdir/supervoxels/conn_comp.py \
$datadir $datastem -M '2Dfilter' -E 1 \
-i ${CC2Dstem} 'stack' -o "${CC2Dstem}_fw_nf" 'stack' \
-p 'label' 'area' 'mean_intensity' 'eccentricity' \
'solidity' 'extent' 'euler_number'
python $scriptdir/supervoxels/conn_comp.py \
$datadir $datastem -M '2Dprops' \
-i ${CC2Dstem} 'stack' -o "${CC2Dstem}_fw_nf" 'stack' \
-p 'label' 'area' 'mean_intensity' 'eccentricity' \
'solidity' 'extent' 'euler_number'

source datastems_09blocks.sh
for propname in 'label' 'area' 'mean_intensity' 'eccentricity' \
  'solidity' 'extent' 'euler_number'; do
    pf="${CC2Dstem}_fw_nf_${propname}"
    for datastem in ${datastems[@]}; do
        python $scriptdir/convert/EM_stack2stack.py \
        $datadir/${dataset}${dspf}${ds}${pf}.h5 $datadir/${datastem}${pf}.nii \
        -e $xe $ye $ze -i 'zyx' -l 'xyz' -p $datastem -b $xo $yo $zo
    done
done
# ds7 ARC
# size (-a -A) is a very good disciminator: between 10 and 400 (this would remove axons transverse to the slice direction)
# eccentricity (-E) is not a very good discriminator
# euler_number (-e) can be a very good discriminator: set to 0
# extent (-x) is not a good discriminator
# solidity (-e) can be a good discriminator: set to 0.50 (this does broaden the gap at nodes of ranvier)
# mean_intensity_mb (-I) can be a very good discriminator: set to 0.5 (for most axons, but fails completely at some others)


# distribute over blocks for proofreading
source datastems_09blocks.sh
# for pf in '' '_maskMM' "${CC2Dstem}_fw_3Dlabeled" "${CC2Dstem}_fw_3Diter1" "${CC2Dstem}_fw_3Diter2" "${CC2Dstem}_fw_3Diter3" '_labelMA_core3D'; do
# for pf in "_labelMM_ws"; do
# for pf in "_labelMA_ws_l0.99_u1.00_s005" "_maskMA_ws_l0.99_u1.00_s005"; do
for pf in "_ws_l0.99_u1.00_s005"; do
    python $scriptdir/convert/EM_stack2stack.py \
    $datadir/${dataset}${dspf}${ds}${pf}.h5 $datadir/${dataset}${dspf}${ds}${pf}.nii \
    -e $xe $ye $ze -i 'zyx' -l 'xyz' &
    for datastem in ${datastems[@]}; do
        python $scriptdir/convert/EM_stack2stack.py \
        $datadir/${dataset}${dspf}${ds}${pf}.h5 $datadir/${datastem}${pf}.nii \
        -e $xe $ye $ze -i 'zyx' -l 'xyz' -p $datastem -b $xo $yo $zo -d uint32 &
    done
done
# delete files after proofreading
datastem="${dataset}${dspf}${ds}"
dellabels=`grep $datastem $deletefile | awk '{$1 = ""; print $0;}'`
python $scriptdir/supervoxels/delete_labels.py \
$datadir $datastem \
-L "${CC2Dstem}_fw_3Diter1" 'stack' -D $dellabels -o "_proofread"
# and redistribute
source datastems_09blocks.sh
for pf in "${CC2Dstem}_fw_3Diter1_proofread"; do
    for datastem in ${datastems[@]}; do
        python $scriptdir/convert/EM_stack2stack.py \
        $datadir/${dataset}${dspf}${ds}${pf}.h5 $datadir/${datastem}${pf}.nii \
        -e $xe $ye $ze -i 'zyx' -l 'xyz' -p $datastem -b $xo $yo $zo &
    done
done
python $scriptdir/convert/EM_stack2stack.py \
$datadir/"M3S1GNUds7${l}.h5" $datadir/"M3S1GNUds7${l}.nii" \
-e $xe $ye $ze -i 'zyx' -l 'xyz' &


# fill holes
methods='42'; close=9
python $scriptdir/supervoxels/fill_holes.py \
$datadir $datastem -w ${methods} \
-L "${CC2Dstem}_fw_3Diter3" 'stack' \
--maskDS '_maskDS' 'stack' \
--maskMM '_maskMM' 'stack' \
--maskMX '_maskMM-0.02' 'stack' \
--maskMA '_maskMA' 'stack' \
-o "_closed" 'stack'



## use the holes for proofreading erroneous merges!
# first do a CC split
# do a manual disconnection pass on ${pvol}_proofread_split.nii.gz >> ${pvol}_proofread_split_manedit.nii.gz
# do CC split again with manually edited aux volume
export splitfile="$datadir/${datastem}${CC2Dstem}_fw_3Diter3_closed_split.txt"
echo "M3S1GNUds7: \
1410 1236 1159 1216 1339 1149 1423 1123 1251 565 1131 622 926 1204 634 1118 \
989 315 1723 295 53 184 379 1758 1923 622 276 1725 249 3176 179 1510 1016 1293 \
2886 3058 2996 2613 1103 \
" > $splitfile
splitlabels=`grep $datastem $splitfile | awk '{$1 = ""; print $0;}'`

python $scriptdir/supervoxels/delete_labels.py \
$datadir $datastem \
-L "${CC2Dstem}_fw_3Diter3_closed" 'stack' \
-o "_proofread" 'stack' -C -q 20 -S $splitlabels
python $scriptdir/supervoxels/delete_labels.py \
$datadir $datastem \
-L "${CC2Dstem}_fw_3Diter3_closed" 'stack' -o "_proofread" 'stack' -O -C -q 20 \
-S $splitlabels -A "${CC2Dstem}_fw_3Diter3_closed_proofread_split_manedit.nii.gz" 'stack'

export splitfile="$datadir/${datastem}${CC2Dstem}_fw_3Diter3_closed_split2.txt"
echo "M3S1GNUds7: \
2117 2015 1347 379 634 622 1148 1232 2410 1853 3213 2273 1067 1260 1021 1222 \
" > $splitfile
splitlabels=`grep $datastem $splitfile | awk '{$1 = ""; print $0;}'`

python $scriptdir/supervoxels/delete_labels.py \
$datadir $datastem \
-L "${CC2Dstem}_fw_3Diter3_closed" 'stack' \
-o "_proofread2" 'stack' -O -C -q 20 -S $splitlabels
splitlabels=`grep $datastem $splitfile | awk '{$1 = ""; print $0;}'`
python $scriptdir/supervoxels/delete_labels.py \
$datadir $datastem \
-L "${CC2Dstem}_fw_3Diter3_closed_proofread" 'stack' -o "_proofread" 'stack' -O -C -q 20 \
-S $splitlabels -A "${CC2Dstem}_fw_3Diter3_closed_proofread2_split_manedit.nii.gz" 'stack'

# export splitfile="$datadir/${datastem}${CC2Dstem}_fw_3Diter3_closed_split2.txt"
# echo "M3S1GNUds7: \
# 963 \
# " > $splitfile


# do hole-filling again
methods='2'; close=9
python $scriptdir/supervoxels/fill_holes.py \
$datadir $datastem -w ${methods} \
-L "${CC2Dstem}_fw_3Diter3_closed_proofread_proofread" 'stack' \
--maskDS '_maskDS' 'stack' \
--maskMM '_maskMM' 'stack' \
--maskMX '_maskMM-0.02' 'stack' \
--maskMA '_maskMA' 'stack' \
-o "_closed" 'stack'


# delete some labels (deletes some labels at the edge as well)
#3490 3571 3583 3528 3533 3394 3514 3542 3456 3401 3236 3325 3362 3355 3379 3593 3299 3096 2982 2857 2982 3467 2596 8 18 107 157 167 203 367 393 510 677 662 485 741 799 820 889 964 1401 1353 994 2119 1326 1960 2534 3316 3406 3605 3506 3363 3206 2997 3206 3182 2215 3485
python $scriptdir/supervoxels/delete_labels.py \
$datadir $datastem \
-L "${CC2Dstem}_fw_3Diter3_closed_proofread_proofread_closed" 'stack' \
-o "_proofread" 'stack' -q 5000 \
-d "$datadir/${datastem}${CC2Dstem}_fw_3Diter3_closed_proofread_manudelete.txt" \
-e "$datadir/${datastem}${CC2Dstem}_fw_3Diter3_closed_proofread_manudelete_except.txt"

cp "${datadir}/${datastem}${CC2Dstem}_fw_3Diter3_closed_proofread_proofread_closed_proofread.h5"  "${datadir}/${datastem}_labelMA_2D.h5"

### some new variables for a new stage
datastem="${dataset}${dspf}${ds}"
l=0.99; u=1.00; s=010;
svoxpf="_ws_l${l}_u${u}_s${s}"
t=0.1
# lvol="${CC2Dstem}_fw_3Diter3"
vol2d="_labelMA_2D"
volws="_labelMA_t${t}${svoxpf}"
NoRpf='_NoR_s5000'

# svox agglomeration
python $scriptdir/supervoxels/agglo_from_labelmask.py \
$datadir $datastem \
-l ${vol2d} 'stack' -s ${svoxpf} 'stack' \
-o "_labelMA_t${t}" 'stack' -t $t

# nodes of ranvier detection
iter=1
python $scriptdir/supervoxels/nodes_of_ranvier.py \
$datadir $datastem \
-l ${volws} 'stack' -s 5000 \
-o "${volws}${NoRpf}_iter${iter}" 'stack' \
-S '_maskDS_invdil' 'stack'
iter=2
python $scriptdir/supervoxels/nodes_of_ranvier.py \
$datadir $datastem -m 'nomerge' \
-l "${volws}${NoRpf}_iter$((iter-1))_automerged" 'stack' \
-o "${volws}${NoRpf}_iter${iter}" 'stack' \
-S '_maskDS_invdil' 'stack'

# proofreading of NoR mask
for pvol in "${vol2d}" "${volws}"; do
python $scriptdir/supervoxels/delete_labels.py \
$datadir $datastem \
-L ${pvol} 'stack' -o "_proofread" 'stack' -O -p 20 \
-D 2390 3460 3423 1846 \
-d "$datadir/${datastem}${volws}${NoRpf}_iter1_smalllabels.pickle" \
-e "$datadir/${datastem}${CC2Dstem}_fw_3Diter3_closed_proofread_manudelete_except.txt" \
-m "$datadir/${datastem}${volws}${NoRpf}_iter1_automerged_proofread.txt"
done

# TODO:
# merge: 1336 + 2625 1233+1264+2046
# split 2154 (2154+2645) 963
# add (105,1166,343)

# final agglo MA
python $scriptdir/supervoxels/agglo_from_labelmask.py \
$datadir $datastem \
-l "_labelMA_2D_proofread" 'stack' \
-s ${svoxpf} 'stack' \
-o "_labelMA_t${t}_final" 'stack' -t $t

python $scriptdir/mesh/EM_separate_sheaths.py \
$datadir $datastem \
-l "_labelMA_2D_proofread" 'stack' -o '_labelMM_local' 'stack' \
--maskMM '_maskMM' 'stack' --maskDS '_maskDS' 'stack' \
--MAdilation 5





















### hopefully not needed anymore now?
# splitting labels in iter3D into labels from iter2D (manually edited label 267 and 295)
pvol="${CC2Dstem}_fw_3Diter3_filled_manedit"
source proofreading_NoR.sh $pvol
datastem="${dataset}${dspf}${ds}"
splitlabels=`grep $datastem $splitfile | awk '{$1 = ""; print $0;}'`
python $scriptdir/supervoxels/delete_labels.py \
$datadir $datastem \
-L ${pvol} 'stack' -o "_proofread" -O -C -q 20 -n -i 'xyz' -o 'zyx' \
-S $splitlabels -A "${CC2Dstem}_fw_3Diter2" 'stack'
# fslmaths M3S1GNUds7_labelMA_core2D_fw_3Diter3 -sub M3S1GNUds7_labelMA_core2D_fw_3Diter3_manedit M3S1GNUds7_labelMA_core2D_fw_3Diter3_manedit_diff
# [53, 295, 276, 2613, 2015, 2117]

# new svox agglomeration (after iter3 split) (renamed from 'zyx'...)
python $scriptdir/supervoxels/agglo_from_labelmask.py \
$datadir $datastem \
-l "${CC2Dstem}_fw_3Diter3_filled_manedit_proofread" 'stack' -s ${svoxpf} 'stack' \
-o "_labelMA_t${t}" 'stack' -t $t


# new NoR (after iter3 split)  # do I reverse the fill_holes here with filtering small labels?
iter=1
python $scriptdir/supervoxels/nodes_of_ranvier.py \
$datadir $datastem \
-l $lvol 'stack' -s 5000 \
-o "${lvol}${NoRpf}_iter${iter}" 'stack' \
-S '_maskDS_invdil' 'stack'
iter=2
python $scriptdir/supervoxels/nodes_of_ranvier.py \
$datadir $datastem \
-l "${lvol}${NoRpf}_iter$((iter-1))_automerged" 'stack' \
-o "${lvol}${NoRpf}_iter${iter}" 'stack' \
-S '_maskDS_invdil' 'stack'


# proofreading of NoR mask
for pvol in "${lvol}" "${CC2Dstem}_fw_3Diter3_filled_manedit_proofread"; do
python $scriptdir/supervoxels/delete_labels.py \
$datadir $datastem \
-L ${pvol} 'stack' -o "_proofread" -O \
-d "$datadir/${datastem}${lvol}${NoRpf}_iter1_smalllabels.pickle" \
"$datadir/${datastem}${lvol}${NoRpf}_manudelete.txt" \
-m "$datadir/${datastem}${lvol}${NoRpf}_iter1_automerged_proofread.txt" \
"$datadir/${datastem}${lvol}${NoRpf}_missinglabels.txt" \
"$datadir/${datastem}${lvol}${NoRpf}_manumerged.txt"
python $scriptdir/convert/EM_stack2stack.py \
$datadir/"${datastem}${pvol}_proofread.h5" \
$datadir/"${datastem}${pvol}_proofread.nii" \
-e $xe $ye $ze -i 'zyx' -l 'xyz' &
done






# new iter
python $scriptdir/supervoxels/nodes_of_ranvier.py \
$datadir $datastem \
-l "${pvol}_proofread" 'stack' \
-o "${pvol}_proofread${NoRpf}" 'stack' \
-S '_maskDS_invdil' 'stack'
# TODO: split 1016 / 1578
# TODO: merge 1742 2076 2348
# TODO: merge 2014 1357
# TODO: merge 3153 3593
# TODO: merge 516 3616
# TODO: merge 988 1821
1926 2141
2273 3623
1949 3319
803 1999
2063 225
293 2201
1233 1264
# or apply automerge?


split 652 / merge 652 3120
split 1347

# 3Diter3_filled splitCC
947 450 1726

# 3Diter3_filled merge
96 1994

# 3Diter3_filled delete
2213



python $scriptdir/convert/EM_stack2stack.py \
$datadir/"${datastem}${pvol}_proofread${NoRpf}.h5" \
$datadir/"${datastem}${pvol}_proofread${NoRpf}.nii" \
-e $xe $ye $ze -i 'zyx' -l 'xyz' &

python $scriptdir/convert/EM_stack2stack.py \
$datadir/"${datastem}${pvol}_proofread${NoRpf}_automerged.h5" \
$datadir/"${datastem}${pvol}_proofread${NoRpf}_automerged.nii" \
-e $xe $ye $ze -i 'zyx' -l 'xyz' &








# final agglo
python $scriptdir/supervoxels/agglo_from_labelmask.py \
$datadir $datastem \
-l "${CC2Dstem}_fw_3Diter3_manedit_proofread_proofread" 'stack' \
-s ${svoxpf} 'stack' \
-o "_labelMA_t${t}_test" 'stack' -t $t

python $scriptdir/convert/EM_stack2stack.py \
$datadir/"${datastem}_labelMA_t${t}_test${svoxpf}".h5 \
$datadir/"${datastem}_labelMA_t${t}_test${svoxpf}".nii \
-e $xe $ye $ze -i 'zyx' -l 'xyz' &


# filtering the NoR from the agglo_ws volume
python $scriptdir/supervoxels/filter_NoR.py \
$datadir $datastem \
-l "${lvol}_proofread" 'stack' \
-L "${CC2Dstem}_fw_3Diter3_manedit_proofread_proofread" 'stack' \
-o "_labelMA" 'stack'

pf="_labelMA"
python $scriptdir/convert/EM_stack2stack.py \
$datadir/"${datastem}${pf}.h5" \
$datadir/"${datastem}${pf}.nii" \
-e $xe $ye $ze -i 'zyx' -l 'xyz' &


# testing separate sheaths on ds7
# l="${CC2Dstem}_fw_3Diter3"
python $scriptdir/mesh/EM_separate_sheaths.py \
$datadir $datastem \
-l "_labelMA" 'stack' \
--maskMM '_maskMM' 'stack' --maskDS '_maskDS' 'stack' \
--MAdilation 5
## the distance_transform is mem intensive (30+GB), but doesn't take very long
## ws is also mem intensive

#dilate maskMA and constrain the watershed within
python $scriptdir/convert/EM_stack2stack.py \
$datadir/M3S1GNUds7_labelMM_ws_MAdilation.h5 \
$datadir/M3S1GNUds7_labelMM_ws_MAdilation.nii \
-e $xe $ye $ze -i 'zyx' -l 'xyz' &
python $scriptdir/convert/EM_stack2stack.py \
$datadir/M3S1GNUds7_labelMM_ws_MM_wsmask.h5 \
$datadir/M3S1GNUds7_labelMM_ws_MM_wsmask.nii \
-e $xe $ye $ze -i 'zyx' -l 'xyz' &
python $scriptdir/convert/EM_stack2stack.py \
$datadir/M3S1GNUds7_labelMM_ws_dist.h5 \
$datadir/M3S1GNUds7_labelMM_ws_dist.nii \
-e $xe $ye $ze -i 'zyx' -l 'xyz' -d 'float' &
python $scriptdir/convert/EM_stack2stack.py \
$datadir/M3S1GNUds7_labelMM_ws_ws.h5 \
$datadir/M3S1GNUds7_labelMM_ws_ws.nii \
-e $xe $ye $ze -i 'zyx' -l 'xyz' &










# test agglo from labelsets
python $scriptdir/supervoxels/agglo_from_labelsets.py \
$datadir $datastem \
-s ${svoxpf} 'stack' \
-l $datadir/${datastem}${lvol}_svoxsets.pickle \
-o "_svoxsets_agglo" 'stack'

python $scriptdir/convert/EM_stack2stack.py \
$datadir/${datastem}${svoxpf}_svoxsets_agglo.h5 \
$datadir/${datastem}${svoxpf}_svoxsets_agglo.nii \
-e $xe $ye $ze -i 'zyx' -l 'xyz' &





python $scriptdir/mesh/EM_seg_stats.py \
$datadir $datastem \
--labelMA '_labelMA_final' 'stack' \
--labelMM '_labelMF_final' 'stack' \
--labelUA '_labelUA_final' 'stack' \
--stats 'area' 'AD' 'centroid'





python $scriptdir/convert/reduceblocks.py \
$datadir $datastem '_maskDS' 'stack' \
-d 1 7 7 -f 'expand' -o '_expanded'

python $scriptdir/convert/EM_stack2stack.py \
$datadir/${datastem}${svoxpf}_svoxsets_agglo.h5 \
$datadir/${datastem}${svoxpf}_svoxsets_agglo.nii \
-e $xe $ye $ze -i 'zyx' -l 'xyz' &






## some observations maskMA_closed
# - missing axon at xyz = 404, 1188, 110
# - partly missing at xyz = 408, 945, 93

# TODO: in vol2d
# delete 2074 3437 3229 3286 3357
# split 1148 1232 2410 1853 3213 2273 1067 1260 1021 1222
