export PATH=/data/ndcn-fmrib-water-brain/ndcn0180/anaconda2/bin:$PATH
# source activate scikit-image-devel_0.13
# conda install h5py scipy
# pip install nibabel

scriptdir="${HOME}/workspace/EM"
datadir="${DATA}/EM/M3/M3_S1_GNU/restored" && cd $datadir
dataset="m000"
xmax=5217
ymax=4460
xs=1000; ys=1000;
z=30; Z=460;
declare -a datastems
i=0
for x in `seq 0 $xs $xmax`; do
[ $x == 5000 ] && X=$xmax || X=$((x+xs))
for y in `seq 0 $ys $ymax`; do
[ $y == 4000 ] && Y=$ymax || Y=$((y+ys))
datastems[$i]=${dataset}_`printf %05d ${x}`-`printf %05d ${X}`_`printf %05d ${y}`-`printf %05d ${Y}`_`printf %05d ${z}`-`printf %05d ${Z}`
i=$((i+1))
done
done


### maskDS, maskMM and maskMM-0.02  # TODO: remove small components
for x in `seq 0 $xs $xmax`; do
[ $x == 5000 ] && X=$xmax || X=$((x+xs))
q="d"
qsubfile=$datadir/EM_prob2mask_submit_${x}-${X}.sh
echo '#!/bin/bash' > $qsubfile
echo "#SBATCH --nodes=2" >> $qsubfile
echo "#SBATCH --ntasks-per-node=10" >> $qsubfile
[ "$q" = "d" ] && echo "#SBATCH --time=00:10:00"  >> $qsubfile || echo "#SBATCH --time=01:00:00" >> $qsubfile
echo "#SBATCH --job-name=EM_s2s" >> $qsubfile
for y in `seq 0 $ys $ymax`; do
[ $y == 4000 ] && Y=$ymax || Y=$((y+ys))
datastem=${dataset}_`printf %05d ${x}`-`printf %05d ${X}`_`printf %05d ${y}`-`printf %05d ${Y}`_`printf %05d ${z}`-`printf %05d ${Z}`
echo "python $scriptdir/convert/prob2mask.py \
$datadir $datastem -p \"\" stack -l 0 -u 10000000 -o _maskDS &" >> $qsubfile
echo "python $scriptdir/convert/prob2mask.py \
$datadir $datastem -p _probs0_eed2 stack -l 0.2 -o _maskMM &" >> $qsubfile
echo "python $scriptdir/convert/prob2mask.py \
$datadir $datastem -p _probs0_eed2 stack -l 0.02 -o _maskMM-0.02 &" >> $qsubfile
echo "python $scriptdir/convert/prob2mask.py \
$datadir $datastem -p '_probs' 'volume/predictions' -c 3 -l 0.3 -o _maskMB &" >> $qsubfile
done
echo "wait" >> $qsubfile
[ "$q" = "d" ] && sbatch -p devel $qsubfile || sbatch $qsubfile
done

### connected components in maskMM-0.02
for x in `seq 0 $xs $xmax`; do
[ $x == 5000 ] && X=$xmax || X=$((x+xs))
q="d"
qsubfile=$datadir/EM_conncomp_submit_${x}-${X}.sh
echo '#!/bin/bash' > $qsubfile
echo "#SBATCH --nodes=1" >> $qsubfile
echo "#SBATCH --ntasks-per-node=5" >> $qsubfile
[ "$q" = "d" ] && echo "#SBATCH --time=00:10:00"  >> $qsubfile || echo "#SBATCH --time=01:00:00" >> $qsubfile
echo "#SBATCH --job-name=EM_s2s" >> $qsubfile
echo "export PATH=/data/ndcn-fmrib-water-brain/ndcn0180/anaconda2/bin:\$PATH" >> $qsubfile
echo "source activate scikit-image-devel_0.13" >> $qsubfile
for y in `seq 0 $ys $ymax`; do
[ $y == 4000 ] && Y=$ymax || Y=$((y+ys))
datastem=${dataset}_`printf %05d ${x}`-`printf %05d ${X}`_`printf %05d ${y}`-`printf %05d ${Y}`_`printf %05d ${z}`-`printf %05d ${Z}`
echo "python $scriptdir/supervoxels/conn_comp.py \
$datadir $datastem --maskMM _maskMM-0.02 stack &" >> $qsubfile
done
echo "wait" >> $qsubfile
[ "$q" = "d" ] && sbatch -p devel $qsubfile || sbatch $qsubfile
done
# to nifti's
for x in `seq 0 $xs $xmax`; do
[ $x == 5000 ] && X=$xmax || X=$((x+xs))
q="d"
qsubfile=$datadir/EM_conncomp_submit_${x}-${X}.sh
echo '#!/bin/bash' > $qsubfile
echo "#SBATCH --nodes=1" >> $qsubfile
echo "#SBATCH --ntasks-per-node=10" >> $qsubfile
[ "$q" = "d" ] && echo "#SBATCH --time=00:10:00"  >> $qsubfile || echo "#SBATCH --time=01:00:00" >> $qsubfile
echo "#SBATCH --job-name=EM_s2s" >> $qsubfile
echo "export PATH=/data/ndcn-fmrib-water-brain/ndcn0180/anaconda2/bin:\$PATH" >> $qsubfile
echo "source activate scikit-image-devel_0.13" >> $qsubfile
for y in `seq 0 $ys $ymax`; do
[ $y == 4000 ] && Y=$ymax || Y=$((y+ys))
datastem=${dataset}_`printf %05d ${x}`-`printf %05d ${X}`_`printf %05d ${y}`-`printf %05d ${Y}`_`printf %05d ${z}`-`printf %05d ${Z}`
pf="_maskMM-0.02"
echo "python $scriptdir/convert/EM_stack2stack.py \
$datadir/${datastem}${pf}.h5 \
$datadir/${datastem}${pf}.nii.gz \
-e 0.0073 0.0073 0.05 -i 'zyx' -l 'xyz' &" >> $qsubfile
pf="_labelMA"
echo "python $scriptdir/convert/EM_stack2stack.py \
$datadir/${datastem}${pf}.h5 \
$datadir/${datastem}${pf}.nii.gz \
-e 0.0073 0.0073 0.05 -i 'zyx' -l 'xyz' &" >> $qsubfile
done
echo "wait" >> $qsubfile
[ "$q" = "d" ] && sbatch -p devel $qsubfile || sbatch $qsubfile
done

### manual deselections from _labelMA (takes about an hour for m000)
editsfile="m000_labelMAmanedit.txt"
echo "m000_00000-01000_00000-01000_00030-00460: 1" > $editsfile
echo "m000_00000-01000_01000-02000_00030-00460: 19 25 28" >> $editsfile
echo "m000_00000-01000_02000-03000_00030-00460: 58" >> $editsfile
echo "m000_00000-01000_03000-04000_00030-00460: 1 61" >> $editsfile
echo "m000_00000-01000_04000-04460_00030-00460: 8 12" >> $editsfile
echo "m000_01000-02000_00000-01000_00030-00460: 8 2 23 62" >> $editsfile
echo "m000_01000-02000_01000-02000_00030-00460: 26 45 43" >> $editsfile
echo "m000_01000-02000_02000-03000_00030-00460: 8 32 35 33" >> $editsfile
echo "m000_01000-02000_03000-04000_00030-00460: 1 35 54 55 81 82" >> $editsfile
echo "m000_01000-02000_04000-04460_00030-00460: 2 24" >> $editsfile
echo "m000_02000-03000_00000-01000_00030-00460: 9 30 55 57" >> $editsfile
echo "m000_02000-03000_01000-02000_00030-00460: 14 38 45 55" >> $editsfile
echo "m000_02000-03000_02000-03000_00030-00460: 12 29 40 69 68 74" >> $editsfile
echo "m000_02000-03000_03000-04000_00030-00460: 17 25 39 35 45 56 55 67 77" >> $editsfile
echo "m000_02000-03000_04000-04460_00030-00460: 4 1 12 25 26 27" >> $editsfile
echo "m000_03000-04000_00000-01000_00030-00460: 28 41" >> $editsfile
echo "m000_03000-04000_01000-02000_00030-00460: 31" >> $editsfile
echo "m000_03000-04000_02000-03000_00030-00460: 1 28 52" >> $editsfile
echo "m000_03000-04000_03000-04000_00030-00460: 36 63 66" >> $editsfile
echo "m000_03000-04000_04000-04460_00030-00460: 1 18 30 32 36 39 44" >> $editsfile
echo "m000_04000-05000_00000-01000_00030-00460: 21" >> $editsfile
echo "m000_04000-05000_01000-02000_00030-00460: 48" >> $editsfile
echo "m000_04000-05000_02000-03000_00030-00460: 35 38 46 49" >> $editsfile
echo "m000_04000-05000_03000-04000_00030-00460: 34 13 46" >> $editsfile
echo "m000_04000-05000_04000-04460_00030-00460: 8 14 18 21 30 24 33 36 35" >> $editsfile
echo "m000_05000-05217_00000-01000_00030-00460: 1 3" >> $editsfile
echo "m000_05000-05217_01000-02000_00030-00460: 1 4 5" >> $editsfile
echo "m000_05000-05217_02000-03000_00030-00460: 1 2 4 5" >> $editsfile
echo "m000_05000-05217_03000-04000_00030-00460: 2 3 5 7" >> $editsfile
echo "m000_05000-05217_04000-04460_00030-00460: 1 2" >> $editsfile
# grep $datastem m000_manedit.txt | awk '{$1 = ""; print $0;}'
for x in `seq 0 $xs $xmax`; do
[ $x == 5000 ] && X=$xmax || X=$((x+xs))
q="d"
qsubfile=$datadir/EM_dellabels_${x}-${X}.sh
echo '#!/bin/bash' > $qsubfile
echo "#SBATCH --nodes=1" >> $qsubfile
echo "#SBATCH --ntasks-per-node=5" >> $qsubfile
[ "$q" = "d" ] && echo "#SBATCH --time=00:10:00"  >> $qsubfile || echo "#SBATCH --time=01:00:00" >> $qsubfile
echo "#SBATCH --job-name=EM_s2s" >> $qsubfile
for y in `seq 0 $ys $ymax`; do
[ $y == 4000 ] && Y=$ymax || Y=$((y+ys))
datastem=${dataset}_`printf %05d ${x}`-`printf %05d ${X}`_`printf %05d ${y}`-`printf %05d ${Y}`_`printf %05d ${z}`-`printf %05d ${Z}`
echo "python $scriptdir/supervoxels/delete_labels.py \
$datadir $datastem -d `grep $datastem $editsfile | awk '{$1 = ""; print $0;}'` &" >> $qsubfile
done
echo "wait" >> $qsubfile
[ "$q" = "d" ] && sbatch -p devel $qsubfile || sbatch $qsubfile
done

### watershed on prob_ics: 20G; 1h  for m000_03000-04000_02000-03000_00030-00460_ws_l0.95_u1.00_s064.h5
# l=0.95; u=1; s=64;
l=0.99; u=1; s=5;
for x in `seq 0 $xs $xmax`; do
[ $x == 5000 ] && X=$xmax || X=$((x+xs))
q=""
qsubfile=$datadir/EM_supervoxels_${x}-${X}.sh
echo '#!/bin/bash' > $qsubfile
echo "#SBATCH --nodes=1" >> $qsubfile
echo "#SBATCH --ntasks-per-node=5" >> $qsubfile
echo "#SBATCH --mem-per-cpu=25000" >> $qsubfile
[ "$q" = "d" ] && echo "#SBATCH --time=00:10:00"  >> $qsubfile || echo "#SBATCH --time=02:00:00" >> $qsubfile
echo "#SBATCH --job-name=EM_s2s" >> $qsubfile
echo "export PATH=/home/ndcn-fmrib-water-brain/ndcn0180/anaconda2/bin:\$PATH" >> $qsubfile
echo "source activate scikit-image-devel_0.13" >> $qsubfile
for y in `seq 0 $ys $ymax`; do
[ $y == 4000 ] && Y=$ymax || Y=$((y+ys))
datastem=${dataset}_`printf %05d ${x}`-`printf %05d ${X}`_`printf %05d ${y}`-`printf %05d ${Y}`_`printf %05d ${z}`-`printf %05d ${Z}`
echo "python $scriptdir/supervoxels/EM_watershed.py \
${datadir} ${datastem} -c 1 -l ${l} -u ${u} -s ${s} &" >> $qsubfile
done
echo "wait" >> $qsubfile
[ "$q" = "d" ] && sbatch -p devel $qsubfile || sbatch $qsubfile
done

### agglomerate watershedMA
# svoxpf='_ws_l0.95_u1.00_s064'
svoxpf='_ws_l0.99_u1.00_s005'
maskpf='_maskMA'
for x in `seq 0 $xs $xmax`; do
[ $x == 5000 ] && X=$xmax || X=$((x+xs))
q="d"
qsubfile=$datadir/EM_aggloMA_${x}-${X}.sh
echo '#!/bin/bash' > $qsubfile
echo "#SBATCH --nodes=1" >> $qsubfile
echo "#SBATCH --ntasks-per-node=5" >> $qsubfile
[ "$q" = "d" ] && echo "#SBATCH --time=00:10:00"  >> $qsubfile || echo "#SBATCH --time=01:00:00" >> $qsubfile
echo "#SBATCH --job-name=EM_agglo" >> $qsubfile
for y in `seq 0 $ys $ymax`; do
[ $y == 4000 ] && Y=$ymax || Y=$((y+ys))
datastem=${dataset}_`printf %05d ${x}`-`printf %05d ${X}`_`printf %05d ${y}`-`printf %05d ${Y}`_`printf %05d ${z}`-`printf %05d ${Z}`
echo "python $scriptdir/supervoxels/agglo_from_labelmask.py \
${datadir} ${datastem} \
-l _labelMAmanedit stack -s ${svoxpf} stack -m ${maskpf} &" >> $qsubfile
done
echo "wait" >> $qsubfile
[ "$q" = "d" ] && sbatch -p devel $qsubfile || sbatch $qsubfile
done
# to nifti's
for x in `seq 0 $xs $xmax`; do
[ $x == 5000 ] && X=$xmax || X=$((x+xs))
q="d"
qsubfile=$datadir/EM_nifti_${x}-${X}.sh
echo '#!/bin/bash' > $qsubfile
echo "#SBATCH --nodes=1" >> $qsubfile
echo "#SBATCH --ntasks-per-node=10" >> $qsubfile
[ "$q" = "d" ] && echo "#SBATCH --time=00:10:00"  >> $qsubfile || echo "#SBATCH --time=01:00:00" >> $qsubfile
echo "#SBATCH --job-name=EM_s2s" >> $qsubfile
echo "export PATH=/data/ndcn-fmrib-water-brain/ndcn0180/anaconda2/bin:\$PATH" >> $qsubfile
echo "source activate scikit-image-devel_0.13" >> $qsubfile
for y in `seq 0 $ys $ymax`; do
[ $y == 4000 ] && Y=$ymax || Y=$((y+ys))
datastem=${dataset}_`printf %05d ${x}`-`printf %05d ${X}`_`printf %05d ${y}`-`printf %05d ${Y}`_`printf %05d ${z}`-`printf %05d ${Z}`
pf="_maskMA"
echo "python $scriptdir/convert/EM_stack2stack.py \
$datadir/${datastem}${pf}.h5 \
$datadir/${datastem}${pf}.nii.gz \
-e 0.0073 0.0073 0.05 -i 'zyx' -l 'xyz' &" >> $qsubfile
pf="${svoxpf}_labelMA"
echo "python $scriptdir/convert/EM_stack2stack.py \
$datadir/${datastem}${pf}.h5 \
$datadir/${datastem}${pf}.nii.gz \
-e 0.0073 0.0073 0.05 -i 'zyx' -l 'xyz' &" >> $qsubfile
done
echo "wait" >> $qsubfile
[ "$q" = "d" ] && sbatch -p devel $qsubfile || sbatch $qsubfile
done

### fill holes in myelinated axons
nodes=3
tasks=10
memcpu=6000
wtime=10:00:00
q=""
for n in `seq 0 $((nodes-1))`; do
qsubfile=$datadir/EM_fillholes_${n}.sh
echo '#!/bin/bash' > $qsubfile
echo "#SBATCH --nodes=1" >> $qsubfile
echo "#SBATCH --ntasks-per-node=${tasks}" >> $qsubfile
echo "#SBATCH --mem-per-cpu=${memcpu}" >> $qsubfile
[ "$q" = "d" ] && echo "#SBATCH --time=00:10:00"  >> $qsubfile || echo "#SBATCH --time=${wtime}" >> $qsubfile
echo "#SBATCH --job-name=EM_fill" >> $qsubfile
echo "export PATH=/data/ndcn-fmrib-water-brain/ndcn0180/anaconda2/bin:\$PATH" >> $qsubfile
echo "source activate scikit-image-devel_0.13" >> $qsubfile
for t in `seq 0 $((tasks-1))`; do
datastem=${datastems[n*tasks+t]}
echo "python $scriptdir/supervoxels/fill_holes.py \
$datadir $datastem \
-l '_ws_l0.99_u1.00_s005_labelMA' 'stack' -m '_maskMA' 'stack' \
--maskMM '_maskMM' 'stack' --maskMA '_maskMA' 'stack' \
-o '_filled' -p '_holes' -w 2 &" >> $qsubfile
done
echo "wait" >> $qsubfile
[ "$q" = "d" ] && sbatch -p devel $qsubfile || sbatch $qsubfile
done

### Neuroproof agglomeration
# cp /data/ndcn-fmrib-water-brain/ndcn0180/EM/Neuroproof/M3_S1_GNU_NP/train/m000_01000-01500_01000-01500_00030-00460_NPminimal_ws_l0.95_u1.00_s064_PA_str2_iter5_parallel.h5 ../../M3/M3_S1_GNU/restored/
# cp /data/ndcn-fmrib-water-brain/ndcn0180/EM/Neuroproof/M3_S1_GNU_NP/train/m000_01000-01500_01000-01500_00030-00460_NPminimal_ws_l0.99_u1.00_s005_PA_str2_iter5_parallel.h5 ../../M3/M3_S1_GNU/restored/
export PATH=/data/ndcn-fmrib-water-brain/ndcn0180/anaconda2/bin:$PATH
CONDA_PATH=$(conda info --root)
PREFIX=${CONDA_PATH}/envs/neuroproof-test
NPdir=/home/ndcn-fmrib-water-brain/ndcn0180/workspace/Neuroproof_minimal
trainset="m000_01000-01500_01000-01500_00030-00460"
# svoxpf='_ws_l0.95_u1.00_s064'  # 50G;>10min for m000_00000-01000_00000-01000
# svoxpf='_ws_l0.95_u1.00_s064_labelMA'
svoxpf='_ws_l0.99_u1.00_s005_labelMA'
classifier="_NPminimal_ws_l0.95_u1.00_s064_PA_str2_iter5_parallel"
cltype='h5'
thr=0.1
alg=1
for x in `seq 0 $xs $xmax`; do
[ $x == 5000 ] && X=$xmax || X=$((x+xs))
for y in `seq 0 $ys $ymax`; do
[ $y == 4000 ] && Y=$ymax || Y=$((y+ys))
q=''
qsubfile=$datadir/EM_NPagglo_${x}-${X}_${y}-${Y}.sh
echo '#!/bin/bash' > $qsubfile
echo "#SBATCH --nodes=1" >> $qsubfile
echo "#SBATCH --ntasks-per-node=1" >> $qsubfile
echo "#SBATCH --mem-per-cpu=60000" >> $qsubfile
[ "$q" = "d" ] && echo "#SBATCH --time=00:10:00"  >> $qsubfile || echo "#SBATCH --time=01:00:00" >> $qsubfile
echo "#SBATCH --job-name=EM_s2s" >> $qsubfile
echo "export PATH=/data/ndcn-fmrib-water-brain/ndcn0180/anaconda2/bin:\$PATH" >> $qsubfile
echo "export LD_LIBRARY_PATH=${PREFIX}/lib" >> $qsubfile
datastem=${dataset}_`printf %05d ${x}`-`printf %05d ${X}`_`printf %05d ${y}`-`printf %05d ${Y}`_`printf %05d ${z}`-`printf %05d ${Z}`
echo "$NPdir/NeuroProof_stack \
-watershed $datadir/${datastem}${svoxpf}.h5 stack \
-prediction $datadir/${datastem}_probs.h5 volume/predictions \
-output $datadir/${datastem}_prediction${classifier}${cltype}_thr${thr}_alg${alg}M.h5 stack \
-classifier $datadir/${trainset}${classifier}.${cltype} \
-threshold ${thr} -algorithm ${alg} &" >> $qsubfile  # -nomito
echo "wait" >> $qsubfile
[ "$q" = "d" ] && sbatch -p devel $qsubfile || sbatch $qsubfile
done
done

### TODO: EED probs3_eed2?

### classify neurons MA/UA
module load mpi4py/1.3.1
module load hdf5-parallel/1.8.14_mvapich2_gcc
module load python/2.7__gcc-4.8
x=2000;X=3000;y=2000;Y=3000;z=30;Z=460;
for x in `seq 0 $xs $xmax`; do
[ $x == 5000 ] && X=$xmax || X=$((x+xs))
for y in `seq 0 $ys $ymax`; do
[ $y == 4000 ] && Y=$ymax || Y=$((y+ys))
datastem="${dataset}_`printf %05d ${x}`-`printf %05d ${X}`_`printf %05d ${y}`-`printf %05d ${Y}`_`printf %05d ${z}`-`printf %05d ${Z}`"
q=""
qsubfile=$datadir/EM_classify_${x}-${X}.sh
echo '#!/bin/bash' > $qsubfile
echo "#SBATCH --nodes=4" >> $qsubfile
echo "#SBATCH --ntasks-per-node=16" >> $qsubfile
echo "#SBATCH --mem-per-cpu=6000" >> $qsubfile  # 4GB per core (16 cores is too much for small nodes)
[ "$q" = "d" ] && echo "#SBATCH --time=00:10:00"  >> $qsubfile || echo "#SBATCH --time=10:00:00" >> $qsubfile
echo "#SBATCH --job-name=EM_agglo" >> $qsubfile
echo "#SBATCH --job-name=classify" >> $qsubfile
echo ". enable_arcus-b_mpi.sh" >> $qsubfile
echo "mpirun \$MPI_HOSTS python $scriptdir/mesh/EM_classify_neurons.py \
$datadir $datastem \
--supervoxels '_prediction_NPminimal_ws_l0.95_u1.00_s064_PA_str2_iter5_parallelh5_thr0.1_alg1' 'stack' \
-o '_per' -m" >> $qsubfile
[ "$q" = "d" ] && sbatch -p devel $qsubfile || sbatch $qsubfile
done
done
