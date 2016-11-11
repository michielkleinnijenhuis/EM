#rsync -avz /Users/michielk/Downloads/Anaconda3-2.4.1-Linux-x86_64.sh ndcn0180@arcus.arc.ox.ac.uk:/home/ndcn-fmrib-water-brain/ndcn0180/
#rsync -avz /Users/michielk/oxdata/P01/EM/M3/M3_S1_GNU/m000_01000-02000_01000-02000_00030-00460_knossos/annotation.xml  ndcn0180@arcus.arc.ox.ac.uk:/data/ndcn-fmrib-water-brain/ndcn0180/EM/M3/M3_S1_GNU/m000_01000-02000_01000-02000_00030-00460_knossos
#rsync -avz /Users/michielk/oxdata/P01/EM/M3/M3_S1_GNU/m000_01000-02000_01000-02000_00030-00460_knossos/annotation-160108T1640.058.k.zip  ndcn0180@arcus.arc.ox.ac.uk:/data/ndcn-fmrib-water-brain/ndcn0180/EM/M3/M3_S1_GNU/m000_01000-02000_01000-02000_00030-00460_knossos
#rsync -avz ndcn0180@arcus.arc.ox.ac.uk:/data/ndcn-fmrib-water-brain/ndcn0180/EM/M3/M3_S1_GNU/m000.h5 /Users/michielk/oxdata/P01/EM/M3/M3_S1_GNU/
#rsync -avz ndcn0180@arcus.arc.ox.ac.uk:/data/ndcn-fmrib-water-brain/ndcn0180/EM/M3/M3_S1_GNU/m000_?????-?????_?????-?????_?????-?????_probs.h5 /Users/michielk/oxdata/P01/EM/M3/M3_S1_GNU/
#rsync -avz ndcn0180@arcus.arc.ox.ac.uk:/data/ndcn-fmrib-water-brain/ndcn0180/EM/M3/M3_S1_GNU/m000_?????-?????_?????-?????_?????-?????_seg.h5 /Users/michielk/oxdata/P01/EM/M3/M3_S1_GNU/
#rsync -avz ndcn0180@arcus.arc.ox.ac.uk:/data/ndcn-fmrib-water-brain/ndcn0180/EM/M3/M3_S1_GNU/m000_?????-?????_?????-?????_?????-?????_seeds_MA.h5 /Users/michielk/oxdata/P01/EM/M3/M3_S1_GNU/
#rsync -avz ndcn0180@arcus.arc.ox.ac.uk:/data/ndcn-fmrib-water-brain/ndcn0180/EM/M3/M3_S1_GNU/m000_?????-?????_?????-?????_?????-?????_seeds_UA_knossos.h5 /Users/michielk/oxdata/P01/EM/M3/M3_S1_GNU/
#rsync -avz ndcn0180@arcus.arc.ox.ac.uk:/data/ndcn-fmrib-water-brain/ndcn0180/EM/M3/M3_S1_GNU/m000_?????-?????_?????-?????_?????-?????_probs_ws_M* /Users/michielk/oxdata/P01/EM/M3/M3_S1_GNU/
#rsync -avz ndcn0180@arcus.arc.ox.ac.uk:/data/ndcn-fmrib-water-brain/ndcn0180/EM/M3/M3_S1_GNU/m000_?????-?????_?????-?????_?????-?????_probs_ws_?A* /Users/michielk/oxdata/P01/EM/M3/M3_S1_GNU/
#rsync -avz ndcn0180@arcus.arc.ox.ac.uk:/data/ndcn-fmrib-water-brain/ndcn0180/EM/M3/M3_S1_GNU/m000_?????-?????_?????-?????_?????-?????_slic* /Users/michielk/oxdata/P01/EM/M3/M3_S1_GNU/
#rsync -avz ndcn0180@arcus.arc.ox.ac.uk:/data/ndcn-fmrib-water-brain/ndcn0180/EM/M3/M3_S1_GNU/m000_?1???-?2???_?1???-?2???_?????-?????_probs0_eed2.h5 /Users/michielk/oxdata/P01/EM/M3/M3_S1_GNU/
#rsync -avz ndcn0180@arcus.arc.ox.ac.uk:/data/ndcn-fmrib-water-brain/ndcn0180/EM/M3/M3_S1_GNU/m000_?????-?????_?????-?????_?????-?????_slic_s00500_c2.000_o0.050.h5 /Users/michielk/oxdata/P01/EM/M3/M3_S1_GNU/
#rsync -avz ndcn0180@arcus.arc.ox.ac.uk:/data/ndcn-fmrib-water-brain/ndcn0180/EM/M3/M3_S1_GNU/m000_?????-?????_?????-?????_?????-?????_segmentation.h5 /Users/michielk/oxdata/P01/EM/M3/M3_S1_GNU/

###========================###
### convert .h5 to .nii.gz ###
###========================###
scriptdir="$HOME/workspace/EM"
DATA="$HOME/oxdata"
datadir="$DATA/P01/EM/M3/M3_S1_GNU" && cd $datadir
dataset='m000'
x=1000; y=1000; xs=500; ys=500; z=30; Z=460;
# z=200; Z=300;
# xs=100; ys=100;
[ $x == 5000 ] && X=5217 || X=$((x+xs))
[ $y == 4000 ] && Y=4460 || Y=$((y+ys))
datastem=${dataset}_`printf %05d ${x}`-`printf %05d ${X}`_`printf %05d ${y}`-`printf %05d ${Y}`_`printf %05d ${z}`-`printf %05d ${Z}`
# raw data to uint8
pf=''
python $scriptdir/convert/EM_stack2stack.py \
${datadir}/${datastem}${pf}.h5 ${datadir}/${datastem}${pf}.nii.gz \
-i 'zyx' -l 'xyz' -e -0.0073 -0.0073 0.05 -u
# other
pf='_seg'
pf='_seeds_MA'
pf='_probs_ws_MA'
pf='_probs_ws_MAfilled'
pf='_probs_ws_MM'
pf='_probs_ws_MMdistsum_distfilter'
pf='_probs_ws_UA'
pf='_probs_ws_UA_tmp2'
pf='_probs_ws_PA'
pf='_seeds_UA'
pf='_seeds_UA_knossos'
pf='_slic'
pf='_MA_seg'
pf='_MA_seg_ws'
pf='_MA_sMAknedges'
pf='_MA_ws_filled'
pf='_MA_sMAkn_sUAkn_ws_filled'
pf='_MA_sMAkn_sUAkn_ws'
pf='_MA_sMAkn_sUAkn_ws_filled_enforceECS'
pf='_MM_ws_sw_df'
pf='_PA'
pf='_PA_enforceECS'
pf='_MA1128_skeletonize_3d'
pf='_notMA1128_edt'
python $scriptdir/convert/EM_stack2stack.py \
${datadir}/${datastem}${pf}.h5 ${datadir}/${datastem}${pf}.nii.gz \
-i 'zyx' -l 'xyz' -e -0.0073 -0.0073 0.05
pf='_slic_s20000_c0.020_o0.010'
pf='_slic_s00500_c2.000_o0.050'
pf='_segmentation'
python $scriptdir/convert/EM_stack2stack.py \
${datadir}/${datastem}${pf}.h5 ${datadir}/${datastem}${pf}.nii.gz \
-i 'zyx' -l 'xyz' -e -0.0073 -0.0073 0.05 -d int32

python $scriptdir/convert/EM_stack2stack.py \
${datadir}/${datastem}${pf}.h5 ${datadir}/${datastem}${pf}.nii.gz \
-i 'zyx' -l 'xyz' -e -0.0073 -0.0073 0.05 -d float

###=======================================================###
### cutouts (430x500x500) from 1000x1000x430 (in .h5 zyx) ###
###=======================================================###
# local
x=1000; X=2000; y=1000; Y=2000; z=30; Z=460;
datastem=${dataset}_`printf %05d ${x}`-`printf %05d ${X}`_`printf %05d ${y}`-`printf %05d ${Y}`_`printf %05d ${z}`-`printf %05d ${Z}`
oX=1000;oY=1000;oZ=30;
nz=30;nZ=460;
nz=200;nZ=300;
for nx in 1000 1500; do
for ny in 1000 1500; do
[ $nx == 5000 ] && nX=5217 || nX=$((nx+500))
[ $ny == 4000 ] && nY=4460 || nY=$((ny+500))
newstem=${dataset}_`printf %05d ${nx}`-`printf %05d ${nX}`_`printf %05d ${ny}`-`printf %05d ${nY}`_`printf %05d ${nz}`-`printf %05d ${nZ}`
for pf in '' '_probs0_eed2'; do
python $scriptdir/convert/EM_stack2stack.py \
${datadir}/${datastem}${pf}.h5 ${datadir}/${newstem}${pf}.h5 \
-i 'zyx' -l 'zyx' -e 0.05 0.0073 0.0073 \
-x $((nx-oX)) -X $((nX-oX)) -y $((ny-oY)) -Y $((nY-oY)) -z $((nz-oZ)) -Z $((nZ-oZ))
done
pf='_probs'; f='volume/predictions'; g='volume/predictions';
python $scriptdir/convert/EM_stack2stack.py \
${datadir}/${datastem}${pf}.h5 ${datadir}/${newstem}${pf}.h5 \
-i 'zyxc' -l 'zyxc' -e 0.05 0.0073 0.0073 -f $f -g $g \
-x $((nx-oX)) -X $((nX-oX)) -y $((ny-oY)) -Y $((nY-oY)) -z $((nz-oZ)) -Z $((nZ-oZ))
pf='_probs'; f='volume/predictions'; g='volume/predictions';
python $scriptdir/convert/EM_stack2stack.py \
${datadir}/${datastem}${pf}.h5 ${datadir}/${newstem}${pf}_xyzc.h5 \
-i 'zyxc' -l 'xyzc' -e 0.05 0.0073 0.0073 -f $f -g $g \
-x $((nx-oX)) -X $((nX-oX)) -y $((ny-oY)) -Y $((nY-oY)) -z $((nz-oZ)) -Z $((nZ-oZ))
done
done
# pf=''
# python $scriptdir/convert/EM_stack2stack.py \
# ${datadir}/${newstem}${pf}.h5 ${datadir}/${newstem}${pf}.nii.gz \
# -i 'zyx' -l 'xyz' -e -0.0073 -0.0073 0.05 -u
# pf='_probs0_eed2'
# python $scriptdir/convert/EM_stack2stack.py \
# ${datadir}/${newstem}${pf}.h5 ${datadir}/${newstem}${pf}.nii.gz \
# -i 'zyx' -l 'xyz' -e -0.0073 -0.0073 0.05

# distributed
qsubfile=$datadir/EM_con.sh
echo '#!/bin/bash' > $qsubfile
echo "#SBATCH --nodes=1" >> $qsubfile
echo "#SBATCH --ntasks-per-node=1" >> $qsubfile
echo "#SBATCH --time=00:10:00" >> $qsubfile
echo "#SBATCH --job-name=EM_con" >> $qsubfile
x=1000; X=2000; y=1000; Y=2000; z=30; Z=460; oX=1000; oY=1000; oZ=30;
datastem=${dataset}_`printf %05d ${x}`-`printf %05d ${X}`_`printf %05d ${y}`-`printf %05d ${Y}`_`printf %05d ${z}`-`printf %05d ${Z}`

xs=100; ys=100; #xs=500; ys=500;
nz=200; nZ=300; #nz=30; nZ=460;
for nx in 1000 1500; do
for ny in 1000 1500; do
nx=1000; ny=1000;
nX=$((nx+xs)); nY=$((ny+ys));
newstem=${dataset}_`printf %05d ${nx}`-`printf %05d ${nX}`_`printf %05d ${ny}`-`printf %05d ${nY}`_`printf %05d ${nz}`-`printf %05d ${nZ}`
for pf in '' '_probs0_eed2'; do
echo "python $scriptdir/convert/EM_stack2stack.py \
${datadir}/${datastem}${pf}.h5 ${datadir}/${newstem}${pf}.h5 \
-i 'zyx' -l 'zyx' -e 0.05 0.0073 0.0073 \
-x $((nx-oX)) -X $((nX-oX)) \
-y $((ny-oY)) -Y $((nY-oY)) \
-z $((nz-oZ)) -Z $((nZ-oZ))"  >> $qsubfile
done
pf='_probs'; f='/volume/predictions'; g='/volume/predictions'; >> $qsubfile
echo "python $scriptdir/convert/EM_stack2stack.py \
${datadir}/${datastem}${pf}.h5 ${datadir}/${newstem}${pf}.h5 \
-i 'zyxc' -l 'zyxc' -e 0.05 0.0073 0.0073 -f $f -g $g \
-x $((nx-oX)) -X $((nX-oX)) \
-y $((ny-oY)) -Y $((nY-oY)) \
-z $((nz-oZ)) -Z $((nZ-oZ))" >> $qsubfile
echo "python $scriptdir/convert/EM_stack2stack.py \
${datadir}/${datastem}${pf}.h5 ${datadir}/${newstem}${pf}_xyzc.h5 \
-i 'zyxc' -l 'xyzc' -e 0.05 0.0073 0.0073 -f $f -g $g \
-x $((nx-oX)) -X $((nX-oX)) \
-y $((ny-oY)) -Y $((nY-oY)) \
-z $((nz-oZ)) -Z $((nZ-oZ))" >> $qsubfile
done
done
sbatch -p devel $qsubfile

# subsetting manual segmentation block (100x500x500)
qsubfile=$datadir/EM_con.sh
echo '#!/bin/bash' > $qsubfile
echo "#SBATCH --nodes=1" >> $qsubfile
echo "#SBATCH --ntasks-per-node=1" >> $qsubfile
echo "#SBATCH --time=00:10:00" >> $qsubfile
echo "#SBATCH --job-name=EM_con" >> $qsubfile
x=1000; X=1500; y=1000; Y=1500; z=30; Z=460;
datastem=${dataset}_`printf %05d ${x}`-`printf %05d ${X}`_`printf %05d ${y}`-`printf %05d ${Y}`_`printf %05d ${z}`-`printf %05d ${Z}`
oX=1000;oY=1000;oZ=30;
nz=200;nZ=300;
nx=1000; ny=1000;
nX=$((nx+500))
nY=$((ny+500))
newstem=${dataset}_`printf %05d ${nx}`-`printf %05d ${nX}`_`printf %05d ${ny}`-`printf %05d ${nY}`_`printf %05d ${nz}`-`printf %05d ${nZ}`
for pf in '_PA'; do
echo "python $scriptdir/convert/EM_stack2stack.py \
${datadir}/${datastem}${pf}.h5 ${datadir}/${newstem}${pf}.h5 \
-i 'zyx' -l 'zyx' -e 0.05 0.0073 0.0073 \
-x $((nx-oX)) -X $((nX-oX)) \
-y $((ny-oY)) -Y $((nY-oY)) \
-z $((nz-oZ)) -Z $((nZ-oZ))"  >> $qsubfile
done
sbatch -p devel $qsubfile
#local
pf='_MA_sMAkn_sUAkn_ws_filled'
python $scriptdir/convert/EM_stack2stack.py \
${datadir}/${datastem}${pf}.h5 ${datadir}/${newstem}${pf}.h5 \
-i 'zyx' -l 'zyx' -e 0.05 0.0073 0.0073 \
-x $((nx-oX)) -X $((nX-oX)) \
-y $((ny-oY)) -Y $((nY-oY)) \
-z $((nz-oZ)) -Z $((nZ-oZ))
python $scriptdir/convert/EM_stack2stack.py \
${datadir}/${newstem}${pf}.h5 ${datadir}/${newstem}${pf}.nii.gz \
-i 'zyx' -l 'xyz' -e -0.0073 -0.0073 0.05 -d int32



# subsetting manual segmentation block (100x100x100)
qsubfile=$datadir/EM_con.sh
echo '#!/bin/bash' > $qsubfile
echo "#SBATCH --nodes=1" >> $qsubfile
echo "#SBATCH --ntasks-per-node=1" >> $qsubfile
echo "#SBATCH --time=00:10:00" >> $qsubfile
echo "#SBATCH --job-name=EM_con" >> $qsubfile
x=1000; X=1500; y=1000; Y=1500; z=30; Z=460;
datastem=${dataset}_`printf %05d ${x}`-`printf %05d ${X}`_`printf %05d ${y}`-`printf %05d ${Y}`_`printf %05d ${z}`-`printf %05d ${Z}`
oX=1000;oY=1000;oZ=30;
nz=200;nZ=300;
nx=1000; ny=1000;
nX=$((nx+100))
nY=$((ny+100))
newstem=${dataset}_`printf %05d ${nx}`-`printf %05d ${nX}`_`printf %05d ${ny}`-`printf %05d ${nY}`_`printf %05d ${nz}`-`printf %05d ${nZ}`
for pf in '_PA'; do
echo "python $scriptdir/convert/EM_stack2stack.py \
${datadir}/${datastem}${pf}.h5 ${datadir}/${newstem}${pf}.h5 \
-i 'zyx' -l 'zyx' -e 0.05 0.0073 0.0073 \
-x $((nx-oX)) -X $((nX-oX)) \
-y $((ny-oY)) -Y $((nY-oY)) \
-z $((nz-oZ)) -Z $((nZ-oZ))"  >> $qsubfile
done
sbatch -p devel $qsubfile
###=======================================================###









z=30; Z=460;
for x in 0 1000 2000 3000 4000 5000; do
[ $x == 5000 ] && X=5217 || X=$((x+1000))
for y in 0 1000 2000 3000 4000; do
[ $y == 4000 ] && Y=4460 || Y=$((y+1000))
datastem=${dataset}_`printf %05d ${x}`-`printf %05d ${X}`_`printf %05d ${y}`-`printf %05d ${Y}`_`printf %05d ${z}`-`printf %05d ${Z}`
datafile=${datadir}/${datastem}.h5
python $scriptdir/convert/EM_stack2stack.py \
${datafile} ${datafile/.h5/.nii.gz} \
-i 'zyx' -l 'xyz' -e -0.0073 -0.0073 0.05 -u
done
done

dataset='m000'
z=30; Z=460;
for x in 0 1000 2000 3000 4000 5000; do
[ $x == 5000 ] && X=5217 || X=$((x+1000))
for y in 0 1000 2000 3000 4000; do
[ $y == 4000 ] && Y=4460 || Y=$((y+1000))
datastem=${dataset}_`printf %05d ${x}`-`printf %05d ${X}`_`printf %05d ${y}`-`printf %05d ${Y}`_`printf %05d ${z}`-`printf %05d ${Z}`
for pf in _probs_ws_MA_probs_ws_MAfilled; do #_seeds_MA2 _probs_ws_MA_probs_ws_MA
python $scriptdir/convert/EM_stack2stack.py \
${datadir}/${datastem}${pf}.h5 ${datadir}/${datastem}${pf}.nii.gz \
-i 'zyx' -l 'xyz' -e -0.0073 -0.0073 0.05
done
done
done




# from scratch4



dataset='m000_00000-01000_00000-01000_00000-00100'
dataset='m000_02000-03000_02000-03000_00000-00100'
python $scriptdir/convert/EM_stack2stack.py \
"${datadir}/${dataset}.h5" \
"${datadir}/${dataset}.nii.gz" -i 'zyx' -l 'xyz' -e -0.0073 -0.0073 0.05
python $scriptdir/convert/EM_stack2stack.py \
"${datadir}/${dataset}_seg.h5" \
"${datadir}/${dataset}_seg.nii.gz" -i 'zyx' -l 'xyz' -e -0.0073 -0.0073 0.05
python $scriptdir/convert/EM_stack2stack.py \
"${datadir}/${dataset}_probs.h5" \
"${datadir}/${dataset}_probs.nii.gz" -i 'zyxc' -l 'xyzc' -e -0.0073 -0.0073 0.05 1 -f 'volume/predictions'
python $scriptdir/convert/EM_stack2stack.py \
"${datadir}/${dataset}_seg.h5" \
"${datadir}/${dataset}_seg.nii.gz" -i 'zyx' -l 'xyz' -e -0.0073 -0.0073 0.05
python $scriptdir/convert/EM_stack2stack.py \
"${datadir}/${dataset}_probs_ws_MA.h5" \
"${datadir}/${dataset}_probs_ws_MA.nii.gz" -i 'zyx' -l 'xyz' -e -0.0073 -0.0073 0.05
python $scriptdir/convert/EM_stack2stack.py \
"${datadir}/${dataset}_probs_ws_MM.h5" \
"${datadir}/${dataset}_probs_ws_MM.nii.gz" -i 'zyx' -l 'xyz' -e -0.0073 -0.0073 0.05
python $scriptdir/convert/EM_stack2stack.py \
"${datadir}/${dataset}_probs_ws_UA.h5" \
"${datadir}/${dataset}_probs_ws_UA.nii.gz" -i 'zyx' -l 'xyz' -e -0.0073 -0.0073 0.05
python $scriptdir/convert/EM_stack2stack.py \
"${datadir}/${dataset}_probs_ws_PA.h5" \
"${datadir}/${dataset}_probs_ws_PA.nii.gz" -i 'zyx' -l 'xyz' -e -0.0073 -0.0073 0.05

for i in 0 1 2; do
python $scriptdir/convert/EM_stack2stack.py \
"${datadir}/${dataset}_probs${i}_eed2.h5" \
"${datadir}/${dataset}_probs${i}_eed2.nii.gz" -i 'zyx' -l 'xyz' -e -0.0073 -0.0073 0.05
done

python $scriptdir/convert/EM_stack2stack.py \
"${datadir}/${dataset}_distance.h5" \
"${datadir}/${dataset}_distance.nii.gz" -i 'zyx' -l 'xyz' -e -0.0073 -0.0073 0.05

python $scriptdir/convert/EM_stack2stack.py \
"${datadir}/${dataset}_probs_ws_MAfilled.h5" \
"${datadir}/${dataset}_probs_ws_MAfilled.nii.gz" -i 'zyx' -l 'xyz' -e -0.0073 -0.0073 0.05
python $scriptdir/convert/EM_stack2stack.py \
"${datadir}/${dataset}_probs_ws_distsum.h5" \
"${datadir}/${dataset}_probs_ws_distsum.nii.gz" -i 'zyx' -l 'xyz' -e -0.0073 -0.0073 0.05
python $scriptdir/convert/EM_stack2stack.py \
"${datadir}/${dataset}_probs_ws_MMdistsum.h5" \
"${datadir}/${dataset}_probs_ws_MMdistsum.nii.gz" -i 'zyx' -l 'xyz' -e -0.0073 -0.0073 0.05
python $scriptdir/convert/EM_stack2stack.py \
"${datadir}/${dataset}_probs_ws_MMdistsum_distfilter.h5" \
"${datadir}/${dataset}_probs_ws_MMdistsum_distfilter.nii.gz" -i 'zyx' -l 'xyz' -e -0.0073 -0.0073 0.05



# from scratch7

scriptdir="$HOME/workspace/EM"
DATA="$HOME/oxdata"
datadir="$DATA/P01/EM/M3/M3_S1_GNU" && cd $datadir
dataset='m000'
python $scriptdir/convert/EM_stack2stack.py \
"${datadir}/${dataset}_01000-02000_01000-02000_00030-00460.h5" \
"${datadir}/${dataset}_01000-02000_01000-02000_00030-00460.nii.gz" \
-i 'zyx' -l 'xyz' -e -0.0073 -0.0073 0.05 -u

scriptdir="$HOME/workspace/EM"
DATA="$HOME/oxdata"
datadir="$DATA/P01/EM/M3/M3_S1_GNU"
dataset='m000_01000-02000_01000-02000_00030-00460'
dataset='m000_01000-02000_02000-03000_00030-00460'
pf='_seg'
pf='_seeds_MA'
pf='_probs_ws_MA'
pf='_probs_ws_MAfilled'
pf='_probs_ws_MMdistsum_distfilter'
pf='_probs_ws_UA'
pf='_probs_ws_PA'
pf='_seeds_UA'
python $scriptdir/convert/EM_stack2stack.py \
"${datadir}/${dataset}${pf}.h5" \
"${datadir}/${dataset}${pf}.nii.gz" \
-i 'zyx' -l 'xyz' -e -0.0073 -0.0073 0.05

#rsync -avz /Users/michielk/oxdata/P01/EM/M3/M3_S1_GNU/m000_01000-02000_01000-02000_00030-00460_probs_ws_MA_probs_ws_MA_manseg.h5 ndcn0180@arcus.arc.ox.ac.uk:/data/ndcn-fmrib-water-brain/ndcn0180/EM/M3/M3_S1_GNU/
#rsync -avz ndcn0180@arcus.arc.ox.ac.uk:/data/ndcn-fmrib-water-brain/ndcn0180/EM/M3/M3_S1_GNU/m000_?????-?????_?????-?????_?????-?????_probs_ws_M* /Users/michielk/oxdata/P01/EM/M3/M3_S1_GNU/
#rsync -avz ndcn0180@arcus.arc.ox.ac.uk:/data/ndcn-fmrib-water-brain/ndcn0180/EM/M3/M3_S1_GNU/m000_?????-?????_?????-?????_?????-?????_seg.h5 /Users/michielk/oxdata/P01/EM/M3/M3_S1_GNU/
#rsync -avz ndcn0180@arcus.arc.ox.ac.uk:/data/ndcn-fmrib-water-brain/ndcn0180/EM/M3/M3_S1_GNU/m000_?????-?????_?????-?????_?????-?????_seeds_MA.h5 /Users/michielk/oxdata/P01/EM/M3/M3_S1_GNU/
#rsync -avz ndcn0180@arcus.arc.ox.ac.uk:/data/ndcn-fmrib-water-brain/ndcn0180/EM/M3/M3_S1_GNU/m000_?????-?????_?????-?????_?????-?????_probs_ws_M*.h5 /Users/michielk/oxdata/P01/EM/M3/M3_S1_GNU/
#rsync -avz ndcn0180@arcus.arc.ox.ac.uk:/data/ndcn-fmrib-water-brain/ndcn0180/EM/M3/M3_S1_GNU/m000_?????-?????_?????-?????_?????-?????_probs_ws_?A.h5 /Users/michielk/oxdata/P01/EM/M3/M3_S1_GNU/
#rsync -avz ndcn0180@arcus.arc.ox.ac.uk:/data/ndcn-fmrib-water-brain/ndcn0180/EM/M3/M3_S1_GNU/m000_05000-06000_00000-01000_?????-?????_probs.h5 /Users/michielk/oxdata/P01/EM/M3/M3_S1_GNU/



# from scratch8

scriptdir="$HOME/workspace/EM"
DATA="$HOME/oxdata"
datadir="$DATA/P01/EM/M3/M3_S1_GNU"

dataset='m000'
z=30; Z=460;
x=5000
[ $x == 5000 ] && X=5217 || X=$((x+1000))
for y in 0 1000 2000 3000 4000; do
[ $y == 4000 ] && Y=4460 || Y=$((y+1000))
datastem=${dataset}_`printf %05d ${x}`-`printf %05d ${X}`_`printf %05d ${y}`-`printf %05d ${Y}`_`printf %05d ${z}`-`printf %05d ${Z}`
datafile=${datadir}/${datastem}.h5
python $scriptdir/convert/EM_stack2stack.py \
${datafile} ${datafile/.h5/.nii.gz} \
-i 'zyx' -l 'xyz' -e -0.0073 -0.0073 0.05 -u
done

for f in `ls m000_?????-?????_?????-?????_00000-00460*`; do
mv $f ${f/00000-00460/00030-00460}
done

dataset='m000'
z=30; Z=460;
for x in 0 1000 2000 3000 4000 5000; do
[ $x == 5000 ] && X=5217 || X=$((x+1000))
for y in 0 1000 2000 3000 4000; do
[ $y == 4000 ] && Y=4460 || Y=$((y+1000))
datastem=${dataset}_`printf %05d ${x}`-`printf %05d ${X}`_`printf %05d ${y}`-`printf %05d ${Y}`_`printf %05d ${z}`-`printf %05d ${Z}`
datafile=${datadir}/${datastem}.h5
python $scriptdir/convert/EM_stack2stack.py \
${datafile} ${datafile/.h5/.nii.gz} \
-i 'zyx' -l 'xyz' -e -0.0073 -0.0073 0.05 -u
done
done

dataset='m000'
z=30; Z=460;
for x in 0 1000 2000 3000 4000 5000; do
[ $x == 5000 ] && X=5217 || X=$((x+1000))
for y in 0 1000 2000 3000 4000; do
[ $y == 4000 ] && Y=4460 || Y=$((y+1000))
datastem=${dataset}_`printf %05d ${x}`-`printf %05d ${X}`_`printf %05d ${y}`-`printf %05d ${Y}`_`printf %05d ${z}`-`printf %05d ${Z}`
for pf in _probs_ws_MA_probs_ws_MAfilled; do #_seeds_MA2 _probs_ws_MA_probs_ws_MA
python $scriptdir/convert/EM_stack2stack.py \
${datadir}/${datastem}${pf}.h5 ${datadir}/${datastem}${pf}.nii.gz \
-i 'zyx' -l 'xyz' -e -0.0073 -0.0073 0.05
done
done
done

dataset='m000'
z=30; Z=460;
x=1000; y=1000;
[ $x == 5000 ] && X=5217 || X=$((x+1000))
[ $y == 4000 ] && Y=4460 || Y=$((y+1000))
datastem=${dataset}_`printf %05d ${x}`-`printf %05d ${X}`_`printf %05d ${y}`-`printf %05d ${Y}`_`printf %05d ${z}`-`printf %05d ${Z}`
pf='_probs_ws_MM'
python $scriptdir/convert/EM_stack2stack.py \
${datadir}/${datastem}${pf}.h5 ${datadir}/${datastem}${pf}.nii.gz \
-i 'zyx' -l 'xyz' -e -0.0073 -0.0073 0.05



# from scratch_misc

#rsync -avz ndcn0180@arcus.arc.ox.ac.uk:/data/ndcn-fmrib-water-brain/ndcn0180/EM/M3/M3_S1_GNU/m000_01000-02000_01000-02000_00030-00460_slic_s500_c10_o0.h5 /Users/michielk/oxdata/P01/EM/M3/M3_S1_GNU/
pf='_slic_s500_c10_o0'
python $scriptdir/convert/EM_stack2stack.py \
${datadir}/m000_01000-02000_01000-02000_00030-00460${pf}.h5 ${datadir}/m000_01000-02000_01000-02000_00030-00460${pf}.nii.gz \
-i 'zyx' -l 'xyz' -e -0.0073 -0.0073 0.05


python $scriptdir/convert/EM_stack2stack.py \
${datadir}/m000_01000-01500_01000-01500_00200-00300.h5 ${datadir}/m000_01000-01500_01000-01500_00200-00300.nii.gz \
-i 'zyx' -l 'xyz' -e -0.0073 -0.0073 0.05 -u
pf='_slic_s500_c0.01_o0'
python $scriptdir/convert/EM_stack2stack.py \
${datadir}/m000_01000-01500_01000-01500_00200-00300${pf}.h5 ${datadir}/m000_01000-01500_01000-01500_00200-00300${pf}.nii.gz \
-i 'zyx' -l 'xyz' -e -0.0073 -0.0073 0.05 -d int32


python $HOME/workspace/EM/convert/EM_stack2stack.py \
/Users/michielk/oxdata/P01/EM/M2/I/training_data0.h5 \
/Users/michielk/oxdata/P01/EM/M2/I/training_data0_tmp.nii.gz \
-i 'zyx' -l 'xyz' -e -0.0073 -0.0073 0.05 -u
pf='_slicvoxels002'
python $HOME/workspace/EM/convert/EM_stack2stack.py \
/Users/michielk/oxdata/P01/EM/M2/I/training_data0${pf}.h5 \
/Users/michielk/oxdata/P01/EM/M2/I/training_data0_tmp${pf}.nii.gz \
-i 'zyx' -l 'xyz' -e -0.0073 -0.0073 0.05 -d int32


scriptdir="$HOME/workspace/EM"
DATA="$HOME/oxdata"
datadir="$DATA/P01/EM/M3/M3_S1_GNU" && cd $datadir
dataset='m000'
x=1000; X=1500; y=1000; Y=1500; z=30; Z=460;
datastem="${dataset}_`printf %05d ${x}`-`printf %05d ${X}`_`printf %05d ${y}`-`printf %05d ${Y}`_`printf %05d ${z}`-`printf %05d ${Z}`"

oX=1000;oY=1000;oZ=30;
nx=1000;nX=1500;ny=1000;nY=1500;nz=30;nZ=80;
newstem=${dataset}_`printf %05d ${nx}`-`printf %05d ${nX}`_`printf %05d ${ny}`-`printf %05d ${nY}`_`printf %05d ${nz}`-`printf %05d ${nZ}`
# raw data to uint8
pf=''
python $scriptdir/convert/EM_stack2stack.py \
${datadir}/${datastem}${pf}.h5 ${datadir}/${newstem}${pf}.h5 \
-i 'zyx' -l 'zyx' -e 0.05 0.0073 0.0073 \
-x $((nx-oX)) -X $((nX-oX)) \
-y $((ny-oY)) -Y $((nY-oY)) \
-z $((nz-oZ)) -Z $((nZ-oZ))
python $scriptdir/convert/EM_stack2stack.py \
${datadir}/${newstem}${pf}.h5 ${datadir}/${newstem}${pf}.nii.gz \
-i 'zyx' -l 'xyz' -e -0.0073 -0.0073 0.05
# other
pf='_slic_s00500_c0.020_o0.010'
python $scriptdir/convert/EM_stack2stack.py \
${datadir}/${datastem}${pf}.h5 ${datadir}/${newstem}${pf}.h5 \
-i 'zyx' -l 'zyx' -e 0.05 -0.0073 -0.0073 \
-x $((nx-oX)) -X $((nX-oX)) \
-y $((ny-oY)) -Y $((nY-oY)) \
-z $((nz-oZ)) -Z $((nZ-oZ))
python $scriptdir/convert/EM_stack2stack.py \
${datadir}/${newstem}${pf}.h5 ${datadir}/${newstem}${pf}.nii.gz \
-i 'zyx' -l 'xyz' -e -0.0073 -0.0073 0.05 -d int32


# from scratch_20160215

scriptdir="$HOME/workspace/EM"
DATA="$HOME/oxdata"
datadir="$DATA/P01/EM/M3/M3_S1_GNU" && cd $datadir
dataset='m000'
paraview --state=$datadir/m000_01000-01500_01000-01500_00200-00300/stack+mesh_compact.pvsm


qsubfile=$datadir/EM_con2.sh
echo '#!/bin/bash' > $qsubfile
echo "#SBATCH --nodes=1" >> $qsubfile
echo "#SBATCH --ntasks-per-node=1" >> $qsubfile
echo "#SBATCH --time=01:00:00" >> $qsubfile
echo "#SBATCH --job-name=EM_con" >> $qsubfile
echo "python $scriptdir/convert/EM_stack2stack.py \
${datadir}/m000.h5 ${datadir}/m000.h5 \
-i 'zyx' -l 'xyz' -e -0.0073 -0.0073 0.05 -u"  >> $qsubfile
sbatch $qsubfile

scriptdir="$HOME/workspace/EM"
DATA="$HOME/oxdata"
datadir="$DATA/P01/EM/M3/M3_S1_GNU" && cd $datadir
dataset='m000'
pf=''
xs=1000; ys=1000;
z=30; Z=460;
for x in 2000 3000; do
for y in 2000 3000; do
X=$((x+xs))
Y=$((y+ys))
datastem=${dataset}_`printf %05d ${x}`-`printf %05d ${X}`_`printf %05d ${y}`-`printf %05d ${Y}`_`printf %05d ${z}`-`printf %05d ${Z}`
python $scriptdir/convert/EM_stack2stack.py \
${datadir}/${datastem}${pf}.h5 ${datadir}/${datastem}${pf}.nii.gz \
-i 'zyx' -l 'xyz' -e -0.0073 -0.0073 0.05 -u
gunzip ${datadir}/${datastem}${pf}.nii.gz
done
done


# from scratch_20160713

###========================###
### convert .h5 to .nii.gz ###
###========================###
scriptdir="$HOME/workspace/EM"
DATA="$HOME/oxdata"
datadir="$DATA/P01/EM/M3/M3_S1_GNU" && cd $datadir
dataset='m000'
x=1000; y=1000; xs=500; ys=500; z=30; Z=460;
[ $x == 5000 ] && X=5217 || X=$((x+xs))
[ $y == 4000 ] && Y=4460 || Y=$((y+ys))
datastem=${dataset}_`printf %05d ${x}`-`printf %05d ${X}`_`printf %05d ${y}`-`printf %05d ${Y}`_`printf %05d ${z}`-`printf %05d ${Z}`
# raw data to uint8
pf=''
python $scriptdir/convert/EM_stack2stack.py \
${datadir}/${datastem}${pf}.h5 ${datadir}/${datastem}${pf}.nii.gz \
-i 'zyx' -l 'xyz' -e -0.0073 -0.0073 0.05 -u

pf='_MA_sMAkn'
pf='_MA_sMAkn_sUAkn'
pf='_MA_sMAkn_sUAkn_ws'
pf='_MA_sMAkn_sUAkn_ws_filled'
pf='_MM_ws'
pf='_MM_ws_sw'
pf='_MM_ws_sw_distsum'
pf='_UA_sUAkn'
pf='_UA_sUAkn_ws'
pf='_PA'
pf='_mito'
python $scriptdir/convert/EM_stack2stack.py \
${datadir}/${datastem}${pf}.h5 ${datadir}/${datastem}${pf}.nii.gz \
-i 'zyx' -l 'xyz' -e -0.0073 -0.0073 0.05


for pf in _MA_sMAkn _MA_sMAkn_sUAkn _MA_sMAkn_sUAkn_ws _MA_sMAkn_sUAkn_ws_filled; do
python $scriptdir/convert/EM_stack2stack.py \
${datadir}/${datastem}${pf}.h5 ${datadir}/${datastem}${pf}.nii.gz \
-i 'zyx' -l 'xyz' -e -0.0073 -0.0073 0.05
done

for pf in _MM_ws _MM_ws_sw_distsum _MM_ws_sw _UA_sUAkn _UA_sUAkn_ws _PA; do
python $scriptdir/convert/EM_stack2stack.py \
${datadir}/${datastem}${pf}.h5 ${datadir}/${datastem}${pf}.nii.gz \
-i 'zyx' -l 'xyz' -e -0.0073 -0.0073 0.05
done



pf='_probs2'
python $scriptdir/convert/EM_stack2stack.py \
${datadir}/${datastem}${pf}.h5 ${datadir}/${datastem}${pf}.nii.gz \
-i 'zyx' -l 'xyz' -e -0.0073 -0.0073 0.05 -d float




# subsetting 1000x1000x430 volume
rsync -avz ndcn0180@arcus.arc.ox.ac.uk:/data/ndcn-fmrib-water-brain/ndcn0180/EM/M3/M3_S1_GNU/restored/m000_01000-02000_01000-02000_00030-00460_probs2_eed2.h5 /Users/michielk/oxdata/P01/EM/M3/M3_S1_GNU/

scriptdir="$HOME/workspace/EM"
DATA="$HOME/oxdata"
datadir="$DATA/P01/EM/M3/M3_S1_GNU" && cd $datadir
dataset='m000'
x=1000; X=2000; y=1000; Y=2000; z=30; Z=460;
datastem=${dataset}_`printf %05d ${x}`-`printf %05d ${X}`_`printf %05d ${y}`-`printf %05d ${Y}`_`printf %05d ${z}`-`printf %05d ${Z}`
oX=1000;oY=1000;oZ=30;
nz=30;nZ=460;
nx=1000
ny=1000
[ $nx == 5000 ] && nX=5217 || nX=$((nx+500))
[ $ny == 4000 ] && nY=4460 || nY=$((ny+500))
newstem=${dataset}_`printf %05d ${nx}`-`printf %05d ${nX}`_`printf %05d ${ny}`-`printf %05d ${nY}`_`printf %05d ${nz}`-`printf %05d ${nZ}`
pf='_probs2_eed2'
python $scriptdir/convert/EM_stack2stack.py \
${datadir}/${datastem}${pf}.h5 ${datadir}/${newstem}${pf}.h5 \
-i 'zyx' -l 'zyx' -e 0.05 0.0073 0.0073 \
-x $((nx-oX)) -X $((nX-oX)) -y $((ny-oY)) -Y $((nY-oY)) -z $((nz-oZ)) -Z $((nZ-oZ))

python $scriptdir/convert/EM_stack2stack.py \
${datadir}/${newstem}${pf}.h5 ${datadir}/${newstem}${pf}.nii.gz \
-i 'zyx' -l 'xyz' -e -0.0073 -0.0073 0.05 -d float


rsync -avz ndcn0180@arcus.arc.ox.ac.uk:/data/ndcn-fmrib-water-brain/ndcn0180/EM/M3/M3_S1_GNU/restored/m000_03000-04000_03000-04000_00030-00460.h5 /Users/michielk/oxdata/P01/EM/M3/M3_S1_GNU/
rsync -avz ndcn0180@arcus.arc.ox.ac.uk:/data/ndcn-fmrib-water-brain/ndcn0180/EM/M3/M3_S1_GNU/restored/m000_03000-04000_03000-04000_00030-00460_probs_ws_MA_probs_ws_MAfilled.h5 /Users/michielk/oxdata/P01/EM/M3/M3_S1_GNU/
rsync -avz ndcn0180@arcus.arc.ox.ac.uk:/data/ndcn-fmrib-water-brain/ndcn0180/EM/M3/M3_S1_GNU/restored/m000_03000-04000_03000-04000_00030-00460_probs_ws_MMdistsum_distfilter.h5 /Users/michielk/oxdata/P01/EM/M3/M3_S1_GNU/
rsync -avz ndcn0180@arcus.arc.ox.ac.uk:/data/ndcn-fmrib-water-brain/ndcn0180/EM/M3/M3_S1_GNU/restored/m000_03000-04000_03000-04000_00030-00460_probs_ws_UA.h5 /Users/michielk/oxdata/P01/EM/M3/M3_S1_GNU/
rsync -avz ndcn0180@arcus.arc.ox.ac.uk:/data/ndcn-fmrib-water-brain/ndcn0180/EM/M3/M3_S1_GNU/restored/m000_03000-04000_03000-04000_00030-00460_probs_ws_PA.h5 /Users/michielk/oxdata/P01/EM/M3/M3_S1_GNU/

remdir='/data/ndcn-fmrib-water-brain/ndcn0180/EM/M3/M3_S1_GNU/restored'
scriptdir="$HOME/workspace/EM"
DATA="$HOME/oxdata"
datadir="$DATA/P01/EM/M3/M3_S1_GNU" && cd $datadir
dataset='m000'
x=3000; y=3000; xs=1000; ys=1000; z=30; Z=460;
[ $x == 5000 ] && X=5217 || X=$((x+xs))
[ $y == 4000 ] && Y=4460 || Y=$((y+ys))
datastem=${dataset}_`printf %05d ${x}`-`printf %05d ${X}`_`printf %05d ${y}`-`printf %05d ${Y}`_`printf %05d ${z}`-`printf %05d ${Z}`

for pf in _probs_ws_MA_probs_ws_MAfilled _probs_ws_MMdistsum_distfilter _probs_ws_UA _probs_ws_PA; do
rsync -avz ndcn0180@arcus.arc.ox.ac.uk:$remdir/${datastem}_probs_ws_PA.h5 /Users/michielk/oxdata/P01/EM/M3/M3_S1_GNU/
done

# raw data to uint8
pf=''
python $scriptdir/convert/EM_stack2stack.py \
${datadir}/${datastem}${pf}.h5 ${datadir}/${datastem}${pf}.nii.gz \
-i 'zyx' -l 'xyz' -e -0.0073 -0.0073 0.05 -u

for pf in _probs_ws_MA_probs_ws_MAfilled _probs_ws_MMdistsum_distfilter _probs_ws_UA _probs_ws_PA; do
python $scriptdir/convert/EM_stack2stack.py \
${datadir}/${datastem}${pf}.h5 ${datadir}/${datastem}${pf}.nii.gz \
-i 'zyx' -l 'xyz' -e -0.0073 -0.0073 0.05
done









dataset='m000'
python $scriptdir/convert/EM_stack2stack.py \
"${datadir}/${dataset}.h5" \
"${datadir}/${dataset}.nii.gz" -f "/stack" -i 'zyx' -l 'xyz'





rsync -avz /Users/michielk/oxdata/P01/EM/M3/M3_S1_GNU/m000_01000-02000_02000-03000_00030-00460_probs_ws_MA_probs_ws_MA_manseg.h5 ndcn0180@arcus.arc.ox.ac.uk:/data/ndcn-fmrib-water-brain/ndcn0180/EM/M3/M3_S1_GNU/
