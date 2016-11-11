#rsync -avz ndcn0180@arcus.arc.ox.ac.uk:/data/ndcn-fmrib-water-brain/ndcn0180/EM/M3/M3_S1_GNU/m000_04000-05000_04000-05000_00000-00460*.h5 /Users/michielk/oxdata/P01/EM/M3/M3_S1_GNU/
z=0; Z=430;
for x in 0 1000 2000 3000 4000; do
[ $x == 5000 ] && X=5217 || X=$((y+1000))
qsubfile=$datadir/EM_p2l_${x}-${X}.sh
echo '#!/bin/bash' > $qsubfile
echo "#SBATCH --nodes=1" >> $qsubfile
echo "#SBATCH --ntasks-per-node=1" >> $qsubfile
echo "#SBATCH --time=00:10:00" >> $qsubfile
echo "#SBATCH --job-name=EM_ws" >> $qsubfile
for y in 0 1000 2000 3000 4000; do
[ $y == 4000 ] && Y=4460 || Y=$((y+1000))
echo "python $scriptdir/mesh/prob2labels.py $datadir $dataset \
-x $x -X $X -y $y -Y $Y -z $z -Z $Z > $datadir/output_${x}-${X}_${y}-${Y} &" >> $qsubfile
done
echo "wait" >> $qsubfile
sbatch -p devel $qsubfile
done
#m000_04000-05000_04000-05000_00000-00460_probs0_eed2.h5




z=30; Z=460;
x=1000; y=1000;
[ $x == 5000 ] && X=5217 || X=$((x+1000))
qsubfile=$datadir/EM_p2l_`printf %05d ${x}`-`printf %05d ${X}`.sh
echo '#!/bin/bash' > $qsubfile
echo "#SBATCH --nodes=1" >> $qsubfile
echo "#SBATCH --ntasks-per-node=1" >> $qsubfile
echo "#SBATCH --time=01:00:00" >> $qsubfile
echo "#SBATCH --mem=256000" >> $qsubfile
echo "#SBATCH --job-name=EM_ws" >> $qsubfile
[ $y == 4000 ] && Y=4460 || Y=$((y+1000))
echo "python $scriptdir/mesh/prob2labels.py $datadir $dataset \
--SEfile '_probs_ws_MA_probs_ws_MA_manseg.h5' --MAfile '_probs_ws_MAfilled.h5' --MMfile '_probs_ws_MMdistsum_distfilter.h5' \
-n 5 -o 250 235 491 -s 460 4460 5217 \
-x $x -X $X -y $y -Y $Y -z $z -Z $Z > \
$datadir/output_`printf %05d ${x}`-`printf %05d ${X}`_`printf %05d ${y}`-`printf %05d ${Y}` &" >> $qsubfile
echo "wait" >> $qsubfile
sbatch -p compute $qsubfile

z=30; Z=460;
x=1000; y=2000;
[ $x == 5000 ] && X=5217 || X=$((x+1000))
qsubfile=$datadir/EM_p2l_`printf %05d ${x}`-`printf %05d ${X}`.sh
echo '#!/bin/bash' > $qsubfile
echo "#SBATCH --nodes=1" >> $qsubfile
echo "#SBATCH --ntasks-per-node=1" >> $qsubfile
echo "#SBATCH --time=05:00:00" >> $qsubfile
echo "#SBATCH --mem=256000" >> $qsubfile
echo "#SBATCH --job-name=EM_ws" >> $qsubfile
[ $y == 4000 ] && Y=4460 || Y=$((y+1000))
echo "python $scriptdir/mesh/prob2labels.py $datadir $dataset \
--SEfile '_probs_ws_MA_probs_ws_MA_manseg.h5' \
-n 5 -o 250 235 491 -s 460 4460 5217 \
-x $x -X $X -y $y -Y $Y -z $z -Z $Z > \
$datadir/output_`printf %05d ${x}`-`printf %05d ${X}`_`printf %05d ${y}`-`printf %05d ${Y}` &" >> $qsubfile
echo "wait" >> $qsubfile
sbatch -p compute $qsubfile

z=30; Z=460;
x=1000; y=1000;
[ $x == 5000 ] && X=5217 || X=$((x+1000))
qsubfile=$datadir/EM_p2l_`printf %05d ${x}`-`printf %05d ${X}`.sh
echo '#!/bin/bash' > $qsubfile
echo "#SBATCH --nodes=1" >> $qsubfile
echo "#SBATCH --ntasks-per-node=1" >> $qsubfile
echo "#SBATCH --time=05:00:00" >> $qsubfile
echo "#SBATCH --mem=256000" >> $qsubfile
echo "#SBATCH --job-name=EM_ws" >> $qsubfile
[ $y == 4000 ] && Y=4460 || Y=$((y+1000))
echo "python $scriptdir/mesh/prob2labels.py $datadir $dataset \
--SEfile '_probs_ws_MA_probs_ws_MA_manseg.h5' \
-n 5 -o 250 235 491 -s 460 4460 5217 \
-x $x -X $X -y $y -Y $Y -z $z -Z $Z > \
$datadir/output_`printf %05d ${x}`-`printf %05d ${X}`_`printf %05d ${y}`-`printf %05d ${Y}` &" >> $qsubfile
echo "wait" >> $qsubfile
sbatch -p compute $qsubfile

z=30; Z=460;
x=0; y=0;
x=1000; y=1000;
[ $x == 5000 ] && X=5217 || X=$((x+1000))
[ $y == 4000 ] && Y=4460 || Y=$((y+1000))
qsubfile=$datadir/EM_p2l_`printf %05d ${x}`-`printf %05d ${X}`.sh
echo '#!/bin/bash' > $qsubfile
echo "#SBATCH --nodes=1" >> $qsubfile
echo "#SBATCH --ntasks-per-node=5" >> $qsubfile
echo "#SBATCH --time=00:10:00" >> $qsubfile
echo "#SBATCH --mem=56000" >> $qsubfile
echo "#SBATCH --job-name=EM_ws" >> $qsubfile
echo "python $scriptdir/mesh/prob2labels.py $datadir $dataset \
--SEfile '_seg.h5' --MAsegfile '_probs_ws_MA' \
-n 5 -o 250 235 491 -s 460 4460 5217 \
-x $x -X $X -y $y -Y $Y -z $z -Z $Z > \
$datadir/output_`printf %05d ${x}`-`printf %05d ${X}`_`printf %05d ${y}`-`printf %05d ${Y}` &" >> $qsubfile
echo "wait" >> $qsubfile
sbatch -p devel $qsubfile

z=30; Z=460;
# x=0; y=0;
x=1000; y=1000;
[ $x == 5000 ] && X=5217 || X=$((x+1000))
[ $y == 4000 ] && Y=4460 || Y=$((y+1000))
qsubfile=$datadir/EM_p2l_`printf %05d ${x}`-`printf %05d ${X}`.sh
echo '#!/bin/bash' > $qsubfile
echo "#SBATCH --nodes=1" >> $qsubfile
echo "#SBATCH --ntasks-per-node=1" >> $qsubfile
echo "#SBATCH --time=05:00:00" >> $qsubfile
echo "#SBATCH --mem=256000" >> $qsubfile
echo "#SBATCH --job-name=EM_ws" >> $qsubfile
echo "python $scriptdir/mesh/prob2labels.py $datadir $dataset \
--SEfile '_seg.h5' --MAfile '_probs_ws_MA.h5' --MMfile '_probs_ws_MMdistsum_distfilter.h5' \
-n 5 -o 250 235 491 -s 460 4460 5217 \
-x $x -X $X -y $y -Y $Y -z $z -Z $Z > \
$datadir/output_`printf %05d ${x}`-`printf %05d ${X}`_`printf %05d ${y}`-`printf %05d ${Y}` &" >> $qsubfile
echo "wait" >> $qsubfile
sbatch -p compute $qsubfile

z=30; Z=460;
# x=0; y=0;
x=1000; y=1000;
[ $x == 5000 ] && X=5217 || X=$((x+1000))
[ $y == 4000 ] && Y=4460 || Y=$((y+1000))
qsubfile=$datadir/EM_p2l_`printf %05d ${x}`-`printf %05d ${X}`.sh
echo '#!/bin/bash' > $qsubfile
echo "#SBATCH --nodes=1" >> $qsubfile
echo "#SBATCH --ntasks-per-node=1" >> $qsubfile
echo "#SBATCH --time=00:10:00" >> $qsubfile
echo "#SBATCH --mem=56000" >> $qsubfile
echo "#SBATCH --job-name=EM_ws" >> $qsubfile
echo "python $scriptdir/mesh/prob2labels.py $datadir $dataset \
--SEfile '_seg.h5' --MAfile '_probs_ws_MA.h5' --MMfile '_probs_ws_MMdistsum_distfilter.h5' \
-n 5 -o 250 235 491 -s 460 4460 5217 \
-x $x -X $X -y $y -Y $Y -z $z -Z $Z > \
$datadir/output_`printf %05d ${x}`-`printf %05d ${X}`_`printf %05d ${y}`-`printf %05d ${Y}` &" >> $qsubfile
echo "wait" >> $qsubfile
sbatch -p devel $qsubfile

z=30; Z=460;
for x in 0 1000 2000 3000 4000 5000; do
[ $x == 5000 ] && X=5217 || X=$((x+1000))
#[ $x == 4000 ] && X=4235 || X=$((y+1000))
qsubfile=$datadir/EM_p2l_`printf %05d ${x}`-`printf %05d ${X}`.sh
echo '#!/bin/bash' > $qsubfile
echo "#SBATCH --nodes=1" >> $qsubfile
echo "#SBATCH --ntasks-per-node=5" >> $qsubfile
echo "#SBATCH --time=100:00:00" >> $qsubfile
echo "#SBATCH --mem=256000" >> $qsubfile
echo "#SBATCH --job-name=EM_ws" >> $qsubfile
for y in 0 1000 2000 3000 4000; do
[ $y == 4000 ] && Y=4460 || Y=$((y+1000))
#[ $y == 4000 ] && Y=4111 || Y=$((y+1000))
echo "python $scriptdir/mesh/prob2labels.py $datadir $dataset \
-n 5 -o 220 235 491 -s 460 4460 5217 \
-x $x -X $X -y $y -Y $Y -z $z -Z $Z > \
$datadir/output_`printf %05d ${x}`-`printf %05d ${X}`_`printf %05d ${y}`-`printf %05d ${Y}` &" >> $qsubfile
done
echo "wait" >> $qsubfile
sbatch -p compute $qsubfile
done

# included knossos
z=30; Z=460;
# x=0; y=0;
x=1000; y=1000;
[ $x == 5000 ] && X=5217 || X=$((x+1000))
[ $y == 4000 ] && Y=4460 || Y=$((y+1000))
qsubfile=$datadir/EM_p2l_`printf %05d ${x}`-`printf %05d ${X}`.sh
echo '#!/bin/bash' > $qsubfile
echo "#SBATCH --nodes=1" >> $qsubfile
echo "#SBATCH --ntasks-per-node=1" >> $qsubfile
echo "#SBATCH --time=01:10:00" >> $qsubfile
echo "#SBATCH --mem=56000" >> $qsubfile
echo "#SBATCH --job-name=EM_ws" >> $qsubfile
echo "python $scriptdir/mesh/prob2labels.py $datadir $dataset \
--SEfile '_seg.h5' --MAfile '_probs_ws_MA_filled.h5' --MMfile '_probs_ws_MMdistsum_distfilter.h5' --KNfile annotation.xml \
-n 5 -o 250 235 491 -s 460 4460 5217 \
-x $x -X $X -y $y -Y $Y -z $z -Z $Z > \
$datadir/output_`printf %05d ${x}`-`printf %05d ${X}`_`printf %05d ${y}`-`printf %05d ${Y}` &" >> $qsubfile
echo "wait" >> $qsubfile
sbatch -p compute $qsubfile


# included knossos for MA
scriptdir="$HOME/workspace/EM"
DATA="$HOME/oxdata"
datadir="$DATA/P01/EM/M3/M3_S1_GNU" && cd $datadir
dataset='m000'
x=1000; X=1500; y=1000; Y=1500; z=200; Z=300;
datastem=${dataset}_`printf %05d ${x}`-`printf %05d ${X}`_`printf %05d ${y}`-`printf %05d ${Y}`_`printf %05d ${z}`-`printf %05d ${Z}`

python $scriptdir/mesh/prob2labels.py $datadir $dataset \
--SEfile '_seg' --MAfile '_' --MMfile '_' --UAfile '_' --PAfile '_' \
-n 5 -o 250 235 491 -s 460 4460 5217 \
-x $x -X $X -y $y -Y $Y -z $z -Z $Z

python $scriptdir/mesh/prob2labels.py $datadir $dataset \
--SEfile '_seg' --MMfile '_' --UAfile '_' --PAfile '_' \
-n 5 -o 250 235 491 -s 460 4460 5217 \
-x $x -X $X -y $y -Y $Y -z $z -Z $Z

python $scriptdir/mesh/prob2labels.py $datadir $dataset \
--MAknossosfile '_KN' --MMfile '_' --UAfile '_' --PAfile '_' \
-n 5 -o 250 235 491 -s 460 4460 5217 \
-x $x -X $X -y $y -Y $Y -z $z -Z $Z

python $scriptdir/mesh/prob2labels.py $datadir $dataset \
--MAknossosfile '_KN' --MMfile '_' --UAfile '_' --UAknossosfile '_KN' --PAfile '_' \
-n 5 -o 250 235 491 -s 460 4460 5217 \
-x $x -X $X -y $y -Y $Y -z $z -Z $Z


python $scriptdir/mesh/prob2labels.py $datadir $dataset \
--SEfile '_seg' --MAknossosfile '_KN' --MMfile '_' --UAfile '_' --PAfile '_' \
-n 5 -o 250 235 491 -s 460 4460 5217 \
-x $x -X $X -y $y -Y $Y -z $z -Z $Z

# python $scriptdir/mesh/prob2labels.py $datadir $dataset \
# --MAknossosfile '_KN' --MMfile '_' --UAfile '_' --PAfile '_' \
# -n 5 -o 250 235 491 -s 460 4460 5217 \
# -x $x -X $X -y $y -Y $Y -z $z -Z $Z -f

python $scriptdir/mesh/prob2labels.py $datadir $dataset \
--MAfile '_MA_KN_ws' --UAfile '_' --PAfile '_' \
-n 5 -o 250 235 491 -s 460 4460 5217 \
-x $x -X $X -y $y -Y $Y -z $z -Z $Z


# pf='_seg'
# pf='_MA_seg_ws'
# pf='_MA_KN'
pf='_MA_KN_ws'
pf='_MA_KN_KN_ws'
# pf='_MA_KN_ws_filled'
# pf='_MA_seg_KN_ws'
# pf='_MM_ws'
pf='_KN'
python $scriptdir/convert/EM_stack2stack.py \
${datadir}/${datastem}${pf}.h5 ${datadir}/${datastem}${pf}.nii.gz \
-i 'zyx' -l 'xyz' -e -0.0073 -0.0073 0.05






# ARC 2 LOCAL
#rsync -avz /Users/michielk/oxdata/P01/EM/M3/M3_S1_GNU/m000_01000-02000_01000-02000_00030-00460_knossos/annotation_??.xml  ndcn0180@arcus.arc.ox.ac.uk:/data/ndcn-fmrib-water-brain/ndcn0180/EM/M3/M3_S1_GNU/m000_01000-01500_01000-01500_00030-00460_knossos

x=1000; X=1500; y=1000; Y=1500; z=30; Z=460;
qsubfile=$datadir/EM_p2l.sh
echo '#!/bin/bash' > $qsubfile
echo "#SBATCH --nodes=1" >> $qsubfile
echo "#SBATCH --ntasks-per-node=1" >> $qsubfile
echo "#SBATCH --time=05:10:00" >> $qsubfile
echo "#SBATCH --mem=56000" >> $qsubfile
echo "#SBATCH --job-name=EM_ws" >> $qsubfile
echo "python $scriptdir/mesh/prob2labels.py $datadir $dataset \
--MAknossosfile '_sMAkn' --UAknossosfile '_sUAkn' \
-n 5 -o 250 235 491 -s 460 4460 5217 \
-x $x -X $X -y $y -Y $Y -z $z -Z $Z -f -w -d" >> $qsubfile
sbatch -p compute $qsubfile

x=1000; X=1500; y=1000; Y=1500; z=30; Z=460;
qsubfile=$datadir/EM_p2l_MA.sh
echo '#!/bin/bash' > $qsubfile
echo "#SBATCH --nodes=1" >> $qsubfile
echo "#SBATCH --ntasks-per-node=1" >> $qsubfile
echo "#SBATCH --time=00:10:00" >> $qsubfile
echo "#SBATCH --mem=56000" >> $qsubfile
echo "#SBATCH --job-name=EM_ws" >> $qsubfile
echo "python $scriptdir/mesh/prob2labels.py $datadir $dataset \
--MAknossosfile '_sMAkn' --MMfile '_' --UAfile '_' --UAknossosfile '_sUAkn' --PAfile '_' \
-n 5 -o 250 235 491 -s 460 4460 5217 \
-x $x -X $X -y $y -Y $Y -z $z -Z $Z -f -w -d" >> $qsubfile
sbatch -p devel $qsubfile
x=1000; X=1500; y=1000; Y=1500; z=30; Z=460;
qsubfile=$datadir/EM_p2l_MM.sh
echo '#!/bin/bash' > $qsubfile
echo "#SBATCH --nodes=1" >> $qsubfile
echo "#SBATCH --ntasks-per-node=1" >> $qsubfile
echo "#SBATCH --time=05:10:00" >> $qsubfile
echo "#SBATCH --mem=56000" >> $qsubfile
echo "#SBATCH --job-name=EM_ws" >> $qsubfile
echo "python $scriptdir/mesh/prob2labels.py $datadir $dataset \
--MAfile '_MA_sMAkn_sUAkn_ws_filled' --UAfile '_' --UAknossosfile '_sUAkn' --PAfile '_' \
-n 5 -o 250 235 491 -s 460 4460 5217 \
-x $x -X $X -y $y -Y $Y -z $z -Z $Z -f -w -d" >> $qsubfile
sbatch -p compute $qsubfile
x=1000; X=1500; y=1000; Y=1500; z=30; Z=460;
qsubfile=$datadir/EM_p2l_UA.sh
echo '#!/bin/bash' > $qsubfile
echo "#SBATCH --nodes=1" >> $qsubfile
echo "#SBATCH --ntasks-per-node=1" >> $qsubfile
echo "#SBATCH --time=00:10:00" >> $qsubfile
echo "#SBATCH --mem=56000" >> $qsubfile
echo "#SBATCH --job-name=EM_ws" >> $qsubfile
echo "python $scriptdir/mesh/prob2labels.py $datadir $dataset \
--MAfile '_MA_sMAkn_sUAkn_ws_filled' --MMfile '_MM_ws_sw_df' --UAknossosfile '_sUAkn' \
-n 5 -o 250 235 491 -s 460 4460 5217 \
-x $x -X $X -y $y -Y $Y -z $z -Z $Z -f -w -d" >> $qsubfile
sbatch -p devel $qsubfile

#rsync -avz ndcn0180@arcus.arc.ox.ac.uk:/data/ndcn-fmrib-water-brain/ndcn0180/EM/M3/M3_S1_GNU/m000_01000-01500_01000-01500_?????-?????_seg.h5 /Users/michielk/oxdata/P01/EM/M3/M3_S1_GNU/
#rsync -avz ndcn0180@arcus.arc.ox.ac.uk:/data/ndcn-fmrib-water-brain/ndcn0180/EM/M3/M3_S1_GNU/m000_01000-01500_01000-01500_?????-?????*.h5 /Users/michielk/oxdata/P01/EM/M3/M3_S1_GNU/
#rsync -avz ndcn0180@arcus.arc.ox.ac.uk:/data/ndcn-fmrib-water-brain/ndcn0180/EM/M3/M3_S1_GNU/m000_01000-01500_01000-01500_?????-?????_MA*.h5 /Users/michielk/oxdata/P01/EM/M3/M3_S1_GNU/

###=======###
### LOCAL ###
###=======###
scriptdir="$HOME/workspace/EM"
DATA="$HOME/oxdata"
datadir="$DATA/P01/EM/M3/M3_S1_GNU" && cd $datadir
dataset='m000'
x=1000; X=1500; y=1000; Y=1500; z=30; Z=460;
datastem=${dataset}_`printf %05d ${x}`-`printf %05d ${X}`_`printf %05d ${y}`-`printf %05d ${Y}`_`printf %05d ${z}`-`printf %05d ${Z}`

python $scriptdir/mesh/prob2labels.py $datadir $dataset \
--MAknossosfile '_sMAknedges' --MMfile '_' --UAfile '_' --UAknossosfile '_sUAkn' --PAfile '_' \
-n 5 -o 250 235 491 -s 460 4460 5217 \
-x $x -X $X -y $y -Y $Y -z $z -Z $Z -f -w -d






# from refineSeg









#         labeled_comprehension(input, labels, index, func, out_dtype, default, pass_positions=False)


MA = loadh5(datadir, dataset + MAfile)
# sesize = (5,5,5)
it = 10  # might be a bit agressive
for l in np.unique(MA)[1:]:
    ### fill holes
    # binary_fill_holes(binim, output=binim)  # does not fill components conneted to boundaries
    labels = label(MA!=l)[0]
    labelCount = np.bincount(labels.ravel())
    background = np.argmax(labelCount)
    MA[labels != background] = l
    ### closing
    # binary_closing(binim, structure=np.ones(sesize), iterations=it, output=binim)
    binim = MA==l
    binim = binary_closing(binim, iterations=it)
    MA[binim] = l
    ### fill holes
    labels = label(MA!=l)[0]
    labelCount = np.bincount(labels.ravel())
    background = np.argmax(labelCount)
    MA[labels != background] = l
    ### update myelin mask
    myelin[MA != 0] = False
    print(l)

writeh5(MA, datadir, dataset + '_probs_ws_MAfilled.h5')


import math
def sigmoid(x):
    return 1 / (1 + math.exp(-x))
from scipy.special import expit

MM = loadh5(datadir, dataset + MMfile)
# distsum = np.zeros_like(MM, dtype='float')
distsum = np.ones_like(MM, dtype='float')
lmask = np.zeros((MM.shape[0],MM.shape[1],MM.shape[2],len(np.unique(MA)[1:])), dtype='bool')
medwidth = {}
for l in np.unique(MA)[1:]:
# for l in [1248,1249,1250,1352]:
    dist = distance_transform_edt(MA!=l, sampling=[0.05,0.0073,0.0073])  # TODO: get from h5 (also for writes)
    # dist[MM!=l] = 0
    # get the median distance at the outer rim:
    MMfilled = MA+MM
    binim = MMfilled == l
    rim = np.logical_xor(erosion(binim), binim)
    medwidth = np.median(dist[rim])
    # nmed = 3
    # maxdist = nmed * medwidth  # np.histogram(dist, bins=100, density=True)
    # MM[np.logical_and(dist > maxdist, MM==l)] = 0
    # weighteddist = dist/medwidth
    weighteddist = expit(dist/medwidth)
    # weighteddist[weighteddist>1] = 1 + (weighteddist[weighteddist>1]-1) / 4
    # weighteddist[dist/medwidth>nmed] = 0
    # writeh5(weighteddist, datadir, dataset + '_probs_ws_wdist' + str(l) + '.h5', dtype='float')
    distsum = np.minimum(distsum, weighteddist)
    # distsum += weighteddist
    # writeh5(distsum, datadir, dataset + '_probs_ws_distsum' + str(l) + '.h5', dtype='float')
    # print(np.count_nonzero(distsum>nmed))
    # distsum[distsum>nmed] = distsum[distsum>nmed] / 2
    print(l)
writeh5(distsum, datadir, dataset + '_probs_ws_distsum.h5', dtype='float')

tmpdistsum = np.copy(distsum)
tmpdistsum[~myelin] = 0
MM = watershed(tmpdistsum, grey_dilation(MA, size=(3,3,3)), mask=np.logical_and(myelin, ~datamask))
writeh5(MM, datadir, dataset + '_probs_ws_MMdistsum.h5')



MM = loadh5(datadir, dataset + MMfile)
distsum = np.ones_like(MM, dtype='float')
lmask = np.zeros((MM.shape[0],MM.shape[1],MM.shape[2],len(np.unique(MA)[1:])), dtype='bool')
medwidth = {}
for i,l in enumerate(np.unique(MA)[1:]):  # TODO: implement mpi
    dist = distance_transform_edt(MA!=l, sampling=[0.05,0.0073,0.0073])  # TODO: get from h5 (also for writes)
    # get the median distance at the outer rim:
    MMfilled = MA+MM
    binim = MMfilled == l
    rim = np.logical_xor(erosion(binim), binim)
    medwidth[l] = np.median(dist[rim])
    # median width weighted sigmoid transform on distance function
    weighteddist = expit(dist/medwidth[l])
    distsum = np.minimum(distsum, weighteddist)
    # labelmask for voxels further than nmed medians from the object
    nmed = 3
    maxdist = nmed * medwidth[l]
    lmask[:,:,:,i] = dist > maxdist
    print(l)

MM = watershed(distsum, grey_dilation(MA, size=(3,3,3)), mask=np.logical_and(myelin, ~datamask))
writeh5(MM, datadir, dataset + '_probs_ws_MMdistsum.h5')

for i,l in enumerate(np.unique(MA)[1:]):
    MM[np.logical_and(lmask[:,:,:,i], MM==l)] = 0

writeh5(MM, datadir, dataset + '_probs_ws_MMdistsum_distfilter.h5')









scriptdir="$HOME/workspace/EM"
DATA="$HOME/oxdata"
datadir="$DATA/P01/EM/M3/M3_S1_GNU/devel"
dataset='m000'
python $scriptdir/mesh/prob2labels.py $datadir $dataset \
--SEfile _seg.h5 \
-x 0 -X 1000 -y 0 -Y 1000 -z 0 -Z 100
python $scriptdir/mesh/prob2labels.py $datadir $dataset \
--SEfile _seg.h5 --MAfile '_probs_ws_MAfilled.h5' \
-x 2000 -X 3000 -y 2000 -Y 3000 -z 0 -Z 100

python $scriptdir/mesh/prob2labels.py $datadir $dataset \
--MAfile '_probs_ws_MAfilled.h5' --MMfile '_probs_ws_MMdistsum_distfilter.h5' --UAfile '_probs_ws_UA.h5' --PAfile '_probs_ws_PA.h5' \
-n 5 -o 50 60 122 -s 100 4111 4235 -e 0.05 0.0073 0.0073 \
-x 0 -X 1000 -y 0 -Y 1000 -z 0 -Z 100


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



# from scratch5

scriptdir="$HOME/workspace/EM"
datadir="$DATA/EM/M3/M3_S1_GNU/testblock" && cd $datadir
dataset='m000'
refsect='0250'

x=1000; X=2000; y=1000; Y=2000; z=0; Z=430;  # mem +- 20GB for MA
qsubfile=$datadir/EM_p2l_${x}-${X}.sh
echo '#!/bin/bash' > $qsubfile
echo "#SBATCH --nodes=1" >> $qsubfile
echo "#SBATCH --ntasks-per-node=1" >> $qsubfile
echo "#SBATCH --time=01:00:00" >> $qsubfile
#echo "#SBATCH --mem=50000" >> $qsubfile
echo "#SBATCH --job-name=EM_ws" >> $qsubfile
echo "python $scriptdir/mesh/prob2labels.py $datadir $dataset \
--SEfile '_segman.h5' --MMfile '_probs_ws_MMdistsum_distfilter.h5' --UAfile '_probs_ws_UA.h5' --PAfile '_probs_ws_PA.h5' \
-n 5 -o 220 235 491 -s 430 4460 5217 \
-x $x -X $X -y $y -Y $Y -z $z -Z $Z > $datadir/output_${x}-${X}_${y}-${Y} &" >> $qsubfile
echo "wait" >> $qsubfile
#sbatch -p compute $qsubfile
sbatch -p devel $qsubfile

--SEfile '_seg.h5' --MAfile '_probs_ws_MAfilled.h5' --MMfile '_probs_ws_MMdistsum_distfilter.h5' --UAfile '_probs_ws_UA.h5' --PAfile '_probs_ws_PA.h5'


# from scratch3

#rsync -avz ndcn0180@arcus.arc.ox.ac.uk:/data/ndcn-fmrib-water-brain/ndcn0180/EM/M3/M3_S1_GNU/m000_04000-05000_04000-05000_00000-00460*.h5 /Users/michielk/oxdata/P01/EM/M3/M3_S1_GNU/
z=0; Z=430;
for x in 0 1000 2000 3000 4000; do
[ $x == 5000 ] && X=5217 || X=$((y+1000))
qsubfile=$datadir/EM_p2l_${x}-${X}.sh
echo '#!/bin/bash' > $qsubfile
echo "#SBATCH --nodes=1" >> $qsubfile
echo "#SBATCH --ntasks-per-node=1" >> $qsubfile
echo "#SBATCH --time=00:10:00" >> $qsubfile
echo "#SBATCH --job-name=EM_ws" >> $qsubfile
for y in 0 1000 2000 3000 4000; do
[ $y == 4000 ] && Y=4460 || Y=$((y+1000))
echo "python $scriptdir/mesh/prob2labels.py $datadir $dataset \
-x $x -X $X -y $y -Y $Y -z $z -Z $Z > $datadir/output_${x}-${X}_${y}-${Y} &" >> $qsubfile
done
echo "wait" >> $qsubfile
sbatch -p devel $qsubfile
done
#m000_04000-05000_04000-05000_00000-00460_probs0_eed2.h5

# from scratch4

#         labeled_comprehension(input, labels, index, func, out_dtype, default, pass_positions=False)

MA = loadh5(datadir, dataset + MAfile)
# sesize = (5,5,5)
it = 10  # might be a bit agressive
for l in np.unique(MA)[1:]:
    ### fill holes
    # binary_fill_holes(binim, output=binim)  # does not fill components conneted to boundaries
    labels = label(MA!=l)[0]
    labelCount = np.bincount(labels.ravel())
    background = np.argmax(labelCount)
    MA[labels != background] = l
    ### closing
    # binary_closing(binim, structure=np.ones(sesize), iterations=it, output=binim)
    binim = MA==l
    binim = binary_closing(binim, iterations=it)
    MA[binim] = l
    ### fill holes
    labels = label(MA!=l)[0]
    labelCount = np.bincount(labels.ravel())
    background = np.argmax(labelCount)
    MA[labels != background] = l
    ### update myelin mask
    myelin[MA != 0] = False
    print(l)

writeh5(MA, datadir, dataset + '_probs_ws_MAfilled.h5')


import math
def sigmoid(x):
    return 1 / (1 + math.exp(-x))
from scipy.special import expit

MM = loadh5(datadir, dataset + MMfile)
# distsum = np.zeros_like(MM, dtype='float')
distsum = np.ones_like(MM, dtype='float')
lmask = np.zeros((MM.shape[0],MM.shape[1],MM.shape[2],len(np.unique(MA)[1:])), dtype='bool')
medwidth = {}
for l in np.unique(MA)[1:]:
# for l in [1248,1249,1250,1352]:
    dist = distance_transform_edt(MA!=l, sampling=[0.05,0.0073,0.0073])  # TODO: get from h5 (also for writes)
    # dist[MM!=l] = 0
    # get the median distance at the outer rim:
    MMfilled = MA+MM
    binim = MMfilled == l
    rim = np.logical_xor(erosion(binim), binim)
    medwidth = np.median(dist[rim])
    # nmed = 3
    # maxdist = nmed * medwidth  # np.histogram(dist, bins=100, density=True)
    # MM[np.logical_and(dist > maxdist, MM==l)] = 0
    # weighteddist = dist/medwidth
    weighteddist = expit(dist/medwidth)
    # weighteddist[weighteddist>1] = 1 + (weighteddist[weighteddist>1]-1) / 4
    # weighteddist[dist/medwidth>nmed] = 0
    # writeh5(weighteddist, datadir, dataset + '_probs_ws_wdist' + str(l) + '.h5', dtype='float')
    distsum = np.minimum(distsum, weighteddist)
    # distsum += weighteddist
    # writeh5(distsum, datadir, dataset + '_probs_ws_distsum' + str(l) + '.h5', dtype='float')
    # print(np.count_nonzero(distsum>nmed))
    # distsum[distsum>nmed] = distsum[distsum>nmed] / 2
    print(l)
writeh5(distsum, datadir, dataset + '_probs_ws_distsum.h5', dtype='float')

tmpdistsum = np.copy(distsum)
tmpdistsum[~myelin] = 0
MM = watershed(tmpdistsum, grey_dilation(MA, size=(3,3,3)), mask=np.logical_and(myelin, ~datamask))
writeh5(MM, datadir, dataset + '_probs_ws_MMdistsum.h5')



MM = loadh5(datadir, dataset + MMfile)
distsum = np.ones_like(MM, dtype='float')
lmask = np.zeros((MM.shape[0],MM.shape[1],MM.shape[2],len(np.unique(MA)[1:])), dtype='bool')
medwidth = {}
for i,l in enumerate(np.unique(MA)[1:]):  # TODO: implement mpi
    dist = distance_transform_edt(MA!=l, sampling=[0.05,0.0073,0.0073])  # TODO: get from h5 (also for writes)
    # get the median distance at the outer rim:
    MMfilled = MA+MM
    binim = MMfilled == l
    rim = np.logical_xor(erosion(binim), binim)
    medwidth[l] = np.median(dist[rim])
    # median width weighted sigmoid transform on distance function
    weighteddist = expit(dist/medwidth[l])
    distsum = np.minimum(distsum, weighteddist)
    # labelmask for voxels further than nmed medians from the object
    nmed = 3
    maxdist = nmed * medwidth[l]
    lmask[:,:,:,i] = dist > maxdist
    print(l)

MM = watershed(distsum, grey_dilation(MA, size=(3,3,3)), mask=np.logical_and(myelin, ~datamask))
writeh5(MM, datadir, dataset + '_probs_ws_MMdistsum.h5')

for i,l in enumerate(np.unique(MA)[1:]):
    MM[np.logical_and(lmask[:,:,:,i], MM==l)] = 0

writeh5(MM, datadir, dataset + '_probs_ws_MMdistsum_distfilter.h5')









scriptdir="$HOME/workspace/EM"
DATA="$HOME/oxdata"
datadir="$DATA/P01/EM/M3/M3_S1_GNU/devel"
dataset='m000'
python $scriptdir/mesh/prob2labels.py $datadir $dataset \
--SEfile _seg.h5 \
-x 0 -X 1000 -y 0 -Y 1000 -z 0 -Z 100
python $scriptdir/mesh/prob2labels.py $datadir $dataset \
--SEfile _seg.h5 --MAfile '_probs_ws_MAfilled.h5' \
-x 2000 -X 3000 -y 2000 -Y 3000 -z 0 -Z 100

python $scriptdir/mesh/prob2labels.py $datadir $dataset \
--MAfile '_probs_ws_MAfilled.h5' --MMfile '_probs_ws_MMdistsum_distfilter.h5' --UAfile '_probs_ws_UA.h5' --PAfile '_probs_ws_PA.h5' \
-n 5 -o 50 60 122 -s 100 4111 4235 -e 0.05 0.0073 0.0073 \
-x 0 -X 1000 -y 0 -Y 1000 -z 0 -Z 100


# from scratch6

x=0000; X=3000; y=0000; Y=3000; z=0; Z=430;  # mem +- 188GB for MA
qsubfile=$datadir/EM_p2l_${x}-${X}.sh
echo '#!/bin/bash' > $qsubfile
echo "#SBATCH --nodes=1" >> $qsubfile
echo "#SBATCH --ntasks-per-node=1" >> $qsubfile
echo "#SBATCH --time=24:00:00" >> $qsubfile
echo "#SBATCH --mem=256000" >> $qsubfile
echo "#SBATCH --job-name=EM_ws" >> $qsubfile
echo "python $scriptdir/mesh/prob2labels.py $datadir $dataset \
--SEfile '_seg.h5' \
-n 5 -o 220 235 491 -s 430 4460 5217 \
-x $x -X $X -y $y -Y $Y -z $z -Z $Z > $datadir/output_${x}-${X}_${y}-${Y} &" >> $qsubfile
echo "wait" >> $qsubfile
sbatch -p compute $qsubfile
sbatch -p devel $qsubfile

--SEfile '_seg.h5' --MAfile '_probs_ws_MAfilled.h5' --MMfile '_probs_ws_MMdistsum_distfilter.h5' --UAfile '_probs_ws_UA.h5' --PAfile '_probs_ws_PA.h5'


# from scratch9

z=30; Z=460;
x=1000; y=1000;
[ $x == 5000 ] && X=5217 || X=$((x+1000))
qsubfile=$datadir/EM_p2l_`printf %05d ${x}`-`printf %05d ${X}`.sh
echo '#!/bin/bash' > $qsubfile
echo "#SBATCH --nodes=1" >> $qsubfile
echo "#SBATCH --ntasks-per-node=1" >> $qsubfile
echo "#SBATCH --time=01:00:00" >> $qsubfile
echo "#SBATCH --mem=256000" >> $qsubfile
echo "#SBATCH --job-name=EM_ws" >> $qsubfile
[ $y == 4000 ] && Y=4460 || Y=$((y+1000))
echo "python $scriptdir/mesh/prob2labels.py $datadir $dataset \
--SEfile '_probs_ws_MA_probs_ws_MA_manseg.h5' --MAfile '_probs_ws_MAfilled.h5' --MMfile '_probs_ws_MMdistsum_distfilter.h5' \
-n 5 -o 250 235 491 -s 460 4460 5217 \
-x $x -X $X -y $y -Y $Y -z $z -Z $Z > \
$datadir/output_`printf %05d ${x}`-`printf %05d ${X}`_`printf %05d ${y}`-`printf %05d ${Y}` &" >> $qsubfile
echo "wait" >> $qsubfile
sbatch -p compute $qsubfile

z=30; Z=460;
x=1000; y=2000;
[ $x == 5000 ] && X=5217 || X=$((x+1000))
qsubfile=$datadir/EM_p2l_`printf %05d ${x}`-`printf %05d ${X}`.sh
echo '#!/bin/bash' > $qsubfile
echo "#SBATCH --nodes=1" >> $qsubfile
echo "#SBATCH --ntasks-per-node=1" >> $qsubfile
echo "#SBATCH --time=05:00:00" >> $qsubfile
echo "#SBATCH --mem=256000" >> $qsubfile
echo "#SBATCH --job-name=EM_ws" >> $qsubfile
[ $y == 4000 ] && Y=4460 || Y=$((y+1000))
echo "python $scriptdir/mesh/prob2labels.py $datadir $dataset \
--SEfile '_probs_ws_MA_probs_ws_MA_manseg.h5' \
-n 5 -o 250 235 491 -s 460 4460 5217 \
-x $x -X $X -y $y -Y $Y -z $z -Z $Z > \
$datadir/output_`printf %05d ${x}`-`printf %05d ${X}`_`printf %05d ${y}`-`printf %05d ${Y}` &" >> $qsubfile
echo "wait" >> $qsubfile
sbatch -p compute $qsubfile

z=30; Z=460;
x=1000; y=1000;
[ $x == 5000 ] && X=5217 || X=$((x+1000))
qsubfile=$datadir/EM_p2l_`printf %05d ${x}`-`printf %05d ${X}`.sh
echo '#!/bin/bash' > $qsubfile
echo "#SBATCH --nodes=1" >> $qsubfile
echo "#SBATCH --ntasks-per-node=1" >> $qsubfile
echo "#SBATCH --time=05:00:00" >> $qsubfile
echo "#SBATCH --mem=256000" >> $qsubfile
echo "#SBATCH --job-name=EM_ws" >> $qsubfile
[ $y == 4000 ] && Y=4460 || Y=$((y+1000))
echo "python $scriptdir/mesh/prob2labels.py $datadir $dataset \
--SEfile '_probs_ws_MA_probs_ws_MA_manseg.h5' \
-n 5 -o 250 235 491 -s 460 4460 5217 \
-x $x -X $X -y $y -Y $Y -z $z -Z $Z > \
$datadir/output_`printf %05d ${x}`-`printf %05d ${X}`_`printf %05d ${y}`-`printf %05d ${Y}` &" >> $qsubfile
echo "wait" >> $qsubfile
sbatch -p compute $qsubfile

z=30; Z=460;
x=0; y=0;
x=1000; y=1000;
[ $x == 5000 ] && X=5217 || X=$((x+1000))
[ $y == 4000 ] && Y=4460 || Y=$((y+1000))
qsubfile=$datadir/EM_p2l_`printf %05d ${x}`-`printf %05d ${X}`.sh
echo '#!/bin/bash' > $qsubfile
echo "#SBATCH --nodes=1" >> $qsubfile
echo "#SBATCH --ntasks-per-node=5" >> $qsubfile
echo "#SBATCH --time=00:10:00" >> $qsubfile
echo "#SBATCH --mem=56000" >> $qsubfile
echo "#SBATCH --job-name=EM_ws" >> $qsubfile
echo "python $scriptdir/mesh/prob2labels.py $datadir $dataset \
--SEfile '_seg.h5' --MAsegfile '_probs_ws_MA' \
-n 5 -o 250 235 491 -s 460 4460 5217 \
-x $x -X $X -y $y -Y $Y -z $z -Z $Z > \
$datadir/output_`printf %05d ${x}`-`printf %05d ${X}`_`printf %05d ${y}`-`printf %05d ${Y}` &" >> $qsubfile
echo "wait" >> $qsubfile
sbatch -p devel $qsubfile



origset = dataset + '_' + str(x).zfill(nzfills) + '-' + str(X).zfill(nzfills) + \
                    '_' + str(y).zfill(nzfills) + '-' + str(Y).zfill(nzfills) + \
                    '_' + str(z).zfill(nzfills) + '-' + str(Z).zfill(nzfills)
nzfills=5
MAfile = '_probs_ws_MA.h5'
MAseedfile = '_seeds_MA.h5'

MA2file = '_probs_ws_MA_probs_ws_MA_manseg'
seeds_MA = loadh5(datadir, origset + MA2file + '.h5')[0]

_, elsize = loadh5(datadir, otherset + '.h5')

side=[-1000,0,0]
otherset = dataset + '_' + str(x + side[0]).zfill(nzfills) + '-' + str(X + side[0]).zfill(nzfills) + \
                     '_' + str(y + side[1]).zfill(nzfills) + '-' + str(Y + side[1]).zfill(nzfills) + \
                     '_' + str(z + side[2]).zfill(nzfills) + '-' + str(Z + side[2]).zfill(nzfills)
sidesection = loadh5(datadir, otherset + MA2file + '.h5')[0]
seeds_MA[:,:,0] = sidesection[:,:,-1]  # right

side=[1000,0,0]  #[0,-1000,0], [0,1000,0]]:
otherset = dataset + '_' + str(x + side[0]).zfill(nzfills) + '-' + str(X + side[0]).zfill(nzfills) + \
                     '_' + str(y + side[1]).zfill(nzfills) + '-' + str(Y + side[1]).zfill(nzfills) + \
                     '_' + str(z + side[2]).zfill(nzfills) + '-' + str(Z + side[2]).zfill(nzfills)
sidesection = loadh5(datadir, otherset + MA2file + '.h5')[0]
seeds_MA[:,:,-1] = sidesection[:,:,0]  # left

side=[0,-1000,0] #, [0,1000,0]]:
otherset = dataset + '_' + str(x + side[0]).zfill(nzfills) + '-' + str(X + side[0]).zfill(nzfills) + \
                     '_' + str(y + side[1]).zfill(nzfills) + '-' + str(Y + side[1]).zfill(nzfills) + \
                     '_' + str(z + side[2]).zfill(nzfills) + '-' + str(Z + side[2]).zfill(nzfills)
sidesection = loadh5(datadir, otherset + MA2file + '.h5')[0]
seeds_MA[:,0,:] = sidesection[:,-1,:]  # anterior

side=[0,1000,0]
otherset = dataset + '_' + str(x + side[0]).zfill(nzfills) + '-' + str(X + side[0]).zfill(nzfills) + \
                     '_' + str(y + side[1]).zfill(nzfills) + '-' + str(Y + side[1]).zfill(nzfills) + \
                     '_' + str(z + side[2]).zfill(nzfills) + '-' + str(Z + side[2]).zfill(nzfills)
sidesection = loadh5(datadir, otherset + MA2file + '.h5')[0]
seeds_MA[:,-1,:] = sidesection[:,0,:]  # posterior

writeh5(seeds_MA, datadir, origset + '_seeds_MA2.h5', element_size_um=elsize)

scriptdir="$HOME/workspace/EM"
DATA="$HOME/oxdata"
datadir="$DATA/P01/EM/M3/M3_S1_GNU" && cd $datadir
dataset='m000'
pf='_probs_ws_MA';
python $scriptdir/convert/EM_stack2stack.py \
"${datadir}/${dataset}_00000-01000_00000-01000_00030-00460${pf}.h5" \
"${datadir}/${dataset}_00000-01000_00000-01000_00030-00460${pf}.nii.gz" \
-i 'zyx' -l 'xyz' -e -0.0073 -0.0073 0.05
python $scriptdir/convert/EM_stack2stack.py \
"${datadir}/${dataset}_01000-02000_01000-02000_00030-00460${pf}.h5" \
"${datadir}/${dataset}_01000-02000_01000-02000_00030-00460${pf}.nii.gz" \
-i 'zyx' -l 'xyz' -e -0.0073 -0.0073 0.05




scriptdir="$HOME/workspace/EM"
DATA="$HOME/oxdata"
datadir="$DATA/P01/EM/M3/M3_S1_GNU" && cd $datadir
dataset='m000'
pf='_seeds_MA2';
python $scriptdir/convert/EM_stack2stack.py \
"${datadir}/${dataset}_01000-02000_01000-02000_00030-00460.h5" \
"${datadir}/${dataset}_01000-02000_01000-02000_00030-00460.nii.gz" \
-i 'zyx' -l 'xyz' -e -0.0073 -0.0073 0.05 -u


z=30; Z=460;
for x in 0 1000 2000 3000 4000 5000; do
[ $x == 5000 ] && X=5217 || X=$((x+1000))
#[ $x == 4000 ] && X=4235 || X=$((y+1000))
qsubfile=$datadir/EM_p2l_`printf %05d ${x}`-`printf %05d ${X}`.sh
echo '#!/bin/bash' > $qsubfile
echo "#SBATCH --nodes=1" >> $qsubfile
echo "#SBATCH --ntasks-per-node=5" >> $qsubfile
echo "#SBATCH --time=100:00:00" >> $qsubfile
echo "#SBATCH --mem=256000" >> $qsubfile
echo "#SBATCH --job-name=EM_ws" >> $qsubfile
for y in 0 1000 2000 3000 4000; do
[ $y == 4000 ] && Y=4460 || Y=$((y+1000))
#[ $y == 4000 ] && Y=4111 || Y=$((y+1000))
echo "python $scriptdir/mesh/prob2labels.py $datadir $dataset \
-n 5 -o 220 235 491 -s 460 4460 5217 \
-x $x -X $X -y $y -Y $Y -z $z -Z $Z > \
$datadir/output_`printf %05d ${x}`-`printf %05d ${X}`_`printf %05d ${y}`-`printf %05d ${Y}` &" >> $qsubfile
done
echo "wait" >> $qsubfile
sbatch -p compute $qsubfile
done


# from scratch_20160713

scriptdir="$HOME/workspace/EM"
DATA="$HOME/oxdata"
datadir="$DATA/P01/EM/M3/M3_S1_GNU" && cd $datadir
dataset='m000'
x=1000; X=1500; y=1000; Y=1500; z=30; Z=460;
datastem=${dataset}_`printf %05d ${x}`-`printf %05d ${X}`_`printf %05d ${y}`-`printf %05d ${Y}`_`printf %05d ${z}`-`printf %05d ${Z}`

# python $scriptdir/mesh/prob2labels.py $datadir $dataset \
# --SEfile '_seg' --MAfile '_' --MMfile '_' --UAfile '_' --PAfile '_' \
# -n 5 -o 250 235 491 -s 460 4460 5217 \
# -x $x -X $X -y $y -Y $Y -z $z -Z $Z

# generates all
python $scriptdir/mesh/prob2labels.py $datadir $dataset \
--MAknossosfile '_sMAkn' --UAknossosfile '_sUAkn' \
-n 5 -o 250 235 491 -s 460 4460 5217 \
-x $x -X $X -y $y -Y $Y -z $z -Z $Z -f -w #-d

# generates MA
python $scriptdir/mesh/prob2labels.py $datadir $dataset \
--MAknossosfile '_sMAkn' --UAknossosfile '_sUAkn' --MMfile '_' --UAfile '_' --PAfile '_' \
-n 5 -o 250 235 491 -s 460 4460 5217 \
-x $x -X $X -y $y -Y $Y -z $z -Z $Z

# generates MM
python $scriptdir/mesh/prob2labels.py $datadir $dataset \
--MAknossosfile '_sMAkn' --UAknossosfile '_sUAkn' --MAfile '_MA_sMAkn_sUAkn_ws' --UAfile '_' --PAfile '_' \
-n 5 -o 250 235 491 -s 460 4460 5217 \
-x $x -X $X -y $y -Y $Y -z $z -Z $Z

# generates UA and PA
python $scriptdir/mesh/prob2labels.py $datadir $dataset \
--MAknossosfile '_sMAkn' --UAknossosfile '_sUAkn' --MAfile '_MA_sMAkn_sUAkn_ws' --MMfile '_MM_ws' \
-n 5 -o 250 235 491 -s 460 4460 5217 \
-x $x -X $X -y $y -Y $Y -z $z -Z $Z
