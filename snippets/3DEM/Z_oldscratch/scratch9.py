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
