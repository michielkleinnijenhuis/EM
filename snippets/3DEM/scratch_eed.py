z=0; Z=100; x=0; X=1000; y=0; Y=1000; layer=1;
mcc -v -R -nojvm -R -singleCompThread -f ./mbuildopts.sh -m $scriptdir/snippets/eed/EM_eed.m
qsubfile=$datadir/EM_eed_submit.sh
echo '#!/bin/bash' > $qsubfile
echo "#SBATCH --nodes=1" >> $qsubfile
echo "#SBATCH --time=00:10:00" >> $qsubfile
echo "#SBATCH --job-name=EM_eed" >> $qsubfile
echo "$datadir/EM_eed '$datadir' '${dataset}_${x}-${X}_${y}-${Y}_${z}-${Z}_probs' '/volume/predictions' 'stack' 1 &" >> $qsubfile
echo "wait" >> $qsubfile
sbatch -p devel $qsubfile


./EM_eed $datadir ${dataset}_${x}-${X}_${y}-${Y}_${z}-${Z}_probs '/volume/predictions' 'stack' 1
$datadir/run_EM_eed.sh "$datadir" 'm000_0-1000_0-1000_0-100_probs' '/volume/predictions' 'stack' 1


datadir = '/data/ndcn-fmrib-water-brain/ndcn0180/EM/M3/M3_S1_GNU';
invol = 'm000_0-1000_0-1000_0-100_probs';
invol = 'm000_0-1000_0-1000_0-100';
infield = '/volume/predictions';
infield = '/stack';
outfield = '/stack';
layers = [1,2,3];
layers = 0;
addpath(genpath('~/oxscripts/matlab/toolboxes/coherencefilter_version5b'));

z=0; Z=100;
for x in 0 1000 2000 3000 4000; do
X=$((x+1000))
for y in 0 1000 2000 3000 4000; do
Y=$((y+1000))
qsubfile=$datadir/EM_eed_${x}-${X}_${y}-${Y}_${z}-${Z}.sh
echo '#!/bin/bash' > $qsubfile
echo "#SBATCH --nodes=1" >> $qsubfile
echo "#SBATCH --ntasks-per-node=1" >> $qsubfile
echo "#SBATCH --time=12:00:00" >> $qsubfile
echo "#SBATCH --job-name=EM_eed" >> $qsubfile
echo "matlab -nojvm -singleCompThread -r \"addpath('$scriptdir/snippets/eed'); \
EM_eed('$datadir', '${dataset}_${x}-${X}_${y}-${Y}_${z}-${Z}_probs', '/volume/predictions', 'stack', 1); \
EM_eed('$datadir', '${dataset}_${x}-${X}_${y}-${Y}_${z}-${Z}_probs', '/volume/predictions', 'stack', 2); \
EM_eed('$datadir', '${dataset}_${x}-${X}_${y}-${Y}_${z}-${Z}_probs', '/volume/predictions', 'stack', 3); \
exit\""  >> $qsubfile
sbatch -p compute $qsubfile
done
done

z=0; Z=460;
x=5000;
X=$((x+1000))
for layer in 1 2 3; do
qsubfile=$datadir/EM_eed_submit_${x}-${X}_${layer}.sh
echo '#!/bin/bash' > $qsubfile
echo "#SBATCH --nodes=1" >> $qsubfile
echo "#SBATCH --ntasks-per-node=5" >> $qsubfile
echo "#SBATCH --time=05:00:00" >> $qsubfile
echo "#SBATCH --mem=250000" >> $qsubfile
echo "#SBATCH --job-name=EM_eed" >> $qsubfile
for y in 0 1000 2000 3000 4000; do
Y=$((y+1000))
[ -f $datadir/${dataset}_`printf %05d ${x}`-`printf %05d ${X}`_`printf %05d ${y}`-`printf %05d ${Y}`_`printf %05d ${z}`-`printf %05d ${Z}`_probs$((layer-1))_eed2.h5 ] || {
echo "$datadir/bin/EM_eed '$datadir' \
'${dataset}_`printf %05d ${x}`-`printf %05d ${X}`_`printf %05d ${y}`-`printf %05d ${Y}`_`printf %05d ${z}`-`printf %05d ${Z}`_probs' \
'/volume/predictions' '/stack' $layer \
> $datadir/${dataset}_`printf %05d ${x}`-`printf %05d ${X}`_`printf %05d ${y}`-`printf %05d ${Y}`_`printf %05d ${z}`-`printf %05d ${Z}`_probs.log &" >> $qsubfile ; }
done
echo "wait" >> $qsubfile
sbatch -p compute $qsubfile
done



# from scratch6



x=0; X=3000; y=0; Y=3000; z=0; Z=430; layer=1;
qsubfile=$datadir/EM_eed_submit_${x}-${X}_${y}-${Y}_${layer}.sh
echo '#!/bin/bash' > $qsubfile
echo "#SBATCH --nodes=1" >> $qsubfile
echo "#SBATCH --ntasks-per-node=1" >> $qsubfile
echo "#SBATCH --time=100:00:00" >> $qsubfile
echo "#SBATCH --mem=256000" >> $qsubfile
echo "#SBATCH --job-name=EM_eed" >> $qsubfile
echo "$datadir/bin/EM_eed '$datadir' \
'${dataset}_`printf %05d ${x}`-`printf %05d ${X}`_`printf %05d ${y}`-`printf %05d ${Y}`_`printf %05d ${z}`-`printf %05d ${Z}`_probs' \
'/volume/predictions' '/stack' $layer \
> $datadir/${dataset}_`printf %05d ${x}`-`printf %05d ${X}`_`printf %05d ${y}`-`printf %05d ${Y}`_`printf %05d ${z}`-`printf %05d ${Z}`_probs.log &" >> $qsubfile
echo "wait" >> $qsubfile
sbatch -p compute $qsubfile
