source ~/.bashrc
module load hdf5-parallel/1.8.14_mvapich2_intel
module load mpi4py/1.3.1
module load python/2.7__gcc-4.8
module load matlab/R2015a

### EED
scriptdir="$HOME/workspace/EM"
datadir='/data/ndcn-fmrib-water-brain/ndcn0180/EM/Neuroproof/M3_S1_GNU_NP'
traindir=$datadir/train/orig
dataset='m000_01000-01500_01000-01500_00030-00460'
mkdir -p $datadir/bin && cd $datadir/bin
mcc -v -R -nojvm -R -singleCompThread -f ./mbuildopts.sh -m $scriptdir/snippets/eed/EM_eed.m -a $HOME/oxscripts/matlab/toolboxes/coherencefilter_version5b

qsubfile=$datadir/EM_eed.sh
echo '#!/bin/bash' > $qsubfile
echo "#SBATCH --nodes=1" >> $qsubfile
echo "#SBATCH --ntasks-per-node=2" >> $qsubfile
echo "#SBATCH --time=05:10:00" >> $qsubfile
echo "#SBATCH --mem=100000" >> $qsubfile
echo "#SBATCH --job-name=EM_eed" >> $qsubfile
for layer in 4 5 6; do
[ -f $traindir/${dataset}_probs$((layer-1))_eed2.h5 ] || {
echo "$datadir/bin/EM_eed '$traindir' '${dataset}_probs' '/volume/predictions' '/stack' $layer \
> $traindir/${dataset}_probs.log &" >> $qsubfile ; }
done
echo "wait" >> $qsubfile
sbatch -p compute $qsubfile



### local matlab
addpath(genpath('~/oxscripts/matlab/toolboxes/coherencefilter_version5b'))
datadir='/Users/michielk/M3_S1_GNU_NP/train/orig'
dataset='m000_01000-01500_01000-01500_00030-00460'
invol='m000_01000-01500_01000-01500_00030-00460_probs'
infield='/volume/predictions'
stackinfo = h5info([datadir filesep invol '.h5'], infield);
layer=4
data = h5read([datadir filesep invol '.h5'], infield, [layer,1,1,1], [1,Inf,Inf,Inf]);
data = squeeze(data(1,:,:,:));
fname = [datadir filesep invol num2str(layer-1) '_eed2.h5'];
u = CoherenceFilter(data, struct('T', 50, 'dt', 1, 'rho', 1, 'Scheme', 'R', 'eigenmode', 2, 'verbose', 'full'));
h5create(fname, outfield, size(u), 'Deflate', 4, 'Chunksize', stackinfo.ChunkSize(2:4));
h5write(fname, outfield, u);
