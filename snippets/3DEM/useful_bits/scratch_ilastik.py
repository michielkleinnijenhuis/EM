source ~/.bashrc
scriptdir="$HOME/workspace/EM"
datadir="$DATA/EM/M3/M3_S1_GNU"
pixprob_trainingset="pixprob_training"
dataset='m000'
# dataset='m000_cutout01'

qsubfile=$datadir/EM_ac2s2_3nodes.sh
echo '#!/bin/bash' > $qsubfile
echo "#SBATCH --nodes=3" >> $qsubfile
echo "#SBATCH --ntasks-per-node=16" >> $qsubfile
echo "#SBATCH --time=00:10:00" >> $qsubfile
echo "#SBATCH --job-name=EM_lc" >> $qsubfile
echo "export PATH=/home/ndcn-fmrib-water-brain/ndcn0180/miniconda/bin:\$PATH" >> $qsubfile
echo "source activate ilastik-devel" >> $qsubfile
echo "CONDA_ROOT=\`conda info --root\`" >> $qsubfile
echo "\${CONDA_ROOT}/envs/ilastik-devel/run_ilastik.sh --headless \
--preconvert_stacks \
--project=$datadir/${pixprob_trainingset}_arcus.ilp \
--output_axis_order=xyzc \
--output_format=hdf5 \
--output_filename_format=$datadir/${dataset}_probs_3nodes.h5 \
--output_internal_path=/volume/predictions \
$datadir/${dataset}.h5/stack" >> $qsubfile
sbatch -p devel $qsubfile
