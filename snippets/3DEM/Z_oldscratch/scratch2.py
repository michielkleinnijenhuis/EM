python $scriptdir/convert/EM_stack2stack.py \
"${datadir}/${dataset}_smooth.h5" \
"${datadir}/${dataset}_smooth.nii.gz" -i 'zyx' -l 'xyz'

./ilastik-1.1.7-OSX.app/Contents/ilastik-release/bin/python -c "import ilastik; print ilastik.__version__"
${CONDA_ROOT}/envs/ilastik-devel/bin/python -c "import ilastik; print ilastik.__version__"


source ~/.bashrc
scriptdir="$HOME/workspace/EM"
datadir="$DATA/EM/M3/M3_S1_GNU"
pixprob_trainingset="pixprob_training"
dataset='m000'
dataset='m000_cutout01'

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
$datadir/$dataset.h5/stack" >> $qsubfile
sbatch -p devel $qsubfile



wsseeds_myelin + parents



scriptdir="$HOME/workspace/EM"
DATA="$HOME/oxdata"
datadir="$DATA/P01/EM/M3/M3_S1_GNU"
pixprob_trainingset="pixprob_training"
dataset='m000'

mkdir -p $datadir/m000_reg_arcus_ds

mpiexec -n 4 python $scriptdir/convert/EM_downsample.py \
$datadir/m000_reg_arcus $datadir/m000_reg_arcus_ds -d 10 -m

mpiexec -n 4 python $scriptdir/convert/EM_series2stack.py \
$datadir/tifs_ds $datadir/tifs_ds.h5 \
-f 'stack' -m -o -e 0.073 0.073 0.05
