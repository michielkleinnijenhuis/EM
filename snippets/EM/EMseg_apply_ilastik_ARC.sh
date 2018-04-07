# http://guide.bash.academy/
# http://www.arc.ox.ac.uk/content/home
# http://www.arc.ox.ac.uk/content/slurm-job-scheduler
# http://ilastik.org/documentation/pixelclassification/headless.html
#
# ssh -Y ndcn0180@arcus-b.arc.ox.ac.uk
#
# ##=========================================================================###
# ## syncing between local and ARC
# ##=========================================================================###
# remdir="ndcn0180@arcus-b.arc.ox.ac.uk:/data/ndcn-fmrib-water-brain/ndcn0180/EM/M3/M3S1GNUds7"
# localdir="/Users/michielk/oxdata/P01/EM/M3/M3_S1_GNU/M3S1GNU/ds7_arc/test"
# localdir="$HOME/oxdata/P01/EM/M3/M3S1GNUds7"
# rsync -avz $localdir/ $remdir
# rsync -avz $remdir/ $localdir
#
# f="M3S1GNUds7.h5"
# rsync -avz $remdir/${f} $localdir
# rsync -avz $localdir/${f} $remdir


###=========================================================================###
### environment prep
###=========================================================================###
export PATH=/data/ndcn-fmrib-water-brain/ndcn0180/anaconda2/bin:$PATH
export CONDA_PATH="$(conda info --root)"
scriptdir="${HOME}/workspace/EM"
datadir="${DATA}/EM/M3/M3S1GNUds7" &&
mkdir -p $datadir && cd $datadir


###=========================================================================###
### apply Ilastik classifier # (step by step)
###=========================================================================###
# no need to load ARC modules for this

# make sure the paths in the .ilp file are relative!
# you can start the GUI version of ilastik with these:
source activate ilastik-devel
$CONDA_PREFIX/run_ilastik.sh &

# create environment variables
export pixprob_trainingset="pixprob_training"
export datastem="M3S1GNU_00000-01050_00000-01050_00030-00460"
export CONDA_ENV="ilastik-devel"
export nodes=1 tasks=16 memcpu=8000 wtime="00:10:00"
export jobname="ilastik"
export qsubfile=$jobname.sh

# write the submission script
echo '#!/bin/bash' > $qsubfile

echo "#SBATCH --nodes=$nodes" >> $qsubfile
echo "#SBATCH --ntasks-per-node=$tasks" >> $qsubfile
echo "#SBATCH --mem-per-cpu=$memcpu" >> $qsubfile
echo "#SBATCH --time=$wtime" >> $qsubfile
echo "#SBATCH --job-name=EM_$jobname" >> $qsubfile

echo "export CONDA_PATH=$(conda info --root)"
echo "export PATH=$CONDA_PATH:\$PATH" >> $qsubfile
echo "source activate $CONDA_ENV" >> $qsubfile

echo "${CONDA_PATH}/envs/ilastik-devel/run_ilastik.sh --headless \
--preconvert_stacks \
--project=$datadir/${pixprob_trainingset}_arcus.ilp \
--output_axis_order=zyxc \
--output_format=hdf5 \
--output_filename_format=$datadir/${datastem}_probs.h5 \
--output_internal_path=volume/predictions \
$datadir/${datastem}.h5/stack" >> $qsubfile

# submit to the queue
sbatch $qsubfile
# or to the development queue (max: 2 nodes and 10 min)
sbatch -p devel $qsubfile


# monitor jobs
squeue
# or my shorthand to my own jobs
sq
# or fully configurable
sacct --format=JobID,Timelimit,elapsed,ReqMem,MaxRss,AllocCPUs,AveVMSize,MaxVMSize,CPUTime,ExitCode

# check output
less slurm*






###=========================================================================###
### apply Ilastik classifier # (via general submission scripts)
###=========================================================================###
# no need to load modules for this
pixprob_trainingset="pixprob_training"

export template='single' additions='conda' CONDA_ENV="ilastik-devel"
export njobs=1 nodes=1 tasks=16 memcpu=8000 wtime="01:30:00" q=""

export jobname="ilastik"
export cmd="${CONDA_PATH}/envs/ilastik-devel/run_ilastik.sh --headless \
--preconvert_stacks \
--project=$datadir/${pixprob_trainingset}_arcus.ilp \
--output_axis_order=zyxc \
--output_format=hdf5 \
--output_filename_format=$datadir/datastem_probs.h5 \
--output_internal_path=volume/predictions \
$datadir/datastem.h5/stack"
source $scriptdir/pipelines/template_job_$template.sh
