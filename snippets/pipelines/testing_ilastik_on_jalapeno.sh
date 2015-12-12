### local 2 jalapeno ###
DATA="$HOME/oxdata"
scriptdir="$HOME/workspace/EM"
datadir="$DATA/P01/EM/M3/M3_S1_GNU"
dataset='m000'
# dataset='m000_cutout01'
pixprob_trainingset="pixprob_training"

scp -r ${datadir}/pipeline_test/${pixprob_trainingset}.ilp jalapeno.fmrib.ox.ac.uk:/vols/Data/km/michielk/P01/EM/M3/M3_S1_GNU/
scp -r ${datadir}/pipeline_test/${pixprob_trainingset}.h5 jalapeno.fmrib.ox.ac.uk:/vols/Data/km/michielk/P01/EM/M3/M3_S1_GNU/
scp -r ${datadir}/${dataset}.h5 jalapeno.fmrib.ox.ac.uk:/vols/Data/km/michielk/P01/EM/M3/M3_S1_GNU/


### jalapeno ###
DATA="/vols/Data/km/michielk"
scriptdir="$HOME/workspace/EM"
datadir="$DATA/P01/EM/M3/M3_S1_GNU"
dataset='m000'
dataset='m000_cutout01'
pixprob_trainingset="pixprob_training"


source activate ilastik-devel
CONDA_ROOT=`conda info --root`
LAZYFLOW_THREADS=4 LAZYFLOW_TOTAL_RAM_MB=8000 ${CONDA_ROOT}/envs/ilastik-devel/run_ilastik.sh --headless \
--preconvert_stacks \
--project="$datadir/${pixprob_trainingset}_jalapeno.ilp" \
--output_axis_order=xyzc \
--output_format=hdf5 \
--output_filename_format="${datadir}/${dataset}_probs.h5" \
--output_internal_path=/volume/predictions \
"${datadir}/${dataset}.h5/stack"

export PATH=/vols/Data/km/michielk/workspace/miniconda/bin:$PATH
source activate neuroproof-devel
neuroproof_graph_learn -h > /vols/Data/km/michielk/workspace/NeuroProof/examples/training_sample2/helpmessage_$1.txt
source activate neuroproof-devel
neuroproof_graph_learn -h > /vols/Data/km/michielk/workspace/NeuroProof/examples/training_sample2/helpmessage_$1.txt


### local test mem
scriptdir="$HOME/workspace/EM"
DATA="$HOME/oxdata"
datadir="$DATA/P01/EM/M3/M3_S1_GNU/pipeline_test"
# dataset='m000'
dataset="pixprob_training"
pixprob_trainingset="pixprob_training"

source activate ilastik-devel
CONDA_ROOT=`conda info --root`
LAZYFLOW_THREADS=4 LAZYFLOW_TOTAL_RAM_MB=4000 ${CONDA_ROOT}/envs/ilastik-devel/run_ilastik.sh --headless \
--preconvert_stacks \
--project="$datadir/${pixprob_trainingset}_2.ilp" \
--output_axis_order=xyzc \
--output_format=hdf5 \
--output_filename_format="$datadir/${dataset}_probs_test.h5" \
--output_internal_path=/volume/predictions \
"$datadir/${dataset}.h5/stack"
