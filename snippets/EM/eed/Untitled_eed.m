#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=9
#SBATCH --mem-per-cpu=50000
#SBATCH --time=03:10:00
#SBATCH --job-name=EM_eed_probs.sum16

/home/ndcn-fmrib-water-brain/ndcn0180/workspace/EM/bin/EM_eed_simple '/data/ndcn-fmrib-water-brain/ndcn0180/EM/Myrf_01/SET-B/B-NT-S10-2f_ROI_00/blocks_0500' 'B-NT-S10-2f_ROI_00_00000-00520_00480-01020_00000-00184_probs' '/sum16' 'B-NT-S10-2f_ROI_00_00000-00520_00480-01020_00000-00184_probs_eed' '/sum16_eed' '0' '50' '1' '1' &

/home/ndcn-fmrib-water-brain/ndcn0180/workspace/EM/bin/EM_eed_simple '/data/ndcn-fmrib-water-brain/ndcn0180/EM/Myrf_01/SET-B/B-NT-S10-2f_ROI_00/blocks_0500' 'B-NT-S10-2f_ROI_00_00000-00520_00980-01520_00000-00184_probs' '/sum16' 'B-NT-S10-2f_ROI_00_00000-00520_00980-01520_00000-00184_probs_eed' '/sum16_eed' '0' '50' '1' '1' &

/home/ndcn-fmrib-water-brain/ndcn0180/workspace/EM/bin/EM_eed_simple '/data/ndcn-fmrib-water-brain/ndcn0180/EM/Myrf_01/SET-B/B-NT-S10-2f_ROI_00/blocks_0500' 'B-NT-S10-2f_ROI_00_00000-00520_02980-03520_00000-00184_probs' '/sum16' 'B-NT-S10-2f_ROI_00_00000-00520_02980-03520_00000-00184_probs_eed' '/sum16_eed' '0' '50' '1' '1' &

/home/ndcn-fmrib-water-brain/ndcn0180/workspace/EM/bin/EM_eed_simple '/data/ndcn-fmrib-water-brain/ndcn0180/EM/Myrf_01/SET-B/B-NT-S10-2f_ROI_00/blocks_0500' 'B-NT-S10-2f_ROI_00_00000-00520_03480-04020_00000-00184_probs' '/sum16' 'B-NT-S10-2f_ROI_00_00000-00520_03480-04020_00000-00184_probs_eed' '/sum16_eed' '0' '50' '1' '1' &

/home/ndcn-fmrib-water-brain/ndcn0180/workspace/EM/bin/EM_eed_simple '/data/ndcn-fmrib-water-brain/ndcn0180/EM/Myrf_01/SET-B/B-NT-S10-2f_ROI_00/blocks_0500' 'B-NT-S10-2f_ROI_00_00000-00520_03980-04520_00000-00184_probs' '/sum16' 'B-NT-S10-2f_ROI_00_00000-00520_03980-04520_00000-00184_probs_eed' '/sum16_eed' '0' '50' '1' '1' &

/home/ndcn-fmrib-water-brain/ndcn0180/workspace/EM/bin/EM_eed_simple '/data/ndcn-fmrib-water-brain/ndcn0180/EM/Myrf_01/SET-B/B-NT-S10-2f_ROI_00/blocks_0500' 'B-NT-S10-2f_ROI_00_00000-00520_04980-05520_00000-00184_probs' '/sum16' 'B-NT-S10-2f_ROI_00_00000-00520_04980-05520_00000-00184_probs_eed' '/sum16_eed' '0' '50' '1' '1' &

/home/ndcn-fmrib-water-brain/ndcn0180/workspace/EM/bin/EM_eed_simple '/data/ndcn-fmrib-water-brain/ndcn0180/EM/Myrf_01/SET-B/B-NT-S10-2f_ROI_00/blocks_0500' 'B-NT-S10-2f_ROI_00_00000-00520_05480-06020_00000-00184_probs' '/sum16' 'B-NT-S10-2f_ROI_00_00000-00520_05480-06020_00000-00184_probs_eed' '/sum16_eed' '0' '50' '1' '1' &

/home/ndcn-fmrib-water-brain/ndcn0180/workspace/EM/bin/EM_eed_simple '/data/ndcn-fmrib-water-brain/ndcn0180/EM/Myrf_01/SET-B/B-NT-S10-2f_ROI_00/blocks_0500' 'B-NT-S10-2f_ROI_00_00000-00520_06480-07020_00000-00184_probs' '/sum16' 'B-NT-S10-2f_ROI_00_00000-00520_06480-07020_00000-00184_probs_eed' '/sum16_eed' '0' '50' '1' '1' &

/home/ndcn-fmrib-water-brain/ndcn0180/workspace/EM/bin/EM_eed_simple '/data/ndcn-fmrib-water-brain/ndcn0180/EM/Myrf_01/SET-B/B-NT-S10-2f_ROI_00/blocks_0500' 'B-NT-S10-2f_ROI_00_00980-01520_00480-01020_00000-00184_probs' '/sum16' 'B-NT-S10-2f_ROI_00_00980-01520_00480-01020_00000-00184_probs_eed' '/sum16_eed' '0' '50' '1' '1' &


wait


#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --mem-per-cpu=50000
#SBATCH --time=01:10:00
#SBATCH --job-name=EM_eed_probs.sum0247

addpath('/home/ndcn-fmrib-water-brain/ndcn0180/workspace/EM/wmem');
datadir = '/data/ndcn-fmrib-water-brain/ndcn0180/EM/Myrf_01/SET-B/B-NT-S10-2f_ROI_00/blocks_0500';
dataset = 'B-NT-S10-2f_ROI_00_00980-01520_00480-01020_00000-00184';

addpath('/home/ndcn-fmrib-water-brain/ndcn0180/workspace/EM/wmem');
datadir = '/data/ndcn-fmrib-water-brain/ndcn0180/EM/Myrf_01/SET-B/B-NT-S10-2f_ROI_00/blocks_0500';
datasets = {
    'B-NT-S10-2f_ROI_00_00000-00520_00480-01020_00000-00184',
    'B-NT-S10-2f_ROI_00_00000-00520_00980-01520_00000-00184',
    'B-NT-S10-2f_ROI_00_00000-00520_02980-03520_00000-00184',
    'B-NT-S10-2f_ROI_00_00000-00520_03480-04020_00000-00184',
    'B-NT-S10-2f_ROI_00_00000-00520_03980-04520_00000-00184',
    'B-NT-S10-2f_ROI_00_00000-00520_04980-05520_00000-00184',
    'B-NT-S10-2f_ROI_00_00000-00520_05480-06020_00000-00184',
    'B-NT-S10-2f_ROI_00_00000-00520_06480-07020_00000-00184'}

for dataset = 1:length(datasets);
    dataset = datasets{1};
    dpf = '_probs';
    invol = [dataset, dpf];
    outvol = [dataset, dpf, '_eed'];
    ds_in = '/sum16';
    ds_out = '/sum16_eed';
    EM_eed_simple(datadir, invol, ds_in, outvol, ds_out, 0, 50, 1, 1);
end

matlab -nodisplay -nojvm -nosplash < $datadir/sum16_7.m &
matlab -nodisplay -nojvm -nosplash < $datadir/sum16_6.m &
matlab -nodisplay -nojvm -nosplash < $datadir/sum16_5.m &
matlab -nodisplay -nojvm -nosplash < $datadir/sum16_4.m &
matlab -nodisplay -nojvm -nosplash < $datadir/sum16_3.m &
matlab -nodisplay -nojvm -nosplash < $datadir/sum16_1.m &

addpath('/home/ndcn-fmrib-water-brain/ndcn0180/workspace/EM/wmem');
datadir = '/data/ndcn-fmrib-water-brain/ndcn0180/EM/Myrf_01/SET-B/B-NT-S10-2f_ROI_00/blocks_0500';
datasets = {
    'B-NT-S10-2f_ROI_00_06480-07020_01480-02020_00000-00184',
    'B-NT-S10-2f_ROI_00_06480-07020_01980-02520_00000-00184',
    'B-NT-S10-2f_ROI_00_06480-07020_02480-03020_00000-00184',
    'B-NT-S10-2f_ROI_00_06480-07020_02980-03520_00000-00184',
    'B-NT-S10-2f_ROI_00_06480-07020_03480-04020_00000-00184',
    'B-NT-S10-2f_ROI_00_06480-07020_03980-04520_00000-00184',
    'B-NT-S10-2f_ROI_00_06480-07020_04480-05020_00000-00184',
    'B-NT-S10-2f_ROI_00_06480-07020_04980-05520_00000-00184',
    'B-NT-S10-2f_ROI_00_06480-07020_05480-06020_00000-00184',
    'B-NT-S10-2f_ROI_00_06480-07020_05980-06520_00000-00184',
    'B-NT-S10-2f_ROI_00_06480-07020_06480-07020_00000-00184',
    'B-NT-S10-2f_ROI_00_06480-07020_06980-07520_00000-00184',
    'B-NT-S10-2f_ROI_00_06480-07020_07480-08020_00000-00184',
    'B-NT-S10-2f_ROI_00_06980-07520_00480-01020_00000-00184'};
for dataset = 1:length(datasets);
    dataset = datasets{1};
    dpf = '_probs';
    invol = [dataset, dpf];
    outvol = [dataset, dpf, '_eed'];
    ds_in = '/sum0247';
    ds_out = '/sum0247_eed';
    EM_eed_simple(datadir, invol, ds_in, outvol, ds_out, 0, 50, 1, 1);
end

% matlab -nodisplay -nojvm < $datadir/sum0247.m &
for i in `seq 0 9`; do
matlab -nodisplay -nojvm -nosplash < $datadir/sum0247_$i.m &
done

module load hdf5-parallel/1.8.17_mvapich2_gcc
module load matlab/R2015a
eed_tgtdir="$scriptdir/bin"
eed_tbxdir="$HOME/oxscripts/matlab/toolboxes/coherencefilter_version5b"
eed_script="$scriptdir/wmem/EM_eed_simple.m"
[ ! -f $eed_tgtdir/EM_eed_simple ] && deployed_eed $eed_tgtdir $eed_tbxdir $eed_script
