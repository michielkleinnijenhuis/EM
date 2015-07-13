#!/bin/bash

#PBS -l nodes=1:ppn=1
#PBS -l walltime=01:00:00
#PBS -N em_reg_ds
#PBS -V
cd $PBS_O_WORKDIR

python SCRIPTDIR/EM_downsample.py -i INPUTDIR/reg -o OUTPUTDIR/reg_ds -d DS_FACTOR -z Z_START -Z Z_END
