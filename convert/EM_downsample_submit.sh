#!/bin/bash

#PBS -l nodes=1:ppn=1
#PBS -l walltime=01:00:00
#PBS -N em_reg_ds
#PBS -V
cd $PBS_O_WORKDIR

python SCRIPTDIR/EM_downsample.py -i INPUTDIR -o OUTPUTDIR -d DS_FACTOR -x X_START -X X_END -y Y_START -Y Y_END -z Z_START -Z Z_END
