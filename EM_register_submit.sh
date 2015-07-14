#!/bin/bash

#PBS -l nodes=1:ppn=1
#PBS -l walltime=10:00:00
#PBS -N em_reg
#PBS -V
cd $PBS_O_WORKDIR

/system/software/linux-x86_64/fiji/20140602/ImageJ-linux64 --headless DATADIR/EM_register.py
