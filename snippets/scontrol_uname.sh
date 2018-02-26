#/bin/bash
for i in $(squeue -u ndcn0180 -h -t PD -o %i)
do
scontrol update jobid=$i partition=compute MinMemoryCPU=60000 TimeLimit=01:30:00
done
