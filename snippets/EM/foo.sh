source datastems_blocks.sh
datastems=( B-NT-S10-2f_ROI_00_00950-02050_04950-06050_00000-00184 B-NT-S10-2f_ROI_00_01950-03050_06950-08050_00000-00184 B-NT-S10-2f_ROI_00_02950-04050_05950-07050_00000-00184 )
nblocks=`echo "${datastems[@]}" | wc -w`

mkdir -p $datadir/blocks

export template='array' additions='conda' CONDA_ENV='root'
export njobs=1 nodes=1 tasks=3 memcpu=6000 wtime='00:10:00' q='h'
export jobname='split'
export cmd="python $scriptdir/wmem/stack2stack.py \
${basepath}.h5/data $datadir/blocks/datastem.h5/data -p datastem"
source $scriptdir/pipelines/template_job_$template.sh

for i in `seq 0 8`; do
sed -i -e 's/node=9/node=1/g' EM_split_$i.sh
sed -i -e 's/ &//g' EM_split_$i.sh
sed -i -e 's/wait//g' EM_split_$i.sh
sbatch -p devel EM_split_$i.sh
done
