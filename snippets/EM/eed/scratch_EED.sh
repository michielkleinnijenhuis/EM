



mkdir -p $datadir/bin && cd $datadir/bin
mcc -v -R -nojvm -R -singleCompThread -f ./mbuildopts.sh -m $scriptdir/wmem/EM_eed.m -a $HOME/oxscripts/matlab/toolboxes/coherencefilter_version5b
cd $datadir

# on blocks (from fullstack file)
export template='array' additions=''
export njobs=6 nodes=1 tasks=16 memcpu=50000 wtime='00:10:00' q='d'
export jobname='eed00'
export cmd="$datadir/bin/EM_eed \
'$datadir/blocks' 'datastem_probs' '/volume/predictions' '/probs_eed' \
'1' '1' '1' '1' \
'184' '550' '550' '2' \
'0' '50' '50' '3' '1' > $datadir/blocks/datastem_probs_$jobname.log"
source $scriptdir/pipelines/template_job_$template.sh
echo $jids


export cmd="$datadir/bin/EM_eed '$datadir/blocks' 'datastem_probs' '/volume/predictions' '/probs_eed' '1' '551'
'1' '1' '184' '550' '550' '2' '0' '50' '50' '3' '1' > $datadir/EM_eed/datastem.log"
source $scriptdir/pipelines/template_job_$template.sh

export cmd="$datadir/bin/EM_eed '$datadir/blocks' 'datastem_probs' '/volume/predictions' '/probs_eed' '1' '1'
'551' '1' '184' '550' '550' '2' '0' '50' '50' '3' '1' > $datadir/EM_eed/datastem.log"
source $scriptdir/pipelines/template_job_$template.sh

export cmd="$datadir/bin/EM_eed '$datadir/blocks' 'datastem_probs' '/volume/predictions' '/probs_eed' '1' '551'
'551' '1' '184' '550' '550' '2' '0' '50' '50' '3' '1' > $datadir/EM_eed/datastem.log"
source $scriptdir/pipelines/template_job_$template.sh





# on the full stack (try a blocksize of 500 (8GB) for maximum cpu utilization)
source datastems_blocks.sh
rm -rf EM_eed*
export xs=500; export ys=500;
mkdir -p $datadir/EM_eed
i=0; j=0;
for x in `seq 0 $xs $xmax`; do
    [ $x == $(((xmax/xs)*xs)) ] && X=$xmax || X=$((x+xs))
    xcount=$((X-x))
    for y in `seq 0 $ys $ymax`; do
        [ $y == $(((ymax/ys)*ys)) ] && Y=$ymax || Y=$((y+ys))
        ycount=$((Y-y))
        echo "$datadir/bin/EM_eed '$datadir' '${dataset}_probs' '/volume/predictions' '/probs_eed' '1' '$((y+1))' '$((x+1))' '1' '$zs' '$ycount' '$xcount' '2' '$zm' '$ym' '$xm' '50' '1' > $datadir/EM_eed/EM.eed.$j.$i.log &" >> $datadir/EM_eed/EM_eed_$j.sh
        [ $i == 15 ] && echo "wait" >> $datadir/EM_eed/EM_eed_$j.sh && j=$((j+1)) && i=0 || i=$((i+1))
    done
done

njobs=`ls EM_eed/EM_eed_*.sh | wc -l`
export nodes=1 tasks=16 memcpu=60000 wtime="05:00:00" jobname="EED"
scriptfile=$datadir/EM_eed_data.sh
echo "export datastems=( "${datastems[@]}" )" >> $scriptfile
echo '#!/bin/bash' > $scriptfile
echo "#SBATCH --nodes=$nodes" >> $scriptfile
echo "#SBATCH --ntasks-per-node=$tasks" >> $scriptfile
echo "#SBATCH --mem-per-cpu=$memcpu" >> $scriptfile
echo "#SBATCH --time=$wtime" >> $scriptfile
echo "#SBATCH --job-name=$jobname" >> $scriptfile
echo ". $datadir/EM_eed/EM_eed_\${SLURM_ARRAY_TASK_ID}.sh" >> $scriptfile
chmod +x $scriptfile
# sbatch -p devel --array=0-1 $scriptfile
sbatch --array=0-$((njobs-1)) $scriptfile  # --dependency=afterok:1299742



# LOCAL TEST
# scriptdir="$HOME/workspace/EM"
# datadir='/Users/michielk/oxdata/P01/EM/Myrf_01/SET-B/B-NT-S9-2a'
#
# mkdir -p $datadir/bin && cd $datadir/bin
# mcc -v -R -nojvm -R -singleCompThread -f ./mbuildopts.sh -m $scriptdir/wmem/EM_eed.m -a $HOME/oxscripts/matlab/toolboxes/coherencefilter_version5b
# cd $datadir
#
# blocksize=500
# scriptfile=$datadir/eed.sh
# for x in `seq 51 $blocksize 1050`; do
# for y in `seq 51 $blocksize 1050`; do  # EM_eed.app/Contents/MacOS/EM_eed
# echo "$datadir/bin/run_EM_eed.sh '$datadir/blocks' \
# 'B-NT-S9-2a_00000-01050_00000-01050_00000-00479_probs' '/volume/predictions' '/probs_eed' \
# '1' '$y' '$x' '1' '479' '$blocksize' '$blocksize' '2' '0' '$margin' '$margin' '2' '1' \
# > $datadir/blocks/B-NT-S9-2a_00000-01050_00000-01050_00000-00479_probs_eed_$x-$y.log &" >> $scriptfile
# done
# done
# echo "wait" >> $scriptfile
# chmod +x $scriptfile
# mpiexec -np 4 $scriptfile
#
# $datadir/bin/EM_eed '$datadir/blocks' 'datastem_probs' '/volume/predictions' '/probs_eed' '1' '1' '1' '1' '100' '100' '100' '2' '50' '50' '50' > $datadir/blocks/datastem_probs_eed2.log
# mpiexec -np 6 \
