fname=B-NT-S10-2f_ROI_00_00000-00520_00000-00520_00000-00184_probs_eed2
fname=B-NT-S10-2f_ROI_00_00980-01520_03480-04020_00000-00184_probs_eed2
rsync -Pazv $host:$datadir_rem/blocks_0500/$fname.h5 $datadir

python $scriptdir/wmem/stack2stack.py \
$datadir/$fname.h5/probs_eed \
$datadir/${fname}_probs_eed.nii.gz






cd blocks_0500
for f in `ls *_probs_eed2.h5`; do
mv $f ${f/'_probs_eed2.h5'/'_probs0247_eed2.h5'}
done
cd -





h5_in='probs'; dset_in='volume/predictions'; h5_out='probs_eed'; dset_out='probs_eed'; wtime='00:10:00';
h5_in='probs'; dset_in='sum0247'; h5_out='probs_eed'; dset_out='sum0247_eed'; wtime='00:10:00';
h5_in='probs'; dset_in='sum16'; h5_out='probs_eed'; dset_out='sum16_eed'; wtime='00:10:00';

export bs='0500' && source datastems_blocks_${bs}.sh
source find_missing_h5.sh ${datadir}/blocks_${bs}/ "_$h5_out" "/$dset_out"
export nstems=${#datastems[@]} && echo $nstems
export tasks=16
export njobs=$(( ($nstems + tasks-1) / $tasks))
export template='array' additions='' CONDA_ENV=''
export nodes=1 memcpu=50000 q='h'
export jobname="$h5_out.$dset_out"
export cmd="$datadir/bin/EM_eed_simple \
'$datadir/blocks_${bs}' \
'datastem_${h5_in}' '/$dset_in' 'datastem_${h5_out}' '/$dset_out' \
'0' '1' '1' '1' \
> $datadir/blocks_${bs}/datastem_$jobname.log"
source $scriptdir/pipelines/template_job_$template.sh
sbatch -p devel ...
source datastems_blocks_${bs}.sh


h5_in='probs'; dset_in='sum0247'; h5_out='probs_eed'; dset_out='sum0247_eed';
export bs='0500' && source datastems_blocks_${bs}.sh
# source find_missing_h5.sh "${datadir}/blocks_${bs}/" "_$h5_in" "/$dset_in"
source find_missing_h5.sh "${datadir}/blocks_${bs}/" "_$h5_out" "/$dset_out"
export nstems=${#datastems[@]} && echo $nstems


        export cmd="python $scriptdir/wmem/mergeblocks.py \
        "${infiles[@]}" $datadir/${dataset}${pf}.h5/$dset \
        -b $zo $yo $xo -p $zs $ys $xs -q $zm $ym $xm -s $zmax $ymax $xmax"







export bs='0500'
datastem='B-NT-S10-2f_ROI_00_00480-01020_00480-01020_00000-00184'

python $scriptdir/wmem/combine_vols.py \
$datadir/blocks_${bs}/${datastem}_probs.h5/volume/predictions \
$datadir/blocks_${bs}/${datastem}_probs.h5/sum0247 \
-i 0 2 4 7
python $scriptdir/wmem/combine_vols.py \
$datadir/blocks_${bs}/${datastem}_probs.h5/volume/predictions \
$datadir/blocks_${bs}/${datastem}_probs.h5/sum16 \
-i 1 6

# % EED

cmd=
mpf='maskDS'; pf=; dset='stack'; arg='-l 0 -u 10000000'; blocksize=20;  # TODO: dilate?
zs=184;
for z in `seq 0 $blocksize $zs`; do
    Z=$((z+blocksize)) && Z=$(( Z < zs ? Z : zs ))
    export cmd+="python $scriptdir/wmem/prob2mask.py \
        $datadir/${dataset}${pf}.h5/$dset \
        $datadir/${dataset}_masks.h5/$mpf \
        -g $arg -D $z $Z 1 0 0 1 0 0 1 ; "
done
$cmd


python $scriptdir/wmem/prob2mask.py \
$datadir/${dataset}${pf}.h5/$dset \
$datadir/${dataset}_masks.h5/$mpf \
-g $arg -D $z $Z 1 0 0 1 0 0 1

read -r -d '' VAR << EOM
for i in  0 1 2 3; do
echo $i
done
EOM
