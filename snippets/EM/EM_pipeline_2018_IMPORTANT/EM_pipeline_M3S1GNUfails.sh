# rm blocks_0500/M3S1GNU_03480-04020_05480-06020_00000-00430_probs_eed.h5
# rm blocks_0500/M3S1GNU_05480-06020_04480-05020_00000-00430_probs_eed.h5
# rm blocks_0500/M3S1GNU_08480-09020_03480-04020_00000-00430_probs_eed.h5
# scriptfile='EM_jp_eed_probs.sum0247_000.m'
# declare fails='138 208 314'
# failscript=$( generate_job_script_partial $scriptfile _fails "$fails" )
# chmod +x $failscript
# fsl_sub -q short.q -t ./$failscript
# fsl_sub -q short.q -t ./EM_jp_eed_probs.sum16_000_fails.m  # copied from sum0247 after deleting files

# declare fails='69 137 207 313'
# unset infiles && declare -a infiles && get_infiles_datastem_indices "$fails"
# mergeblocks '' '' '0' $ipf $ids $opf $ods "$args"

# # find . -maxdepth 1 -type f -size 1
# scriptfile='EM_jp_eed_probs.sum16_000.m'
# declare fails='70 105 113 116 136 138 207 208 314'
# failscript=$( generate_job_script_partial $scriptfile _fails "$fails" )
# chmod +x $failscript
# fsl_sub -q long.q -t ./EM_jp_eed_probs.sum16_000_fails.m
# # fail2: already exists for matlab
# rm blocks_0500/M3S1GNU_01480-02020_07480-08020_00000-00430_probs_eed.h5
# rm blocks_0500/M3S1GNU_02980-03520_01980-02520_00000-00430_probs_eed.h5
# rm blocks_0500/M3S1GNU_02980-03520_03480-04020_00000-00430_probs_eed.h5
# rm blocks_0500/M3S1GNU_03480-04020_04480-05020_00000-00430_probs_eed.h5
# fsl_sub -q long.q -t ./EM_jp_eed_probs.sum16_000_fails.m
# fsl_sub -q long.q -t ./EM_jp_eed_probs.sum0247_000_fails.m  # copied from sum16 after deleting files

# declare fails='207'
# unset infiles && declare -a infiles && get_infiles_datastem_indices "$fails"
# mergeblocks '' '' '0' $ipf $ids $opf $ods "$args"

# scriptfile='EM_jp_rb_probs1.probMA_000.sh'
# declare fails='16 17 18 19'
# failscript=$( generate_job_script_partial $scriptfile _fails "$fails" )
# chmod +x $failscript
# fsl_sub -q veryshort.q ./$failscript

# scriptfile='EM_jp_eed_probs1.probMA_000.m'
# declare fails='184'
# failscript=$( generate_job_script_partial $scriptfile _fails "$fails" )
# chmod +x $failscript
# fsl_sub -q long.q -t ./$failscript

# declare fails='203'
# unset infiles && declare -a infiles && get_infiles_datastem_indices "$fails"
# mergeblocks '' '' '0' $ipf $ids $opf $ods "$args"

# scriptfile='EM_jp_p2m_dstems_maskMM_000.sh'
# declare fails='3 49 138 208 314'
# failscript=$( generate_job_script_partial $scriptfile _fails "$fails" )
# chmod +x $failscript
# fsl_sub -q veryshort.q -t ./$failscript

# declare fails='2'
# unset infiles && declare -a infiles && get_infiles_datastem_indices "$fails"
# mergeblocks '' '' '0' $ipf $ids $opf $ods ''

# scriptfile='EM_jp_p2m_maskMM_000.sh'
# declare fails='25 26'
# failscript=$( generate_job_script_partial $scriptfile _fails "$fails" )
# chmod +x $failscript
# fsl_sub -q veryshort.q -t ./$failscript

# scriptfile='EM_jp_p2m_dstems_maskICS_000.sh'
# declare fails='105 207'
# failscript=$( generate_job_script_partial $scriptfile _fails "$fails" )
# chmod +x $failscript
# fsl_sub -q veryshort.q -t ./$failscript

# scriptfile='EM_jp_p2m_dstems_maskMA_000.sh'
# declare fails='205'
# failscript=$( generate_job_script_partial $scriptfile _fails "$fails" )
# chmod +x $failscript
# fsl_sub -q veryshort.q -t ./$failscript









# FIXME: infiles not updated properly
get_datastem_index B-NT-S10-2f_ROI_01_03980-04520_00480-01020_00000-00184  # 137
get_datastem_index B-NT-S10-2f_ROI_01_04980-05520_00000-00520_00000-00184  # 170
declare fails=`seq 137 170`
unset infiles && declare -a infiles && get_infiles_datastem_indices $fails
declare ipf='_probs_eed_sum0247' ids='sum0247_eed' opf='_probs_eed_sum0247' ods='sum0247_eed' args='-d float16'
mergeblocks 'h' '' '0' $ipf $ids $opf $ods $args
unset infiles && declare -a infiles && get_infiles_datastem_indices $fails
declare ipf='_probs_eed_sum16' ids='sum16_eed' opf='_probs_eed_sum16' ods='sum16_eed' args='-d float16'
mergeblocks 'h' '' '0' $ipf $ids $opf $ods $args

get_datastem_index B-NT-S10-2f_ROI_02_05480-06020_00980-01520_00000-00184  # 189
get_datastem_index B-NT-S10-2f_ROI_02_06480-07020_03480-04020_00000-00184  # 228
declare fails=`seq 189 228`
unset infiles && declare -a infiles && get_infiles_datastem_indices $fails
declare ipf='_probs_eed_sum0247' ids='sum0247_eed' opf='_probs_eed_sum0247' ods='sum0247_eed' args='-d float16'
mergeblocks 'h' '' '0' $ipf $ids $opf $ods $args
unset infiles && declare -a infiles && get_infiles_datastem_indices $fails
declare ipf='_probs_eed_sum16' ids='sum16_eed' opf='_probs_eed_sum16' ods='sum16_eed' args='-d float16'
mergeblocks 'h' '' '0' $ipf $ids $opf $ods $args





### BS102fROI00 fails
# declare fails='10 18 30'
# unset infiles && declare -a infiles && get_infiles_datastem_indices "$fails"
# mergeblocks '' '' '0' $ipf $ids $opf $ods "$args"



scriptfile='EM_jp_eed_probs_sums.sum16_000.m'
declare fails='159 225'
failscript=$( generate_job_script_partial $scriptfile _fails "$fails" )
chmod +x $failscript
fsl_sub -q short.q -t ./$failscript









# FIXME: mergeblocks 'h' '_probs_eed' 'sum16_eed' '_probs_eed_sum16_brtest' 'sum16_eed' '' '' '-B 1 7 7 -f np.mean'


# # FIXME: fix missing probs1_eed
# datastems=()
# datastems+=( B-NT-S10-2f_ROI_00_00000-00520_04980-05520_00000-00184 )  # 10
# datastems+=( B-NT-S10-2f_ROI_00_00480-01020_00480-01020_00000-00184 )  # 18
# datastems+=( B-NT-S10-2f_ROI_00_00480-01020_06480-07020_00000-00184 )  # 30
# eed '' '' '_probs1' 'probMA' '_probs1_eed' 'probMA_eed' '03:10:00'  # (quickerfix) OKAY?
# eed '' '' '_probs' 'volume/predictions' '_probs_eed' 'probs_eed' '10:10:00'

# FIXME: also needs full reprocessing...: blocks_0500/B-NT-S10-2f_ROI_00_00480-01020_00480-01020_00000-00184_probs_eed.h5




jid01=$( fsl_sub -q short.q ./EM_jp_mergeblocks_probs_eed.sum0247_eed_000.sh )
fsl_sub -q long.q -j $jid01 ./EM_jp_rb_probs_eed_sum0247.sum0247_eed_000.sh
jid01=$( fsl_sub -q short.q ./EM_jp_mergeblocks_probs_eed.sum16_eed_000.sh )
fsl_sub -q long.q -j $jid01 ./EM_jp_rb_probs_eed_sum16.sum16_eed_000.sh


jid01=$( fsl_sub -q short.q -j 7713883 ./EM_jp_mergeblocks_probs_eed.sum16_eed_000.sh )
jid02=$( fsl_sub -q short.q -j 7714969 ./EM_jp_mergeblocks_probs_eed.sum16_eed_000.sh )
fsl_sub -q long.q -j $jid02 ./EM_jp_rb_probs_eed_sum16.sum16_eed_000.sh


fsl_sub -q long.q -j 7714970 -t ./EM_jp_rb_probs_eed_sum0247.sum0247_eed_000.sh
fsl_sub -q long.q -j 7714971 -t ./EM_jp_rb_probs_eed_sum0247.sum0247_eed_000.sh

jid=$( sbatch EM_sb_mergeblocks_probs1.probMA_000.sh )
sbatch --dependency=after:${jid##* } EM_sb_rb_probs1.probMA_000.sh


# sbatch --dependency=after:7714970 EM_jp_rb_probs_eed_sum0247.sum0247_eed_000.sh
# sbatch --dependency=after:7714971 EM_jp_rb_probs_eed_sum0247.sum0247_eed_000.sh
# 1635899 (ROI01) slab=23
# 1635898 (ROI02) slab=24


declare fails=`seq 137 170`
unset infiles && declare -a infiles && get_infiles_datastem_indices $fails
declare ipf='_probs_eed' ids='sum0247_eed' opf='_probs_eed_sum0247' ods='sum0247_eed' args='-d float16'
mergeblocks 'h' '' '0' $ipf $ids $opf $ods "$args"
jid01=$( fsl_sub -q short.q ./EM_jp_mergeblocks_probs_eed.sum0247_eed_000.sh )
jid02=$( fsl_sub -q short.q -j $jid01 ./EM_jp_rb_probs_eed_sum0247.sum0247_eed_000.sh )
declare fails=`seq 137 170`
unset infiles && declare -a infiles && get_infiles_datastem_indices $fails
declare ipf='_probs_eed' ids='sum16_eed' opf='_probs_eed_sum16' ods='sum16_eed' args='-d float16'
mergeblocks 'h' '' '0' $ipf $ids $opf $ods "$args"
jid03=$( fsl_sub -q short.q -j $jid01 ./EM_jp_mergeblocks_probs_eed.sum16_eed_000.sh )
jid04=$( fsl_sub -q short.q -j $jid03 ./EM_jp_rb_probs_eed_sum16.sum16_eed_000.sh )


declare fails=`seq 189 228`
unset infiles && declare -a infiles && get_infiles_datastem_indices $fails
declare ipf='_probs_eed' ids='sum0247_eed' opf='_probs_eed_sum0247' ods='sum0247_eed' args='-d float16'
mergeblocks 'h' '' '0' $ipf $ids $opf $ods "$args"
jid01=$( fsl_sub -q short.q ./EM_jp_mergeblocks_probs_eed.sum0247_eed_000.sh )
jid02=$( fsl_sub -q short.q -j $jid01 ./EM_jp_rb_probs_eed_sum0247.sum0247_eed_000.sh )
declare fails=`seq 189 228`
unset infiles && declare -a infiles && get_infiles_datastem_indices $fails
declare ipf='_probs_eed' ids='sum16_eed' opf='_probs_eed_sum16' ods='sum16_eed' args='-d float16'
mergeblocks 'h' '' '0' $ipf $ids $opf $ods "$args"
jid03=$( fsl_sub -q short.q -j $jid01 ./EM_jp_mergeblocks_probs_eed.sum16_eed_000.sh )
jid04=$( fsl_sub -q short.q -j $jid03 ./EM_jp_rb_probs_eed_sum16.sum16_eed_000.sh )








# EED jalapeno pipeline
jid01=$( fsl_sub -q short.q -t ./EM_jp_eed$ipf.${ids////-}_000.m )
jid02=$( fsl_sub -q short.q -j $jid01 ./EM_jp_mergeblocks$ipf.${ids////-}_000.sh )
jid03=$( fsl_sub -q short.q -j $jid02 ./EM_jp_rb$ipf.${ids////-}_000.sh )
# EED fails jalapeno pipeline
scriptfile="EM_jp_eed$ipf.${ids////-}_000.m"
declare fails='207'
failscript=$( generate_job_script_partial $scriptfile _fails "$fails" )
chmod +x $failscript
jid01=$( fsl_sub -q short.q -t ./$failscript )
unset infiles && declare -a infiles && get_infiles_datastem_indices $fails
declare ipf='_probs_eed' ids='probMA' opf='_probs_eed_probMA' ods='probMA_eed' args='-d float16'
mergeblocks 'h' '' '0' $ipf $ids $opf $ods "$args"
jid02=$( fsl_sub -q short.q -j $jid01 ./EM_jp_mergeblocks_probs_eed.probMA_000.sh )
declare ipf='_probs_eed_probMA' ids='probMA_eed' opf='_probs_eed_probMA' ods='probMA_eed' \
    brfun='np.mean' brvol='' slab=12 memcpu=60000 wtime='02:00:00' vol_slice=''
blockreduce 'h' $ipf $ids $opf $ods $brfun '' $slab $memcpu $wtime "$vol_slice"
jid03=$( fsl_sub -q short.q -j $jid02 ./EM_jp_rb_probs_eed_probMA.probMA_eed_000.sh )
fsl_sub -q short.q ./EM_jp_rb_probs_eed_probMA.probMA_eed_000.sh



declare ipf='_probs_sum0247' ids='sum0247' opf='_probs_eed_sum0247' ods='sum0247_eed' wtime='03:10:00'
eed 'h' 'a' $ipf $ids $opf $ods $wtime
jid01=$( fsl_sub -q short.q -t ./EM_jp_eed$ipf.${ids////-}_000.m )
declare ipf='_probs_eed_sum0247' ids='sum0247_eed' opf='_probs_eed_sum0247' ods='sum0247_eed' args='-d float16'
mergeblocks '' '' '' $ipf $ids $opf $ods $args
jid02=$( fsl_sub -q short.q -j $jid01 ./EM_jp_mergeblocks$ipf.${ids////-}_000.sh )
declare ipf='_probs_eed_sum0247' ids='sum0247_eed' opf='_probs_eed_sum0247' ods='sum0247_eed' \
    brfun='np.mean' brvol='' slab=12 memcpu=60000 wtime='02:00:00' vol_slice=''
blockreduce '' $ipf $ids $opf $ods $brfun '' $slab $memcpu $wtime $vol_slice
jid03=$( fsl_sub -q short.q -j $jid02 ./EM_jp_rb$ipf.${ids////-}_000.sh )


# masks
# # maskMM jalapeno pipeline
# jid01=$( fsl_sub -q veryshort.q -t ./EM_jp_p2m_dstems_${ods}_000.sh )
# jid02=$( fsl_sub -q veryshort.q -j $jid01 ./EM_jp_mergeblocks$ipf.${ids////-}_000.sh )
# jid03=$( fsl_sub -q veryshort.q -j $jid02 ./EM_jp_p2m_${ods}_000.sh )
# jid04=$( fsl_sub -q veryshort.q -j $jid03 ./EM_jp_rb$ipf.${ids////-}_000.sh )
#
# # maskICS jalapeno pipeline
# jid01=$( fsl_sub -q veryshort.q -t ./EM_jp_p2m_dstems_${ods}_000.sh )
# jid02=$( fsl_sub -q veryshort.q -j $jid01 ./EM_jp_mergeblocks$ipf.${ids////-}_000.sh )
# jid03=$( fsl_sub -q veryshort.q -j $jid02 ./EM_jp_rb$ipf.${ids////-}_000.sh )

declare ipf='_masks_maskDS' ids='maskDS' opf='_masks_maskDS' ods='maskDS' \
    brfun='np.amax' brvol='' slab=20 memcpu=60000 wtime='00:10:00' vol_slice=''
blockreduce 'h' $ipf $ids $opf $ods $brfun '' $slab $memcpu $wtime "$vol_slice"
fsl_sub -q veryshort.q ./EM_jp_rb$ipf.${ids////-}_000.sh

declare ipf='_probs_eed' ids='sum0247_eed' opf='_masks_maskMM' ods='maskMM' arg='-g -l 0.5 -s 2000 -d 1 -S'
prob2mask_datastems 'h' 'a' $ipf $ids $opf $ods "$arg"
jid01=$( fsl_sub -q veryshort.q -t ./EM_jp_p2m_dstems_${ods}_000.sh )
declare ipf='_masks_maskMM' ids='maskMM_steps/raw' opf='_masks_maskMM' ods='maskMM_raw'
mergeblocks 'h' '' '' $ipf $ids $opf $ods ''
jid02=$( fsl_sub -q veryshort.q -j $jid01 ./EM_jp_mergeblocks$ipf.${ids////-}_000.sh )
declare ipf='_masks_maskMM' ids='maskMM_raw' opf='_masks_maskMM' ods='maskMM' \
    slab=12 arg='-g -l 0 -u 0 -s 2000 -d 1'
prob2mask 'h' $ipf $ids $opf $ods $slab $arg
jid03=$( fsl_sub -q veryshort.q -j $jid02 ./EM_jp_p2m_${ods}_000.sh )
declare ipf='_masks_maskMM' ids='maskMM' opf='_masks_maskMM' ods='maskMM' \
    brfun='np.amax' brvol='' slab=27 memcpu=60000 wtime='00:10:00' vol_slice=''
blockreduce 'h' $ipf $ids $opf $ods $brfun '' $slab $memcpu $wtime "$vol_slice"
jid04=$( fsl_sub -q veryshort.q -j $jid03 ./EM_jp_rb$ipf.${ids////-}_000.sh )

declare ipf='_probs_eed' ids='sum16_eed' opf='_masks_maskICS' ods='maskICS' arg='-g -l 0.2'
prob2mask_datastems 'h' 'a' $ipf $ids $opf $ods "$arg"
jid01=$( fsl_sub -q short.q -t ./EM_jp_p2m_dstems_${ods}_000.sh )
declare ipf='_masks_maskICS' ids='maskICS' opf='_masks_maskICS' ods='maskICS'
mergeblocks 'h' '' '' $ipf $ids $opf $ods ''
jid02=$( fsl_sub -q short.q -j $jid01 ./EM_jp_mergeblocks$ipf.${ids////-}_000.sh )
declare ipf='_masks_maskICS' ids='maskICS' opf='_masks_maskICS' ods='maskICS' \
    brfun='np.amax' brvol='' slab=27 memcpu=60000 wtime='00:10:00' vol_slice=''
blockreduce 'h' $ipf $ids $opf $ods $brfun '' $slab $memcpu $wtime "$vol_slice"
jid03=$( fsl_sub -q short.q -j $jid02 ./EM_jp_rb$ipf.${ids////-}_000.sh )

declare ipf='_probs_eed' ids='probMA' opf='_masks_maskMA' ods='maskMA' arg='-g -l 0.2'  # FIXME: change to probMA_eed
prob2mask_datastems 'h' 'a' $ipf $ids $opf $ods "$arg"
jid01=$( fsl_sub -q veryshort.q -t ./EM_jp_p2m_dstems_${ods}_000.sh )
declare ipf='_masks_maskMA' ids='maskMA' opf='_masks_maskMA' ods='maskMA'
mergeblocks 'h' '' '' $ipf $ids $opf $ods ''
jid02=$( fsl_sub -q veryshort.q -j $jid01 ./EM_jp_mergeblocks$ipf.${ids////-}_000.sh )
declare ipf='_masks_maskMA' ids='maskMA' opf='_masks_maskMA' ods='maskMA' \
    brfun='np.amax' brvol='' slab=24 memcpu=60000 wtime='00:10:00' vol_slice=''
blockreduce 'h' $ipf $ids $opf $ods $brfun '' $slab $memcpu $wtime "$vol_slice"
jid03=$( fsl_sub -q veryshort.q -j $jid02 ./EM_jp_rb$ipf.${ids////-}_000.sh )

for dset in 'maskDS' 'maskMM' 'maskICS' 'maskMA'; do
    h5copy -p -i ${dataset}_masks_$dset.h5 -s $dset -o ${dataset}_masks.h5 -d $dset
    h5copy -p -i ${dataset_ds}_masks_$dset.h5 -s $dset -o ${dataset_ds}_masks.h5 -d $dset
done


# connected components
declare ipf='_masks_maskMM' ids='maskMM' opf='_labels_labelMA_core2D' ods='labelMA_core2D' meanint='dummy'
conncomp 'h' '2D' $dataset $ipf $ids $opf $ods $meanint
jid01=$( fsl_sub -q long.q ./EM_jp_conncomp_$dataset$ipf.${ids////-}.2D_000.sh )
declare ipf='_labels_labelMA_core2D' ids='labelMA_core2D' opf='_labels_mapall' ods='dummy' meanint='_probs_eed_probMA.h5/probMA_eed'
conncomp 'h' '2Dfilter' $dataset $ipf $ids $opf $ods $meanint
jid02=$( fsl_sub -q long.q -j $jid01 ./EM_jp_conncomp_$dataset$ipf.${ids////-}.2Dfilter_000.sh )

declare ipf='_masks_maskMM' ids='maskMM' opf='_labels_labelMA_core2D' ods='labelMA_core2D' meanint='dummy'
conncomp 'h' '2D' $dataset_ds $ipf $ids $opf $ods $meanint
jid01=$( fsl_sub -q long.q ./EM_jp_conncomp_$dataset_ds$ipf.${ids////-}.2D_000.sh )
declare ipf='_labels_labelMA_core2D' ids='labelMA_core2D' opf='_labels_mapall' ods='dummy' meanint='_probs_eed_probMA.h5/probMA_eed'
conncomp 'h' '2Dfilter' $dataset_ds $ipf $ids $opf $ods $meanint
jid02=$( fsl_sub -q long.q -j $jid01 ./EM_jp_conncomp_$dataset_ds$ipf.${ids////-}.2Dfilter_000.sh )
declare ipf='_labels_labelMA_core2D' ids='labelMA_core2D' opf='_labels_mapall' ods='dummy' meanint='dummy'
conncomp 'h' '2Dprops' $dataset_ds $ipf $ids $opf $ods $meanint
jid03=$( fsl_sub -q long.q -j $jid02 ./EM_jp_conncomp_$dataset_ds$ipf.${ids////-}.2Dprops_000.sh )


###=========================================================================###
### watershed
###=========================================================================###
xs=2000 ys=2000
xm=50 ym=50 zm=0  # margins
bs='2000'
blockdir=$datadir/blocks_$bs &&
    mkdir -p $blockdir &&
        datastems_blocks

declare ipf='_probs_eed_sum16' ids='sum16_eed' opf='_probs_eed_sum16' ods='sum16_eed'
split_blocks 'h' 'a' $ipf $ids $opf $ods
jid01=$( fsl_sub -q short.q ./EM_jp_splitblocks_bs2000_000.sh )
declare ipf='_probs_eed_sum16' ids='sum16_eed' opf='_ws' ods='l0.99_u1.00_s010' l=0.99 u=1.00 s=010
watershed 'h' '' $ipf $ids $opf $ods $l $u $s
jid02=$( fsl_sub -q bigmem.q -j $jid01 -t ./EM_jp_ws_${ods}_000.sh )

declare ipf='_ws' ids='l0.99_u1.00_s010' opf='_ws' ods='l0.99_u1.00_s010'
mergeblocks 'h' '' '' $ipf $ids $opf $ods '-l -r -F'
jid03=$( fsl_sub -q bigmem.q ./EM_jp_mergeblocks$ipf.${ids////-}_000.sh )
declare ipf='_ws' ids='l0.99_u1.00_s010' opf='_ws' ods='l0.99_u1.00_s010' \
    brfun='np.amax' brvol='' slab=24 memcpu=60000 wtime='00:10:00' vol_slice=''
blockreduce 'h' $ipf $ids $opf $ods $brfun '' $slab $memcpu $wtime "$vol_slice"
jid04=$( fsl_sub -q veryshort.q -j $jid03 ./EM_jp_rb$ipf.${ids////-}_000.sh )

jid03=$( fsl_sub -q bigmem.q ./EM_jp_mergeblocks$ipf.${ids////-}_000.sh )
jid04=$( fsl_sub -q short.q ./EM_jp_rb$ipf.${ids////-}_000.sh )


declare ipf='_ws_noopt' ids='l0.99_u1.00_s010' opf='_ws' ods='l0.99_u1.00_s010'
mergeblocks 'h' '' '' $ipf $ids $opf $ods '-l'
fsl_sub -q short.q ./EM_jp_mergeblocks$ipf.${ids////-}_000.sh




###=========================================================================###
### delete probs_eed
###=========================================================================###
import os
from glob import glob
import h5py
dataset = 'B-NT-S10-2f_ROI_00'
datadir = '/data/ndcn-fmrib-water-brain/ndcn0180/EM/Myrf_01/SET-B/' + dataset
ppath = os.path.join(datadir, 'blocks_0500', dataset + '*_probs_eed.h5')
pfiles = glob(ppath)
for pfile in pfiles:
    try:
        h5file = h5py.File(pfile, 'r+')
        if 'probs_eed' in h5file.keys():
            try:
                del(h5file['probs_eed'])
            except Exception as e:
                print(e)
        h5file.close()
    except Exception as e:
        print(e)


###=========================================================================###
### correct elsize
###=========================================================================###

import os
import numpy as np
from wmem import utils
from glob import glob
import h5py

dataset = 'B-NT-S10-2f_ROI_01'
dataset = 'B-NT-S10-2d_ROI_00'
dataset = 'B-NT-S10-2d_ROI_02'
datadir = '/data/ndcn-fmrib-water-brain/ndcn0180/EM/Myrf_01/SET-B/' + dataset

es = [0.1, 0.007, 0.007]
es_ds = [0.1, 0.049, 0.049]
es_probs = [0.1, 0.007, 0.007, 1]
es_probs_ds = [0.1, 0.049, 0.049, 1]

def write_elsize(h5path_out, es):
    h5_out, ds_out, _, _ = utils.h5_load(h5path_out)
    utils.h5_write_attributes(ds_out, element_size_um=es)
    h5_out.close()

write_elsize(os.path.join(datadir, 'B-NT-S10-2d_ROI_00_03000-03500_03000-03500_00000-00135.h5/data'), es)
write_elsize(os.path.join(datadir, 'B-NT-S10-2d_ROI_02_03000-03500_03000-03500_00000-00240.h5/data'), es)
write_elsize(os.path.join(datadir, 'B-NT-S10-2f_ROI_00_03000-03500_03000-03500_00000-00184.h5/data'), es)
write_elsize(os.path.join(datadir, dataset + '.h5/data'), es)
write_elsize(os.path.join(datadir, dataset + '_masks_maskDS.h5/maskDS'), es)
write_elsize(os.path.join(datadir, dataset + '_probs1.h5/probMA'), es)
write_elsize(os.path.join(datadir, dataset + 'ds7.h5/data'), es_ds)
write_elsize(os.path.join(datadir, dataset + '_probs.h5/volume/predictions'), es_probs)

ppath = os.path.join(datadir, 'blocks_0500', dataset + '*_probs.h5')
pfiles = glob(ppath)
for pfile in pfiles:
    write_elsize(pfile + '/volume/predictions', es_probs)
    write_elsize(pfile + '/sum0247', es)
    write_elsize(pfile + '/sum16', es)

ppath = os.path.join(datadir, 'blocks_0500', dataset + '*_probs1.h5')
pfiles = glob(ppath)
for pfile in pfiles:
    write_elsize(pfile + '/probMA', es)

ppath = os.path.join(datadir, 'blocks_0500', dataset + '*_probs_sum0247.h5')
pfiles = glob(ppath)
for pfile in pfiles:
    write_elsize(pfile + '/sum0247', es)

ppath = os.path.join(datadir, 'blocks_0500', dataset + '*_probs_sum16.h5')
pfiles = glob(ppath)
for pfile in pfiles:
    write_elsize(pfile + '/sum16', es)

ppath = os.path.join(datadir, 'blocks_0500', dataset + '*_probs_probMA.h5')
pfiles = glob(ppath)
for pfile in pfiles:
    write_elsize(pfile + '/probMA', es)

ppath = os.path.join(datadir, 'blocks_0500', dataset + '*_probs_eed_sum0247.h5')
pfiles = glob(ppath)
for pfile in pfiles:
    write_elsize(pfile + '/sum0247_eed', es)

ppath = os.path.join(datadir, 'blocks_0500', dataset + '*_probs_eed_sum16.h5')
pfiles = glob(ppath)
for pfile in pfiles:
    write_elsize(pfile + '/sum16_eed', es)

ppath = os.path.join(datadir, 'blocks_0500', dataset + '*_probs_eed_probMA.h5')
pfiles = glob(ppath)
for pfile in pfiles:
    write_elsize(pfile + '/probMA_eed', es)


dataset = 'B-NT-S10-2f_ROI_02'
dataset = 'B-NT-S10-2d_ROI_00'
datadir = '/vols/Data/km/michielk/oxdata/P01/EM/Myrf_01/SET-B/' + dataset

write_elsize(os.path.join(datadir, dataset + 'ds7_masks.h5/maskMM'), es_ds)
write_elsize(os.path.join(datadir, dataset + 'ds7_masks.h5/maskICS'), es_ds)
write_elsize(os.path.join(datadir, dataset + 'ds7_masks.h5/maskMA'), es_ds)
write_elsize(os.path.join(datadir, dataset + 'ds7_masks_maskMM.h5/maskMM'), es_ds)
write_elsize(os.path.join(datadir, dataset + 'ds7_masks_maskICS.h5/maskICS'), es_ds)
write_elsize(os.path.join(datadir, dataset + 'ds7_masks_maskMA.h5/maskMA'), es_ds)
write_elsize(os.path.join(datadir, dataset + 'ds7_probs_eed_sum0247.h5/sum0247_eed'), es_ds)
write_elsize(os.path.join(datadir, dataset + 'ds7_probs_eed_sum16.h5/sum16_eed'), es_ds)
write_elsize(os.path.join(datadir, dataset + 'ds7_probs_eed_probMA.h5/probMA_eed'), es_ds)

write_elsize(os.path.join(datadir, dataset + '_masks.h5/maskMM'), es)
write_elsize(os.path.join(datadir, dataset + '_masks.h5/maskICS'), es)
write_elsize(os.path.join(datadir, dataset + '_masks.h5/maskMA'), es)
write_elsize(os.path.join(datadir, dataset + '_masks.h5/maskMM_raw'), es)
write_elsize(os.path.join(datadir, dataset + '_masks_maskMM.h5/maskMM'), es)
write_elsize(os.path.join(datadir, dataset + '_masks_maskICS.h5/maskICS'), es)
write_elsize(os.path.join(datadir, dataset + '_masks_maskMA.h5/maskMA'), es)
write_elsize(os.path.join(datadir, dataset + '_masks_maskMM_raw.h5/maskMM_raw'), es)
write_elsize(os.path.join(datadir, dataset + '_probs_eed_sum0247.h5/sum0247_eed'), es)
write_elsize(os.path.join(datadir, dataset + '_probs_eed_sum16.h5/sum16_eed'), es)
write_elsize(os.path.join(datadir, dataset + '_probs_eed_probMA.h5/probMA_eed'), es)

for mask in ['maskICS', 'maskMA']:
    ppath = os.path.join(datadir, 'blocks_0500', dataset + '*_masks_{}.h5'.format(mask))
    pfiles = glob(ppath)
    for pfile in pfiles:
        write_elsize(pfile + '/{}'.format(mask), es)

ppath = os.path.join(datadir, 'blocks_0500', dataset + '*_probs1.h5')
pfiles = glob(ppath)
for pfile in pfiles:
    write_elsize(pfile + '/probMA'.format(mask), es)

ppath = os.path.join(datadir, 'blocks_0500', dataset + '*_probs_eed.h5')
pfiles = glob(ppath)
for pfile in pfiles:
    write_elsize(pfile + '/probMA'.format(mask), es)
    write_elsize(pfile + '/sum0247_eed'.format(mask), es)
    write_elsize(pfile + '/sum16_eed'.format(mask), es)

ppath = os.path.join(datadir, 'blocks_0500', dataset + '*_probs_sums.h5')
pfiles = glob(ppath)
for pfile in pfiles:
    write_elsize(pfile + '/sum0247'.format(mask), es)
    write_elsize(pfile + '/sum16'.format(mask), es)

# NOTE: maskMM_steps was deleted during copy
mask = 'maskMM'
ppath = os.path.join(datadir, 'blocks_0500', dataset + '*_masks_{}.h5'.format(mask))
pfiles = glob(ppath)
for pfile in pfiles:
    # write_elsize(pfile + '/{}'.format(mask), es)
    write_elsize(pfile + '/{}_steps/dil'.format(mask), es)
    write_elsize(pfile + '/{}_steps/mito'.format(mask), es)
    write_elsize(pfile + '/{}_steps/raw'.format(mask), es)






import os
import numpy as np
from wmem import utils
from glob import glob
import h5py

dataset = 'B-NT-S10-2f_ROI_02'
datadir = '/vols/Data/km/michielk/oxdata/P01/EM/Myrf_01/SET-B/' + dataset
datadir = '/Users/michielk/oxdata/P01/EM/Myrf_01/SET-B/' + dataset

elsizes = {
    'es': [0.1, 0.007, 0.007],
    'es_ds': [0.1, 0.049, 0.049],
    'es_probs': [0.1, 0.007, 0.007, 1],
    'es_probs_ds': [0.1, 0.049, 0.049, 1]}

def write_elsize(datadir, filename, dsets, es):
    def write_attr(name, obj):
        utils.h5_write_attributes(ds_out, element_size_um=es)
    h5path_out = os.path.join(datadir, )
    h5file = h5py.File(os.path.join(datadir, filename), 'r+')
    if not dsets:
        dsets = h5file.keys()
        # h5file.visititems(write_attr)
    for dset in dsets:
        ds_out = h5file[dset]
        utils.h5_write_attributes(ds_out, element_size_um=es)
    h5file.close()


write_elsize(datadir, dataset + '_probs_eed_probMA.h5', ['probMA_eed'], elsizes['es'])
write_elsize(datadir, dataset + 'ds7_probs_eed_probMA.h5', ['probMA_eed'], elsizes['es_ds'])

write_elsize(datadir, dataset + '_labels_labelMA_core2D.h5', ['labelMA_core2D'], elsizes['es'])
write_elsize(datadir, dataset + 'ds7_labels_labelMA_core2D.h5', ['labelMA_core2D'], elsizes['es_ds'])
write_elsize(datadir, dataset + 'ds7_labels_mapall.h5', [], elsizes['es_ds'])


# import h5py
# def print_attrs(name, obj):
#     print name
#     for key, val in obj.attrs.iteritems():
#         print "    %s: %s" % (key, val)
#
# h5file = h5py.File(pfiles[0], 'r+')
# h5file.visititems(print_attrs)








for f in `ls blocks_0500/*_probs_eed_probMA.h5`; do
    h5copy -p -i $f -s 'probMA' -o ${f/_probs_eed_probMA.h5/_probs_eed_probMA.h5tmp} -d 'probMA_eed'
done
ls -lh blocks_0500/*_probs_eed_probMA.h5tmp
ls blocks_0500/*_probs_eed_probMA.h5tmp | wc -l
rm blocks_0500/*_probs_eed_probMA.h5
rename _probs_eed_probMA.h5tmp _probs_eed_probMA.h5 blocks_0500/*_probs_eed_probMA.h5tmp

rename _probs1.h5 _probs_probMA.h5 blocks_0500/*_probs1.h5

for f in `ls blocks_0500/*_masks.h5`; do
    for dset in 'maskMM' 'maskICS'; do  # 'maskDS'
        h5copy -p -i $f -s $dset -o ${f/_masks.h5/_masks_$dset.h5} -d $dset
    done
done
rm blocks_0500/*_masks.h5
