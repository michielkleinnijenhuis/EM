###=========================================================================###
### dataset parameters
###=========================================================================###
basedir="${DATA}/EM/Myrf_01/SET-B"
dataset='B-NT-S9-2a'
datadir=$basedir/${dataset} && mkdir -p $datadir && cd $datadir
bs=0500 && mkdir -p $datadir/blocks_${bs} && source datastems_blocks_${bs}.sh


###=========================================================================###
### dataset parameters # Image Width: 8649 Image Length: 8308
###=========================================================================###
basedir="${DATA}/EM/Myrf_01/SET-B"
dataset='B-NT-S10-2d_ROI_00'
datadir=$basedir/${dataset} && mkdir -p $datadir && cd $datadir
bs=0500 && mkdir -p $datadir/blocks_${bs} && source datastems_blocks_${bs}.sh


###=========================================================================###
### dataset parameters # Image Width: 8844 Image Length: 8521
###=========================================================================###
basedir="${DATA}/EM/Myrf_01/SET-B"
dataset='B-NT-S10-2d_ROI_02'
datadir=$basedir/${dataset} && mkdir -p $datadir && cd $datadir
bs=0500 && mkdir -p $datadir/blocks_${bs} && source datastems_blocks_${bs}.sh

for tif in 'tif_ds'; do  # 'tif'
    mkdir $datadir/$tif/artefacts
    for i in `seq 4 9`; do
        mv $datadir/$tif/002$i?.tif $datadir/$tif/artefacts
    done
    mv $datadir/$tif/003??.tif $datadir/$tif/artefacts
    mv $datadir/$tif/004??.tif $datadir/$tif/artefacts
done


###=========================================================================###
### dataset parameters # Image Width: 8423 Image Length: 8316
###=========================================================================###
basedir="${DATA}/EM/Myrf_01/SET-B"
dataset='B-NT-S10-2f_ROI_00'
datadir=$basedir/${dataset} && mkdir -p $datadir && cd $datadir
bs=0500 && mkdir -p $datadir/blocks_${bs} && source datastems_blocks_${bs}.sh


###=========================================================================###
### dataset parameters # Image Width: 8649 Image Length: 8287
###=========================================================================###
basedir="${DATA}/EM/Myrf_01/SET-B"
dataset='B-NT-S10-2f_ROI_01'
datadir=$basedir/${dataset} && mkdir -p $datadir && cd $datadir
bs=0500 && mkdir -p $datadir/blocks_${bs} && source datastems_blocks_${bs}.sh


###=========================================================================###
### dataset parameters # Image Width: 8457 Image Length: 8453
###=========================================================================###
basedir="${DATA}/EM/Myrf_01/SET-B"
dataset='B-NT-S10-2f_ROI_02'
datadir=$basedir/${dataset} && mkdir -p $datadir && cd $datadir
bs=0500 && mkdir -p $datadir/blocks_${bs} && source datastems_blocks_${bs}.sh
