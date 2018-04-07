#!/bin/bash


function dataset_parameters {
    # Parameters for datasets.

    local dataset=$1

    if [ "$dataset" == 'M3S1GNU' ]
    then
        basedir="$DATA/EM/M3"
        datadir_loc="/Users/michielk/oxdata/P01/EM/M3/$dataset"
        datadir_jal="/vols/Data/km/michielk/oxdata/P01/EM/M3/$dataset"
        datadir_arc="/data/ndcn-fmrib-water-brain/ndcn0180/EM/M3/$dataset"
#         dm3dir="$DATA/EM/Myrf_01/SET-B/3View/$dataset"  # FIXME
        regref='00250.tif'  # unmoving image in registration
        regname='reg'  # name of registered data
        dspf='ds' ds=7  # inplane downsample factor
        xmax=9179 ymax=8786 zmax=430  # matrix size
        xs=500 ys=500 zs=430  # blocksizes
        xm=20 ym=20 zm=0  # margins
        xo=0 yo=0 zo=0  # offsets
        xe=0.0073 ye=0.0073 ze=0.05  # voxel sizes
        # chunksize data: 20, 20, 20
        # chunksize data_ds: 14, 79, 82
        # chunksize probs: 
        # chunksize probs_eed: 14, 79, 82
        # chunksize probs_eed_ds: 14, 79, 82
        # chunksize masks_ds: 14, 79, 82
        # chunksize labels: 7, 138, 144
        # chunksize labels_ds: 14, 40, 82
        # chunksize maskDS: 7, 275, 287
        # chunksize maskMM: 7, 275, 287
        # chunksize maskICS: 27, 33, 65
        # chunksize maskMA: 27, 33, 65
        # chunksize ws: 14, 65, 65
        # chunksize probs_0500: 
        # chunksize probs_eed_0500: NA
        # chunksize sums_0500: NA
        # chunksize probs1_0500: 27, 33, 33
        # chunksize probs_eed_sums_0500: 27, 33, 33
        # chunksize masks: 27, 33, 65
    fi

    # 'M3S1SPL'

    if [ "$dataset" == 'B-NT-S9-2a' ]
    then
        basedir="$DATA/EM/Myrf_01/SET-B"
        datadir_loc="/Users/michielk/oxdata/P01/EM/Myrf_01/SET-B/$dataset"
        datadir_jal="/vols/Data/km/michielk/oxdata/P01/EM/Myrf_01/SET-B/$dataset"
        datadir_arc="/data/ndcn-fmrib-water-brain/ndcn0180/EM/Myrf_01/SET-B/$dataset"
        dm3dir="$DATA/EM/Myrf_01/SET-B/3View/$dataset"
        regref='00250.tif'  # unmoving image in registration
        regname='reg'  # name of registered data
        dspf='ds' ds=7  # inplane downsample factor
        xmax=9849 ymax=9590 zmax=479  # matrix size
        xs=500 ys=500 zs=479  # blocksizes
        xm=20 ym=20 zm=0  # margins
        xo=0 yo=0 zo=0  # offsets
        xe=0.007 ye=0.007 ze=0.07  # voxel sizes
    fi

    if [ "$dataset" == 'B-NT-S10-2d_ROI_00' ]
    then
        basedir="$DATA/EM/Myrf_01/SET-B"
        datadir_loc="/Users/michielk/oxdata/P01/EM/Myrf_01/SET-B/$dataset"
        datadir_jal="/vols/Data/km/michielk/oxdata/P01/EM/Myrf_01/SET-B/$dataset"
        datadir_arc="/data/ndcn-fmrib-water-brain/ndcn0180/EM/Myrf_01/SET-B/$dataset"
        dm3dir="$DATA/EM/Myrf_01/SET-B/3View/$dataset"
        regref='00065.tif'  # unmoving image in registration
        regname='reg'  # name of registered data
        dspf='ds' ds=7  # inplane downsample factor
        xmax=8649 ymax=8308 zmax=135  # matrix size
        xs=500 ys=500 zs=135  # blocksizes
        xm=20 ym=20 zm=0  # margins
        xo=0 yo=0 zo=0  # offsets
        xe=0.007 ye=0.007 ze=0.1  # voxel sizes
        # chunksize data: 20, 20, 20
        # chunksize data_ds: 9, 75, 78
        # chunksize probs: 50, 51, 50, 1
        # chunksize probs_0500: 9, 65, 65, 1
        # chunksize probs_eed_0500: NA
        # chunksize sums_0500: 9, 33, 65
        # chunksize probs_eed_sums:
    fi

    if [ "$dataset" == 'B-NT-S10-2d_ROI_02' ]
    then
        basedir="$DATA/EM/Myrf_01/SET-B"
        datadir_loc="/Users/michielk/oxdata/P01/EM/Myrf_01/SET-B/$dataset"
        datadir_jal="/vols/Data/km/michielk/oxdata/P01/EM/Myrf_01/SET-B/$dataset"
        datadir_arc="/data/ndcn-fmrib-water-brain/ndcn0180/EM/Myrf_01/SET-B/$dataset"
        dm3dir="$DATA/EM/Myrf_01/SET-B/3View/B-NT-S10-2d/scan2_25Oct17"
        regref='00120.tif'  # unmoving image in registration
        regname='reg'  # name of registered data
        dspf='ds' ds=7  # inplane downsample factor
        xmax=8844 ymax=8521 zmax=240  # matrix size
        xs=500 ys=500 zs=240  # blocksizes
        xm=20 ym=20 zm=0  # margins
        xo=0 yo=0 zo=0  # offsets
        xe=0.007 ye=0.007 ze=0.1  # voxel sizes
        # chunksize data: 20, 20, 20
        # chunksize data_ds: 8, 77, 79
        # chunksize probs: 50, 51, 50, 1
        # chunksize probs_0500: 50, 51, 50, 1
        # chunksize probs_eed_0500: NA
        # chunksize sums_0500: 15, 33, 65
        # chunksize probs_eed_sums:
    fi

    if [ "$dataset" == 'B-NT-S10-2f_ROI_00' ]
    then
        basedir="$DATA/EM/Myrf_01/SET-B"
        datadir_loc="/Users/michielk/oxdata/P01/EM/Myrf_01/SET-B/$dataset"
        datadir_jal="/vols/Data/km/michielk/oxdata/P01/EM/Myrf_01/SET-B/$dataset"
        datadir_arc="/data/ndcn-fmrib-water-brain/ndcn0180/EM/Myrf_01/SET-B/$dataset"
        dm3dir="$DATA/EM/Myrf_01/SET-B/3View/$dataset/ROI_00"
        regref='00092.tif'  # unmoving image in registration
        regname='reg'  # name of registered data
        dspf='ds' ds=7  # inplane downsample factor
        xmax=8423 ymax=8316 zmax=184  # matrix size
        xs=500 ys=500 zs=184  # blocksizes
        xm=20 ym=20 zm=0  # margins
        xo=0 yo=0 zo=0  # offsets
        xe=0.007 ye=0.007 ze=0.1  # voxel sizes
    fi

    if [ "$dataset" == 'B-NT-S10-2f_ROI_01' ]
    then
        basedir="$DATA/EM/Myrf_01/SET-B"
        datadir_loc="/Users/michielk/oxdata/P01/EM/Myrf_01/SET-B/$dataset"
        datadir_jal="/vols/Data/km/michielk/oxdata/P01/EM/Myrf_01/SET-B/$dataset"
        datadir_arc="/data/ndcn-fmrib-water-brain/ndcn0180/EM/Myrf_01/SET-B/$dataset"
        dm3dir="$DATA/EM/Myrf_01/SET-B/3View/$dataset/ROI_01"
        regref='00092.tif'  # unmoving image in registration
        regname='reg'  # name of registered data
        dspf='ds' ds=7  # inplane downsample factor
        xmax=8649 ymax=8287 zmax=184  # matrix size
        xs=500 ys=500 zs=184  # blocksizes
        xm=20 ym=20 zm=0  # margins
        xo=0 yo=0 zo=0  # offsets
        xe=0.007 ye=0.007 ze=0.1  # voxel sizes
        # chunksize data: 20, 20, 20
        # chunksize data_ds: 
        # chunksize probs: 50, 51, 50
        # chunksize probs_0500: 23, 39, 22
        # chunksize probs_eed_0500: 23, 39, 22
        # chunksize sums_0500: 12, 39, 22
        # chunksize probs_eed_sums: 12, 34, 65
    fi

    if [ "$dataset" == 'B-NT-S10-2f_ROI_02' ]
    then
        basedir="$DATA/EM/Myrf_01/SET-B"
        datadir_loc="/Users/michielk/oxdata/P01/EM/Myrf_01/SET-B/$dataset"
        datadir_jal="/vols/Data/km/michielk/oxdata/P01/EM/Myrf_01/SET-B/$dataset"
        datadir_arc="/data/ndcn-fmrib-water-brain/ndcn0180/EM/Myrf_01/SET-B/$dataset"
        dm3dir="$DATA/EM/Myrf_01/SET-B/3View/$dataset/ROI_02"
        regref='00092.tif'  # unmoving image in registration
        regname='reg'  # name of registered data
        dspf='ds' ds=7  # inplane downsample factor
        xmax=8457 ymax=8453 zmax=184  # matrix size
        xs=500 ys=500 zs=184  # blocksizes
        xm=20 ym=20 zm=0  # margins
        xo=0 yo=0 zo=0  # offsets
        xe=0.007 ye=0.007 ze=0.1  # voxel sizes
        # chunksize data: 20, 20, 20
        # chunksize data_ds: 12, 76, 76
        # chunksize probs: 50, 51, 50
        # chunksize probs_0500: 12, 60, 60
        # chunksize probs_eed_0500: 12, 60, 60
        # chunksize sums_0500: 12, 30, 60
        # chunksize probs_eed_sums: 12, 33, 65
    fi

}


function filter_artefact_sections {
    # Move identified artefact sections to subdirectory.

    local dataset=$1
    local tif

    if [ "$dataset" == 'B-NT-S10-2d_ROI_02' ]
    then
        (
        for tif in 'tif' 'tif_ds'; do
            mkdir $datadir/$tif/artefacts
            for i in `seq 4 9`; do
                mv $datadir/$tif/002$i?.tif $datadir/$tif/artefacts
            done
            mv $datadir/$tif/003??.tif $datadir/$tif/artefacts
            mv $datadir/$tif/004??.tif $datadir/$tif/artefacts
        done
        )
    fi
}
