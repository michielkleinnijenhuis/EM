local_datadir=/Users/michielk/oxdata/P01/EM/M3
rem_datadir=/data/ndcn-fmrib-water-brain/ndcn0180/EM/M3

intpath=M3_S1_SPL/stitched
intpath=M3_S1_GNU_old/stitched
fname=0000.tif
fname=0250.tif
fname=0459.tif
locdir=$local_datadir/$intpath
mkdir -p $locdir
rsync -avz ndcn0180@arcus.arc.ox.ac.uk:$rem_datadir/$intpath/$fname $locdir

intpath=M3_S1_GNU_regpointpairs
intpath=M3_S1_GNU_regpointpairs/reg_d4_low
intpath=M3_S1_GNU_regpointpairs/reg_d4
fname=tifs
fname=reg_d4
fname=0*.tif
fname=betas*
locdir=$local_datadir/$intpath
mkdir -p $locdir
rsync -avz ndcn0180@arcus.arc.ox.ac.uk:$rem_datadir/$intpath/$fname $locdir

intpath=M3_S1_GNU_stitch
locdir=$local_datadir/$intpath
fname=localPP
rsync -avz $locdir/$fname ndcn0180@arcus.arc.ox.ac.uk:$rem_datadir/$intpath
