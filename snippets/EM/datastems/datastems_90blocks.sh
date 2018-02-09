#!/bin/bash

export PATH="$DATA/anaconda2/bin:$PATH"
export CONDA_PATH="$(conda info --root)"
export scriptdir="$HOME/workspace/EM"
export basedir="$DATA/EM/Myrf_01/SET-B"
export dataset='B-NT-S9-2a'
export dm3dir="$DATA/EM/Myrf_01/SET-B/3View/$dataset"
export datadir="$basedir/${dataset}"  && cd $datadir
export basepath="$datadir/${dataset}"
export regref='00250.tif'

# 90 blocks
xmax=9849; ymax=9590; zmax=479;
xs=1000; ys=1000; zs=479;
xm=50; ym=50; zm=0;
z=0; Z=479;
xo=0; yo=0; zo=0;
xe=0.007; ye=0.007; ze=0.07;
export ds_factor=8

unset datastems
declare -a datastems

i=0
for x in `seq 0 $xs $xmax`; do
    [ $x == $(((xmax/xs)*xs)) ] && X=$xmax || X=$((x+xs+xm))
    [ $x == 0 ] || x=$((x-xm))
    for y in `seq 0 $ys $ymax`; do
        [ $y == $(((ymax/ys)*ys)) ] && Y=$ymax || Y=$((y+ys+ym))
        [ $y == 0 ] || y=$((y-ym))
        xrange=`printf %05d ${x}`-`printf %05d ${X}`
        yrange=`printf %05d ${y}`-`printf %05d ${Y}`
        zrange=`printf %05d ${z}`-`printf %05d ${Z}`
        echo ${dataset}_${xrange}_${yrange}_${zrange}
        datastems[$i]=${dataset}_${xrange}_${yrange}_${zrange}
        i=$((i+1))
    done
done

export datastems
