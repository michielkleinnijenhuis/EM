scriptdir="$HOME/workspace/EM"
datadir='/Users/michielk/oxdata/P01/EM/Kashturi11'

### cutout01
cutout='cutout01'
# http://openconnecto.me/ocp/ca/kasthuri11cc/hdf5/0/10000,11000/16000,17000/1100,1200/
# http://openconnecto.me/ocp/ca/kat11segments/hdf5/0/10000,11000/16000,17000/1100,1200/
# http://openconnecto.me/ocp/ca/kat11mito/hdf5/0/10000,11000/16000,17000/1100,1200/
# http://openconnecto.me/ocp/ca/kat11vesicles/hdf5/0/10000,11000/16000,17000/1100,1200/
# http://openconnecto.me/ocp/ca/kat11synapses/hdf5/0/10000,11000/16000,17000/1100,1200/
# http://openconnecto.me/ocp/ca/kasthuri11cc/hdf5/1/5000,5500/8000,8500/1100,1200/
# http://openconnecto.me/ocp/ca/kat11segments/hdf5/1/5000,5500/8000,8500/1100,1200/
# http://openconnecto.me/ocp/ca/kat11mito/hdf5/1/5000,5500/8000,8500/1100,1200/
# http://openconnecto.me/ocp/ca/kat11vesicles/hdf5/1/5000,5500/8000,8500/1100,1200/
# http://openconnecto.me/ocp/ca/kat11synapses/hdf5/1/5000,5500/8000,8500/1100,1200/
# http://openconnecto.me/ocp/ca/kasthuri11cc/hdf5/2/2500,2750/4000,4250/1100,1200/
# http://openconnecto.me/ocp/ca/kat11segments/hdf5/2/2500,2750/4000,4250/1100,1200/
# http://openconnecto.me/ocp/ca/kat11mito/hdf5/2/2500,2750/4000,4250/1100,1200/
# http://openconnecto.me/ocp/ca/kat11vesicles/hdf5/2/2500,2750/4000,4250/1100,1200/
# http://openconnecto.me/ocp/ca/kat11synapses/hdf5/2/2500,2750/4000,4250/1100,1200/
# http://openconnecto.me/ocp/ca/kasthuri11cc/hdf5/3/1250,1375/2000,2125/1100,1200/
# http://openconnecto.me/ocp/ca/kat11segments/hdf5/3/1250,1375/2000,2125/1100,1200/
# http://openconnecto.me/ocp/ca/kat11mito/hdf5/3/1250,1375/2000,2125/1100,1200/
# http://openconnecto.me/ocp/ca/kat11vesicles/hdf5/3/1250,1375/2000,2125/1100,1200/
# http://openconnecto.me/ocp/ca/kat11synapses/hdf5/3/1250,1375/2000,2125/1100,1200/
scale=0; zs=0.03; ys=0.003; xs=0.003; zo=1100; yo=16000; xo=10000; zl=100; yl=1000; xl=1000
scale=1; zs=0.03; ys=0.006; xs=0.006; zo=1100; yo=8000; xo=5000; zl=100; yl=500; xl=500
scale=3; zs=0.03; ys=0.024; xs=0.024; zo=1100; yo=2000; xo=1250; zl=100; yl=125; xl=125


### cutout03
# http://openconnecto.me/ocp/ca/kasthuri11cc/hdf5/0/5200,14800/13600,20000/1000,1400/
# http://openconnecto.me/ocp/ca/kasthuri11cc/hdf5/1/2600,7400/6800,10000/1000,1400/
# http://openconnecto.me/ocp/ca/kasthuri11cc/hdf5/2/1300,3700/3400,5000/1000,1400/
# http://openconnecto.me/ocp/ca/kasthuri11cc/hdf5/3/650,1850/1700,2500/1000,1400/
# http://openconnecto.me/ocp/ca/kat11segments/hdf5/3/650,1850/1700,2500/1000,1400/
# http://openconnecto.me/ocp/ca/kat11mito/hdf5/3/650,1850/1700,2500/1000,1400/
# http://openconnecto.me/ocp/ca/kat11vesicles/hdf5/3/650,1850/1700,2500/1000,1400/
# http://openconnecto.me/ocp/ca/kat11synapses/hdf5/3/650,1850/1700,2500/1000,1400/
# http://openconnecto.me/ocp/ca/kat11mojocylinder/hdf5/3/650,1850/1700,2500/1000,1400/
# http://openconnecto.me/ocp/ca/kat11redcylinder/hdf5/3/650,1850/1700,2500/1000,1400/
# http://openconnecto.me/ocp/ca/kat11greencylinder/hdf5/3/650,1850/1700,2500/1000,1400/
cutout='cutout03'
scale=3; zs=0.03; ys=0.024; xs=0.024; zo=1000; yo=1700; xo=650; zl=400; yl=800; xl=1200

cudir=$datadir/$cutout
cutoutstr='-default-hdf5-'${scale}'-'${xo}'_'$((xo+xl))'-'${yo}'_'$((yo+yl))'-'${zo}'_'$((zo+zl))'-ocpcutout.h5'

# TODO: function that downloads data
python $scriptdir/mesh/label2stl.py \
$cudir -S $scale -s $zs $ys $xs -o $zo $yo $xo -l $zl $yl $xl -r 1 1 1 \
-L 'segments' 'mito' 'redcylinder' -M 'redcylinder'

blender -b -P $scriptdir/mesh/stl2blender.py -- $cudir/dmcsurf testcylinders -L 'greencylinder' 'redcylinder' 'mojocylinder'
blender -b -P $scriptdir/mesh/stl2blender.py -- $cudir/dmcsurf testvesicles -L 'vesicles'
blender -b -P $scriptdir/mesh/stl2blender.py -- $cudir/dmcsurf testsynapses -L 'synapses'
blender -b -P $scriptdir/mesh/stl2blender.py -- $cudir/dmcsurf cylinders_red -L 'redcylinder'
blender -b -P $scriptdir/mesh/stl2blender.py -- $cudir/dmcsurf mito -L 'mito'
blender -b -P $scriptdir/mesh/stl2blender.py -- $cudir/dmcsurf MA -L 'MA'
blender -b -P $scriptdir/mesh/stl2blender.py -- $cudir/dmcsurf MM -L 'MM'
blender -b -P $scriptdir/mesh/stl2blender.py -- $cudir/dmcsurf DD -L 'DD'
blender -b -P $scriptdir/mesh/stl2blender.py -- $cudir/dmcsurf UA -L 'UA'
blender -b -P $scriptdir/mesh/stl2blender.py -- $cudir/dmcsurf NN -L 'ECS'

blender -b -P $scriptdir/mesh/stl2blender.py -- $cudir/dmcsurf testMA_sl -L 'MA'

### h5-to-nifti conversions (in bash for now;  TODO: create module to import functions into python)

xvs=-0.024; yvs=-0.024; zvs=0.03;
python $scriptdir/convert/EM_stack2stack.py \
"${cudir}/kasthuri11cc${cutoutstr}" "${cudir}/data.nii.gz" \
-i 'zyx' -l 'xyz' -e $xvs $yvs $zvs -f '/default/CUTOUT'
for l in 'segments' 'mito' 'vesicles' 'synapses'; do
python $scriptdir/convert/EM_stack2stack.py \
"${cudir}/kat11${l}${cutoutstr}" "${cudir}/${l}.nii.gz" \
-i 'zyx' -l 'xyz' -e $xvs $yvs $zvs -f '/default/CUTOUT'
done

# 
# pf='_orig'; xvs=-0.006; yvs=-0.006; zvs=0.03;
# pf='_ds'; xvs=-0.024; yvs=-0.024; zvs=0.03;
# pf='_us'; xvs=-0.006; yvs=-0.006; zvs=0.006;
# 
# # pf='_orig'; xvs=-0.024; yvs=-0.024; zvs=0.03;
# 
# python $scriptdir/EM_stack2stack.py "${datadir}/data${pf}.h5" "${datadir}/data${pf}.nii.gz" -i 'zyx' -l 'xyz' -e $xvs $yvs $zvs
# python $scriptdir/EM_stack2stack.py "${datadir}/data_smooth${pf}.h5" "${datadir}/data_smooth${pf}.nii.gz" -i 'zyx' -l 'xyz' -e $xvs $yvs $zvs
# for l in 'segments' 'mito' 'vesicles' 'synapses'; do
# python $scriptdir/EM_stack2stack.py "${datadir}/${l}${pf}.h5" "${datadir}/${l}${pf}.nii.gz" -i 'zyx' -l 'xyz' -e $xvs $yvs $zvs
# done
# python $scriptdir/EM_stack2stack.py "${datadir}/segments_ws${pf}.h5" "${datadir}/segments_ws${pf}.nii.gz" -i 'zyx' -l 'xyz' -e $xvs $yvs $zvs
# python $scriptdir/EM_stack2stack.py "${datadir}/L_ECS${pf}.h5" "${datadir}/L_ECS${pf}.nii.gz" -i 'zyx' -l 'xyz' -e $xvs $yvs $zvs

