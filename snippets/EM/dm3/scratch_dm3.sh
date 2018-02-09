from skimage.io import imsave
from skimage.io import imshow
from skimage import img_as_float, img_as_int

filename = os.path.split(filepath)[1]
fileref = os.path.splitext(filename)[0]
dm3f = dm3.DM3(filepath, debug=0)

imshow(dm3f.imagedata)
im = Image.fromarray(dm3f.imagedata)

im = img_as_int(dm3f.imagedata)


fname = os.path.join(datadir, session,  'test.tif')
imsave(fname, dm3f.imagedata)



scriptdir="$HOME/workspace/EM"
export PYTHONPATH=$PYTHONPATH:$scriptdir
export PYTHONPATH=$PYTHONPATH:/Users/michielk/workspace/pyDM3reader

datadir="$HOME/oxdata/originaldata/P01/Myrf_00/test"
dataset="test"
basepath=$datadir/${dataset}
python $scriptdir/wmem/series2stack.py $datadir $basepath.h5/data -o 'zyx' -s 4 20 20 -r '*.dm3'

datadir="$HOME/oxdata/originaldata/P01/Myrf_01/test"
dataset="test"
basepath=$datadir/${dataset}
python $scriptdir/wmem/series2stack.py $datadir $basepath.h5/data -o 'zyx' -s 4 20 20 -r '*.dm3'
