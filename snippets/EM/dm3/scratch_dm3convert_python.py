scriptdir="$HOME/workspace/EM"
export PYTHONPATH=$PYTHONPATH:$scriptdir
export PYTHONPATH=$PYTHONPATH:/Users/michielk/workspace/pyDM3reader
ipython

import os.path
import sys

import numpy as np
from PIL import Image

# sys.path.append(r'/Users/michielk/Downloads/pyDM3reader')
import DM3lib as dm3
from glob import glob

# Myrf_00
savedir = "/Users/michielk/oxdata/P01/EM/Myrf_00_201707/TEM_jpg"
if not os.path.exists(savedir):
    os.makedirs(savedir)
datadir = "/Users/michielk/oxdata/originaldata/P01/Myrf_00"
sessions = ['TEM_20170801', 'TEM_20170802', 'TEM_20170805']

# Myrf_00
savedir = "/Users/michielk/oxdata/P01/EM/Myrf_00_201707/3View_jpg"
if not os.path.exists(savedir):
    os.makedirs(savedir)
datadir = "/Users/michielk/oxdata/originaldata/P01/Myrf_00/3View"
sessions = ['25Aug17', '25Aug17/test', '25Aug17/540slices_70nm_8k_25Aug17']

# Myrf_01 - SET-A - 01
savedir = "/Users/michielk/oxdata/P01/EM/Myrf_01_201708/TEM_A_jpg"
if not os.path.exists(savedir):
    os.makedirs(savedir)
datadir = "/Users/michielk/oxdata/originaldata/P01/Myrf_01/SET-A"
sessions = ['TEM_04092017', '20170918_TEM']

# Myrf_01 - SET-B - 01
savedir = "/Users/michielk/oxdata/P01/EM/Myrf_01_201708/TEM_B_jpg"
if not os.path.exists(savedir):
    os.makedirs(savedir)
datadir = "/Users/michielk/oxdata/originaldata/P01/Myrf_01/SET-B"
sessions = ['20170927_TEM', '20170929_TEM']

savedir = "/Users/michielk/oxdata/P01/EM/Myrf_01_201708/3View"
if not os.path.exists(savedir):
    os.makedirs(savedir)
datadir = "/Users/michielk/oxdata/originaldata/P01/Myrf_01/SET-B/3View"
sessions = ['setup_3Oct17']

savedir = "/Users/michielk/oxdata/P01/EM/Myrf_01_201708/3View/3Oct17"
if not os.path.exists(savedir):
    os.makedirs(savedir)
datadir = "/Volumes/NO NAME"
sessions = ['3Oct17']

for session in sessions:
    filepaths = glob(os.path.join(datadir, session, '*.dm3'))
    for filepath in filepaths[::20]:  #
        save_im(savedir, filepath)
        # save_im(savedir, filepath, imformat='.tif', dumptags=True)

def save_im(savedir, filepath, imformat='.jpg', dumptags=False):

    # get filename
    filename = os.path.split(filepath)[1]
    fileref = os.path.splitext(filename)[0]

    dm3f = dm3.DM3(filepath, debug=0)
    cuts = dm3f.cuts
    if dumptags:
        dm3f.dumpTags(savedir)
    aa = dm3f.imagedata

    # save image as TIFF
    if '.tif' in imformat:
        tif_file = os.path.join(savedir, fileref + imformat)
        im = Image.fromarray(aa)
        im.save(tif_file)
        # check TIFF dynamic range
        if Image.open(tif_file).mode == 'L':
            tif_range = "8-bit"
        else:
            tif_range = "32-bit"
    else:
        # - normalize image for conversion to 8-bit
        aa_norm = aa.copy()
        # -- apply cuts (optional)
        if cuts[0] != cuts[1]:
            aa_norm[ (aa <= min(cuts)) ] = float(min(cuts))
            aa_norm[ (aa >= max(cuts)) ] = float(max(cuts))
        # -- normalize
        aa_norm = (aa_norm - np.min(aa_norm)) / (np.max(aa_norm) - np.min(aa_norm))
        # -- scale to 0--255, convert to (8-bit) integer
        aa_norm = np.uint8(np.round( aa_norm * 255 ))
        # - save as <imformat>
        im_dsp = Image.fromarray(aa_norm)
        im_dsp.save(os.path.join(savedir, fileref + imformat))






scriptdir='/Users/michielk/Downloads/pyDM3reader'
export PYTHONPATH=$PYTHONPATH:$scriptdir
dm3dir="${HOME}/TempMich2_0208"
$scriptdir/demo.py $dm3dir/T2_4_4B_0020.dm3 -v --convert

scriptdir="${HOME}/workspace/EM"
dm3dir="${HOME}/TempMich2_0208"
datadir="${HOME}/TEM_0208"
imagej=/Applications/Fiji.app/Contents/MacOS/ImageJ-macosx
mkdir -p $datadir/tifs
sed "s?INPUTDIR?$dm3dir?;\
    s?OUTPUTDIR?$datadir/tifs?;\
    s?OUTPUT_POSTFIX?_m000?g" \
    $scriptdir/wmem/tiles2tif.py \
    > $datadir/EM_tiles2tif_m000.py
$imagej --headless $datadir/EM_tiles2tif_m000.py


https://github.com/jrminter/snippets.git
