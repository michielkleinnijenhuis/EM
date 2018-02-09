conda create --name ndio python=3.5 tifffile pillow numpy h5py requests jsonschema nibabel
# conda install pymcubes pycollada blosc==1.3.2 json-spec
conda install ipython jupyter matplotlib
pip install git+git://github.com/scikit-image/scikit-image@master
# pip install ndio
cd ~/workspace/
git clone https://github.com/neurodata/ndio.git
pip install ~/workspace/ndio --upgrade

conda create --name ndmg python=2.7
# pip install ndmg
cd ~/workspace/
git clone https://github.com/neurodata/ndmg.git
pip install ~/workspace/ndmg --upgrade
# python ndmg/scripts/ndmg_pipeline.py tests/data/KKI2009_113_1_DTI_s4.nii \
# tests/data/KKI2009_113_1_DTI_s4.bval tests/data/KKI2009_113_1_DTI_s4.bvec \
# tests/data/KKI2009_113_1_MPRAGE_s4.nii tests/data/MNI152_T1_1mm_s4.nii.gz \
# tests/data/MNI152_T1_1mm_brain_mask_s4.nii.gz \
# tests/data/outputs tests/data/desikan_s4.nii.gz
