# This attempts to solve a bug in NP
# see https://github.com/janelia-flyem/NeuroProof/issues/4


### neuroproof/CMakeFiles/NeuroProofRag.dir/build.make
336 + ../lib/libNeuroProofRag.so: /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX10.11.sdk/System/Library/Frameworks/OpenGL.framework
337 - # ../lib/libNeuroProofRag.so: /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX10.10.sdk/System/Library/Frameworks/OpenGL.framework
### neuroproof/CMakeFiles/NeuroProofRag.dir/link.txt

https://github.com/phracker/MacOSX-SDKs/releases
# cp -r ~/Downloads/MacOSX10.10.sdk /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/

./configure-for-conda.sh ~/anaconda/envs/neuroproof-devel/
cd build
ccmake -DCMAKE_BUILD_TYPE=Debug ..
ccmake -G"Eclipse CDT4 - Unix Makefiles" -DCMAKE_BUILD_TYPE=Debug ..
make
make install

export LD_LIBRARY_PATH=~/anaconda/envs/neuroproof-devel/lib

# -DCMAKE_OSX_DEPLOYMENT_TARGET:STRING="" \
# -DCMAKE_OSX_SYSROOT:STRING=/ \




# This tests a version of NP provided by Toufiq

# local
cd ~/workspace/Neuroproof_minimal
source activate neuroproof-test
PREFIX=$(conda info --root)/envs/neuroproof-test
export DYLD_FALLBACK_LIBRARY_PATH=${PREFIX}/lib
bash batch-build

#ARC
rsync -avz /Users/michielk/workspace/NP_* ndcn0180@arcus.arc.ox.ac.uk:/home/ndcn-fmrib-water-brain/ndcn0180/workspace
rsync -avz /Users/michielk/workspace/Neuroproof_minimal.zip ndcn0180@arcus.arc.ox.ac.uk:/home/ndcn-fmrib-water-brain/ndcn0180/workspace
cd ~/workspace; unzip Neuroproof_minimal.zip; cd Neuroproof_minimal;
# module load gcc/4.8.2
# module load opencv/2.4.5
cd ~/workspace/Neuroproof_minimal
# source activate neuroproof-test
PREFIX=$(conda info --root)/envs/neuroproof-test
export LD_LIBRARY_PATH=${PREFIX}/lib
bash batch-build

# on OSX?? -std=c++11 -stdlib=libc++

g++ -c -w -O3 -I$PREFIX/include  -I$PREFIX/include/vigra -I$PREFIX/include/python2.7 -I$PREFIX/include/boost  FeatureManager/FeatureManager.cpp FeatureManager/Features.cpp DataStructures/Stack.cpp DataStructures/StackPredict.cpp Algorithms/MergePriorityFunction.cpp Classifier/vigraRFclassifier.cpp Classifier/vigraRFclassifierP.cpp Classifier/opencvRFclassifier.cpp Classifier/opencvRFclassifierP.cpp Classifier/compositeRFclassifier.cpp NeuroProof_stack.cpp

 # Algorithms/BatchMergeMRFh.cpp

g++ -o NeuroProof_stack -L$PREFIX/lib  -L$HOME/libDAI/libDAI-0.2.7/lib FeatureManager.o Features.o MergePriorityFunction.o vigraRFclassifier.o vigraRFclassifierP.o opencvRFclassifier.o opencvRFclassifierP.o compositeRFclassifier.o Stack.o StackPredict.o NeuroProof_stack.o  -lhdf5 -lboost_python -lpython2.7 -lpng -lvigraimpex -lhdf5_hl -lopencv_ml -lopencv_core -lboost_thread -lboost_system -lboost_chrono

# BatchMergeMRFh.o  -ldai

g++ -c -w -O3 -I$PREFIX/include  -I$PREFIX/include/vigra -I$PREFIX/include/python2.7 -I$PREFIX/include/boost -I$HOME/OpenCV/opencv-install/include/ FeatureManager/FeatureManager.cpp FeatureManager/Features.cpp DataStructures/Stack.cpp DataStructures/StackLearn.cpp Algorithms/MergePriorityFunction.cpp Classifier/vigraRFclassifier.cpp Classifier/vigraRFclassifierP.cpp Classifier/opencvRFclassifier.cpp Classifier/opencvABclassifier.cpp Classifier/opencvSVMclassifier.cpp  NeuroProof_stack_learn.cpp

g++ -o NeuroProof_stack_learn -L$PREFIX/lib -L$HOME/OpenCV/opencv-install/lib FeatureManager.o Features.o MergePriorityFunction.o vigraRFclassifier.o  vigraRFclassifierP.o opencvRFclassifier.o opencvABclassifier.o  opencvSVMclassifier.o  Stack.o StackLearn.o NeuroProof_stack_learn.o  -lhdf5 -lboost_python -lpython2.7 -lvigraimpex -lhdf5_hl -lopencv_ml -lopencv_core  -lboost_system -lboost_thread -lboost_chrono


rm -f *.o


### THIS IS A JALAPENO VERSION: THE FUNCTIONS FOR SMALL LABELS (absorb_small_regions) HAD TO BE TAKEN OUT TO MAKE IT WORK

### Toufiq Parag data
scp -r ~/workspace/Neuroproof_minimal/TP_data jalapeno.fmrib.ox.ac.uk:/vols/Data/km/michielk/workspace/Neuroproof_minimal/

./NeuroProof_stack_learn \
-watershed trn/250-1_chris_cc3_ws2.h5 stack \
-prediction trn/STACKED_prediction_chris_zyxc.h5 volume/predictions \
-groundtruth trn/250-1_groundtruth_orig.h5 stack \
-classifier classifier_itr2_chris.xml \
-iteration 2 -strategy 2

./NeuroProof_stack \
-watershed tst/250-2_cc3_ws_o3.h5 stack \
-prediction tst/STACKED_prediction.h5 volume/predictions \
-output 250-2_result_chris.h5 stack \
-classifier classifier_itr2_chris.xml \
-threshold 0.2 -algorithm 1


### Neuroproof examples data
scp -r ~/workspace/FlyEM/Neuroproof/examples jalapeno.fmrib.ox.ac.uk:/vols/Data/km/michielk/workspace/Neuroproof_minimal/

NPdir=/vols/Data/km/michielk/workspace/Neuroproof_minimal
datadir=$NPdir/examples

dataset=training_sample2
datavol=oversegmented_stack_labels
iter=2
strtype=2
$NPdir/NeuroProof_stack_learn \
-watershed $datadir/$dataset/$datavol.h5 stack \
-prediction $datadir/$dataset/boundary_prediction.h5 volume/predictions \
-groundtruth $datadir/$dataset/groundtruth.h5 stack \
-classifier $datadir/training_sample2/classifier_${datavol}_str${strtype}_iter${iter}_NPminimal.xml \
-iteration $iter -strategy ${strtype}

dataset=validation_sample
datavol=oversegmented_stack_labels
thr=0.2
alg=1
$NPdir/NeuroProof_stack \
-watershed $datadir/$dataset/$datavol.h5 stack \
-prediction $datadir/$dataset/boundary_prediction.h5 volume/predictions \
-output $datadir/$dataset/classifier_${datavol}_str${strtype}_iter${iter}_segmentation.h5 stack \
-classifier $datadir/training_sample2/classifier_${datavol}_str${strtype}_iter${iter}_NPminimal.xml \
-threshold $thr -algorithm $alg


rsync -avz jalapeno.fmrib.ox.ac.uk:/vols/Data/km/michielk/workspace/Neuroproof_minimal/examples ~/Neuroproof_examples
