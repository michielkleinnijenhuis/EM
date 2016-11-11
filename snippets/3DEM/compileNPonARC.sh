rsync -avz /Users/michielk/workspace/Neuroproof_minimal.zip ndcn0180@arcus.arc.ox.ac.uk:/home/ndcn-fmrib-water-brain/ndcn0180/workspace
cd ~/workspace
unzip Neuroproof_minimal.zip
cd Neuroproof_minimal
PREFIX=$(conda info --root)/envs/neuroproof-test
export LD_LIBRARY_PATH=${PREFIX}/lib

g++ -c -w -O3 -I$PREFIX/include  -I$PREFIX/include/vigra -I$PREFIX/include/python2.7 -I$PREFIX/include/boost  FeatureManager/FeatureManager.cpp FeatureManager/Features.cpp DataStructures/Stack.cpp DataStructures/StackPredict.cpp Algorithms/MergePriorityFunction.cpp Classifier/vigraRFclassifier.cpp Classifier/vigraRFclassifierP.cpp Classifier/opencvRFclassifier.cpp Classifier/opencvRFclassifierP.cpp Classifier/compositeRFclassifier.cpp NeuroProof_stack.cpp
# Algorithms/BatchMergeMRFh.cpp

g++ -o NeuroProof_stack -L$PREFIX/lib  -L$HOME/libDAI/libDAI-0.2.7/lib FeatureManager.o Features.o MergePriorityFunction.o vigraRFclassifier.o vigraRFclassifierP.o opencvRFclassifier.o opencvRFclassifierP.o compositeRFclassifier.o Stack.o StackPredict.o NeuroProof_stack.o  -lhdf5 -lboost_python -lpython2.7 -lpng -lvigraimpex -lhdf5_hl -lopencv_ml -lopencv_core -lboost_thread -lboost_system -lboost_chrono
# BatchMergeMRFh.o  -ldai

g++ -c -w -O3 -I$PREFIX/include  -I$PREFIX/include/vigra -I$PREFIX/include/python2.7 -I$PREFIX/include/boost -I$HOME/OpenCV/opencv-install/include/ FeatureManager/FeatureManager.cpp FeatureManager/Features.cpp DataStructures/Stack.cpp DataStructures/StackLearn.cpp Algorithms/MergePriorityFunction.cpp Classifier/vigraRFclassifier.cpp Classifier/vigraRFclassifierP.cpp Classifier/opencvRFclassifier.cpp Classifier/opencvABclassifier.cpp Classifier/opencvSVMclassifier.cpp  NeuroProof_stack_learn.cpp

g++ -o NeuroProof_stack_learn -L$PREFIX/lib -L$HOME/OpenCV/opencv-install/lib FeatureManager.o Features.o MergePriorityFunction.o vigraRFclassifier.o  vigraRFclassifierP.o opencvRFclassifier.o opencvABclassifier.o  opencvSVMclassifier.o  Stack.o StackLearn.o NeuroProof_stack_learn.o  -lhdf5 -lboost_python -lpython2.7 -lvigraimpex -lhdf5_hl -lopencv_ml -lopencv_core  -lboost_system -lboost_thread -lboost_chrono

rm -f *.o


### testdata Toufiq Parag
scp -r ~/workspace/Neuroproof_minimal/TP_data ndcn0180@arcus-b.arc.ox.ac.uk:/home/ndcn-fmrib-water-brain/ndcn0180/workspace/Neuroproof_minimal

NPdir=/home/ndcn-fmrib-water-brain/ndcn0180/workspace/Neuroproof_minimal
CONDA_PATH=$(conda info --root)
PREFIX=${CONDA_PATH}/envs/neuroproof-test
datadir=/home/ndcn-fmrib-water-brain/ndcn0180/workspace/Neuroproof_minimal
qsubfile=$datadir/submit.sh
echo '#!/bin/bash' > $qsubfile
echo "#SBATCH --nodes=1" >> $qsubfile
echo "#SBATCH --ntasks-per-node=1" >> $qsubfile
echo "#SBATCH --time=00:10:00" >> $qsubfile
echo "#SBATCH --job-name=NPlearn" >> $qsubfile
echo "export PATH=${CONDA_PATH}/bin:\$PATH" >> $qsubfile
# echo "source activate neuroproof-test" >> $qsubfile
echo "export LD_LIBRARY_PATH=${PREFIX}/lib" >> $qsubfile
echo "$NPdir/NeuroProof_stack_learn \
-watershed $datadir/trn/250-1_chris_cc3_ws2.h5 stack \
-prediction $datadir/trn/STACKED_prediction_chris_zyxc.h5 volume/predictions \
-groundtruth $datadir/trn/250-1_groundtruth_orig.h5 stack \
-classifier $datadir/classifier_itr2_chris.xml \
-iteration 2 -strategy 2" >> $qsubfile
sbatch -p devel $qsubfile

qsubfile=$datadir/submit.sh
echo '#!/bin/bash' > $qsubfile
echo "#SBATCH --nodes=1" >> $qsubfile
echo "#SBATCH --ntasks-per-node=1" >> $qsubfile
echo "#SBATCH --time=00:10:00" >> $qsubfile
echo "#SBATCH --job-name=NPtest" >> $qsubfile
echo "export PATH=/data/ndcn-fmrib-water-brain/ndcn0180/anaconda2/bin:\$PATH" >> $qsubfile
# echo "source activate neuroproof-test" >> $qsubfile
echo "export LD_LIBRARY_PATH=${PREFIX}/lib" >> $qsubfile
echo "$datadir/NeuroProof_stack \
-watershed $NPdir/tst/250-2_cc3_ws_o3.h5 stack \
-prediction $datadir/tst/STACKED_prediction.h5 volume/predictions \
-output $datadir/250-2_result_chris.h5 stack \
-classifier $datadir/classifier_itr2_chris.xml \
-threshold 0.2 -algorithm 1" >> $qsubfile
sbatch -p devel $qsubfile
