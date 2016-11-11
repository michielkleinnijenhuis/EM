cd ~/workspace
git clone https://github.com/slash-segmentation/cytoseg.git

python run_pipeline_test.py /home/rgiuly/images/fua/testing output \
--trainingImage=/home/rgiuly/images/fua/training \
--trainingSeg=/home/rgiuly/images/fua/training_groundtruth \
--voxelTrainingLowerBound=*,*,* \
--voxelTrainingUpperBound=*,*,* \
--voxelProcessingLowerBound=*,*,* \
--voxelProcessingUpperBound=*,*,* \
--contourTrainingLowerBound=*,*,* \
--contourTrainingUpperBound=*,*,* \
--contourProcessingLowerBound=*,*,* \
--contourProcessingUpperBound=*,*,* \
--accuracyCalcLowerBound=*,*,* \
--accuracyCalcUpperBound=*,*,* \
--labelConfigFile=settings2.py \
--voxelWeights=0.0026,0.000128 \
--contourListWeights=7,1 \
--contourListThreshold=0.5 \
--step1 --step2 --step3
