% data = h5read('test_data_segmentation.h5', '/stack');
% 
% 
% for label = unique(data)
%     
%     h5create(['test_data_segmentation_' num2str(label, '%06d')], '/stack', [200,200,50], 'Datatype', 'int8');
%     h5write(['test_data_segmentation_' num2str(label, '%06d')], '/stack', data);
% end

segm = h5read('training_data_Simple Segmentation.h5', '/stack');
segm = squeeze(segm - 1);
slic = h5read('training_data_slicvoxels.h5', '/stack');
slicsegm = int8(zeros(size(slic)));
svoxlabels = unique(slic);

for svox = 1:length(svoxlabels)
    svox_idx = find(slic==svoxlabels(svox));
    if length(nonzeros(segm(svox_idx))) > floor(length(svox_idx) / 4)
        slicsegm(svox_idx) = 1;
    end
    disp(svox)
end

h5create('training_data_slicsegmentation.h5', '/stack', size(slic), 'Datatype', 'int8');
h5write('training_data_slicsegmentation.h5', '/stack', slicsegm);

%%

cd ~/oxdata/P01/EM/M2/I/
slicsegm = h5read('training_data_slicsegmentation.h5', '/stack');
slicsegm = imcomplement(logical(slicsegm));
% labels = bwlabel(slicsegm, n)
cc = bwconncomp(slicsegm, 6);
labels = labelmatrix(cc);

h5create('training_data_slicsegmentation_labels.h5', '/stack', size(slicsegm), 'Datatype', 'int8');
h5write('training_data_slicsegmentation_labels.h5', '/stack', labels);

