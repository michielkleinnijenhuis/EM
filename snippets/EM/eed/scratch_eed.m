addpath(genpath('~/workspace/EM/snippets/eed'));
addpath(genpath('~/oxscripts/matlab/toolboxes/coherencefilter_version5b'));

datadir = '/Users/michielk/oxdata/P01/EM/scratch_wmem_package/ds7_arc';
invol = 'M3S1GNUds7';
infield = '/stack';

stackinfo = h5info([datadir filesep invol '.h5'], infield);

blockfield = '/ROIsets/blocks';
blockinfo = h5info([datadir filesep invol '.h5'], blockfield);

for block = 1:blockinfo.Dataspace.Size
    
end

h5disp([datadir filesep invol '.h5'], blockfield);
blocks = h5read([datadir filesep invol '.h5'], blockfield);

block3 = h5read([datadir filesep invol '.h5'], blockfield, 3, 1);
% data = h5read('example.h5','/g3/reference')




datadir = '/Users/michielk/oxdata/P01/EM/scratch_wmem_package';
invol = 'test';
ds_in = '/stack';
ds_out = '/eed2';

filepath = [datadir filesep invol '.h5'];
stackinfo = h5info(filepath, ds_in);

h5create(filepath, ds_out, stackinfo.Dataspace.Size, ...
    'Deflate', 4, 'Chunksize', stackinfo.ChunkSize);

u = EM_eed(datadir, invol, ds_in, ds_out, 0, 5, 5, 5, 1, 50, 100, 200, 10, 10, 10);


EM_eed(datadir, invol, ds_in, ds_out, ...
    start_c, start_z, start_y, start_x, ...
    count_c, count_z, count_y, count_x, ...
    margin_z, margin_y, margin_x)