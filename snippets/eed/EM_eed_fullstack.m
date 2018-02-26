function ul = EM_eed(datadir, invol, ds_in, ds_out, ...
    start_z, start_y, start_x, start_c, ...
    count_z, count_y, count_x, count_c, ...
    margin_z, margin_y, margin_x)

if isdeployed
    % make sure numeric args are numeric
    start_z = str2double(start_z);
    start_y = str2double(start_y);
    start_x = str2double(start_x);
    start_c = str2double(start_c);
    count_z = str2double(count_z);
    count_y = str2double(count_y);
    count_x = str2double(count_x);
    count_c = str2double(count_c);
    margin_z = str2double(margin_z);
    margin_y = str2double(margin_y);
    margin_x = str2double(margin_x);
else
    cf_path = '~/oxscripts/matlab/toolboxes/coherencefilter_version5b';
    addpath(genpath(cf_path));
end

filepath = [datadir filesep invol '.h5'];
stackinfo = h5info(filepath, ds_in);
% NOTE: ilastik zyxc is transposed to cxyz;
% NOTE: axistags not read properly;

% create outputfile
outpath = [datadir filesep invol '_eed.h5'];
try
h5create(outpath, ds_out, stackinfo.Dataspace.Size, ...
    'Deflate', 4, ...
    'Chunksize', stackinfo.ChunkSize);
catch
end

% determine start indices
margin_z_lower = min([margin_z, start_z - 1]);
margin_y_lower = min([margin_y, start_y - 1]);
margin_x_lower = min([margin_x, start_x - 1]);
start = [start_z - margin_z_lower, ...
         start_y - margin_y_lower, ...
         start_x - margin_x_lower];

% determine counts
count = [...
    margin_z_lower + count_z + margin_z, ...
    margin_y_lower + count_y + margin_y, ...
    margin_x_lower + count_x + margin_x];
count(1) = min([count(1), stackinfo.Dataspace.Size(4) - start_z + 1]);
count(2) = min([count(2), stackinfo.Dataspace.Size(3) - start_y + 1]);
count(3) = min([count(3), stackinfo.Dataspace.Size(2) - start_x + 1]);

% read data
start = [start_c, start(3), start(2), start(1)];
count = [count_c, count(3), count(2), count(1)];
data = h5read(filepath, ds_in, start, count);

% calculate EED
for c = start_c : start_c + count_c - 1
    u = CoherenceFilter(squeeze(data(c,:,:,:)), ...
        struct('T', 50, 'dt', 1, 'rho', 1, ...
        'Scheme', 'R', 'eigenmode', 2, 'verbose', 'full'));
    data(c,:,:,:) = u(margin_x_lower + 1 : count_x + margin_x_lower, ...
          margin_y_lower + 1 : count_y + margin_y_lower, ...
          margin_z_lower + 1 : count_z + margin_z_lower);
end

% write to file
start_out = [start_c, start_x, start_y, start_z];
count_out = [count_c, count_x, count_y, count_z];
h5write(outpath, ds_out, data, start_out, count_out);
