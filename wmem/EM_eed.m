function ul = EM_eed(datadir, invol, ds_in, ds_out, ...
    start_z, start_y, start_x, start_c, ...
    count_z, count_y, count_x, count_c, ...
    margin_z, margin_y, margin_x, T, dt)

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
    T = str2double(T);
    dt = str2double(dt);
else
    cf_path = '~/oxscripts/matlab/toolboxes/coherencefilter_version5b';
    addpath(genpath(cf_path));
end

% NOTE: ilastik zyxc is transposed to cxyz;
% NOTE: axistags not read properly;
filepath = [datadir filesep invol '.h5'];
stackinfo = h5info(filepath, ds_in);
dsize = stackinfo.Dataspace.Size(end-2:end);
if length(stackinfo.Dataspace.Size) == 3
    start_c = 1;
    count_c = 1;
end

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
count(1) = min([count(1), dsize(3) - start_z + 1]);
count(2) = min([count(2), dsize(2) - start_y + 1]);
count(3) = min([count(3), dsize(1) - start_x + 1]);


start = [start_c, start(3), start(2), start(1)];
count = [count_c, count(3), count(2), count(1)];
start_out = [start_c, start_x, start_y, start_z];
count_out = [count_c, count_x, count_y, count_z];
ul = zeros([count_c, count_x, count_y, count_z]);
if length(stackinfo.Dataspace.Size) == 3
    start(1) = [];
    count(1) = [];
    start_out(1) = [];
    count_out(1) = [];
end

% read data
data = h5read(filepath, ds_in, start, count);
if length(stackinfo.Dataspace.Size) == 3
    data = permute(data, [4, 1, 2, 3]);
end

% calculate EED
for c = start_c : start_c + count_c - 1
    u = CoherenceFilter(squeeze(data(c,:,:,:)), ...
        struct('T', T, 'dt', dt, 'rho', 1, ...
        'Scheme', 'R', 'eigenmode', 2, 'verbose', 'full'));
    lower = [margin_x_lower + 1, margin_y_lower + 1, margin_z_lower + 1];
    upper = [margin_x_lower + count_x, ...
             margin_y_lower + count_y, ...
             margin_z_lower + count_z];
    upper(1) = min([upper(1), size(u, 1)]);
    upper(2) = min([upper(2), size(u, 2)]);
    upper(3) = min([upper(3), size(u, 3)]);
    ul(c,:,:,:) = u(lower(1):upper(1), lower(2):upper(2), lower(3):upper(3));
end

% write to file
h5write(outpath, ds_out, squeeze(ul), start_out, count_out);
