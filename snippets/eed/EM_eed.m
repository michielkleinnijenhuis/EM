function u = EM_eed(datadir, invol, ds_in, ds_out, ...
    start_c, start_z, start_y, start_x, ...
    count_c, count_z, count_y, count_x, ...
    margin_z, margin_y, margin_x)

if isdeployed
    % make sure numeric args are numeric
    start_c = str2double(start_c);
    start_z = str2double(start_z);
    start_y = str2double(start_y);
    start_x = str2double(start_x);
    count_c = str2double(count_c);
    count_z = str2double(count_z);
    count_y = str2double(count_y);
    count_x = str2double(count_x);
    margin_z = str2double(margin_z);
    margin_y = str2double(margin_y);
    margin_x = str2double(margin_x);
else
    cf_path = '~/oxscripts/matlab/toolboxes/coherencefilter_version5b';
    addpath(genpath(cf_path));
end

filepath = [datadir filesep invol '.h5'];
stackinfo = h5info(filepath, ds_in);

margin_z_lower = min([margin_z, start_z - 1])
margin_y_lower = min([margin_y, start_y - 1])
margin_x_lower = min([margin_x, start_x - 1])
start = [start_z - margin_z_lower, ...
         start_y - margin_y_lower, ...
         start_x - margin_x_lower]

margin_z_upper = min([...
    margin_z_lower + start_z + count_z + margin_z, ...
    stackinfo.Dataspace.Size(1)])
margin_y_upper = min([...
    margin_y_lower + start_y + count_y + margin_y, ...
    stackinfo.Dataspace.Size(2)])
margin_x_upper = min([...
    margin_x_lower + start_x + count_x + margin_x, ...
    stackinfo.Dataspace.Size(3)])
count = [...
    margin_z_lower + count_z + margin_z_upper, ...
    margin_y_lower + count_y + margin_y_upper, ...
    margin_x_lower + count_x + margin_x_upper]

if start_c == 0
    data = h5read(filepath, ds_in, start, count);
else
    start = [start_c, start_z, start_y, start_x];
    count = [count_c, count_z, count_y, count_x];
    data = h5read(filepath, ds_in, start, count);
    data = squeeze(data(1,:,:,:));
end

u = CoherenceFilter(data, ...
    struct('T', 50, 'dt', 1, 'rho', 1, ...
    'Scheme', 'R', 'eigenmode', 2, 'verbose', 'full'));

% fname = [datadir filesep invol num2str(start_c - 1) '_eed2.h5'];
% h5create(fname, outfield, size(u), 'Deflate', 4, ...
%     'Chunksize', stackinfo.ChunkSize(2:4));

u = u(margin_z + 1:count_z, ...
      margin_y + 1:count_y, ...
      margin_x + 1:count_x);

h5write(filepath, ds_out, u, ...
    [start_c, start_z, start_y, start_x], ...
    [count_c, count_z, count_y, count_x]);
