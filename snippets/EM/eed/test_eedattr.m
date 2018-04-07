datadir = '/Users/michielk/oxdata/P01/EM/Myrf_01/SET-B/B-NT-S10-2f_ROI_00/old';
dataset = 'B-NT-S10-2f_ROI_00ds7';
dpf = '_probs';
infile = [dataset, dpf];
infield = '/volume/predictions';
outfile = [dataset, dpf, 'foo'];
outfield = '/probs_eed_foo';
vol_idxs=0;
T=1;
dt=1;
rho=1;

%%
addpath('/Users/michielk/workspace/EM/wmem')
datadir = '/Users/michielk/oxdata/P01/EM/Myrf_01/SET-B/B-NT-S10-2f_ROI_00/blocks_0500';
dataset = 'B-NT-S10-2f_ROI_00_00480-01020_00480-01020_00000-00184';

dpf = '_probs';
invol = [dataset, dpf];
outvol = [dataset, dpf, '_bar'];

ds_in = '/volume/predictions';
ds_out = '/sum0247_eed';

EM_eed_simple(datadir, invol, ds_in, outvol, ds_out, 1, 1, 1, 1);


%%
% ATTVAL = h5readatt(inputfile, infield, 'element_size_um')
% ATTVAL = h5readatt(inputfile, infield, 'DIMENSION_LABELS')
% h5writeatt(FILENAME,LOCATION,ATTNAME,ATTVALUE)
% h5writeatt(outputfile, outfield, stackinfo.Attributes(al_idx).Name, ATTVAL)




fid = H5F.open(inputfile);
gid = H5G.open(fid, '/');
did = H5D.open(gid, infield);
aid = H5A.open(did, 'DIMENSION_LABELS');

tid = H5A.get_type(aid);
sid = H5A.get_space(aid);
data = H5A.read(aid);

H5A.close(aid);
H5D.close(did);
H5G.close(gid);
H5F.close(fid);



fid = H5F.open(outputfile,'H5F_ACC_RDWR','H5P_DEFAULT');
gid = H5G.open(fid, '/');
did = H5D.open(gid, outfield);

pid = H5P.create('H5P_ATTRIBUTE_CREATE');
aid = H5A.create(did, 'DIMENSION_LABELS', tid, sid, pid);
H5A.write(aid, tid, data)

H5A.close(aid);
H5D.close(did);
H5G.close(gid);
H5F.close(fid);







dsetName = 'DIMENSION_LABELS';
dataDims = [4 1];
h5DataDims = fliplr(dataDims);
h5MaxDims = h5DataDims;
spaceID = H5S.create_simple(2,h5DataDims,h5MaxDims);
dsetID = H5D.create(grpID,dsetName,typeID,spaceID,...
             'H5P_DEFAULT','H5P_DEFAULT','H5P_DEFAULT');
H5A.write(dsetID,'H5ML_DEFAULT','H5S_ALL',...
               'H5S_ALL','H5P_DEFAULT',dataToWrite);



dataToWrite = {char([12487 12540 12479]) 'hello' ...
                   char([1605 1585 1581 1576 1575]); ...
               'world' char([1052 1080 1088])    ...
                   char([954 972 963 956 959 962])};
disp(dataToWrite)

fileName = 'outfile.h5';
fileID = H5F.create(fileName,'H5F_ACC_TRUNC',...
                     'H5P_DEFAULT', 'H5P_DEFAULT');


lcplID = H5P.create('H5P_LINK_CREATE'); 
H5P.set_char_encoding(lcplID,H5ML.get_constant_value('H5T_CSET_UTF8'));
plist = 'H5P_DEFAULT';


grpName = char([12464 12523 12540 12503]);
grpID = H5G.create(fileID,grpName,lcplID,plist,plist);

typeID = H5T.copy('H5T_C_S1');
H5T.set_size(typeID,'H5T_VARIABLE');
H5T.set_cset(typeID,H5ML.get_constant_value('H5T_CSET_UTF8'));

dsetName = 'datasetUtf8';
dataDims = [2 3];
h5DataDims = fliplr(dataDims);
h5MaxDims = h5DataDims;
spaceID = H5S.create_simple(2,h5DataDims,h5MaxDims);
dsetID = H5D.create(grpID,dsetName,typeID,spaceID,...
             'H5P_DEFAULT','H5P_DEFAULT','H5P_DEFAULT');

H5D.write(dsetID,'H5ML_DEFAULT','H5S_ALL',...
               'H5S_ALL','H5P_DEFAULT',dataToWrite);

dataRead = h5read('outfile.h5',['/' grpName '/' dsetName])

isequal(dataRead,dataToWrite)

H5D.close(dsetID);
H5S.close(spaceID);
H5T.close(typeID);
H5G.close(grpID);
H5P.close(lcplID);
H5F.close(fileID);

