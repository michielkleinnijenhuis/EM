ms1 = 1; 
ms2 = 51;

datadir = '~/workspace/DifSim/utilities/geoms/2DEM/M3_S1_GNU/0250_m000';
mvol1 = nii_zip([datadir filesep '0250_m000_seg.nii']);
datadir = '/Users/michielk/oxdata/P01/EM/M3/M3_S1_GNU/tifs/reg';
mvol2 = nii_zip([datadir filesep 'reg.nii']);

xoffset = 123;
yoffset = 61;

mvol2.img = int16(zeros(size(mvol2.img)));
mvol2.img(xoffset:xoffset + size(mvol1.img,1)-1,...
    yoffset:yoffset + size(mvol1.img,2)-1,ms2) = mvol1.img(:,:,ms1);
save_untouch_nii(mvol2, [datadir filesep 'reg_seg.nii']);


%%
ms1 = 1; 
ms2 = 51;

datadir = '~/workspace/DifSim/utilities/geoms/2DEM/M3_S1_GNU/0250_m000';
mvol1 = nii_zip([datadir filesep '0250_m000_seg.nii']);

datadir = '/Users/michielk/oxdata/P01/EM/M3/M3_S1_GNU/tifs/reg';
!fslroi /Users/michielk/oxdata/P01/EM/M3/M3_S1_GNU/tifs/reg/reg.nii.gz /Users/michielk/oxdata/P01/EM/M3/M3_S1_GNU/tifs/reg/reg_cutout00.nii.gz 122 500 60 500 0 -1
mvol2 = nii_zip([datadir filesep 'reg_cutout00.nii']); xstart = 123; ystart = 61;
!fslroi /Users/michielk/oxdata/P01/EM/M3/M3_S1_GNU/tifs/reg/reg.nii.gz /Users/michielk/oxdata/P01/EM/M3/M3_S1_GNU/tifs/reg/reg_cutout01.nii.gz 1000 500 1000 500 0 -1
mvol2 = nii_zip([datadir filesep 'reg_cutout01.nii']); xstart = 1001; ystart = 1001;
!fslroi /Users/michielk/oxdata/P01/EM/M3/M3_S1_GNU/tifs/reg/reg.nii.gz /Users/michielk/oxdata/P01/EM/M3/M3_S1_GNU/tifs/reg/reg_cutout02.nii.gz 1000 250 1000 250 0 -1
mvol2 = nii_zip([datadir filesep 'reg_cutout02.nii']); xstart = 1001; ystart = 1001;

xoffset = 123;
yoffset = 61;

xsize = size(mvol2.img,1);
ysize = size(mvol2.img,2);

seg = mvol1.img(xstart - xoffset + 1:xstart - xoffset + xsize,ystart - yoffset + 1:ystart - yoffset + ysize, ms1);

mvol1.img = int16(zeros(size(mvol2.img)));
mvol1.hdr.dime.dim(2:4) = size(mvol1.img);
mvol1.img(:,:,ms2) = seg;
save_untouch_nii(mvol1, [datadir filesep 'reg_cutout01_seg.nii']);

%%
ms1 = 1; 
ms2 = 51;

segdir = '~/workspace/DifSim/utilities/geoms/2DEM/M3_S1_GNU/0250_m000';
mvol1 = nii_zip([segdir filesep '0250_m000_seg.nii']);

datadir='/Users/michielk/oxdata/P01/EM/M3/M3_S1_GNU';
invol = 'm000_cutout01'; xstart = 1001; ystart = 1001;
h5disp([datadir filesep invol '.h5']);
data_info = h5info([datadir filesep invol '.h5']); % data_info.Datasets.Attributes.Value
data = h5read([datadir filesep invol '.h5'], '/stack');

xoffset = 123;
yoffset = 61;
xsize = size(data,1);
ysize = size(data,2);

seg = int16(zeros(size(data)));
seg(:,:,ms2) = mvol1.img(xstart - xoffset + 1:xstart - xoffset + xsize, ...
    ystart - yoffset + 1:ystart - yoffset + ysize, ms1);

h5create([datadir filesep invol '_seg.h5'], '/stack', size(seg), 'Datatype','int16');
h5write([datadir filesep invol '_seg.h5'], '/stack', seg);

av = h5readatt([datadir filesep invol '.h5'],'/stack','element_size_um');
h5writeatt([datadir filesep invol '_seg.h5'],'/stack','element_size_um', av);
% av = h5readatt([datadir filesep invol '.h5'],'/stack','DIMENSION_LABELS');
% h5writeatt([datadir filesep invol '_seg.h5'],'/stack','DIMENSION_LABELS', av);
% h5disp([datadir filesep invol '_seg.h5']);
