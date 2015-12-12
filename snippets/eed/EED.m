addpath(genpath('/Users/michielk/oxscripts/matlab/toolboxes/coherencefilter_version5b'));
addpath /Users/michielk/oxscripts/matlab/toolboxes/NIFTI_20090703

% cd ~/oxdata/P01/EM/NeuroMorph_Toolkit/NM_Samples/EM_stack
% inputvolume = 'EM_stack_OR.nii';
cd ~/oxdata/P01/EM/M2/J/crop2000
inputvolume = 'EM_stack0000-0499.nii';

Options.rho = 1;

system(['gunzip ' inputvolume '.gz']);
img = load_untouch_nii(inputvolume);
system(['gzip ' inputvolume]);
image = single(img.img);

% difference of gaussian filtering at different scales
u0 = imgaussian(image,0.5,2);
img.img = u0; save_untouch_nii(img, 'EM_stack_G00.nii');
u1 = imgaussian(image,1,4);
img.img = u1; save_untouch_nii(img, 'EM_stack_G01.nii');
u2 = imgaussian(image,2,8);
img.img = u2; save_untouch_nii(img, 'EM_stack_G02.nii');
u3 = imgaussian(image,5,20);
img.img = u3; save_untouch_nii(img, 'EM_stack_G03.nii');
u4 = imgaussian(image,10,40);
img.img = u4; save_untouch_nii(img, 'EM_stack_G04.nii');

img.img = u0-u1; save_untouch_nii(img, 'EM_stack_D01.nii');
img.img = u0-u2; save_untouch_nii(img, 'EM_stack_D02.nii');
img.img = u0-u3; save_untouch_nii(img, 'EM_stack_D03.nii');
img.img = u0-u4; save_untouch_nii(img, 'EM_stack_D04.nii');
img.img = u1-u2; save_untouch_nii(img, 'EM_stack_D12.nii');
img.img = u1-u3; save_untouch_nii(img, 'EM_stack_D13.nii');
img.img = u1-u4; save_untouch_nii(img, 'EM_stack_D14.nii');
img.img = u2-u3; save_untouch_nii(img, 'EM_stack_D23.nii');
img.img = u2-u4; save_untouch_nii(img, 'EM_stack_D24.nii');
img.img = u3-u4; save_untouch_nii(img, 'EM_stack_D34.nii');

clear u*

% Calculate the gradients
ux = derivatives(image,'x');
uy = derivatives(image,'y');
uz = derivatives(image,'z');

% Gradient magnitude squared
u = ux.^2+uy.^2+uz.^2;
img.img = u; save_untouch_nii(img, 'EM_stack_GRM.nii');

% hessian eigenvalues
[Jxx, Jxy, Jxz, Jyy, Jyz, Jzz] = StructureTensor3D(ux,uy,uz, Options.rho);
clear u*

[mu1,mu2,mu3,v3x,v3y,v3z,v2x,v2y,v2z,v1x,v1y,v1z]=EigenVectors3D(Jxx, Jxy, Jxz, Jyy, Jyz, Jzz);
clear J* v*

img.img = mu1; save_untouch_nii(img, 'EM_stack_HE1.nii');
img.img = mu2; save_untouch_nii(img, 'EM_stack_HE2.nii');
img.img = mu3; save_untouch_nii(img, 'EM_stack_HE3.nii');

u = CoherenceFilter(img.img, struct('T',10,'dt',0.1,'rho',1,'Scheme','R','eigenmode',2,'verbose','full'));
img.img = u; save_untouch_nii(img, 'EM_stack_STR.nii');

