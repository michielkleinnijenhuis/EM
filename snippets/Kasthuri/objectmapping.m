% importdata mmc2.xls

cd('/Users/michielk/oxdata/P01/EM/Kashturi11')

axons = unique(AxonNo);
axons(isnan(axons)) = [];
fid = fopen('axons.txt', 'w');
fprintf(fid, '%d\n', axons);
fclose(fid);

MAlog = Axontype==2;  % is this the axon or the sheet?
MA = AxonNo(MAlog);
MA = unique(MA);
MA = [MA 2064];  % MM/MA: 2064/4247
fid = fopen('MA.txt', 'w');
fprintf(fid, '%d\n', MA);
fclose(fid);


dendrites = unique(DendriteNo);
dendrites(isnan(dendrites)) = [];
fid = fopen('dendrites.txt', 'w');
fprintf(fid, '%d\n', dendrites);
fclose(fid);
