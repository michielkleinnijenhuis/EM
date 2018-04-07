datadir = '/Users/michielk/oxdata/P01/EM/M3/M3_S1_GNU/M3S1GNU/ds7_arc/';
datadir = '/Users/michielk/oxdata/P01/EM/M3/M3_S1_GNU/M3S1GNU/ds7_arc/stats';
fs = 18;

comp = {'MF'; 'MA'; 'UA'};
names = {'myelinated fibre'; 'myelinated axon'; 'unmyelinated axon'};
xl = [0 2];
nbins = [200, 200, 200];
for i = 1:3
    slcws_AD = dlmread([datadir filesep 'stats_' comp{i} '_slcws_AD.txt']);
%     slcws_AD = dlmread([datadir filesep 'stats_' comp{i} '_slcws_eccentricity.txt']);
%     slcws_AD = dlmread([datadir filesep 'stats_' comp{i} '_slcws_solidity.txt']);
    AD = slcws_AD(~isnan(slcws_AD));
    AD(AD<1.8) = [];  %0.0999 %0.1153
    AD = AD * 0.0511;
    subplot(2,3,i); histfit(AD, nbins(i), 'gamma')
    set(gca,'fontsize', fs); xlim(xl);
    title([names{i} ' diameter'], 'fontsize', fs);
    axis square;
    mean(AD(:), 'omitnan')
    std(AD(:), 'omitnan')
    p = gamfit(AD(:))
end

slcws_AD = dlmread([datadir filesep 'stats_MF_slcws_AD.txt']);
met = slcws_AD*0.0511; xl = [0, 0.2]; name = 'axon diameter';
sd_ax = var(met, '', 1, 'omitnan');
rp = met(randperm(numel(met)));
rp = reshape(rp, size(met));
sd_rp = var(rp, '', 1, 'omitnan');

nbins = 250;
subplot(2,4,6); hist(sd_ax(~isnan(sd_ax)), nbins);
xlim(xl);
set(gca,'fontsize', fs); axis square;
title([name ' variance within axon'], 'fontsize', fs);
subplot(2,4,7); hist(sd_rp(~isnan(sd_rp)), nbins);
xlim(xl); set(gca,'fontsize', fs); axis square;
title([name ' variance random permutation'], 'fontsize', fs);


% gratios calculated from areas or read from file
calcg = 0;
if calcg
    xl = [0 1];
    areasMM = dlmread([datadir filesep 'stats_MF_slcws_area.txt']);
    areaMM = sum(areasMM, 'omitnan');
    areasMA = dlmread([datadir filesep 'stats_MA_slcws_area.txt']);
    areaMA = sum(areasMA, 'omitnan');
    areaML = areaMM - areaMA; plot([areaMA; areaMM; areaML]')
    gratios = sqrt(1 - areaML ./ areaMM);
    for i = 1:430
        for j = 1:2281
            areaML = areasMM(i,j) - areasMA(i,j);
            gratios_slcws(i,j) = sqrt(1 - areaML / areasMM(i, j));
        end
    end
else
    gratios = dlmread([datadir filesep 'stats_gratios.txt']);
    gratios_slcws = dlmread([datadir filesep 'stats_gratios_slcws.txt']);
end
xl = [0 1];
subplot(2,3,1); hist(gratios(~isnan(gratios)), 50)
set(gca,'fontsize', fs); xlim(xl); axis square;
title('g-ratio', 'fontsize', fs);
GR = gratios_slcws(~isnan(gratios_slcws));
subplot(2,3,2); histfit(GR, 200, 'gamma')
set(gca,'fontsize', fs); xlim(xl); axis square;
title('slicewise g-ratio', 'fontsize', fs);
mean(GR(:), 'omitnan')
std(GR(:), 'omitnan')
p = gamfit(GR(:))


% within vs. between variance of g-ratio
met = gratios_slcws; xl = [0,0.03]; name = 'g-ratio';

sd_ax = var(met, '', 1, 'omitnan');
rp = met(randperm(numel(met)));
rp = reshape(rp, size(met));
sd_rp = var(rp, '', 1, 'omitnan');

nbins = 250;
subplot(2,4,6); hist(sd_ax(~isnan(sd_ax)), nbins);
xlim(xl); set(gca,'fontsize', fs); axis square;
title([name ' variance within axon'], 'fontsize', fs);
subplot(2,4,7); hist(sd_rp(~isnan(sd_rp)), nbins);
xlim(xl); set(gca,'fontsize', fs); axis square;
title([name ' variance random permutation'], 'fontsize', fs);


% figure, plot(gratios_slcws', slcws_AD'*0.0511, 'Linestyle', 'none', 'Marker', '.' )
% axis square

slcws_AD = dlmread([datadir filesep 'stats_MA_slcws_AD.txt']);
meanAD = mean(slcws_AD*0.0511, 1);
subplot(2,3,1); plot(meanAD', gratios, 'Linestyle', 'none', 'Marker', 'o')
set(gca,'fontsize', fs); axis square;

% slcws_CX = dlmread([datadir filesep 'slcws_centroid_x.txt']);
% slcws_CY = dlmread([datadir filesep 'slcws_centroid_y.txt']);
% skel = cat(3, slcws_CX', slcws_CY');
% plot(squeeze(skel(4,:,1)), squeeze(skel(4,:,2)));
% 

mean(slcws_AD(:)*0.0511, 'omitnan')     ; % 0.1827

% r = sqrt(A/pi);
for a=1:4
d = 2*sqrt(a/pi)
end
* 0.0511

r = slcws_AD;
hist(r(:), 100)
mean(r(:)*0.0511, 'omitnan'); % 0.1827
histfit(r(:), 100, 'gamma')
p = gamfit(r(:)); % 2.0822    0.0878

