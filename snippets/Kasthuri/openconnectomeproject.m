function img = openconnectomeproject(dataset, xmin, xmax, ymin, ymax, zmin, zmax)

%%% get a cube of EM data from openconnectomeproject.org
%
%
%
%%% by Michiel Kleinnijenhuis (michiel.kleinnijenhuis@ndcn.ox.ac.uk)

dataset = 'kasthuri11';
xmin = 4000;
xmax = 5000;
ymin = 4000;
ymax = 5000;
zmin = 1000;
zmax = 2000;
xmi = num2str(xmin);
xma = num2str(xmax);
ymi = num2str(ymin);
yma = num2str(ymax);
zmi = num2str(zmin);
zma = num2str(zmax);

filename = [dataset '_' ...
	xmi '-' xma '_' ...
	ymi '-' yma '_' ...
	zmi '-' zma '.hdf5'];

urlwrite (['http://openconnecto.me/emca/' dataset '/hdf5/1/' ...
	xmi ',' xma '/' ...
	ymi ',' yma '/' ...
	zmi ',' zma '/'], filename);

img = h5read(filename, '/cube');
