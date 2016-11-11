from os import path

fibrepaths = ['/Users/michielk/oxdata/P01/EM/M3/M3_S1_GNU/dmcsurf_PA/NN.blend',
              '/Users/michielk/oxdata/P01/EM/M3/M3_S1_GNU/dmcsurf_PA/GP.blend']
for fibre in fibrepaths:
    compname = path.basename(fibre).split('.')[0]
    with D.libraries.load(fibre) as (data_from, data_to):
        data_to.objects = [obj for obj in data_from.objects
                           if obj.startswith(compname)]
    for obj in data_to.objects:
        if obj is not None:
            obj.name = obj.name.replace(compname, 'UA')
            C.scene.objects.link(obj)
