import h5py
import healpy as hp
import numpy as np
from pathlib import Path
import os
import matplotlib

matplotlib.use('Agg')


def map_proj(input_maps, target_maps, rot=[0.0, 90.0, 0.0], **kwargs):
    n_maps = input_maps.shape[0]
    proj_m0 = hp.azeqview(map=input_maps[0, 0], return_projected_map=True, **kwargs)
    fig_size = proj_m0.shape
    
    input_map_cut = np.zeros((n_maps, ) + fig_size, dtype=np.float64)
    target_map_cut = np.zeros((n_maps, ) + fig_size, dtype=np.float64)

    for i in range(n_maps):
        if i % 20 == 0:
            print('map No.', i)
        input_map_cut[i] = azimuthal_projmap(input_maps[i, 0], rot=rot)
        target_map_cut[i] = azimuthal_projmap(target_maps[i, 0], rot=rot)

    return input_map_cut, target_map_cut


def azimuthal_projmap(map_, rot=None, coord=None, nest=False, **kwargs):
    nside = hp.npix2nside(map_.shape[-1])
    f = lambda x, y, z: hp.pixelfunc.vec2pix(nside, x, y, z, nest=nest)
    azproj = hp.projector.AzimuthalProj(rot=rot, coord=coord, **kwargs)

    return azproj.projmap(map_, f)


if __name__ == '__main__':
    input_maps_dir = '/public/home/wufq/sfzuo/workspace/dish_simNP_750MHz/ps_map/'
    re_maps_dir = '/public/home/wufq/sfzuo/workspace/dish_simNP_750MHz/output_sim_750MHz/map'
    input_path = Path(input_maps_dir)
    re_path = Path(re_maps_dir)
    
    # output_path = Path(target_maps_dir)

    input_file  = sorted(input_path.glob('*.hdf5'))
    n_file = len(input_file)
    re_file = [re_path.joinpath("ts_750_gen_data_%s" % i, 'ts', 'map_full.hdf5') for i in range(n_file)]

    input_map_cut = []
    re_map_cut = []


    for i in range(n_file):
        print('file No.', i)
        input_map_i = h5py.File(input_file[i], 'r')['map']
        re_map_i = h5py.File(re_file[i], 'r')['map']

        input_map_cut_i, re_map_cut_i = map_proj(input_map_i, re_map_i, rot=[0.0, 90.0, 0.0])

        input_map_cut.append(input_map_cut_i)
        re_map_cut.append(re_map_cut_i)

    
    input_map_cut = np.concatenate(input_map_cut, axis=0)
    re_map_cut = np.concatenate(re_map_cut, axis=0)

    with h5py.File('train_data.hdf5', 'w') as f:
        f.create_dataset('input', data=input_map_cut)
        f.create_dataset('reconstruction', data=re_map_cut)
