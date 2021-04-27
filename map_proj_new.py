import os
import h5py
import healpy as hp
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
try:
    # for new version matplotlib
    import matplotlib.style as mstyle
    mstyle.use('classic')
except ImportError:
    pass
import matplotlib.pyplot as plt


def map_proj(input_maps, target_maps, rot=[0.0, 90.0, 0.0], xsize=800, reso=1.5, **kwargs):
    n_maps = input_maps.shape[0]

    input_map_cut = np.zeros((n_maps, xsize, xsize), dtype=np.float64)
    target_map_cut = np.zeros((n_maps, xsize, xsize), dtype=np.float64)

    for i in range(n_maps):
        if i % 20 == 0:
            print('map No.', i)
        input_map_cut[i] = azimuthal_projmap(input_maps[i, 0], rot=rot, xsize=xsize, reso=reso)
        target_map_cut[i] = azimuthal_projmap(target_maps[i, 0], rot=rot, xsize=xsize, reso=reso)

    return input_map_cut, target_map_cut


def azimuthal_projmap(map_, rot=None, coord=None, xsize=800, reso=1.5, nest=False, **kwargs):
    nside = hp.npix2nside(map_.shape[-1])
    f = lambda x, y, z: hp.pixelfunc.vec2pix(nside, x, y, z, nest=nest)
    azproj = hp.projector.AzimuthalProj(rot=rot, coord=coord, xsize=xsize, reso=reso, **kwargs)

    return azproj.projmap(map_, f)


if __name__ == '__main__':
    # input_maps_dir1 = '/public/home/wufq/sfzuo/workspace/dish_simNP_750MHz/ps_map/'
    input_maps_dir2 = '/public/home/wufq/sfzuo/workspace/dish_simNP_750MHz/750_maps/'
    re_maps_dir = '/public/home/wufq/sfzuo/workspace/dish_simNP_750MHz/output_sim_750MHz_2/map'
    # input_path1 = Path(input_maps_dir1)
    input_path2 = Path(input_maps_dir2)
    re_path = Path(re_maps_dir)

    # output_path = Path(target_maps_dir)

    # input_file1  = sorted(input_path1.glob('ps_*.hdf5'))
    input_file2  = sorted(input_path2.glob('21cm_*.hdf5'))
    input_file3  = sorted(input_path2.glob('fg_*.hdf5'))
    n_file = len(input_file2)
    re_file = [re_path.joinpath("ts_750_gen_data_%s" % i, 'ts', 'map_full.hdf5') for i in range(n_file)]

    input_map_cut = []
    re_map_cut = []


    for i in range(n_file):
        print('file No.', i)
        # input_map1_i = h5py.File(input_file1[i], 'r')['map'][:]
        input_map2_i = h5py.File(input_file2[i], 'r')['map'][:]
        input_map3_i = h5py.File(input_file3[i], 'r')['map'][:]
        # input_map_i = input_map1_i + input_map2_i + input_map3_i
        input_map_i = input_map2_i + input_map3_i
        re_map_i = h5py.File(re_file[i], 'r')['map'][:]

        input_map_cut_i, re_map_cut_i = map_proj(input_map_i, re_map_i, rot=[0.0, 90.0, 0.0], xsize=100, reso=12.0)

        input_map_cut.append(input_map_cut_i)
        re_map_cut.append(re_map_cut_i)


    input_map_cut = np.concatenate(input_map_cut, axis=0)
    re_map_cut = np.concatenate(re_map_cut, axis=0)


    output_dir = './train_data'
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    with h5py.File(output_dir + '/train_data.hdf5', 'w') as f:
        f.create_dataset('input', data=input_map_cut)
        f.create_dataset('reconstruction', data=re_map_cut)


    # subtract mean of the maps
    with h5py.File(output_dir + '/train_data.hdf5', 'r') as f:
        input_map_cut = f['input'][:]
        re_map_cut = f['reconstruction'][:]

    for i in range(input_map_cut.shape[0]):
        input_map_cut[i] -= np.mean(input_map_cut[i])
        re_map_cut[i] -= np.mean(re_map_cut[i])

    # save mean subtracted maps
    with h5py.File(output_dir + '/train_data_zero_mean.hdf5', 'w') as f:
        f.create_dataset('input', data=input_map_cut)
        f.create_dataset('reconstruction', data=re_map_cut)