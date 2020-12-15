import os
from pathlib import Path
import numpy as np
import h5py
import healpy as hp
import rotate
import matplotlib
matplotlib.use('Agg')
try:
    # for new version matplotlib
    import matplotlib.style as mstyle
    mstyle.use('classic')
except ImportError:
    pass
import matplotlib.pyplot as plt


def azimuthal_projmap(map_, rot=None, coord=None, xsize=800, reso=1.5, nest=False, **kwargs):
    nside = hp.npix2nside(map_.shape[-1])
    f = lambda x, y, z: hp.pixelfunc.vec2pix(nside, x, y, z, nest=nest)
    azproj = hp.projector.AzimuthalProj(rot=rot, coord=coord, xsize=xsize, reso=reso, **kwargs)

    return azproj.projmap(map_, f)



if __name__ == '__main__':
    input_maps_dir1 = '/public/home/wufq/sfzuo/workspace/dish_simNP_750MHz/ps_map/'
    input_maps_dir2 = '/public/home/wufq/sfzuo/workspace/dish_simNP_750MHz/750_maps/'
    re_maps_dir = '/public/home/wufq/sfzuo/workspace/dish_simNP_750MHz/output_sim_750MHz_2/map'
    input_path1 = Path(input_maps_dir1)
    input_path2 = Path(input_maps_dir2)
    re_path = Path(re_maps_dir)

    # output_path = Path(target_maps_dir)

    input_file1  = sorted(input_path1.glob('ps_*.hdf5'))
    input_file2  = sorted(input_path2.glob('21cm_*.hdf5'))
    input_file3  = sorted(input_path2.glob('fg_*.hdf5'))
    n_file = len(input_file1)
    re_file = [re_path.joinpath("ts_750_gen_data_%s" % i, 'ts', 'map_full.hdf5') for i in range(n_file)]

    input_maps = []
    rec_maps = []

    for i in range(n_file):
        print('file No.', i)
        input_map1_i = h5py.File(input_file1[i], 'r')['map'][:, 0, :]
        input_map2_i = h5py.File(input_file2[i], 'r')['map'][:, 0, :]
        input_map3_i = h5py.File(input_file3[i], 'r')['map'][:, 0, :]
        input_map_i = input_map1_i + input_map2_i + input_map3_i
        re_map_i = h5py.File(re_file[i], 'r')['map'][:, 0, :]

        input_maps.append(input_map_i)
        rec_maps.append(re_map_i)

    input_maps = np.concatenate(input_maps, axis=0)
    rec_maps = np.concatenate(rec_maps, axis=0)
    # print input_maps.shape, rec_maps.shape

    n_map = input_maps.shape[0]
    n_aug = 20 # aug 20 times for each map

    reso = 12.0 # arcmin
    xsize = 100
    input_img = np.zeros((n_map * n_aug, xsize, xsize), dtype=input_maps.dtype)
    rec_img = np.zeros((n_map * n_aug, xsize, xsize), dtype=input_maps.dtype)

    np.random.seed(0)
    rands = np.random.rand(n_map * n_aug)

    for i in range(n_map):
        for j in range(n_aug):
            mi = i * n_aug + j
            if j == 0:
                input_rot_map = input_maps[i].copy()
                rec_rot_map = rec_maps[i].copy()
            else:
                angle = 360 * rands[mi]
                input_rot_map = rotate.np_rotate(input_maps[i].copy(), angle)
                rec_rot_map = rotate.np_rotate(rec_maps[i].copy(), angle)

            input_img[mi] = azimuthal_projmap(input_rot_map, rot=[0.0, 90.0, 0.0], xsize=xsize, reso=reso)
            rec_img[mi] = azimuthal_projmap(rec_rot_map, rot=[0.0, 90.0, 0.0], xsize=xsize, reso=reso)

    print input_img.shape, rec_img.shape


    output_dir = './train_data'
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    with h5py.File(output_dir + '/train_data_aug.hdf5', 'w') as f:
        f.create_dataset('input', data=input_img)
        f.create_dataset('reconstruction', data=rec_img)
        f.create_dataset('aug_rands', data=rands)


    # # subtract mean of the maps
    # with h5py.File(output_dir + '/train_data.hdf5', 'r') as f:
    #     input_img = f['input'][:]
    #     rec_img = f['reconstruction'][:]

    for i in range(input_img.shape[0]):
        input_img[i] -= np.mean(input_img[i])
        rec_img[i] -= np.mean(rec_img[i])

    # save mean subtracted maps
    with h5py.File(output_dir + '/train_data_zero_mean_aug.hdf5', 'w') as f:
        f.create_dataset('input', data=input_img)
        f.create_dataset('reconstruction', data=rec_img)
        f.create_dataset('aug_rands', data=rands)

    print 'Done...'