import numpy as np
from cora.foreground import galaxy, pointsource
import h5py
import healpy as hp

def gen_galaxy(nside, freq, num):

    gal_map = np.zeros((num, 1, hp.nside2npix(nside)), dtype=np.float64)

    gal = galaxy.ConstrainedGalaxy()
    gal.nside = nside
    # gal.frequencies = freq
    gal.frequencies = np.array([freq, freq + 0.1]) # the number of frequencies must be more than two.

    for i in range(num):
        gal_ = gal.getsky()
        gal_map[i] = gal_.mean(axis=0, keepdims=True)
    
    return gal_map


def gen_ps(nside, freq, num, maxflux=1e6):

    ps_map = np.zeros((num, 1, hp.nside2npix(nside)), dtype=np.float64)

    ps = pointsource.CombinedPointSources()
    ps.nside = nside
    ps.frequencies = np.array([freq, freq + 0.1]) # the number of frequencies must be more than two.
    ps.flux_max = maxflux

    for i in range(num):
        ps_ = ps.getsky()
        ps_map[i] = ps_.mean(axis=0, keepdims=True)
    
    return ps_map
    

def combine_gal_ps(galaxy_map, ps_map, num, weighted=False, ps_weight=False):

    n_gal = galaxy_map.shape[0]
    n_ps = ps_map.shape[0]
    nside = hp.npix2nside(galaxy_map.shape[-1])
    npol = galaxy_map.shape[1]

    comb_map = np.zeros((num, npol, hp.nside2npix(nside)), dtype=np.float64)

    for i in range(num):

        if weighted:
            weight = np.random.uniform(0, 1)
            gal_ind = np.random.randint(0, n_gal, 2) # only use two galaxy maps to constructe a combined one.
            gal_map_ = weight * galaxy_map[gal_ind[0]] + (1 - weight) * galaxy_map[gal_ind[1]]

            if ps_weight:
                ps_weight = np.random.uniform(0, 1)
                ps_ind = np.random.randint(0, n_ps, 2)
                ps_map_ = ps_weight * ps_map[ps_ind[0]] + (1 - ps_weight) * ps_map[ps_ind[1]]
            else:
                ps_ind = np.random.randint(0, n_ps)
                ps_map_ = ps_map[ps_ind]
        
        else:
            gal_ind = np.random.randint(0, n_gal)
            gal_map_ = galaxy_map[gal_ind]
            ps_ind = np.random.randint(0, n_ps)
            ps_map_ = ps_map[ps_ind]

        comb_map[i] = gal_map_ + ps_map_

    return comb_map


def write_map(filename, data):
    with h5py.File(filename, 'w') as f:
        f.create_dataset('map', data=data)

if __name__ == '__main__':

    freq = 750. #MHz
    nside = 256
    gal_num = 10
    ps_num = 200
    maxflux = 1e6 #Jy

    fg_num = 10

    # gal = gen_galaxy(nside, freq, gal_num)
    # ps = gen_ps(nside, freq, ps_num, maxflux=maxflux)

    # combined_map = combine_gal_ps(gal, ps, fg_num, weighted=False, ps_weight=False)

    # write_map("galaxy_" + str(freq) + "_num_" + str(gal_num) + ".hdf5", data=gal)
    # write_map("ps_" + str(freq) + "_num_" + str(ps_num) + ".hdf5", data=ps)
    # write_map("fg_" + str(freq) + "_num_" + str(fg_num) + ".hdf5", data=combined_map)

    for i in range(5):
        ps = gen_ps(nside, freq, ps_num, maxflux=maxflux)
        write_map("ps_map/ps_" + str(freq) + "_num_" + str(ps_num) + str(i) + ".hdf5", data=ps)

    print('DONE!')
