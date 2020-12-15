import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
try:
    # for new version matplotlib
    import matplotlib.style as mstyle
    mstyle.use('classic')
except ImportError:
    pass
import matplotlib.pyplot as plt


input_dir = './train_data'

# ii = 300
for ii in range(0, 1000, 99):
    with h5py.File(input_dir + '/train_data.hdf5', 'r') as f:
    # with h5py.File(input_dir + '/train_data_zero_mean.hdf5', 'r') as f:
        in_map = f['input'][ii]
        rec_map = f['reconstruction'][ii]

    print in_map.mean(), rec_map.mean()
    # in_map -= in_map.mean()
    # rec_map -= rec_map.mean()

    plt.figure(figsize=(13, 5))
    plt.subplot(121)
    # plt.imshow(in_map, origin='lower', aspect='equal', vmin=0, vmax=3)
    plt.imshow(in_map, origin='lower', aspect='equal', vmin=0, vmax=7)
    plt.colorbar()
    plt.subplot(122)
    plt.imshow(rec_map, origin='lower', aspect='equal', vmin=0, vmax=3)
    plt.colorbar()
    plt.savefig(input_dir + '/train_%04d.png' % ii)
    # plt.savefig(input_dir + '/train_zero_mean_%04d.png' % ii)
    plt.close()