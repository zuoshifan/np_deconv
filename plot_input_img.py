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
# input_file = '%s/train_data.hdf5' % input_dir
input_file = '%s/train_data_zero_mean.hdf5' % input_dir

with h5py.File(input_file, 'r') as f:
    in_map = f['input'][0]
    rec_map = f['reconstruction'][0]

print in_map.mean(), rec_map.mean()
# in_map -= in_map.mean()
# rec_map -= rec_map.mean()

plt.figure()
plt.imshow(in_map, origin='lower', aspect='equal', vmin=0, vmax=3)
# plt.imshow(in_map, origin='lower', aspect='equal', vmin=-3, vmax=3)
plt.colorbar()
plt.savefig(input_dir + '/input_0.png')
plt.close()

plt.figure()
plt.imshow(rec_map, origin='lower', aspect='equal', vmin=0, vmax=3)
# plt.imshow(rec_map, origin='lower', aspect='equal', vmin=-3, vmax=3)
plt.colorbar()
plt.savefig(input_dir + '/rec_0.png')
plt.close()
