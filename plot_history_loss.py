import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
# try:
#     # for new version matplotlib
#     import matplotlib.style as mstyle
#     mstyle.use('classic')
# except ImportError:
#     pass
import matplotlib.pyplot as plt


net = 'ae'
# net = 'unet'
# net = 'unet1'
normalization = False
# normalization = True


loss_dir = '%s%s_result' % (net, '_normalization' if normalization else '')
loss_file = '%s/history_loss.hdf5' % loss_dir

with h5py.File(loss_file, 'r') as f:
    loss = f['loss'][:]
    val_loss = f['val_loss'][:]

epoch = len(loss)

# plot loss curve
plt.figure()
plt.plot(np.arange(1, 1+epoch), loss)
plt.plot(np.arange(1, 1+epoch), val_loss)
plt.xlabel('Epoch')
plt.ylabel('MSE loss')
plt.legend(['train', 'validation'], loc='upper right')
plt.savefig(loss_dir + '/history_loss.png')
plt.close()
