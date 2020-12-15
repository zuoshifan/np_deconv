import os
import numpy as np
from scipy.stats import pearsonr
import h5py
import healpy as hp
import keras
# from keras.models import load_model
from keras.models import model_from_json
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Activation, Dropout, Concatenate
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
# from keras.optimizers import Adam
# import pickle

import matplotlib
matplotlib.use('Agg')
try:
    # for new version matplotlib
    import matplotlib.style as mstyle
    mstyle.use('classic')
except ImportError:
    pass
import matplotlib.pyplot as plt


epochs = 50
# net = 'ae'
# net = 'unet'
# net = 'unet1'
net = 'unet1_cosh_loss'

input_file = './train_data/train_data_zero_mean.hdf5'
output_dir = './%s_result' % net

if not os.path.isdir(output_dir):
    os.mkdir(output_dir)

with h5py.File(input_file, 'r') as f:
    in_map = f['input'][:]
    rec_map = f['reconstruction'][:]

# read in dataset
train1 = rec_map[:600][:, :, :, np.newaxis]
val1 = rec_map[600:900][:, :, :, np.newaxis]
test1 = rec_map[900:][:, :, :, np.newaxis]

train2 = in_map[:600][:, :, :, np.newaxis]
val2 = in_map[600:900][:, :, :, np.newaxis]
test2 = in_map[900:][:, :, :, np.newaxis]


# load model
with open('%s.json' % net, 'r') as f:
    json_string = f.read()

model = model_from_json(json_string)
# model.summary()

rs_train = []
rs_val = []

for i in range(epochs):
    print '%d of %d...' % (i, epochs)
    model.load_weights(output_dir + '/model_weights_%04d.h5' % i)

    train_predict = model.predict(train1)
    val_predict = model.predict(val1)
    # print test_predict.shape

    rs_train_ = []
    rs_val_ = []
    for i in range(0, train1.shape[0]):
        # compute pearson r
        r, p = pearsonr(train2[i, :, :, 0].flatten(), train_predict[i, :, :, 0].flatten())
        rs_train_.append(r)
    rs_train.append(np.mean(rs_train_))

    for i in range(0, val1.shape[0]):
        r, p = pearsonr(val2[i, :, :, 0].flatten(), val_predict[i, :, :, 0].flatten())
        rs_val_.append(r)
    rs_val.append(np.mean(rs_val_))


# compute r for test dataset
model.load_weights(output_dir + '/model_weights_%04d.h5' % (epochs - 1))
test_predict = model.predict(test1)

rs_test_ = []
for i in range(0, test1.shape[0]):
    # compute pearson r
    r, p = pearsonr(test2[i, :, :, 0].flatten(), test_predict[i, :, :, 0].flatten())
    rs_test_.append(r)
rs_test = np.mean(rs_test_)


# save data
with h5py.File(output_dir + '/pearson_r.hdf5', 'w') as f:
    f.create_dataset('rs_train', data=np.array(rs_train))
    f.create_dataset('rs_val', data=np.array(rs_val))
    f.create_dataset('rs_test', data=np.array(rs_test))

# plot r
plt.figure()
plt.plot(np.arange(1, epochs+1), rs_train, 'b', linewidth=1.5, label='train')
plt.plot(np.arange(1, epochs+1), rs_val, 'g', linewidth=1.5, label='validation')
plt.axhline(rs_test, linewidth=1.5, color='r', label='test')
plt.legend(fontsize=15, loc='best')
plt.xlabel(r'Epoch', fontsize=16)
plt.ylabel(r'Pearson correlation coefficient', fontsize=16)
plt.savefig(output_dir + '/pearson_r.png')
plt.close()
