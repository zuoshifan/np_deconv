import os
import numpy as np
import h5py
import keras
# from keras.models import load_model
from keras.models import model_from_json
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Activation, Dropout, Concatenate
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib
matplotlib.use('Agg')
try:
    # for new version matplotlib
    import matplotlib.style as mstyle
    mstyle.use('classic')
except ImportError:
    pass
import matplotlib.pyplot as plt

from structural_similarity import structural_similarity


# input_dir = './train_data'

# # ii = 300
# for ii in range(0, 1000, 99):
#     # with h5py.File(input_dir + '/train_data.hdf5', 'r') as f:
#     with h5py.File(input_dir + '/train_data_zero_mean.hdf5', 'r') as f:
#         in_map = f['input'][ii]
#         rec_map = f['reconstruction'][ii]

#     # print in_map.mean(), rec_map.mean()
#     # in_map -= in_map.mean()
#     # rec_map -= rec_map.mean()

#     print structural_similarity(in_map, rec_map)


epochs = 50
# net = 'ae'
# net = 'unet'
net = 'unet1'

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

ssim_train = []
ssim_val = []

for i in range(epochs):
    print '%d of %d...' % (i, epochs)
    model.load_weights(output_dir + '/model_weights_%04d.h5' % i)

    train_predict = model.predict(train1)
    val_predict = model.predict(val1)
    # print test_predict.shape

    ssim_train_ = []
    ssim_val_ = []
    for i in range(0, train1.shape[0]):
        in_map = train2[i, :, :, 0]
        rec_map = train_predict[i, :, :, 0]
        # compute ssim
        s = structural_similarity(in_map, rec_map)
        ssim_train_.append(s)
    ssim_train.append(np.mean(ssim_train_))

    for i in range(0, val1.shape[0]):
        in_map = val2[i, :, :, 0]
        rec_map = val_predict[i, :, :, 0]
        # compute ssim
        s = structural_similarity(in_map, rec_map)
        ssim_val_.append(s)
    ssim_val.append(np.mean(ssim_val_))


# compute ssim for test dataset
model.load_weights(output_dir + '/model_weights_%04d.h5' % (epochs - 1))
test_predict = model.predict(test1)

ssim_test_ = []
for i in range(0, test1.shape[0]):
    in_map = test2[i, :, :, 0]
    rec_map = test_predict[i, :, :, 0]
    # compute ssim
    s = structural_similarity(in_map, rec_map)
    ssim_test_.append(s)
ssim_test = np.mean(ssim_test_)


# save data
with h5py.File(output_dir + '/ssim.hdf5', 'w') as f:
    f.create_dataset('ssim_train', data=np.array(ssim_train))
    f.create_dataset('ssim_val', data=np.array(ssim_val))
    f.create_dataset('ssim_test', data=np.array(ssim_test))

# plot r
plt.figure()
plt.plot(np.arange(1, epochs+1), ssim_train, 'b', linewidth=1.5, label='train')
plt.plot(np.arange(1, epochs+1), ssim_val, 'g', linewidth=1.5, label='validation')
plt.axhline(ssim_test, linewidth=1.5, color='r', label='test')
plt.legend(fontsize=15, loc='best')
plt.xlabel(r'Epoch', fontsize=16)
plt.ylabel(r'SSIM', fontsize=16)
plt.savefig(output_dir + '/ssim.png')
plt.close()
