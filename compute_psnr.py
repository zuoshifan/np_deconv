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

from simple_metrics import peak_signal_noise_ratio


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

#     print peak_signal_noise_ratio(in_map, rec_map, data_range=in_map.max()-in_map.min())


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

psnr_train = []
psnr_val = []

for i in range(epochs):
    print '%d of %d...' % (i, epochs)
    model.load_weights(output_dir + '/model_weights_%04d.h5' % i)

    train_predict = model.predict(train1)
    val_predict = model.predict(val1)
    # print test_predict.shape

    psnr_train_ = []
    psnr_val_ = []
    for i in range(0, train1.shape[0]):
        in_map = train2[i, :, :, 0]
        rec_map = train_predict[i, :, :, 0]
        # compute psnr
        p = peak_signal_noise_ratio(in_map, rec_map, data_range=in_map.max()-in_map.min())
        psnr_train_.append(p)
    psnr_train.append(np.mean(psnr_train_))

    for i in range(0, val1.shape[0]):
        in_map = val2[i, :, :, 0]
        rec_map = val_predict[i, :, :, 0]
        # compute psnr
        p = peak_signal_noise_ratio(in_map, rec_map, data_range=in_map.max()-in_map.min())
        psnr_val_.append(p)
    psnr_val.append(np.mean(psnr_val_))


# compute psnr for test dataset
model.load_weights(output_dir + '/model_weights_%04d.h5' % (epochs - 1))
test_predict = model.predict(test1)

psnr_test_ = []
for i in range(0, test1.shape[0]):
    in_map = test2[i, :, :, 0]
    rec_map = test_predict[i, :, :, 0]
    # compute psnr
    p = peak_signal_noise_ratio(in_map, rec_map, data_range=in_map.max()-in_map.min())
    psnr_test_.append(p)
psnr_test = np.mean(psnr_test_)


# save data
with h5py.File(output_dir + '/psnr.hdf5', 'w') as f:
    f.create_dataset('psnr_train', data=np.array(psnr_train))
    f.create_dataset('psnr_val', data=np.array(psnr_val))
    f.create_dataset('psnr_test', data=np.array(psnr_test))

# plot r
plt.figure()
plt.plot(np.arange(1, epochs+1), psnr_train, 'b', linewidth=1.5, label='train')
plt.plot(np.arange(1, epochs+1), psnr_val, 'g', linewidth=1.5, label='validation')
plt.axhline(psnr_test, linewidth=1.5, color='r', label='test')
plt.legend(fontsize=15, loc='best')
plt.xlabel(r'Epoch', fontsize=16)
plt.ylabel(r'PSNR', fontsize=16)
plt.savefig(output_dir + '/psnr.png')
plt.close()
