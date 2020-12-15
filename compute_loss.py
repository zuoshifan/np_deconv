import os
import numpy as np
from scipy.stats import pearsonr
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


epochs = 50
net = 'ae'
# net = 'unet'
# net = 'unet1'

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

loss_train = []
loss_val = []

for i in range(epochs):
    print '%d of %d...' % (i, epochs)
    model.load_weights(output_dir + '/model_weights_%04d.h5' % i)

    train_predict = model.predict(train1)
    val_predict = model.predict(val1)
    # print test_predict.shape

    loss_train_ = []
    loss_val_ = []
    for i in range(0, train1.shape[0]):
        # compute loss
        l = np.mean((train2[i, :, :, 0] - train_predict[i, :, :, 0])**2)
        loss_train_.append(l)
    loss_train.append(np.mean(loss_train_))

    for i in range(0, val1.shape[0]):
        l = np.mean((val2[i, :, :, 0] - val_predict[i, :, :, 0])**2)
        loss_val_.append(l)
    loss_val.append(np.mean(loss_val_))


# compute loss for test dataset
model.load_weights(output_dir + '/model_weights_%04d.h5' % (epochs - 1))
test_predict = model.predict(test1)

loss_test_ = []
for i in range(0, test1.shape[0]):
    # compute loss
    l = np.mean((test2[i, :, :, 0] - test_predict[i, :, :, 0])**2)
    loss_test_.append(l)
loss_test = np.mean(loss_test_)


# save data
with h5py.File(output_dir + '/loss.hdf5', 'w') as f:
    f.create_dataset('loss_train', data=np.array(loss_train))
    f.create_dataset('loss_val', data=np.array(loss_val))
    f.create_dataset('loss_test', data=np.array(loss_test))

# plot r
plt.figure()
plt.plot(np.arange(1, epochs+1), loss_train, 'b', linewidth=1.5, label='train')
plt.plot(np.arange(1, epochs+1), loss_val, 'g', linewidth=1.5, label='validation')
plt.axhline(loss_test, linewidth=1.5, color='r', label='test')
plt.legend(fontsize=15, loc='best')
plt.xlabel(r'Epoch', fontsize=16)
plt.ylabel(r'MSE loss', fontsize=16)
plt.savefig(output_dir + '/loss.png')
plt.close()
