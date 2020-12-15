import os
import numpy as np
import h5py
import healpy as hp
import keras
# from keras.models import load_model
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


input_file = './train_data/train_data_zero_mean_aug.hdf5'
output_dir = './unet1_aug_result'

if not os.path.isdir(output_dir):
    os.mkdir(output_dir)

with h5py.File(input_file, 'r') as f:
    in_map = f['input'][:]
    rec_map = f['reconstruction'][:]

# shuffle the data
inds = np.arange(in_map.shape[0])
np.random.shuffle(inds[1:]) # leave 0 (the true NP) unchanged
in_map = in_map[inds] # randomly shuffle the datasets
rec_map = rec_map[inds] # randomly shuffle the datasets


# read in dataset
train1 = rec_map[:12000][:, :, :, np.newaxis]
val1 = rec_map[12000:18000][:, :, :, np.newaxis]
test1 = rec_map[18000:][:, :, :, np.newaxis]

train2 = in_map[:12000][:, :, :, np.newaxis]
val2 = in_map[12000:18000][:, :, :, np.newaxis]
test2 = in_map[18000:][:, :, :, np.newaxis]


# unet
nsample, npix, _, _ = train1.shape

input_img = Input(shape=(npix, npix, 1))  #  1 for one pol, adapt this if using `channels_first` image data format

conv1 = Conv2D(32, (3, 3), activation = 'relu', padding='same')(input_img)
conv1 = Conv2D(32, (3, 3), activation = 'relu', padding='same')(conv1)
pool1 = MaxPooling2D((2, 2))(conv1)
conv2 = Conv2D(64, (3, 3), activation = 'relu', padding='same')(pool1)
conv2 = Conv2D(64, (3, 3), activation = 'relu', padding='same')(conv2)
pool2 = MaxPooling2D((2, 2))(conv2)

conv3 = Conv2D(128, (3, 3), activation = 'relu', padding='same')(pool2)
conv3 = Conv2D(128, (3, 3), activation = 'relu', padding='same')(conv3)
drop3 = Dropout(0.5)(conv3)

up4 = UpSampling2D((2, 2))(drop3)
merge4 = Concatenate()([conv2, up4])
conv4 = Conv2D(64, (3, 3), activation = 'relu', padding='same')(merge4)
conv4 = Conv2D(64, (3, 3), activation = 'relu', padding='same')(merge4)

up5 = UpSampling2D((2, 2))(conv4)
merge5 = Concatenate()([conv1, up5])
conv5 = Conv2D(32, (3, 3), activation = 'relu', padding='same')(merge5)
conv5 = Conv2D(32, (3, 3), activation = 'relu', padding='same')(merge5)

conv6 = Conv2D(1, 1, activation = 'linear')(conv5)


model = Model(input_img, conv6)

# # save model
# json_string = model.to_json()
# with open('unet1.json', 'w') as f:
#         f.write(json_string)

# model.compile(optimizer='adam', loss='binary_crossentropy')
# model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
model.compile(optimizer='adam', loss='mse')
# model.compile(optimizer='adam', loss='mae') # use mae to promote sparsity for point sources

model.summary()


batch_size = 128
# epochs = 100
epochs = 1


# start index
si = 38
# load weights
model.load_weights(output_dir + '/model_weights_%04d.h5' % (si - 1))

# load in previous loss
with h5py.File(output_dir + '/history_loss.hdf5', 'r') as f:
    loss = f['loss'][:si].tolist()
    val_loss = f['val_loss'][:si].tolist()

# loss = []
# val_loss = []
# for ii in range(50):
for ii in range(si, 50):

    # es_cb = EarlyStopping(monitor='val_loss', patience=2, verbose=1, mode='auto')
    # chkpt = saveDir + 'AutoEncoder_Cifar10_denoise_weights.{epoch:02d}-{loss:.2f}-{val_loss:.2f}.hdf5'
    # cp_cb = ModelCheckpoint(filepath = chkpt, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')

    history = model.fit(train1, train2,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        validation_data=(val1, val2),
                        # callbacks=[es_cb,],
                        # callbacks=[es_cb, cp_cb],
                        shuffle=True)

    # save weights
    model.save_weights(output_dir + '/model_weights_%04d_new.h5' % ii)

    predict = model.predict(train1[0].reshape(1, npix, npix, 1))

    # plot predict
    plt.figure()
    plt.imshow(predict[0, :, :, 0], origin='lower', aspect='equal', vmin=0, vmax=3)
    plt.colorbar()
    plt.savefig(output_dir + '/predict_%04d_new.png' % ii)
    plt.close()

    # save loss
    loss.append(history.history['loss'][-1])
    val_loss.append(history.history['val_loss'][-1])

    # plot history for loss
    plt.figure()
    # plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])
    plt.plot(loss)
    plt.plot(val_loss)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['train', 'validation'], loc='upper right')
    plt.savefig(output_dir + '/loss_%04d_new.png' % ii)
    plt.close()


# save loss for plot
with h5py.File(output_dir + '/history_loss_new.hdf5', 'w') as f:
    f.create_dataset('loss', data=np.array(loss))
    f.create_dataset('val_loss', data=np.array(val_loss))
