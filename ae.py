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


normalization = True

input_file = './train_data/train_data_zero_mean.hdf5'
if normalization:
    output_dir = './ae_normalization_result'
else:
    output_dir = './ae_result'

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

if normalization:
    train_mean = np.mean(train1, axis=(1, 2, 3))
    val_mean = np.mean(val1, axis=(1, 2, 3))
    test_mean = np.mean(test1, axis=(1, 2, 3))
    train_std = np.std(train1, axis=(1, 2, 3))
    val_std = np.std(val1, axis=(1, 2, 3))
    test_std = np.std(test1, axis=(1, 2, 3))

    train1 = (train1 - train_mean[:, np.newaxis, np.newaxis, np.newaxis]) / train_std[:, np.newaxis, np.newaxis, np.newaxis]
    train2 = (train2 - train_mean[:, np.newaxis, np.newaxis, np.newaxis]) / train_std[:, np.newaxis, np.newaxis, np.newaxis]
    val1 = (val1 - val_mean[:, np.newaxis, np.newaxis, np.newaxis]) / val_std[:, np.newaxis, np.newaxis, np.newaxis]
    val2 = (val2 - val_mean[:, np.newaxis, np.newaxis, np.newaxis]) / val_std[:, np.newaxis, np.newaxis, np.newaxis]
    test1 = (test1 - test_mean[:, np.newaxis, np.newaxis, np.newaxis]) / test_std[:, np.newaxis, np.newaxis, np.newaxis]
    test2 = (test2 - test_mean[:, np.newaxis, np.newaxis, np.newaxis]) / test_std[:, np.newaxis, np.newaxis, np.newaxis]


# autoencoder
nsample, npix, _, _ = train1.shape

input_img = Input(shape=(npix, npix, 1))  #  1 for one pol, adapt this if using `channels_first` image data format

x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

x = Conv2D(64, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)

decoded = Conv2D(1, 1, activation='linear', padding='same')(x)


model = Model(input_img, decoded)
model.summary()

# save model
json_string = model.to_json()
with open('ae.json', 'w') as f:
        f.write(json_string)

# model.compile(optimizer='adam', loss='binary_crossentropy')
# model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
model.compile(optimizer='adam', loss='mse')
# model.compile(optimizer='adam', loss='mae') # use mae to promote sparsity for point sources



batch_size = 32
epochs = 1

loss = []
val_loss = []
for ii in range(100):

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
    model.save_weights(output_dir + '/model_weights_%04d.h5' % ii)

    predict = model.predict(train1[0].reshape(1, npix, npix, 1))
    if normalization:
        predict = predict * test_std[0] + test_mean[0]

    # plot predict
    plt.figure()
    plt.imshow(predict[0, :, :, 0], origin='lower', aspect='equal', vmin=0, vmax=3)
    plt.colorbar()
    plt.savefig(output_dir + '/predict_%04d.png' % ii)
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
    plt.savefig(output_dir + '/loss_%04d.png' % ii)
    plt.close()


# save loss for plot
with h5py.File(output_dir + '/history_loss.hdf5', 'w') as f:
    f.create_dataset('loss', data=np.array(loss))
    f.create_dataset('val_loss', data=np.array(val_loss))
