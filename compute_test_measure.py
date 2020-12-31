import numpy as np
from scipy.stats import pearsonr
import h5py
from keras.models import model_from_json

from simple_metrics import peak_signal_noise_ratio
from simple_metrics import mean_squared_error
from structural_similarity import structural_similarity


epoch = 99
normalization = False

input_file = './train_data/train_data_zero_mean.hdf5'

with h5py.File(input_file, 'r') as f:
    in_map = f['input'][:]
    rec_map = f['reconstruction'][:]

# read in dataset
# train1 = rec_map[:600][:, :, :, np.newaxis]
# val1 = rec_map[600:900][:, :, :, np.newaxis]
test1 = rec_map[900:][:, :, :, np.newaxis]

# train2 = in_map[:600][:, :, :, np.newaxis]
# val2 = in_map[600:900][:, :, :, np.newaxis]
test2 = in_map[900:][:, :, :, np.newaxis]


mse = []
r = []
psnr = []
ssim = []

nets = [ 'ae', 'unet', 'unet1' ]
for net in nets:
    # net = 'ae'
    # net = 'unet'
    # net = 'unet1'

    if normalization:
        model_weight_dir = './%s_normalization_result' % net
    else:
        model_weight_dir = './%s_result' % net

    # load model
    with open('%s.json' % net, 'r') as f:
        json_string = f.read()

    model = model_from_json(json_string)
    # model.summary()

    model.load_weights(model_weight_dir + '/model_weights_%04d.h5' % epoch)
    test_predict = model.predict(test1)

    mse_test_ = []
    r_test_ = []
    psnr_test_ = []
    ssim_test_ = []
    for i in range(0, test1.shape[0]):
        in_map = test2[i, :, :, 0]
        rec_map = test_predict[i, :, :, 0]

        # compute mse
        # l = np.mean((in_map - rec_map)**2)
        l = mean_squared_error(in_map, rec_map)
        mse_test_.append(l)

        # compute pearson r
        r_, p = pearsonr(in_map.flatten(), rec_map.flatten())
        r_test_.append(r_)

        # compute psnr
        p = peak_signal_noise_ratio(in_map, rec_map, data_range=in_map.max()-in_map.min())
        psnr_test_.append(p)

        # compute ssim
        s = structural_similarity(in_map, rec_map)
        ssim_test_.append(s)

    mse_test = np.mean(mse_test_)
    r_test = np.mean(r_test_)
    psnr_test = np.mean(psnr_test_)
    ssim_test = np.mean(ssim_test_)

    mse.append(mse_test)
    r.append(r_test)
    psnr.append(psnr_test)
    ssim.append(ssim_test)



print '%6s %6s %6s %6s %6s' % ('model', 'MSE', 'r', 'PSNR', 'SSIM')
for m, l, r, p, s in zip(nets, mse, r, psnr, ssim):
    print '%6s %6.2f %6.2f %6.2f %6.2f' % (m, l, r, p, s)