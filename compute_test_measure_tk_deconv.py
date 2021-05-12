import numpy as np
from scipy.stats import pearsonr
import h5py
from keras.models import model_from_json

from simple_metrics import peak_signal_noise_ratio
from simple_metrics import mean_squared_error
from structural_similarity import structural_similarity


input_file = './train_data/train_data_zero_mean.hdf5'
with h5py.File(input_file, 'r') as f:
    in_map = f['input'][0, :]
    # rec_map = f['reconstruction'][:]

# fl = './tk_deconv/dish_map_cut_iter3000_loop0.5.hdf5'
fl = './tk_deconv/dish_map_cut_iter20000_loop0.5.hdf5'
with h5py.File(fl, 'r') as f:
    tk_map = f['map_cut'][:]

print(in_map.shape, tk_map.shape)


l = mean_squared_error(in_map, tk_map)

# compute pearson r
r, p = pearsonr(in_map.flatten(), tk_map.flatten())

# compute psnr
p = peak_signal_noise_ratio(in_map, tk_map, data_range=in_map.max()-in_map.min())

# compute ssim
s = structural_similarity(in_map, tk_map)

print '%6s %6s %6s %6s %6s' % ('model', 'MSE', 'r', 'PSNR', 'SSIM')
print '%6s %6.2f %6.2f %6.2f %6.2f' % ('tk', l, r, p, s)