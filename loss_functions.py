######################
###     IMPORTS    ###
######################


import time
import tensorflow as tf
import numpy as np

from matplotlib import pyplot as plt
from astropy import stats

from photutils.detection import DAOStarFinder
from photutils.aperture import CircularAperture, aperture_photometry

from multiprocessing import Pool

######################
### LOSS FUNCTIONS ###
######################
@tf.function
def huber_loss(gen_output, target, δ):
    err = gen_output - target
    err_abs = tf.abs(err)
    Lh = tf.where(err_abs < δ, 0.5 * tf.square(err), δ * (err_abs - 0.5 * δ))
    return tf.reduce_sum(Lh)
    
@tf.function
def comp_stats(gen_output, target):
    def comp_mean_median(x, sigma):
        mean, median, _ = stats.sigma_clipped_stats(x, sigma=sigma, maxiters=3)
        return np.float32(mean), np.float32(median)
    gen_mean, gen_median = tf.numpy_function(comp_mean_median, [gen_output, 3], Tout=[tf.float32, tf.float32])
    target_mean, target_median = tf.numpy_function(comp_mean_median, [target, 3], Tout=[tf.float32, tf.float32])
    l_mean = tf.abs(gen_mean - target_mean)
    l_median = tf.abs(gen_median - target_median)
    return l_mean + l_median

# def flux_handler(gen_output, target):
#     target_finder = DAOStarFinder(fwhm=10, threshold=5*np.std(target))
#     aper_flux_diff = 0
#     for i in range(gen_output.shape[0]):
#         target_sources = target_finder(target[i])
#         try:
#             tr = np.transpose((target_sources['xcentroid'], target_sources['ycentroid']))
#             apertures = CircularAperture(tr, r=10)
#             target_table = aperture_photometry(target[i], apertures)
#             aper_flux_diff += np.sum(np.abs(aperture_photometry(gen_output[i], apertures)['aperture_sum'] - target_table['aperture_sum']))/len(target_table['aperture_sum'])
#         except:
#             continue
#     return aper_flux_diff

# def comp_flux(gen_output, target):
    
#     total_aper_flux_diff = 0
#     gen_output = np.squeeze(gen_output)
#     target = np.squeeze(target)
#     # CPU_CORES = 4
#     # N = gen_output.shape[0]//CPU_CORES

#     # # # Multiprocessing
#     # with Pool(CPU_CORES) as p:
#     #   results = [p.apply_async(flux_handler, args=(gen_output[i*N:(i+1)*N], target[i*N:(i+1)*N])) if i!= (CPU_CORES-1) else p.apply_async(flux_handler, args=(gen_output[i*N:], target[i*N:])) for i in range(CPU_CORES)]
#     #   for r in results:
#     #     total_aper_flux_diff += r.get()
    
#     # Commented Single Processing
#     target_finder = DAOStarFinder(fwhm=10, threshold=5*np.std(target))
#     for i in range(gen_output.shape[0]):
#         target_sources = target_finder(target[i])
#         tr = np.transpose((target_sources['xcentroid'], target_sources['ycentroid']))
#         apertures = CircularAperture(tr, r=10)
#         target_table = aperture_photometry(target[i], apertures)
#         total_aper_flux_diff += np.sum(np.abs(aperture_photometry(gen_output[i], apertures)['aperture_sum'] - target_table['aperture_sum']))/len(target_table['aperture_sum'])
#     return np.float32(total_aper_flux_diff)

def comp_flux(gen_output, target, Y_source_cat):
    total_aperflux_diff = 0
    gen_output = np.squeeze(gen_output)
    target = np.squeeze(target)
    apertures = CircularAperture(Y_source_cat[:,-3:-1], r=7.9)
    for i in tf.range(target.shape[0]):
        mask = np.where(np.int16(Y_source_cat[:,-1]) == i)[0]
        #print(Y_source_cat[mask,3:5])
        # MASKS = apertures[mask].to_mask(method='center')
        # plt.imshow(MASKS[0].multiply(target[i]))
        # plt.close()
        target_aperflux = aperture_photometry(target[i], apertures[mask], method='center')['aperture_sum']
        #target_aperflux = target_aperflux[~np.isnan(target_aperflux).any(axis=0), :]
        gen_aperflux = aperture_photometry(gen_output[i], apertures[mask], method='center')['aperture_sum']
        aperfluxdiff = np.abs(target_aperflux - gen_aperflux)
        aperfluxdiff = aperfluxdiff[~np.isnan(aperfluxdiff)]
        #print(aperfluxdiff)
        if len(aperfluxdiff) != 0:
            total_aperflux_diff += np.sum(aperfluxdiff)/len(aperfluxdiff)
    return np.float32(total_aperflux_diff)

# Function used during weight training
@tf.function
def loss(gen_output, target, Y_source_cat, α):
    Lh = huber_loss(gen_output, target, 5*tf.math.reduce_std(target))
    Lstats = comp_stats(gen_output, target)
    Lflux = tf.numpy_function(comp_flux, [gen_output, target, Y_source_cat], Tout=[tf.float32])
    return α*Lh + Lstats + (1.0 - tf.cast(α, tf.float32)) * Lflux

# Function used for validation loss computations
def loss_valid(gen_output, target, Y_source_cat, α):
    Lh = huber_loss(gen_output, target, 5*tf.math.reduce_std(target))
    Lstats = comp_stats(gen_output, target)
    Lflux = tf.numpy_function(comp_flux, [gen_output, target, Y_source_cat], Tout=[tf.float32])
    return α*Lh, Lstats, (1.0 - tf.cast(α, tf.float32)) * Lflux