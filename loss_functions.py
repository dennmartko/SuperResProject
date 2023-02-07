######################
###     IMPORTS    ###
######################


import time
import tensorflow as tf
import numpy as np

from matplotlib import pyplot as plt
from astropy import stats

from photutils.detection import DAOStarFinder
from photutils.segmentation import detect_sources
from photutils.aperture import CircularAperture, aperture_photometry

from multiprocessing import Pool

######################
### LOSS FUNCTIONS ###
######################
# @tf.function
# def huber_loss(gen_output, target, δ):
#     err = gen_output - target
#     err_abs = tf.abs(err)
#     Lh = tf.where(err_abs < δ, 0.5 * tf.square(err), δ * (err_abs - 0.5 * δ))
#     return tf.reduce_sum(Lh)
@tf.function
def huber_loss(gen_output, target):
    err = gen_output - target
    return tf.reduce_sum(tf.math.log(tf.math.cosh(err)))
# @tf.function
# def huber_loss(gen_output, target, δ):
#     err = gen_output - target
#     return tf.reduce_sum(tf.math.abs(err))
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

# def comp_aperflux(gen_output, target, Y_source_cat):
#     total_flux_diff = 0
#     gen_output = np.squeeze(gen_output)
#     target = np.squeeze(target)
#     apertures = CircularAperture(Y_source_cat[:,1:-1], r=7.9)
#     for i in tf.range(target.shape[0]):
#         mask = np.where(np.int16(Y_source_cat[:,-1]) == i)[0]
#         #print(Y_source_cat[mask,3:5])
#         # MASKS = apertures[mask].to_mask(method='center')
#         # plt.imshow(MASKS[0].multiply(target[i]))
#         # plt.close()
#         target_aperflux = aperture_photometry(target[i], apertures[mask], method='center')['aperture_sum']
#         #target_aperflux = target_aperflux[~np.isnan(target_aperflux).any(axis=0), :]
#         gen_aperflux = aperture_photometry(gen_output[i], apertures[mask], method='center')['aperture_sum']
#         gen_aperflux[np.isnan(gen_aperflux)] = 0
#         aperfluxdiff = np.abs(target_aperflux - gen_aperflux) # L2 format may work better to retrieve peak fluxes + use spritz in training to let model feel largerpixelvalues
#         #aperfluxdiff = aperfluxdiff[~np.isnan(aperfluxdiff)]
#         #print(aperfluxdiff)
#         if len(aperfluxdiff) != 0:
#             total_flux_diff += np.sum(aperfluxdiff)/len(aperfluxdiff)
#     return np.float32(total_flux_diff)

# def comp_aperflux(gen, target):
#     #gen = np.squeeze(gen)
#     #target = np.squeeze(target)

#     daofind_target = DAOStarFinder(fwhm=7.9, threshold=5*2.8/1000)
#     sources_target = daofind_target(np.squeeze(np.float32(target[0])))
    
#     daofind_gen = DAOStarFinder(fwhm=7.9, threshold=5*2.8/1000)
#     sources_gen = daofind_gen(np.squeeze(np.float32(gen[0])))
    
#     #flux_list_target, median_target, mean_target = gal_flux(target, gal_fwhm, gal_threshold, gal_sigma)
#     #flux_list_gen, median_gen, mean_gen = gal_flux(gen, gal_fwhm, gal_threshold, gal_sigma)
    
#     flux_list_target = sources_target
#     flux_list_gen = sources_gen
    
#     positions = np.transpose((flux_list_target['xcentroid'], flux_list_target['ycentroid']))
#     apertures = CircularAperture(positions, r=7.9)
    
#     target_table = aperture_photometry(np.squeeze(np.float32(target[0])), apertures)
#     target_table['aperture_sum'].info.format = '%.8g'
    
#     gen_table = aperture_photometry(np.squeeze(np.float32(gen[0])), apertures)
#     gen_table['aperture_sum'].info.format = '%.8g'
    
#     #print (gen_table['aperture_sum'], target_table['aperture_sum'])
#     #diff = tf.reduce_sum(np.abs(target_table['aperture_sum'] - gen_table['aperture_sum']))/len(target_table['aperture_sum'])
#     diff = tf.reduce_sum(tf.abs(tf.expand_dims(target_table['aperture_sum'], 0) - tf.expand_dims(gen_table['aperture_sum'], 0))) / len(target_table['aperture_sum'])
#     #print (np.shape(apertures), np.median(target_table['aperture_sum']), np.median(gen_table['aperture_sum']), \
#     #       diff.astype(np.float32), np.abs(median_target - median_gen).astype(np.float32), np.abs(mean_target - mean_gen).astype(np.float32))
    
#     return tf.cast(diff, tf.float32)

# def comp_peakflux(gen_output, target, Y_source_cat):
#     total_peakflux_diff = 0
#     gen_output = np.squeeze(gen_output)
#     target = np.squeeze(target)
#     apertures = CircularAperture(Y_source_cat[:,1:-1], r=7.9)
#     for i in tf.range(target.shape[0]):
#         mask = np.where(np.int16(Y_source_cat[:,-1]) == i)[0]
#         aperture_masks = apertures[mask].to_mask(method='center')
#         for idx, k in enumerate(mask):
#             total_peakflux_diff += abs(Y_source_cat[k,0] - np.max(aperture_masks[idx].multiply(gen_output[i])))/len(mask)
#     return np.float32(total_peakflux_diff)

def comp_peakflux(gen_output, target, Y_source_cat):
    total_peakflux_diff = tf.constant(0, dtype=tf.float32)
    gen_output = tf.squeeze(gen_output)
    target = tf.squeeze(target)

    for i in tf.range(tf.shape(target)[0]):
        mask = tf.where(tf.cast(Y_source_cat[:,-1], tf.int16) == tf.cast(i, tf.int16))
        indices = tf.gather(mask, 0)
        for j in tf.range(tf.shape(indices)[0]):
            idx = indices[j]
            x = tf.cast(tf.round(Y_source_cat[idx, 2]), tf.int32)
            y = tf.cast(tf.round(Y_source_cat[idx, 1]), tf.int32)
            total_peakflux_diff += tf.abs(Y_source_cat[idx, 0] - gen_output[i, x, y])
    return total_peakflux_diff




# Function used during weight training
@tf.function
def non_adversarial_loss(gen_output, target, Y_source_cat, α):
    # cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    Lh = 0#huber_loss(gen_output, target)
    Lstats = 0#comp_stats(gen_output, target)
    #Lflux = tf.numpy_function(comp_aperflux, [gen_output, target, Y_source_cat], Tout=[tf.float32])
    Lflux = tf.py_function(comp_peakflux, [gen_output, target, Y_source_cat], Tout=[tf.float32])
    #Lflux = tf.py_function(func=comp_aperflux, inp=[gen_output, target], Tout=tf.float32)
    # Ldisc = cross_entropy(tf.ones_like(fake_output), fake_output)
    return α*Lh + Lstats + (1.0 - tf.cast(α, tf.float32)) * Lflux

# Function used for validation loss computations
def non_adversarial_loss_valid(gen_output, target, Y_source_cat, α):
    Lh = huber_loss(gen_output, target)
    Lstats = comp_stats(gen_output, target)
    Lflux = tf.numpy_function(comp_aperflux, [gen_output, target, Y_source_cat], Tout=[tf.float32])
    # cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    # Ldisc = cross_entropy(tf.ones_like(fake_output), fake_output)
    return α*Lh, Lstats, (1.0 - tf.cast(α, tf.float32)) * Lflux

@tf.function
def Wasserstein_loss(y_true, y_pred):
    return tf.math.reduce_mean(y_true * y_pred)

# @tf.function()
# def discriminator_loss(real_output, fake_output):
#     cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
#     real_loss = cross_entropy(tf.ones_like(real_output), real_output)
#     fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
#     total_loss = real_loss + fake_loss
#     return total_loss
    
def binary_accuracy(y_true, y_pred):
    '''Calculates the mean accuracy rate across all predictions for binary
    classification problems.
    '''
    return np.mean(np.equal(y_true, np.array(y_pred>= 0.5).astype(int)))
    