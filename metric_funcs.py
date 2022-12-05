import tensorflow as tf
tf.random.set_seed(42)

from tensorflow.keras.layers import Conv2D, BatchNormalization, Input, GlobalAveragePooling2D, Dense, LeakyReLU, Conv2DTranspose
from tensorflow.keras.optimizers import Adam

import os
import time
import PIL
import imageio
import glob
import random

import numpy as np

from matplotlib import pyplot as plt
from tqdm import tqdm

from IPython import display
from astropy.io import fits
from astropy import stats

from photutils.detection import DAOStarFinder
from photutils.aperture import CircularAperture, aperture_photometry

from matplotlib.gridspec import GridSpec
from multiprocessing import Pool

import seaborn as sns
import gc


def huber_loss(gen_output, target, δ):
    err = gen_output - target
    err_abs = tf.abs(err)
    Lh = tf.where(err_abs < δ, 0.5 * tf.square(err), δ * (err_abs - 0.5 * δ))
    return tf.reduce_sum(Lh)
    
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

# Function used for validation loss computations
def loss_test(gen_output, target, Y_source_cat, α):
    Lh = huber_loss(gen_output, target, 5*tf.math.reduce_std(target))
    Lstats = comp_stats(gen_output, target)
    Lflux = comp_flux(gen_output, target, Y_source_cat)
    return α*Lh, Lstats, (1.0 - α) * Lflux


def completeness(gen_output, target, Y_source_cat, flux_bins):
    # This metric is more of a positional nature: "if the true source has X flux, do I generate this source within 10 arcseconds of true location?"
    # Detect sources in generated image
    # Threshold >= 0.9 mJy, since we also attempt to measure the generated sources whose true fluxes are 1 mJy
    # Sources with positions that are within 10'' of a real source are counted as True Positive (TP) else False Positive
    # 

    target = np.squeeze(target)
    gen_output = np.squeeze(gen_output)
    source_finder = DAOStarFinder(fwhm=7.9, threshold=3.9/1000)
    search_r = 10 / 1 # 10 [pixels] / 1 [pixel/('')]
    TP = np.zeros(len(flux_bins))
    FN = np.zeros(len(flux_bins))

    for i in range(target.shape[0]):
        gen_sources = source_finder(gen_output[i])
        try:
            tr_gen = np.transpose((gen_sources['xcentroid'], gen_sources['ycentroid']))
        except:
            continue
        # Iterate over all true sources in image
        mask = np.where(np.int16(Y_source_cat[:,-1]) == i)[0]
        for s in Y_source_cat[mask,-4:-1]:
            r = np.sqrt((s[1] - tr_gen[:,0])**2 + (s[2] - tr_gen[:,1])**2)
            rmin_idx = np.argmin(r)
            if r[rmin_idx] <= search_r:
                for idx, bin in enumerate(flux_bins):
                    if bin[0] <= s[0] < bin[1]:
                        TP[idx] += 1;
            else:
                for idx, bin in enumerate(flux_bins):
                    if bin[0] <= s[0] < bin[1]:
                        FN[idx] += 1;
    return TP, FN

def reliability(gen_output, target, Y_source_cat, flux_bins):
    # This metric is more of a positional nature: "if the generated source has X flux, is there a true source within 10 arcseconds of generated source location?"
    # Detect sources in generated image
    # Threshold >= 1mJy
    # real sources within 10'' of generated sources are counted as True Positive else False Positive
    # Quantifies precision
    # Note to self: if plots display values < 1mjy impose s['peak'] >= 1/1000

    target = np.squeeze(target)
    gen_output = np.squeeze(gen_output)

    source_finder = DAOStarFinder(fwhm=7.9, threshold=4/1000)
    search_r = 10 / 1 # 10 [pixels] / 1 [pixel/('')]
    TP = np.zeros(len(flux_bins))
    FP = np.zeros(len(flux_bins))

    for i in range(gen_output.shape[0]):
        gen_sources = source_finder(gen_output[i])
        try:
            tr_gen = np.transpose((gen_sources['xcentroid'], gen_sources['ycentroid']))
        except:
            continue
        # Iterate over all true sources in image
        mask = np.where(np.int16(Y_source_cat[:,-1]) == i)[0]
        for s in gen_sources:
            r = np.sqrt((s['xcentroid'] - Y_source_cat[mask, -3])**2 + (s['ycentroid'] - Y_source_cat[mask, -2])**2)
            # SafeGuard
            if len(r) == 0:
                continue
            rmin_idx = np.argmin(r)
            if r[rmin_idx] <= search_r:
                for idx, bin in enumerate(flux_bins):
                    if bin[0] <= s['peak'] < bin[1]:
                        TP[idx] += 1;
            else:
                for idx, bin in enumerate(flux_bins):
                    if bin[0] <= s['peak'] < bin[1]:
                        FP[idx] += 1;
    return TP, FP


# def completeness(gen_output, target):
#     # Detect sources in target image
#     # Detect sources in generated image
#     # SNR >= 5

#     target_finder = DAOStarFinder(fwhm=10, threshold=5*np.std(target))
#     search_r = 10 # pixels --> must correspond to 10 arcseconds
    
#     # Initialize TP, FP, aper_flux arrays
#     n_bins = 100 # number of bins
#     max_binval = .5 # Jy/beam
#     TP = [0 for i in range(n_bins)]
#     FN = [0 for i in range(n_bins)]
#     aper_flux_bins = [(i*max_binval/n_bins, (i+1)*max_binval/n_bins) for i in range(n_bins)]
#     for i in range(target.shape[0]):
#         target_sources = target_finder(target[i])
#         gen_sources = target_finder(gen_output[i])
#         try:
#             tr_target = np.transpose((target_sources['xcentroid'], target_sources['ycentroid']))
#             tr_gen = np.transpose((gen_sources['xcentroid'], gen_sources['ycentroid']))
#         except:
#             continue

#         # Iterate over all sources in target
#         for targ_source in tr_target:
#             source_found = False
#             # Calculate the peak flux of the target source
#             mask = CircularAperture(targ_source, r=search_r).to_mask(method='center')
#             target_peakflux = np.max(mask.multiply(target[i]))
#             # plt.imshow(mask.multiply(gen_output[i]), cmap="gnuplot2", aspect="auto")
#             # plt.colorbar()
#             # plt.show()
#             # First look if there is a source within 10 arcseconds
#             for gen_source in tr_gen:
#                 if np.sqrt((gen_source[0] - targ_source[0])**2 + (gen_source[1] - targ_source[1])**2) <= search_r: # arcseconds
#                     for bin_idx, bin in enumerate(aper_flux_bins):                        
#                         if bin[0] <= target_peakflux < bin[1]:
#                             TP[bin_idx] += 1
#                             source_found = True
#                             break;
                
#             if source_found == False:
#                 for bin_idx, bin in enumerate(aper_flux_bins):
#                         if bin[0] <= target_peakflux < bin[1]:
#                             FN[bin_idx] += 1
#     return aper_flux_bins, np.array(TP), np.array(FN)

# def reliability(gen_output, target):
#     # Detect sources in target image
#     # Detect sources in generated image
#     # SNR >= 1
#     target_finder = DAOStarFinder(fwhm=10, threshold=np.std(target))
#     search_r = 10 # pixels --> must correspond to 10 arcseconds
    
#     # Initialize TP, FP, aper_flux arrays
#     n_bins = 100 # number of bins
#     max_binval = .5 # Jy/beam
#     TP = [0 for i in range(n_bins)]
#     FP = [0 for i in range(n_bins)]
#     aper_flux_bins = [(i*max_binval/n_bins, (i+1)*max_binval/n_bins) for i in range(n_bins)]
#     for i in range(gen_output.shape[0]):
#         target_sources = target_finder(target[i])
#         gen_sources = target_finder(gen_output[i])
#         try:
#             tr_target = np.transpose((target_sources['xcentroid'], target_sources['ycentroid']))
#             tr_gen = np.transpose((gen_sources['xcentroid'], gen_sources['ycentroid']))
#         except:
#             continue
#         # Iterate over all sources in target
#         for gen_source in tr_gen:
#             source_found = False
#             # Calculate the peak flux of the target source
#             mask = CircularAperture(gen_source, r=search_r).to_mask(method='center')
#             gen_peakflux = np.max(mask.multiply(gen_output[i]))
#             # First look if there is a source within 10 arcseconds in real image
#             for targ_source in tr_target:
#                 if np.sqrt((gen_source[0] - targ_source[0])**2 + (gen_source[1] - targ_source[1])**2) <= search_r: # arcseconds
#                     for bin_idx, bin in enumerate(aper_flux_bins):
#                         if bin[0] <= gen_peakflux < bin[1]:
#                             TP[bin_idx] += 1
#                             source_found = True
#                             break;                  
                
#             if source_found == False:
#                 for bin_idx, bin in enumerate(aper_flux_bins):
#                         if bin[0] <= gen_peakflux < bin[1]:
#                             FP[bin_idx] += 1

#     return aper_flux_bins, np.array(TP), np.array(FP)


def flux_correlation(gen_output, target, Y_source_cat):
    # Link every true source flux with its corresponding generated flux
    # True source flux are embedded in the catalogue
    # generared source flux is calculated by calculating peak flux in aperture at position of the true source in the generated image

    source_finder = DAOStarFinder(fwhm=7.9, threshold=3.9/1000)
    gen_output = np.squeeze(gen_output)
    target = np.squeeze(target)

    search_r = 10 / 1 # 10 [pixels] / 1 [pixel/('')]

    target_galx_fluxes = []
    gen_galx_fluxes = []

    target_galx_aperfluxes = []
    gen_galx_aperfluxes = []

    for i in range(target.shape[0]):
        gen_sources = source_finder(gen_output[i])
        try:
            tr_gen = np.transpose((gen_sources['xcentroid'], gen_sources['ycentroid']))
        except:
            continue
        # Iterate over all true sources in image
        mask = np.where(np.int16(Y_source_cat[:,-1]) == i)[0]
        #print(np.int16(Y_source_cat[:,-1]))
        for s in Y_source_cat[mask,-4:-1]:
            r = np.sqrt((s[1] - tr_gen[:,0])**2 + (s[2] - tr_gen[:,1])**2)
            rmin_idx = np.argmin(r)
            if r[rmin_idx] <= search_r:
                target_galx_fluxes.append(s[0])
                gen_galx_fluxes.append(gen_sources['peak'][rmin_idx])

                aperture_target = CircularAperture(s[1:,], r=10)
                # masks = aperture_target.to_mask(method='center')
                # plt.imshow(masks.multiply(target[i]))
                # plt.show()
                aperture_gen = CircularAperture(tr_gen[rmin_idx], r=10)
                target_galx_aperfluxes.append(aperture_photometry(target[i], aperture_target)['aperture_sum'])
                gen_galx_aperfluxes.append(aperture_photometry(gen_output[i], aperture_gen)['aperture_sum'])


    return target_galx_fluxes, gen_galx_fluxes, target_galx_aperfluxes, gen_galx_aperfluxes