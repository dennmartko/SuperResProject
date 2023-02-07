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
from photutils.segmentation import detect_sources
from photutils.aperture import CircularAperture, aperture_photometry

from matplotlib.gridspec import GridSpec
from multiprocessing import Pool
from skimage.metrics import structural_similarity as ssim
from astropy.stats import sigma_clipped_stats
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

def comp_aperflux(gen_output, target, Y_source_cat):
    total_aperflux_diff = 0
    gen_output = np.squeeze(gen_output)
    target = np.squeeze(target)
    apertures = CircularAperture(Y_source_cat[:,1:-1], r=7.9)
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

def comp_peakflux(gen_output, target, Y_source_cat):
    total_peakflux_diff = 0
    gen_output = np.squeeze(gen_output)
    target = np.squeeze(target)
    apertures = CircularAperture(Y_source_cat[:,1:-1], r=7.9)
    for i in tf.range(target.shape[0]):
        mask = np.where(np.int16(Y_source_cat[:,-1]) == i)[0]
        aperture_masks = apertures[mask].to_mask(method='center')
        for idx, k in enumerate(mask):
            total_peakflux_diff += abs(Y_source_cat[k,0] - np.max(aperture_masks[idx].multiply(gen_output[i])))/len(mask)
    return np.float32(total_peakflux_diff)

# Function used for validation loss computations
def loss_test(gen_output, target, Y_source_cat, α):
    Lh = huber_loss(gen_output, target, 5*tf.math.reduce_std(target))
    Lstats = comp_stats(gen_output, target)
    Laperflux = comp_aperflux(gen_output, target, Y_source_cat)
    Lpeakflux = comp_peakflux(gen_output, target, Y_source_cat)
    for i in range(len(gen_output)):
        if i == 0:
            Lssim = ssim(np.squeeze(gen_output)[i], np.squeeze(target)[i], gaussian_weights=True, k1=0.01, k2=0.03, sigma=1.5)
        else:
            Lssim += ssim(np.squeeze(gen_output)[i], np.squeeze(target)[i], gaussian_weights=True, k1=0.01, k2=0.03, sigma=1.5)
    return Lh, Lstats, Laperflux, Lpeakflux, Lssim/len(gen_output)
    
def ssim_map(gen_output, target):
    for i in range(len(gen_output)):
        if i == 0:
            _, ssim_img = ssim(np.squeeze(gen_output)[i], np.squeeze(target)[i], gaussian_weights=True, k1=0.01, k2=0.03, sigma=1.5, full=True)
        else:
            _, ssim_img_tmp = ssim(np.squeeze(gen_output)[i], np.squeeze(target)[i], gaussian_weights=True, k1=0.01, k2=0.03, sigma=1.5, full=True)
            ssim_img += ssim_img_tmp
    return ssim_img/len(gen_output)


# def completeness(gen_output, target, Y_source_cat, flux_bins):
#     # This metric is more of a positional nature: "if the true source has X flux, do I generate this source within 10 arcseconds of true location?"
#     # Detect sources in generated image
#     # Threshold >= 3.9 mJy, since we also attempt to measure the generated sources whose true fluxes are 1 mJy
#     # Sources with positions that are within 8'' of a real source are counted as True Positive (TP) else False Positive
#     # 

#     target = np.squeeze(target)
#     gen_output = np.squeeze(gen_output)
#     source_finder = DAOStarFinder(fwhm=7.9, threshold=3.8/1000)
#     search_r = 8 / 1 # 8 [pixels] / 1 [pixel/('')]
#     TP = np.zeros(len(flux_bins))
#     FN = np.zeros(len(flux_bins))

#     for i in range(target.shape[0]):
#         gen_sources = source_finder(gen_output[i])
#         try:
#             tr_gen = np.transpose((gen_sources['xcentroid'], gen_sources['ycentroid']))
#         except:
#             continue
#         # Iterate over all true sources in image
#         mask = np.where(np.int16(Y_source_cat[:,-1]) == i)[0]
#         for s in Y_source_cat[mask,-4:-1]:
#             r = np.sqrt((s[1] - tr_gen[:,0])**2 + (s[2] - tr_gen[:,1])**2)
#             rmin_idx = np.argmin(r)
#             if r[rmin_idx] <= search_r:
#                 for idx, bin in enumerate(flux_bins):
#                     if bin[0] <= s[0] < bin[1]:
#                         TP[idx] += 1;
#             else:
#                 for idx, bin in enumerate(flux_bins):
#                     if bin[0] <= s[0] < bin[1]:
#                         FN[idx] += 1;
#     return TP, FN

# def reliability(gen_output, target, Y_source_cat, flux_bins):
#     # This metric is more of a positional nature: "if the generated source has X flux, is there a true source within 10 arcseconds of generated source location?"
#     # Detect sources in generated image
#     # Threshold >= 1mJy
#     # real sources within 10'' of generated sources are counted as True Positive else False Positive
#     # Quantifies precision
#     # Note to self: if plots display values < 1mjy impose s['peak'] >= 1/1000

#     target = np.squeeze(target)
#     gen_output = np.squeeze(gen_output)

#     source_finder = DAOStarFinder(fwhm=7.9, threshold=4/1000)
#     search_r = 8 / 1 # 10 [pixels] / 1 [pixel/('')]
#     TP = np.zeros(len(flux_bins))
#     FP = np.zeros(len(flux_bins))

#     for i in range(gen_output.shape[0]):
#         gen_sources = source_finder(gen_output[i])
#         try:
#             tr_gen = np.transpose((gen_sources['xcentroid'], gen_sources['ycentroid']))
#         except:
#             continue
#         # Iterate over all true sources in image
#         mask = np.where(np.int16(Y_source_cat[:,-1]) == i)[0]
#         for s in gen_sources:
#             r = np.sqrt((s['xcentroid'] - Y_source_cat[mask, -3])**2 + (s['ycentroid'] - Y_source_cat[mask, -2])**2)
#             # SafeGuard
#             if len(r) == 0:
#                 continue
#             rmin_idx = np.argmin(r)
#             if r[rmin_idx] <= search_r:
#                 for idx, bin in enumerate(flux_bins):
#                     if bin[0] <= s['peak'] < bin[1]:
#                         TP[idx] += 1;
#             else:
#                 for idx, bin in enumerate(flux_bins):
#                     if bin[0] <= s['peak'] < bin[1]:
#                         FP[idx] += 1;
#     return TP, FP

def can_connect_circles(circles, PSF):
    # Depth-First Search based method
    def distance(c1, c2):
        return np.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)

    def can_connect(c1, c2):
        return distance(c1, c2) <= PSF

    def dfs(node, visited):
        visited.add(node)
        for neighbor in adjacency_list[node]:
            if neighbor not in visited:
                if dfs(neighbor, visited):
                    return True
        return False

    # create an adjacency list representation of the graph
    adjacency_list = {i: [] for i in range(len(circles))}
    for i in range(len(circles)):
        for j in range(i+1, len(circles)):
            if can_connect(circles[i], circles[j]):
                adjacency_list[i].append(j)
                adjacency_list[j].append(i)

    # check if all circles can be connected using a depth-first search
    visited = set()
    for i in range(len(circles)):
        if dfs(i, visited):
            return True
    return False

def confusion_score(gen_output, target, Y_source_cat, confusion_df):
    target = np.squeeze(target)
    gen_output = np.squeeze(gen_output)
    _, __, std_noise = sigma_clipped_stats(target, sigma=2, maxiters=50)
    source_finder = DAOStarFinder(fwhm=7.9, threshold=1*std_noise)
    source_finder_target = DAOStarFinder(fwhm=7.9, threshold=5*std_noise)
    search_r = 8 / 1 # 8 [pixels] / 1 [pixel/('')]

    # Completeness
    for i in range(target.shape[0]):
        gen_sources = source_finder(gen_output[i])
        try:
            tr_gen = np.transpose((gen_sources['xcentroid'], gen_sources['ycentroid']))
        except:
            continue
        # Iterate over all true sources in image
        mask = np.where(np.int16(Y_source_cat[:,-1]) == i)[0]
        for s in Y_source_cat[mask, 0:-1]:
            r = np.sqrt((s[1] - tr_gen[:,0])**2 + (s[2] - tr_gen[:,1])**2)
            rmin_idx = np.argmin(r)
            if r[rmin_idx] <= search_r:
                for idx, bin in enumerate(confusion_df['Flux bins']):
                    if bin[0] <= s[0] < bin[1]:
                        confusion_df.loc[idx, 'TPc'] += 1;
            else:
                for idx, bin in enumerate(confusion_df['Flux bins']):
                    if bin[0] <= s[0] < bin[1]:
                        confusion_df.loc[idx, 'FNc'] += 1;

    # Reliability
    for i in range(gen_output.shape[0]):
        gen_sources = source_finder(gen_output[i])
        try:
            tr_gen = np.transpose((gen_sources['xcentroid'], gen_sources['ycentroid']))
        except:
            continue
        # Iterate over all generated sources in image
        mask = np.where(np.int16(Y_source_cat[:,-1]) == i)[0]
        for s in gen_sources:
            r = np.sqrt((s['xcentroid'] - Y_source_cat[mask, 1])**2 + (s['ycentroid'] - Y_source_cat[mask, 2])**2)
            # SafeGuard, no sources in either generated or true image
            if len(r) == 0:
                continue
            rmin_idx = np.argmin(r)
            if r[rmin_idx] <= search_r:
                # Reliability Issue Detection
                potential_flags = np.where(r <= search_r)[0]
                if len(potential_flags) > 1:
                    true_sources_in_fhwm = Y_source_cat[mask, 1:-1][potential_flags]
                    # Try to connect each true source under the PSF 'umbrella' of a generated source
                    # If this is possible, we have no issue and the generated source is "fine"
                    # If this is not possible, there is a source-source seperation > PSF. 
                    # These should not be under the PSF of the generated source, and we should have atleast 2 seperate generated sources.
                    if not can_connect_circles(true_sources_in_fhwm, search_r):
                        # Flag found! --> Update table
                        for idx, bin in enumerate(confusion_df['Flux bins']):
                            if bin[0] <= s['peak'] < bin[1]:
                                confusion_df.loc[idx, 'flag_TPr'] += 1;
                        
                # Currently flags are considered matches, therefore continue anyways
                for idx, bin in enumerate(confusion_df['Flux bins']):
                    if bin[0] <= s['peak'] < bin[1]:
                        confusion_df.loc[idx, 'TPr'] += 1;
            else:
                for idx, bin in enumerate(confusion_df['Flux bins']):
                    if bin[0] <= s['peak'] < bin[1]:
                        confusion_df.loc[idx, 'FPr'] += 1;
    #print(confusion_df)
    return confusion_df

def find_matches(gen_output, target, Y_source_cat, matches_dict):
    # First we find all matches generated sourves vs true sources
    _, __, std_noise = sigma_clipped_stats(target, sigma=2, maxiters=50)
    source_finder = DAOStarFinder(fwhm=7.9, threshold=1*std_noise)
    source_finder_target = DAOStarFinder(fwhm=7.9, threshold=5*std_noise)
    gen_output = np.squeeze(gen_output)
    target = np.squeeze(target)
    search_r = 8 / 1 # 10 [pixels] / 1 [pixel/('')]
    aper_area = np.pi*7.9**2 # Area in arcseconds --> Each arcsecond = 1 pixel, hence no pixel conversion needed

    for i in range(gen_output.shape[0]):
        gen_sources = source_finder(gen_output[i])
        try:
            tr_gen = np.transpose((gen_sources['xcentroid'], gen_sources['ycentroid']))
        except:
            continue
        # Iterate over "true" sources in image
        mask = np.where(np.int16(Y_source_cat[:,-1]) == i)[0]
        Y_source_cat_modified = Y_source_cat[mask].copy()
        Y_source_cat_modified[:,1:-1] += np.random.uniform(-30, 30, size=Y_source_cat_modified[:,1:-1].shape) 
        for s in Y_source_cat[mask,0:-1]:
            r = np.sqrt((s[1] - tr_gen[:,0])**2 + (s[2] - tr_gen[:,1])**2)

            # SafeGuard, no sources in either generated or true image
            if len(r) == 0:
                continue
            rmin_idx = np.argmin(r)
            if r[rmin_idx] <= search_r:
                matches_dict['Sout1'].append(gen_sources['peak'][rmin_idx])
                matches_dict['Sin_true'].append(s[0])
                matches_dict['true_offset'].append(r[rmin_idx])
                matches_dict['true_xy_offset'][0].append(s[1] - tr_gen[rmin_idx,0])
                matches_dict['true_xy_offset'][1].append(s[2] - tr_gen[rmin_idx,1]) 
        for s in Y_source_cat_modified[:,0:-1]:
            r = np.sqrt((s[1] - tr_gen[:,0])**2 + (s[2] - tr_gen[:,1])**2)

            # SafeGuard, no sources in either generated or true image
            if len(r) == 0:
                continue
            rmin_idx = np.argmin(r)
            if r[rmin_idx] <= search_r:
                matches_dict['Sout2'].append(gen_sources['peak'][rmin_idx])
                matches_dict['Sin_rnd'].append(s[0])
                matches_dict['rnd_offset'].append(r[rmin_idx])
                matches_dict['rnd_xy_offset'][0].append(s[1] - tr_gen[rmin_idx,0])
                matches_dict['rnd_xy_offset'][1].append(s[2] - tr_gen[rmin_idx,1]) 

        # Flag distribution
        for s in Y_source_cat[mask,0:-1]:
            r = np.sqrt((s[1] - tr_gen[:,0])**2 + (s[2] - tr_gen[:,1])**2)

            # SafeGuard, no sources in either generated or true image
            if len(r) == 0:
                continue
            rmin_idx = np.argmin(r)
            if r[rmin_idx] <= search_r:
                r_from_gen = np.sqrt((tr_gen[:,0][rmin_idx] - Y_source_cat[mask, 1])**2 + (tr_gen[:,1][rmin_idx] - Y_source_cat[mask, 2])**2)
                # Reliability Issue Detection
                potential_flags = np.where(r_from_gen <= search_r)[0]
                if len(potential_flags) > 1:
                    true_sources_in_fhwm = Y_source_cat[mask, 1:-1][potential_flags]
                    # Try to connect each true source under the PSF 'umbrella' of a generated source
                    # If this is possible, we have no issue and the generated source is "fine"
                    # If this is not possible, there is a source-source seperation > PSF. 
                    # These should not be under the PSF of the generated source, and we should have atleast 2 seperate generated sources.
                    if not can_connect_circles(true_sources_in_fhwm, search_r):
                        matches_dict['Sout3'].append(gen_sources['peak'][rmin_idx])
                        matches_dict['Sin_flag'].append(s[0])
                        matches_dict['flag_offset'].append(r[rmin_idx])
                        matches_dict['flag_xy_offset'][0].append(s[1] - tr_gen[rmin_idx,0])
                        matches_dict['flag_xy_offset'][1].append(s[2] - tr_gen[rmin_idx,1])
    return matches_dict

# def flux_correlation(gen_output, target, Y_source_cat):
#     # Link every true source flux with its corresponding generated flux
#     # True source flux are embedded in the catalogue
#     # generared source flux is calculated by calculating peak flux in aperture at position of the true source in the generated image

#     source_finder = DAOStarFinder(fwhm=7.9, threshold=3.8/1000)
#     gen_output = np.squeeze(gen_output)
#     target = np.squeeze(target)

#     search_r = 8 / 1 # 10 [pixels] / 1 [pixel/('')]
#     aper_area = np.pi*7.9**2 # Area in arcseconds --> Each arcsecond = 1 pixel, hence no pixel conversion needed

#     target_galx_fluxes = []
#     gen_galx_fluxes = []

#     distances = []
#     x_offset = []
#     y_offset = []

#     target_galx_aperfluxes = []
#     gen_galx_aperfluxes = []

#     for i in range(target.shape[0]):
#         gen_sources = source_finder(gen_output[i])
#         try:
#             tr_gen = np.transpose((gen_sources['xcentroid'], gen_sources['ycentroid']))
#         except:
#             continue
#         # Iterate over all true sources in image
#         mask = np.where(np.int16(Y_source_cat[:,-1]) == i)[0]
#         #print(np.int16(Y_source_cat[:,-1]))
#         for s in Y_source_cat[mask,-4:-1]:
#             r = np.sqrt((s[1] - tr_gen[:,0])**2 + (s[2] - tr_gen[:,1])**2)
#             rmin_idx = np.argmin(r)

#             if r[rmin_idx] <= search_r:
#                 target_galx_fluxes.append(s[0])
#                 gen_galx_fluxes.append(gen_sources['peak'][rmin_idx])

#                 aperture_target = CircularAperture(s[1:,], r=7.9)
#                 # masks = aperture_target.to_mask(method='center')
#                 # plt.imshow(masks.multiply(target[i]))
#                 # plt.show()
#                 aperture_gen = CircularAperture(tr_gen[rmin_idx], r=7.9)
#                 target_galx_aperfluxes.append(aperture_photometry(target[i], aperture_target)['aperture_sum'].data[0]/aper_area)
#                 gen_galx_aperfluxes.append(aperture_photometry(gen_output[i], aperture_gen)['aperture_sum'].data[0]/aper_area)

#                 distances.append(r[rmin_idx])
#                 x_offset.append(s[1] - tr_gen[rmin_idx,0])
#                 y_offset.append(s[2] - tr_gen[rmin_idx,1])

#     return target_galx_fluxes, gen_galx_fluxes, target_galx_aperfluxes, gen_galx_aperfluxes, distances, x_offset, y_offset 