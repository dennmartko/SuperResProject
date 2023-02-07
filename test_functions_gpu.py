######################
###     IMPORTS    ###
######################

import tensorflow as tf
tf.random.set_seed(42)

import os
import configparser
import gc

import numpy as np
import pandas as pd

from astropy.io import fits
from PlotLib.PlotFunctionsTest import confusion_plot, hexplot, insetplot, PS_plot, offsetplot, SSIM_plot, write_metrics_to_file #reliability_plot, completeness_plot, comparison_plot, hexplot
from metric_funcs import loss_test, find_matches, ssim_map, confusion_score

#######################
###    TEST CLASS   ###
#######################
class SRTesterGPU():
    def __init__(self, path_to_config, gridmode = False, idx = None) -> None:
        # Load configuration file for training
        self.config = configparser.ConfigParser()
        self.config.read(path_to_config)

        # Create any missing directories and initialize necessary parameters
        self.DIM = (424, 424)

        ## Paths with train data
        self.path_test = self.config['COMMON']['path_test'].rstrip().lstrip()

        ## Indicate purpose of run
        self.RUN_NAME = self.config['COMMON']['RUN_NAME'].rstrip().lstrip()
        if gridmode == True:
            self.RUN_NAME += f'_gridmode_{idx}'

        ## Load path to models dir
        self.models_lib_path = self.config['COMMON']['model_outdir'].rstrip().lstrip()
        self.model_path = os.path.join(self.models_lib_path, self.RUN_NAME)

        ## Set classes
        self.classes = [i.strip(' ') for i in self.config['COMMON']['input'].rstrip().lstrip().split(",")] + [self.config['COMMON']['target'].rstrip().lstrip()] # Inp first, target last

        self.TOTAL_SAMPLES = len([entry for entry in os.listdir(os.path.join(self.path_test, self.classes[0]))])
        self.tdir_out = os.path.join(self.model_path, "test_results")

        if not os.path.isdir(self.tdir_out):
            os.mkdir(self.tdir_out)

        #Register GridMode
        self.gridmode = gridmode
    def LoadTestData(self):
        self.TEST_BATCH_SIZE = 10
        self.test_arr_X = np.zeros((self.TOTAL_SAMPLES, 3, 106, 106))
        self.test_arr_Y = np.zeros((self.TOTAL_SAMPLES, 1, 424, 424))

        for i in range(self.TOTAL_SAMPLES):
            for k in range(len(self.classes)):
                with fits.open(os.path.join(self.path_test, f"{os.path.join(self.classes[k],self.classes[k])}_{i}.fits")) as hdu:
                    if k == len(self.classes) - 1:
                        self.test_arr_Y[i] = hdu[0].data
                        arr = np.array([list(row) for row in hdu[1].data])
                        if i == 0:
                            arr = np.column_stack((arr, np.full(len(arr), i)))
                            self.target_image_sources_cat_test = arr.copy()
                        else:
                            arr = np.column_stack((arr, np.full(len(arr), i)))
                            self.target_image_sources_cat_test = np.vstack((self.target_image_sources_cat_test, arr))

                        del arr;
                    else:
                        self.test_arr_X[i][k] = hdu[0].data    

        # Free memory
        gc.collect()

    def LoadModel(self, GridVector = None):
        self.generator = tf.keras.models.load_model(os.path.join(self.model_path, 'BestValid_Model'))
        gc.collect()

    def TestAnalysis(self, α = None):
        # Check α
        if self.gridmode == True:
            assert α is not None, "α must be given"
        else:
            α = float(self.config['TRAINING PARAMETERS']['alpha'].rstrip().lstrip())

        # Calculate Metrics
        self.Lh = 0; self.Lstats = 0; self.Lflux = 0; self.Laperflux = 0; self.Lpeakflux = 0; self.Lssim = 0;
        ## Create an array of logarithmically spaced values from 4 mJy to max bin value
        max_bin_value = 150 # mJy
        num_bins = 30
        values = np.logspace(np.log10(4), np.log10(max_bin_value), num_bins + 1, base=10)/1000
        ## Zip the values array with itself shifted by one position to the left to create tuples of the left and right bounds of each bin
        self.flux_bins = list(zip(values[:-1], values[1:]))
        zero_list = np.zeros(len(self.flux_bins))
        confusion_dict = {'Flux bins':self.flux_bins, 'TPc': zero_list, 'TPr': zero_list, 'FNc': zero_list, 'FPr': zero_list, 'flag_TPr': zero_list, 'C': zero_list, 'R': zero_list, 'flag_R': zero_list}
        confusion_df = pd.DataFrame(confusion_dict)

        matches_dict = {'Sout1':[], 'Sout2':[], "Sout3":[], "Sin_true":[], "Sin_rnd":[],"Sin_flag":[], "true_offset":[], "rnd_offset":[], "flag_offset":[], "true_xy_offset":[[],[]], "rnd_xy_offset":[[],[]], "flag_xy_offset":[[],[]]}

        for batch_idx in range(self.test_arr_X.shape[0]//self.TEST_BATCH_SIZE):
            X = self.test_arr_X[batch_idx*self.TEST_BATCH_SIZE:self.TEST_BATCH_SIZE*(batch_idx + 1)].astype(np.float32) if batch_idx != (self.test_arr_X.shape[0]//self.TEST_BATCH_SIZE - 1) else self.test_arr_X[batch_idx*self.TEST_BATCH_SIZE:].astype(np.float32)
            Y = self.test_arr_Y[batch_idx*self.TEST_BATCH_SIZE:self.TEST_BATCH_SIZE*(batch_idx + 1)].astype(np.float32) if batch_idx != (self.test_arr_Y.shape[0]//self.TEST_BATCH_SIZE - 1) else self.test_arr_Y[batch_idx*self.TEST_BATCH_SIZE:].astype(np.float32)
            Y_source_cat = list()
            idx_arr = np.arange(batch_idx*self.TEST_BATCH_SIZE, self.TEST_BATCH_SIZE*(batch_idx + 1)) if batch_idx != (self.test_arr_X.shape[0]//self.TEST_BATCH_SIZE - 1) else np.arange(batch_idx*self.TEST_BATCH_SIZE, self.test_arr_X.shape[0])
            for j, k in enumerate(idx_arr):
                Y_source_cat.append(self.target_image_sources_cat_test[np.where(self.target_image_sources_cat_test[:,-1] == k)])
                # Synchronize batch idx and catalogue image idx
                Y_source_cat[-1][:,-1] = j
            Y_source_cat = np.vstack(Y_source_cat)
            gen_valid = self.generator(X, training=False).numpy()

            # Loss metrics
            Lh_batch, Lstats_batch, Laperflux_batch, Lpeakflux_batch, Lssim = loss_test(gen_valid, Y, Y_source_cat, α)
            self.Lh += Lh_batch; self.Lstats += Lstats_batch; self.Laperflux += Laperflux_batch; self.Lpeakflux += Lpeakflux_batch; self.Lssim += Lssim/(self.test_arr_X.shape[0]//self.TEST_BATCH_SIZE)
            
            # Completeness and Reliability metrics
            confusion_df = confusion_score(gen_valid, Y, Y_source_cat, confusion_df)
            # if batch_idx == 0:
            #     TPc, FNc = completeness(gen_valid, Y, Y_source_cat, self.flux_bins)
            #     TPr, FPr = reliability(gen_valid, Y, Y_source_cat, self.flux_bins)
            # else:
            #     tpc, fnc = completeness(gen_valid, Y, Y_source_cat, self.flux_bins)
            #     tpr, fpr = reliability(gen_valid, Y, Y_source_cat, self.flux_bins)
            #     TPc += tpc; FNc += fnc;
            #     TPr += tpr; FPr += fpr;

            # Hexplot metrics
            matches_dict = find_matches(gen_valid, Y, Y_source_cat, matches_dict)
            # if batch_idx == 0:
            #     target_galx_fluxes, gen_galx_fluxes, target_galx_aperfluxes, gen_galx_aperfluxes, distances, x_offset, y_offset = flux_correlation(gen_valid, Y, Y_source_cat)
            # else:
            #     flux_out = flux_correlation(gen_valid, Y, Y_source_cat)
            #     target_galx_fluxes += flux_out[0]; gen_galx_fluxes += flux_out[1]; target_galx_aperfluxes += flux_out[2]; gen_galx_aperfluxes += flux_out[3]; distances += flux_out[4]; x_offset += flux_out[5]; y_offset += flux_out[6]

            # SSIM metric
            if batch_idx == 0:
                self.ssim_img = ssim_map(gen_valid, Y)/(self.test_arr_X.shape[0]//self.TEST_BATCH_SIZE)
            else:
                self.ssim_img += ssim_map(gen_valid, Y)/(self.test_arr_X.shape[0]//self.TEST_BATCH_SIZE)

            # Insetplot
            # Tweak the number of loops to produce the number of desired plots, note that this number does not equal number of plots!
            # additional plot spam filter
            if batch_idx < 25:
                for i in range(2):
                    insetplot(gen_valid[i], Y[i], Y_source_cat[np.where(Y_source_cat[:,-1] == i)], os.path.join(self.tdir_out, f"insetplot_{batch_idx}_{i}.png"))

        # Calculate completeness and reliability of test sample
        ## If needed resolve zero-occurences
        for i in range(len(confusion_df['Flux bins'])):
            if confusion_df.loc[i, 'TPc'] + confusion_df.loc[i, 'FNc'] != 0:
                confusion_df.loc[i, 'C'] = confusion_df.loc[i, 'TPc']/(confusion_df.loc[i, 'TPc'] + confusion_df.loc[i, 'FNc'])
        
            if confusion_df.loc[i, 'TPr'] + confusion_df.loc[i, 'FPr'] != 0:
                confusion_df.loc[i, 'R'] = confusion_df.loc[i, 'TPr']/(confusion_df.loc[i, 'TPr'] + confusion_df.loc[i, 'FPr'])
                confusion_df.loc[i, 'flag_R'] = confusion_df.loc[i, 'flag_TPr']/(confusion_df.loc[i, 'TPr'] + confusion_df.loc[i, 'FPr'])
        
        # C = np.zeros_like(TPc)
        # R = np.zeros_like(TPr)

        # for bin_idx in range(C.shape[0]):
        #     if TPc[bin_idx] + FNc[bin_idx] == 0:
        #         C[bin_idx] = 0
        #     else:
        #         C[bin_idx] = TPc[bin_idx]/(TPc[bin_idx] + FNc[bin_idx])
        #     if TPr[bin_idx] + FPr[bin_idx] == 0:
        #         R[bin_idx] = 0
        #     else:
        #         R[bin_idx] = TPr[bin_idx]/(TPr[bin_idx] + FPr[bin_idx])
       
        # Set average completeness, reliability
        self.avg_c = np.mean(confusion_df['C'])
        self.avg_r = np.mean(confusion_df['R'])
        # self.avg_c = np.mean(C)
        # self.avg_r = np.mean(R)
        # Make plots
        confusion_plot(self.flux_bins, confusion_df, os.path.join(self.tdir_out, "confusionplot.png"))
        hexplot(matches_dict, os.path.join(self.tdir_out, "2DHistMatchesS.png"), os.path.join(self.tdir_out, "relSdiffratio_plot.png"))
        PS_plot(matches_dict, os.path.join(self.tdir_out, "2DHistMatchesPS.png"))
        offsetplot(matches_dict, os.path.join(self.tdir_out, "offsetplot.png"))
        # offsetplot(target_galx_aperfluxes, gen_galx_aperfluxes, distances, x_offset, y_offset, os.path.join(self.tdir_out, "radial_offset_scatterplot.png"), os.path.join(self.tdir_out, "xy_offset_scatterplot.png"), os.path.join(self.tdir_out, "radial_offset_histplot.png"))
        #TestMetricPlot(self.Lh, self.Lstats, self.Lflux, self.avg_r, self.avg_c, distances, self.Lssim, os.path.join(self.tdir_out, "metric_barplot.png"))
        SSIM_plot(self.ssim_img, os.path.join(self.tdir_out, "ssim_plot.png"))
        write_metrics_to_file([self.Lh, self.Lstats, self.Laperflux, self.Lpeakflux, self.avg_c, self.avg_r], ["Huber Loss", "Stats Loss", "AperFlux Loss", "PeakFlux Loss", "avg_c", "avg_r"], confusion_df, os.path.join(self.tdir_out, "TestMetricResults.txt"))
        # completeness_plot(c_aper_flux_bins, C, α, os.path.join(self.tdir_out, "completenessplot.png"))
        # reliability_plot(r_aper_flux_bins, R, α, os.path.join(self.tdir_out, "reliabilityplot.png"))

        # gen_slice, target_slice = np.squeeze(self.generator(self.test_arr_X[:4], training = False).numpy()), np.squeeze(self.test_arr_Y[:4])
        # comparison_plot(gen_slice, target_slice, os.path.join(self.tdir_out, "generated-validation.png"))


        # Compute Average Completeness, Reliability
        # os.path.join(tdir_out_analysis, "generated-validation.pdf")
        # self.avg_c = np.mean(C)
        # self.avg_r = np.mean(R)

        # # HexPlot
        # gc.collect()
        # gen_galx_fluxes = []
        # target_galx_fluxes = []

        # for i in range(self.test_arr_X.shape[0]//self.TEST_BATCH_SIZE):
        #     if i != self.test_arr_X.shape[0]//self.TEST_BATCH_SIZE - 1:
        #         target = np.squeeze(self.test_arr_Y[i*self.TEST_BATCH_SIZE:(i + 1)*self.TEST_BATCH_SIZE])
        #         gen_output = np.squeeze(self.generator(self.test_arr_X[i*self.TEST_BATCH_SIZE:(i + 1)*self.TEST_BATCH_SIZE], training = False).numpy())
        #     else:
        #         target = np.squeeze(self.test_arr_Y[i*self.TEST_BATCH_SIZE:])
        #         gen_output = np.squeeze(self.generator(self.test_arr_X[i*self.TEST_BATCH_SIZE:], training = False).numpy())

        #     partial_gen_galx_fluxes, partial_target_galx_fluxes = comp_flux_galx(gen_output, target)
        #     gen_galx_fluxes += partial_gen_galx_fluxes; target_galx_fluxes += partial_target_galx_fluxes

        # # Hexplot
        # hexplot(gen_galx_fluxes, target_galx_fluxes, os.path.join(self.tdir_out, f"hexplot.png"))

if __name__ == "__main__":
    SRModelTest = SRTesterGPU("TrainingConfig.ini")
    SRModelTest.LoadTestData()
    SRModelTest.LoadModel()
    SRModelTest.TestAnalysis(α = 0.01)