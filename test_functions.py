######################
###     IMPORTS    ###
######################

import tensorflow as tf
tf.random.set_seed(42)

import os
import configparser
import gc

import numpy as np

from astropy.io import fits
from PlotLib.PlotFunctionsTest import confusion_plot, hexplot, insetplot #reliability_plot, completeness_plot, comparison_plot, hexplot
from metric_funcs import completeness, reliability, loss_test, flux_correlation

#######################
###    TEST CLASS   ###
#######################
class SRTester():
    def __init__(self, path_to_config, gridmode = False, idx = None) -> None:
        # Load configuration file for training
        self.config = configparser.ConfigParser()
        self.config.read(path_to_config)

        # Toggle GPU/CPU
        if self.config['COMMON']['use_gpu'].rstrip().lstrip() == "False":
            tf.config.set_visible_devices([], 'GPU')

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
                with fits.open(os.path.join(self.path_test, f"{self.classes[k]}/{self.classes[k]}_{i}.fits")) as hdu:
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

        # NHWC
        self.test_arr_X = np.swapaxes(np.swapaxes(self.test_arr_X, 1, 2), 2, 3)        
        self.test_arr_Y = np.swapaxes(np.swapaxes(self.test_arr_Y, 1, 2), 2, 3)        

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
        self.Lh = 0; self.Lstats = 0; self.Lflux = 0;
        max_binval = 20/1000; n_bins = 20;
        self.flux_bins = [(i*max_binval/n_bins, (i+1)*max_binval/n_bins) for i in range(n_bins)]
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
            Lh_batch, Lstats_batch, Lflux_batch = loss_test(gen_valid, Y, Y_source_cat, α)
            self.Lh += Lh_batch/α; self.Lstats += Lstats_batch; self.Lflux += Lflux_batch/(1-α) if α != 1 else 0;      
            
            # Completeness and Reliability metrics
            if batch_idx == 0:
                TPc, FNc = completeness(gen_valid, Y, Y_source_cat, self.flux_bins)
                TPr, FPr = reliability(gen_valid, Y, Y_source_cat, self.flux_bins)
            else:
                tpc, fnc = completeness(gen_valid, Y, Y_source_cat, self.flux_bins)
                tpr, fpr = reliability(gen_valid, Y, Y_source_cat, self.flux_bins)
                TPc += tpc; FNc += fnc;
                TPr += tpr; FPr += fpr;

            # Hexplot metrics
            if batch_idx == 0:
                target_galx_fluxes, gen_galx_fluxes, target_galx_aperfluxes, gen_galx_aperfluxes = flux_correlation(gen_valid, Y, Y_source_cat)
            else:
                flux_out = flux_correlation(gen_valid, Y, Y_source_cat)
                target_galx_fluxes += flux_out[0]; gen_galx_fluxes += flux_out[1]; target_galx_aperfluxes += flux_out[2]; gen_galx_aperfluxes += flux_out[3];


            # Insetplot
            for i in range(6):
                insetplot(gen_valid[i], Y[i], Y_source_cat[np.where(Y_source_cat[:,-1] == i)], os.path.join(self.tdir_out, f"insetplot_{batch_idx}_{i}.png"))

        # Calculate completeness and reliability of test sample
        ## If needed resolve zero-occurences
        C = np.zeros_like(TPc)
        R = np.zeros_like(TPr)

        for bin_idx in range(C.shape[0]):
            if TPc[bin_idx] + FNc[bin_idx] == 0:
                C[bin_idx] = 0
            else:
                C[bin_idx] = TPc[bin_idx]/(TPc[bin_idx] + FNc[bin_idx])
            if TPr[bin_idx] + FPr[bin_idx] == 0:
                R[bin_idx] = 0
            else:
                R[bin_idx] = TPr[bin_idx]/(TPr[bin_idx] + FPr[bin_idx])

        # Make plots
        confusion_plot(self.flux_bins, C, R, os.path.join(self.tdir_out, "confusionplot.png"))
        hexplot(target_galx_fluxes, gen_galx_fluxes, target_galx_aperfluxes, gen_galx_aperfluxes, os.path.join(self.tdir_out, "hexplotpeak.png"), os.path.join(self.tdir_out, "hexplotaper.png"))
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
    SRModelTest = SRTester("TrainingConfig.ini")
    SRModelTest.LoadTestData()
    SRModelTest.LoadModel()
    SRModelTest.TestAnalysis(α = 0.01)