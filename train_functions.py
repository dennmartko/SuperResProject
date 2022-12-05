######################
###     IMPORTS    ###
######################

import tensorflow as tf
tf.random.set_seed(42)

import os
import random
import configparser
import datetime
import gc
import imageio
import glob
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from tqdm import tqdm
from astropy.io import fits
from ModelArchitectures.PaperModel import CreatePaperModel, CreateGridPaperModel
from ModelArchitectures.CustomModel1 import CustomModel1, GridCustomModel1
from loss_functions import loss, loss_valid
from PlotLib.PlotFunctionsTrain import TrainingSnapShot, LossComponentPlot, FilteredLossComponentPlot

#######################
###   TRAIN CLASS   ###
#######################
def printlog(msg, outfile):
    with open(outfile, 'a') as f:
        f.write(msg + '\n')

class SRTrainer():
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
        self.path_train = self.config['COMMON']['path_train'].rstrip().lstrip()

        ## Indicate purpose of run
        self.RUN_NAME = self.config['COMMON']['RUN_NAME'].rstrip().lstrip()
        if gridmode == True:
            self.RUN_NAME += f'_gridmode_{idx}'

        ## Create folder with trained models
        self.models_lib_path = self.config['COMMON']['model_outdir'].rstrip().lstrip()
        self.model_path = os.path.join(self.models_lib_path, self.RUN_NAME)

        ### if verbose is TRUE create file TrainingLog.log in model_path
        self.verbose = bool(self.config['COMMON']['training_verbose'].rstrip().lstrip())
        self.logfile_path = os.path.join(self.model_path, "TrainingLog.log")

        if not os.path.isdir(self.models_lib_path):
            os.mkdir(self.models_lib_path)
        if not os.path.isdir(self.model_path):
            os.mkdir(self.model_path)
        if self.verbose == True:
            open(self.logfile_path, 'w').close()

        ## Set classes
        self.classes = [i.strip(' ') for i in self.config['COMMON']['input'].rstrip().lstrip().split(",")] + [self.config['COMMON']['target'].rstrip().lstrip()] # Inp first, target last

        self.TOTAL_SAMPLES = len([entry for entry in os.listdir(os.path.join(self.path_train, self.classes[0]))])
        tdir_out = os.path.join(self.model_path, "train_results")
        self.tdir_out_progress = os.path.join(tdir_out, "train_progress")
        self.tdir_out_analysis = os.path.join(tdir_out, "train_analysis")

        if not os.path.isdir(tdir_out):
            os.mkdir(tdir_out)
        if not os.path.isdir(self.tdir_out_progress):
            os.mkdir(self.tdir_out_progress)
        if not os.path.isdir(self.tdir_out_analysis):
            os.mkdir(self.tdir_out_analysis)

        if self.verbose == True:
            printlog(f"{datetime.datetime.now()} - Successfully Configured Training Settings!", self.logfile_path)
            
        #Register GridMode
        self.gridmode = gridmode
        if self.gridmode == True:
            printlog(f"{datetime.datetime.now()} - Training finds itself in GridMode!", self.logfile_path)

    def LoadTrainingData(self):
        # Write to log
        if self.verbose == True:
            printlog(f"{datetime.datetime.now()} - Call to Load Data....", self.logfile_path)

        # Split Training data into validation and training set according to parameter validation_ratio
        validation_ratio = float(self.config['TRAINING PARAMETERS']['validation_ratio'].rstrip().lstrip())
        indices = np.arange(0, self.TOTAL_SAMPLES)
        TOTAL_SAMPLES_VALID = round(validation_ratio * self.TOTAL_SAMPLES)
        n_valid_indx = np.array(random.sample(indices.tolist(), TOTAL_SAMPLES_VALID))

        ## Load training and validation data into memory
        self.train_arr_X = np.zeros((self.TOTAL_SAMPLES - TOTAL_SAMPLES_VALID, 3, 106, 106))
        self.train_arr_Y = np.zeros((self.TOTAL_SAMPLES - TOTAL_SAMPLES_VALID, 1, 424, 424))
        self.valid_arr_X = np.zeros((TOTAL_SAMPLES_VALID, 3, 106, 106))
        self.valid_arr_Y = np.zeros((TOTAL_SAMPLES_VALID, 1, 424, 424))

        valid_idx = 0
        train_idx = 0
        for i in range(self.TOTAL_SAMPLES):
            if i not in n_valid_indx:
                for k in range(len(self.classes)):
                    with fits.open(os.path.join(self.path_train, f"{self.classes[k]}\{self.classes[k]}_{i}.fits"), memmap=False) as hdu:
                        if k == len(self.classes) - 1:
                            self.train_arr_Y[train_idx] = hdu[0].data
                            arr = np.array([list(row) for row in hdu[1].data])
                            if train_idx == 0:
                                arr = np.column_stack((arr, np.full(len(arr), train_idx)))
                                self.target_image_sources_cat_train = arr.copy()
                            else:
                                arr = np.column_stack((arr, np.full(len(arr), train_idx)))
                                self.target_image_sources_cat_train = np.vstack((self.target_image_sources_cat_train, arr))

                            del arr;
                        else:
                            self.train_arr_X[train_idx][k] = hdu[0].data

                train_idx += 1;
            elif i in n_valid_indx:
                for k in range(len(self.classes)):
                    with fits.open(os.path.join(self.path_train, f"{self.classes[k]}\{self.classes[k]}_{i}.fits"), memmap=False) as hdu:
                        if k == len(self.classes) - 1:
                            self.valid_arr_Y[valid_idx] = hdu[0].data
                            arr = np.array([list(row) for row in hdu[1].data])
                            if valid_idx == 0:
                                arr = np.column_stack((arr, np.full(len(arr), valid_idx)))
                                self.target_image_sources_cat_valid = arr.copy()
                            else:
                                arr = np.column_stack((arr, np.full(len(arr), valid_idx)))
                                self.target_image_sources_cat_valid = np.vstack((self.target_image_sources_cat_valid, arr))

                            del arr;
                        else:
                            self.valid_arr_X[valid_idx][k] = hdu[0].data
                valid_idx += 1;

        #print(self.target_image_sources_cat_valid[self.target_image_sources_cat_valid[:,-2] < 0])

        # NCHW --> NHWC for CPU compatiblity
        self.train_arr_X = np.swapaxes(np.swapaxes(self.train_arr_X, 1, 2), 2, 3)
        self.train_arr_Y = np.swapaxes(np.swapaxes(self.train_arr_Y, 1, 2), 2, 3)

        self.valid_arr_X = np.swapaxes(np.swapaxes(self.valid_arr_X, 1, 2), 2, 3)
        self.valid_arr_Y = np.swapaxes(np.swapaxes(self.valid_arr_Y, 1, 2), 2, 3)

        # Write to log
        if self.verbose == True:
            printlog(f"{datetime.datetime.now()} - Data Loaded in memory!", self.logfile_path)
            printlog(f"{datetime.datetime.now()} - Training Samples: (Xshape, Yshape) = {(self.train_arr_X.shape, self.train_arr_Y.shape)}, Validation Samples: (Xshape, Yshape) = {(self.valid_arr_X.shape, self.valid_arr_Y.shape)}", self.logfile_path)
        
        # Free memory
        gc.collect()

    def get_real_images(self):
        # X = np.empty(shape=(self.BATCH_SIZE, self.DIM[0], self.DIM[1], 3), dtype="float32")
        X = np.empty(shape=(self.BATCH_SIZE, 106, 106, 3), dtype="float32")
        Y = np.empty(shape=(self.BATCH_SIZE, self.DIM[0], self.DIM[1], 1), dtype="float32")
        draw = random.sample(range(0, self.train_arr_X.shape[0]), self.BATCH_SIZE)
        Y_source_cat = list()
        for i, k in enumerate(draw):
            Y_source_cat.append(self.target_image_sources_cat_train[np.where(self.target_image_sources_cat_train[:,-1] == k)])
            # Synchronize batch idx and catalogue image idx
            Y_source_cat[-1][:,-1] = i
            X[i] = self.train_arr_X[k]
            Y[i] = self.train_arr_Y[k]

        # Convert to a tensor
        Y_source_cat = tf.constant(np.float32(np.vstack(Y_source_cat)))

        # Maybe a np.cstack(Y_source_cat) needed
        return X, Y, Y_source_cat

    def BuildModel(self, GridVector = None):
        # Build the model given the chosen architecture
        if int(self.config['COMMON']['model_architecture_id'].rstrip().lstrip()) == 0: 
            if self.gridmode == True:
                if not isinstance(GridVector, list):
                    GridVector = list(GridVector)
                self.generator = CreateGridPaperModel(*GridVector)
            else:
                self.generator = CreatePaperModel()

            if self.verbose == True:
                if self.gridmode == True:
                    printlog(f"{datetime.datetime.now()} - Grid parameter values: {GridVector}", self.logfile_path)
                printlog(f"{datetime.datetime.now()} - Model Architecture Loaded!", self.logfile_path)
                self.generator.summary(print_fn=lambda x: printlog(x, self.logfile_path))

        # Note to self: Not compatible with different dimensions input and output
        if int(self.config['COMMON']['model_architecture_id'].rstrip().lstrip()) == 1:
            if self.gridmode == True:
                if not isinstance(GridVector, list):
                    GridVector = list(GridVector)
                self.generator = GridCustomModel1(*GridVector)
            else:
                self.generator = CustomModel1()

            if self.verbose == True:
                if self.gridmode == True:
                    printlog(f"{datetime.datetime.now()} - Grid parameter values: {GridVector}", self.logfile_path)
                printlog(f"{datetime.datetime.now()} - Model Architecture Loaded!", self.logfile_path)
                self.generator.summary(print_fn=lambda x: printlog(x, self.logfile_path))


    def SetOptimizer(self):
        self.BATCH_SIZE = int(self.config['TRAINING PARAMETERS']['batch_size'].rstrip().lstrip())
        # Initialise the optimizer and learning rate function
        if self.config['TRAINING PARAMETERS']['use_polynomial_decay'].rstrip().lstrip() == "True":
            lr_params = [float(i.strip(' ')) for i in self.config['TRAINING PARAMETERS']['polynomial_decay'].rstrip().lstrip().split(",")]
            lr_fn = tf.optimizers.schedules.PolynomialDecay(lr_params[0], lr_params[1]*self.BATCH_SIZE, lr_params[2], lr_params[3])
        else:
            lr_fn = 1e-4
        self.generator_optimizer = tf.keras.optimizers.Adam(lr_fn, beta_1=0.9)

    def Train(self, α = None):
        self.BATCH_SIZE = int(self.config['TRAINING PARAMETERS']['batch_size'].rstrip().lstrip())
        # Initialise the optimizer and learning rate function
        if self.config['TRAINING PARAMETERS']['use_polynomial_decay'].rstrip().lstrip() == "True":
            lr_params = [float(i.strip(' ')) for i in self.config['TRAINING PARAMETERS']['polynomial_decay'].rstrip().lstrip().split(",")]
            lr_fn = tf.optimizers.schedules.PolynomialDecay(lr_params[0], lr_params[1]*self.BATCH_SIZE, lr_params[2], lr_params[3])
        else:
            lr_fn = 1e-4
        self.generator_optimizer = tf.keras.optimizers.Adam(lr_fn, beta_1=0.9)

        # Graphed function performing a training step
        @tf.function
        def train_step(X, Y, Y_source_cat, generator, generator_optimizer, α):
            with tf.GradientTape() as gen_tape:
                gen_output = generator(X, training=True)

                gen_loss = loss(gen_output, Y, Y_source_cat, α)

                gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)

            generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
            return gen_loss
        # Free memory
        gc.collect()

        # Start Training
        self.n_epochs = int(self.config['TRAINING PARAMETERS']['number_of_epochs'].rstrip().lstrip())
        if self.gridmode == True:
            assert α is not None, "α must be given"
        else:
            α = float(self.config['TRAINING PARAMETERS']['alpha'].rstrip().lstrip())
            
        num_vis = 9
        GRACE_PERIOD = 80 # Epochs
        MIN_TIME_CKPT = 25 # Epochs

        # Number of its per epoch
        its = self.train_arr_X.shape[0]//self.BATCH_SIZE

        train_loss = []
        valid_epoch = []
        valid_loss = [[], [], []]

        val_epochapp = valid_epoch.append
        val_appL2 = valid_loss[0].append
        val_appLstat = valid_loss[1].append
        val_appLflux = valid_loss[2].append
        for epoch in tqdm(range(self.n_epochs), desc="Training GAN..."):
            it_train_loss = 0

            # Load all data batches, then apply training function
            for i in range(its):
                X, Y, Y_source_cat = self.get_real_images()
                #print(Y_source_cat.dtype)
                it_train_loss += float(train_step(X, Y, Y_source_cat, self.generator, self.generator_optimizer, α))
                #print(it_train_loss)
            # Append train_loss
            train_loss.append(it_train_loss)
            # Every 10 epochs show progress of generated images based on validation array
            if epoch % 10 == 0:
                TrainingSnapShot(self.generator, epoch + 1, self.valid_arr_X[:9], self.tdir_out_progress)

            # Write training loss to log
            if self.verbose == True:
                printlog(f"{datetime.datetime.now()} - Epoch: {epoch}", self.logfile_path)
                printlog(f"{datetime.datetime.now()} - Gen loss: {train_loss[-1]:.2f}", self.logfile_path)

            # Store the model after the grace_period
            if epoch == GRACE_PERIOD:
                Lh, Lstats, Lflux = 0, 0, 0
                for batch_idx in range(self.valid_arr_X.shape[0]//self.BATCH_SIZE):
                    X = self.valid_arr_X[batch_idx*self.BATCH_SIZE:self.BATCH_SIZE*(batch_idx + 1)].astype(np.float32) if batch_idx != (self.valid_arr_X.shape[0]//self.BATCH_SIZE - 1) else self.valid_arr_X[batch_idx*self.BATCH_SIZE:].astype(np.float32)
                    Y = self.valid_arr_Y[batch_idx*self.BATCH_SIZE:self.BATCH_SIZE*(batch_idx + 1)].astype(np.float32) if batch_idx != (self.valid_arr_Y.shape[0]//self.BATCH_SIZE - 1) else self.valid_arr_Y[batch_idx*self.BATCH_SIZE:].astype(np.float32)
                    Y_source_cat = list()
                    idx_arr = np.arange(batch_idx*self.BATCH_SIZE, self.BATCH_SIZE*(batch_idx + 1)) if batch_idx != (self.valid_arr_X.shape[0]//self.BATCH_SIZE - 1) else np.arange(batch_idx*self.BATCH_SIZE, self.valid_arr_X.shape[0])
                    for k in idx_arr:
                        Y_source_cat.append(self.target_image_sources_cat_valid[np.where(self.target_image_sources_cat_valid[:,-1] == k)])
                        # Synchronize batch idx and catalogue image idx
                        Y_source_cat[-1][:,-1] = k
                    Y_source_cat = np.vstack(Y_source_cat)
                    gen_valid = self.generator(X, training=False).numpy()
                    Lh_batch, Lstats_batch, Lflux_batch = tf.numpy_function(loss_valid, [gen_valid, Y, Y_source_cat, α], Tout=[tf.float32, tf.float32, tf.float32])
                    Lh += Lh_batch; Lstats += Lstats_batch; Lflux += Lflux_batch

                min_l_valid = Lh.numpy() + Lstats.numpy() + Lflux.numpy()
                min_lh_valid = Lh.numpy()
                min_lstats_valid = Lstats.numpy()
                min_lflux_valid = Lflux.numpy()
                val_appL2(Lh.numpy()); val_appLstat(Lstats.numpy()); val_appLflux(Lflux.numpy())
                val_epochapp(epoch)

                # Write summary to log
                if self.verbose == True:
                    printlog(f"{datetime.datetime.now()} - Validation loss improved: Validation loss: {min_l_valid}", self.logfile_path)
                    printlog(f"{datetime.datetime.now()} - Model saved!", self.logfile_path)

                self.generator.save(os.path.join(self.model_path, 'BestValid_Model'))
                epoch_ckpt = GRACE_PERIOD
                epoch_ckpt_huber = GRACE_PERIOD
                epoch_ckpt_stats = GRACE_PERIOD
                epoch_ckpt_flux = GRACE_PERIOD

            # Store best parameters if validation loss decreased
            if epoch > GRACE_PERIOD:
                Lh, Lstats, Lflux = 0, 0, 0
                for batch_idx in range(self.valid_arr_X.shape[0]//self.BATCH_SIZE):
                    X = self.valid_arr_X[batch_idx*self.BATCH_SIZE:self.BATCH_SIZE*(batch_idx + 1)].astype(np.float32) if batch_idx != (self.valid_arr_X.shape[0]//self.BATCH_SIZE - 1) else self.valid_arr_X[batch_idx*self.BATCH_SIZE:].astype(np.float32)
                    Y = self.valid_arr_Y[batch_idx*self.BATCH_SIZE:self.BATCH_SIZE*(batch_idx + 1)].astype(np.float32) if batch_idx != (self.valid_arr_Y.shape[0]//self.BATCH_SIZE - 1) else self.valid_arr_Y[batch_idx*self.BATCH_SIZE:].astype(np.float32)
                    Y_source_cat = list()
                    idx_arr = np.arange(batch_idx*self.BATCH_SIZE, self.BATCH_SIZE*(batch_idx + 1)) if batch_idx != (self.valid_arr_X.shape[0]//self.BATCH_SIZE - 1) else np.arange(batch_idx*self.BATCH_SIZE, self.valid_arr_X.shape[0])
                    for k in idx_arr:
                        Y_source_cat.append(self.target_image_sources_cat_valid[np.where(self.target_image_sources_cat_valid[:,-1] == k)])
                        # Synchronize batch idx and catalogue image idx
                        Y_source_cat[-1][:,-1] = k
                    Y_source_cat = np.vstack(Y_source_cat)

                    gen_valid = self.generator(X, training=False).numpy()
                    Lh_batch, Lstats_batch, Lflux_batch = tf.numpy_function(loss_valid, [gen_valid, Y, Y_source_cat, α], Tout=[tf.float32, tf.float32, tf.float32])
                    Lh += Lh_batch; Lstats += Lstats_batch; Lflux += Lflux_batch
                # L2_, Lstats, Lflux = loss_valid(generator(valid_arr[:,:3], training=False), valid_arr[:,3])
                Lh = Lh.numpy(); Lstats = Lstats.numpy(); Lflux = Lflux.numpy();
                val_appL2(Lh); val_appLstat(Lstats); val_appLflux(Lflux)
                val_epochapp(epoch)

                if Lh < min_lh_valid:
                    # Write summary to log
                    if self.verbose == True:
                        printlog(f"{datetime.datetime.now()} - Huber Validation loss improved! Huber loss: {Lh}", self.logfile_path)
                    min_lh_valid = Lh
                    # if abs(epoch_ckpt_huber-epoch) > MIN_TIME_CKPT:
                    #     # Write summary to log
                    #     if self.verbose == True:
                    #         printlog(f"{datetime.datetime.now()} - Best Huber Model saved!", self.logfile_path)
                            
                    #     self.generator.save(os.path.join(self.model_path, 'BestHuber_Model'))
                    #     epoch_ckpt_huber = epoch

                if Lstats < min_lstats_valid:
                    if self.verbose == True:
                        printlog(f"{datetime.datetime.now()} - Stats Validation loss improved! Stats loss: {Lstats}", self.logfile_path)

                    min_lstats_valid = Lstats
                    # if abs(epoch_ckpt_stats-epoch) > MIN_TIME_CKPT:
                    #     if self.verbose == True:
                    #         printlog(f"{datetime.datetime.now()} - Best Stats Model saved!", self.logfile_path)

                    #     self.generator.save(os.path.join(self.model_path, 'BestStats_Model'))
                    #     epoch_ckpt_stats = epoch

                if Lflux < min_lflux_valid:
                    if self.verbose == True:
                        printlog(f"{datetime.datetime.now()} - Flux Validation loss improved! Flux loss: {Lflux}", self.logfile_path)

                    min_lflux_valid = Lflux
                    # if abs(epoch_ckpt_flux-epoch) > MIN_TIME_CKPT:
                    #     if self.verbose == True:
                    #         printlog(f"{datetime.datetime.now()} - Best Flux Model saved!", self.logfile_path)
                    #     self.generator.save(os.path.join(self.model_path, 'BestFlux_Model'))
                    #     epoch_ckpt_flux = epoch

                if (Lh + Lstats + Lflux) < min_l_valid:
                    min_l_valid = Lh + Lstats + Lflux
                    if self.verbose == True:
                        printlog(f"{datetime.datetime.now()} - Total Validation loss improved! Validation loss: {min_l_valid}", self.logfile_path)

                    if abs(epoch_ckpt-epoch) > MIN_TIME_CKPT:
                        if self.verbose == True:
                            printlog(f"{datetime.datetime.now()} - Best Model saved!", self.logfile_path)
                        self.generator.save(os.path.join(self.model_path, 'BestValid_Model'))
                        epoch_ckpt = epoch

            if (epoch) % 25 == 0:
                gc.collect()

        TrainingSnapShot(self.generator, epoch + 1, self.valid_arr_X[:9], self.tdir_out_progress)
        # Store the validation loss values
        printlog(f"{datetime.datetime.now()} - Stored Training/Validation Loss History!", self.logfile_path)
        save_path = os.path.join(self.model_path, 'Model')
        np.savez(os.path.join(self.model_path, 'ValidationLossHistory.npz'), train_epochs = np.array(range(self.n_epochs)), valid_epochs=np.array(valid_epoch), train_loss=train_loss,  Ntrain=self.train_arr_X.shape[0], Nvalid = self.valid_arr_X.shape[0] ,Lh=np.array(valid_loss[0]), Lstats=np.array(valid_loss[1]), Lflux=np.array(valid_loss[2]), 
                                                                        min_lhuber_save=min_lh_valid, min_lstats_save=min_lstats_valid, min_lflux_save=min_lflux_valid, min_l_save=min_l_valid, alpha=α)
                
        return train_loss, valid_epoch, valid_loss

    def TrainingAnalysis(self):
        printlog(f"{datetime.datetime.now()} - Analysing Training Run!", self.logfile_path)
        # Make plots that describe the model's performance during training
        losshist = np.load(os.path.join(self.model_path, 'ValidationLossHistory.npz'))
        LossComponentPlot(losshist, self.tdir_out_analysis)
        FilteredLossComponentPlot(losshist, self.tdir_out_analysis)

        anim_file = os.path.join(self.tdir_out_analysis, 'dcgan.gif')
        with imageio.get_writer(anim_file, mode='I') as writer:
            filenames = glob.glob(os.path.join(self.tdir_out_progress, 'image_at_epoch*.png'))
            filenames = sorted(filenames)
            for filename in filenames:
                image = imageio.imread(filename)
                writer.append_data(image)

        printlog(f"{datetime.datetime.now()} - Analysis Complete!", self.logfile_path)