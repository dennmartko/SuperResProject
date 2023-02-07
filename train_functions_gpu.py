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
from ModelArchitectures.PaperModel import Generator, Discriminator, CreatePaperModel, CreateGridPaperModel
from ModelArchitectures.CustomModel1 import CustomModel1, GridCustomModel1
from loss_functions import non_adversarial_loss, non_adversarial_loss_valid, Wasserstein_loss
from PlotLib.PlotFunctionsTrain import TrainingSnapShot, NonAdversarialLossComponentPlot, NonAdversarialFilteredLossComponentPlot, AdversarialLossComponentPlot

random.seed(10)
#######################
###   TRAIN CLASS   ###
#######################
def printlog(msg, outfile):
    with open(outfile, 'a') as f:
        f.write(msg + '\n')

class SRTrainerGPU():
    def __init__(self, path_to_config, gridmode = False, idx = None) -> None:
        # Load configuration file for training
        self.config = configparser.ConfigParser()
        self.config.read(path_to_config)

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

            # List device used: GPU/CPU
            printlog(f"{datetime.datetime.now()} - GPU assigned!", self.logfile_path)

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
                    with fits.open(os.path.join(self.path_train, f"{os.path.join(self.classes[k],self.classes[k])}_{i}.fits"), memmap=False) as hdu:
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
                    with fits.open(os.path.join(self.path_train, f"{os.path.join(self.classes[k],self.classes[k])}_{i}.fits"), memmap=False) as hdu:
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
        # Write to log
        if self.verbose == True:
            printlog(f"{datetime.datetime.now()} - Data Loaded in memory!", self.logfile_path)
            printlog(f"{datetime.datetime.now()} - Training Samples: (Xshape, Yshape) = {(self.train_arr_X.shape, self.train_arr_Y.shape)}, Validation Samples: (Xshape, Yshape) = {(self.valid_arr_X.shape, self.valid_arr_Y.shape)}", self.logfile_path)
        
        # Free memory
        gc.collect()

    def get_real_images(self):
        # X = np.empty(shape=(self.BATCH_SIZE, self.DIM[0], self.DIM[1], 3), dtype="float32")
        X = np.empty(shape=(self.BATCH_SIZE, 3, 106, 106), dtype="float32")
        # NCHW --> GPU
        # NHWC --> CPU
        Y = np.empty(shape=(self.BATCH_SIZE, 1, self.DIM[0], self.DIM[1]), dtype="float32")
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
        self.enable_adversarial = True if self.config['TRAINING PARAMETERS']['enable_adversarial'].rstrip().lstrip() == "True" else False
        # Build the model given the chosen architecture
        if int(self.config['COMMON']['model_architecture_id'].rstrip().lstrip()) == 0: 
            if self.gridmode == True:
                self.generator = Generator((3, 106, 106), "channels_first", 32, 4, GridVector)
            else:
                self.generator = Generator((3, 106, 106), "channels_first", 32, 4, (1.5, 2.5, 4, 6, 16))
                if self.enable_adversarial:
                    self.discriminator = Discriminator((1, 424, 424), "channels_first", 16, 4, 0.01)
                else:
                    self.discriminator = None
            if self.verbose == True:
                if self.gridmode == True:
                    printlog(f"{datetime.datetime.now()} - Grid parameter values: {GridVector}", self.logfile_path)
                printlog(f"{datetime.datetime.now()} - Model Architecture Loaded!", self.logfile_path)
                self.generator.summary(print_fn=lambda x: printlog(x, self.logfile_path))
                #self.discriminator.summary(print_fn=lambda x: printlog(x, self.logfile_path))
        # Note to self: Not compatible with different dimensions input and output
        if int(self.config['COMMON']['model_architecture_id'].rstrip().lstrip()) == 1:
            if self.gridmode == True:
                if not isinstance(GridVector, list):
                    GridVector = list(GridVector)
                self.generator = GridCustomModel1((3, 106, 106), "channels_first", *GridVector)
            else:
                self.generator = CustomModel1((3, 106, 106), "channels_first")

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
        self.n_stopping_threshold = int(self.config['TRAINING PARAMETERS']['early_stopping'].rstrip().lstrip())
        self.n_crit = int(self.config['TRAINING PARAMETERS']['n_crit'].rstrip().lstrip())
        # Initialise the optimizer and learning rate function
        if self.config['TRAINING PARAMETERS']['use_polynomial_decay'].rstrip().lstrip() == "True":
            lr_params = [float(i.strip(' ')) for i in self.config['TRAINING PARAMETERS']['polynomial_decay'].rstrip().lstrip().split(",")]
            lr_fn = tf.optimizers.schedules.PolynomialDecay(lr_params[0], lr_params[1]*self.BATCH_SIZE, lr_params[2], lr_params[3])
        else:
            lr_fn = 1e-4
        self.generator_optimizer = tf.keras.optimizers.Adam(lr_fn, beta_1=0.9)
        self.discriminator_optimizer = tf.keras.optimizers.RMSprop(5e-5)
        #self.discriminator_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.9)


        # Graphed function performing a training step
        @tf.function
        def train_step_generator(X, Y, Y_source_cat, generator, generator_optimizer, α, discriminator = None, enable_adversarial=False): #discriminator, discriminator_optimizer
            # First update generator with adversarial loss
            if enable_adversarial:
                with tf.GradientTape() as gen_tape:
                    gen_output = generator(X, training=True)
                    
                    fake_output = discriminator(gen_output, training=False)

                    avg_score_fake = Wasserstein_loss(-tf.ones_like(fake_output), fake_output)

                    gradients_of_generator = gen_tape.gradient(avg_score_fake, generator.trainable_variables)

                generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
            else:
                avg_score_fake = 0
            # Finally update generator with non-adversarial loss

            with tf.GradientTape() as gen_tape:
                gen_output = generator(X, training=True)
                gen_loss = non_adversarial_loss(gen_output, Y, Y_source_cat, α)

                #print(gen_loss)
                gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)

            generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
            return avg_score_fake, gen_loss

        @tf.function
        def train_step_discriminator(X, Y, generator, discriminator, discriminator_optimizer):
            with tf.GradientTape() as disc_tape:
                gen_output = generator(X, training=False)
                
                real_output = discriminator(Y, training=True)
                fake_output = discriminator(gen_output, training=True)

                avg_score_real = Wasserstein_loss(-tf.ones_like(real_output), real_output)
                avg_score_fake = Wasserstein_loss(tf.ones_like(fake_output), fake_output)

                gradients_of_discriminator = disc_tape.gradient(avg_score_real + avg_score_fake, discriminator.trainable_variables)

            discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
            return avg_score_real, avg_score_fake
        # Free memory
        gc.collect()

        # Start Training
        self.n_epochs = int(self.config['TRAINING PARAMETERS']['number_of_epochs'].rstrip().lstrip())
        if self.gridmode == True:
            assert α is not None, "α must be given"
        else:
            α = float(self.config['TRAINING PARAMETERS']['alpha'].rstrip().lstrip())
            
        num_vis = 9
        EPOCH_THRESHOLD_MODELSAVING = 150 
        n_stopping = 0
        # Number of its per epoch
        its = self.train_arr_X.shape[0]//self.BATCH_SIZE

        train_dict = {"Non-Adversarial loss":[], "Adversarial loss":{"G":[], "D":{"real_score":[], "fake_score":[]}}, "epochs":[]}
        valid_dict = {"Non-Adversarial loss":{"Lh":[], "Lstats":[], "Lflux":[]}, "Adversarial loss":{"G":[], "D":{"real_score":[], "fake_score":[]}}, "epochs":[]}
        for epoch in tqdm(range(self.n_epochs), desc="Training GAN..."):
            train_non_adv_loss, train_adv_loss_G = 0, 0
            train_adv_loss_D_real, train_adv_loss_D_fake = 0, 0
            # Load all data batches, then apply training function
            for i in range(its):
                # Adverserial training of discriminator/critic
                if self.enable_adversarial:
                    it_train_adv_loss_real = 0
                    it_train_adv_loss_fake = 0
                    for t in range(self.n_crit):
                        X, Y, Y_source_cat = self.get_real_images()
                        avg_score_real, avg_score_fake = train_step_discriminator(X, Y, self.generator, self.discriminator, self.discriminator_optimizer)
                        it_train_adv_loss_real += float(avg_score_real.numpy())/self.n_crit
                        it_train_adv_loss_fake += float(avg_score_fake.numpy())/self.n_crit

                    train_adv_loss_D_real += it_train_adv_loss_real
                    train_adv_loss_D_fake += it_train_adv_loss_fake

                # Training of generator, if adversarial is enabled we also train with wasserstein loss
                X, Y, Y_source_cat = self.get_real_images()
                avg_score_fake, gen_loss  = train_step_generator(X, Y, Y_source_cat, self.generator, self.generator_optimizer, α, discriminator = self.discriminator, enable_adversarial=self.enable_adversarial)
                train_non_adv_loss += float(gen_loss.numpy())
                train_adv_loss_G += float(avg_score_fake.numpy())

            # Append training losses
            train_dict["Non-Adversarial loss"].append(train_non_adv_loss)
            train_dict["Adversarial loss"]["G"].append(train_adv_loss_G)
            train_dict["Adversarial loss"]["D"]["real_score"].append(train_adv_loss_D_real)
            train_dict["Adversarial loss"]["D"]["fake_score"].append(train_adv_loss_D_fake)

            train_dict["epochs"].append(epoch)
            # Every 10 epochs show progress of generated images based on validation array
            if epoch % 10 == 0:
                TrainingSnapShot(self.generator, epoch + 1, self.valid_arr_X[:9], self.tdir_out_progress)

            # Write training loss to log
            if self.verbose == True:
                printlog(f"{datetime.datetime.now()} - Epoch: {train_dict['epochs'][-1]}", self.logfile_path)
                printlog(f"{datetime.datetime.now()} - Non-Adversarial loss: {train_dict['Non-Adversarial loss'][-1]:.2f}", self.logfile_path)
                printlog(f"{datetime.datetime.now()} - Adversarial loss (G, D): ({train_dict['Adversarial loss']['G'][-1]:.2f}, {(train_dict['Adversarial loss']['D']['real_score'][-1] + train_dict['Adversarial loss']['D']['fake_score'][-1]):.2f})", self.logfile_path)


            # Store the model after the epoch threshold
            Lh, Lstats, Lflux, LwG, LwD = 0, 0, 0, 0, 0
            valid_its = self.valid_arr_X.shape[0]//self.BATCH_SIZE
            for batch_idx in range(valid_its):
                X = self.valid_arr_X[batch_idx*self.BATCH_SIZE:self.BATCH_SIZE*(batch_idx + 1)].astype(np.float32) if batch_idx != (valid_its - 1) else self.valid_arr_X[batch_idx*self.BATCH_SIZE:].astype(np.float32)
                Y = self.valid_arr_Y[batch_idx*self.BATCH_SIZE:self.BATCH_SIZE*(batch_idx + 1)].astype(np.float32) if batch_idx != (valid_its - 1) else self.valid_arr_Y[batch_idx*self.BATCH_SIZE:].astype(np.float32)
                Y_source_cat = list()
                idx_arr = np.arange(batch_idx*self.BATCH_SIZE, self.BATCH_SIZE*(batch_idx + 1)) if batch_idx != (valid_its - 1) else np.arange(batch_idx*self.BATCH_SIZE, self.valid_arr_X.shape[0])
                for k in idx_arr:
                    Y_source_cat.append(self.target_image_sources_cat_valid[np.where(self.target_image_sources_cat_valid[:,-1] == k)])
                    # Synchronize batch idx and catalogue image idx
                    Y_source_cat[-1][:,-1] = k
                Y_source_cat = np.vstack(Y_source_cat)

                gen_valid = self.generator(X, training=False).numpy()
                Lh_batch, Lstats_batch, Lflux_batch = tf.numpy_function(non_adversarial_loss_valid, [gen_valid, Y, Y_source_cat, α], Tout=[tf.float32, tf.float32, tf.float32])

                if self.enable_adversarial:
                    real_output = self.discriminator(Y, training=False)
                    fake_output = self.discriminator(gen_valid, training=False)

                    real_score = float(Wasserstein_loss(-tf.ones_like(real_output), real_output).numpy())
                    fake_score = float(Wasserstein_loss(tf.ones_like(fake_output), fake_output).numpy())

                    LwG_batch = float(Wasserstein_loss(-tf.ones_like(fake_output), fake_output).numpy())
                else:
                    LwG_batch = 0
                    real_score = 0
                    fake_score = 0
                Lh += Lh_batch; Lstats += Lstats_batch; Lflux += Lflux_batch; LwG += LwG_batch;
            # L2_, Lstats, Lflux = loss_valid(generator(valid_arr[:,:3], training=False), valid_arr[:,3])
            Lh = Lh.numpy(); Lstats = Lstats.numpy(); Lflux = Lflux.numpy();
            valid_dict["Non-Adversarial loss"]["Lh"].append(Lh)
            valid_dict["Non-Adversarial loss"]["Lstats"].append(Lstats)
            valid_dict["Non-Adversarial loss"]["Lflux"].append(Lflux)
            valid_dict["Adversarial loss"]["G"].append(LwG)
            valid_dict["Adversarial loss"]["D"]["real_score"].append(real_score)
            valid_dict["Adversarial loss"]["D"]["fake_score"].append(fake_score)

            if epoch == 0:
                min_l_valid = valid_dict["Non-Adversarial loss"]["Lh"][-1] + valid_dict['Non-Adversarial loss']['Lstats'][-1] + valid_dict['Non-Adversarial loss']['Lflux'][-1]
                min_Lh_valid = valid_dict["Non-Adversarial loss"]["Lh"][-1]
                min_Lstats_valid = valid_dict["Non-Adversarial loss"]["Lstats"][-1]
                min_Lflux_valid = valid_dict["Non-Adversarial loss"]["Lflux"][-1]
                min_LwG_valid = valid_dict['Adversarial loss']['G'][-1]
                epoch_best_model_save = epoch

            if valid_dict["Non-Adversarial loss"]["Lh"][-1] < min_Lh_valid:
                # Write summary to log
                min_Lh_valid = valid_dict["Non-Adversarial loss"]["Lh"][-1]
                if self.verbose == True:
                    printlog(f"{datetime.datetime.now()} - Huber Validation loss improved! Huber loss: {valid_dict['Non-Adversarial loss']['Lh'][-1]}", self.logfile_path)

            if valid_dict['Non-Adversarial loss']['Lstats'][-1] < min_Lstats_valid:
                min_Lstats_valid = valid_dict['Non-Adversarial loss']['Lstats'][-1]
                if self.verbose == True:
                    printlog(f"{datetime.datetime.now()} - Stats Validation loss improved! Stats loss: {valid_dict['Non-Adversarial loss']['Lstats'][-1]}", self.logfile_path)

            if valid_dict['Non-Adversarial loss']['Lflux'][-1] < min_Lflux_valid:
                min_Lflux_valid = valid_dict['Non-Adversarial loss']['Lflux'][-1]
                if self.verbose == True:
                    printlog(f"{datetime.datetime.now()} - Flux Validation loss improved! Flux loss: {valid_dict['Non-Adversarial loss']['Lflux'][-1]}", self.logfile_path)

            if valid_dict['Adversarial loss']['G'][-1] < min_LwG_valid:
                min_LwG_valid = valid_dict['Adversarial loss']['G'][-1]
                if self.verbose == True:
                    printlog(f"{datetime.datetime.now()} - Wasserstein (G) Validation loss improved! Wasserstein (G) loss: {valid_dict['Adversarial loss']['G'][-1]}", self.logfile_path)

            if (valid_dict["Non-Adversarial loss"]["Lh"][-1] + valid_dict['Non-Adversarial loss']['Lstats'][-1] + valid_dict['Non-Adversarial loss']['Lflux'][-1]) < min_l_valid:
                min_l_valid = valid_dict["Non-Adversarial loss"]["Lh"][-1] + valid_dict['Non-Adversarial loss']['Lstats'][-1] + valid_dict['Non-Adversarial loss']['Lflux'][-1]
                n_stopping = 0                
                if self.verbose == True:
                    printlog(f"{datetime.datetime.now()} - Total Validation loss improved! Validation loss: {min_l_valid}", self.logfile_path)

                if epoch >= EPOCH_THRESHOLD_MODELSAVING:
                    if self.verbose == True:
                        printlog(f"{datetime.datetime.now()} - Best Model saved!", self.logfile_path)
                    self.generator.save(os.path.join(self.model_path, 'BestValid_Model'))
                    epoch_best_model_save = epoch
            else:
                n_stopping += 1

            if n_stopping == self.n_stopping_threshold:
                break

            # Memory CleanUp
            if (epoch) % 25 == 0:
                gc.collect()

        TrainingSnapShot(self.generator, epoch + 1, self.valid_arr_X[:9], self.tdir_out_progress)
        # Store the validation loss values
        printlog(f"{datetime.datetime.now()} - Stored Training/Validation Loss History!", self.logfile_path)
        #save_path = os.path.join(self.model_path, 'Model')
        train_loss = np.array(valid_dict['Non-Adversarial loss']['Lh']) + np.array(valid_dict['Non-Adversarial loss']['Lstats']) + np.array(valid_dict['Non-Adversarial loss']['Lflux'])
        np.savez(os.path.join(self.model_path, 'ValidationLossHistory.npz'), train_epochs = np.array(train_dict['epochs']), train_loss=train_loss,  Ntrain=self.train_arr_X.shape[0], Nvalid = self.valid_arr_X.shape[0], Lh=np.array(valid_dict['Non-Adversarial loss']['Lh']), Lstats=np.array(valid_dict['Non-Adversarial loss']['Lstats']), Lflux=np.array(valid_dict['Non-Adversarial loss']['Lflux']), 
                                                                        min_lhuber_save=np.min(np.array(valid_dict['Non-Adversarial loss']['Lh'])), epoch_best_model_save=epoch_best_model_save, alpha=α, LwG=np.array(valid_dict['Adversarial loss']['G']), LwD_real_score=np.array(valid_dict['Adversarial loss']['D']['real_score']), LwD_fake_score=np.array(valid_dict['Adversarial loss']['D']['fake_score']))

    def TrainingAnalysis(self):
        printlog(f"{datetime.datetime.now()} - Analysing Training Run!", self.logfile_path)
        # Make plots that describe the model's performance during training
        losshist = np.load(os.path.join(self.model_path, 'ValidationLossHistory.npz'))
        NonAdversarialLossComponentPlot(losshist, self.tdir_out_analysis)
        #NonAdversarialFilteredLossComponentPlot(losshist, self.tdir_out_analysis)
        AdversarialLossComponentPlot(losshist, self.tdir_out_analysis)
        anim_file = os.path.join(self.tdir_out_analysis, 'dcgan.gif')
        with imageio.get_writer(anim_file, mode='I') as writer:
            filenames = glob.glob(os.path.join(self.tdir_out_progress, 'image_at_epoch*.png'))
            filenames = sorted(filenames)
            for filename in filenames:
                image = imageio.imread(filename)
                writer.append_data(image)

        printlog(f"{datetime.datetime.now()} - Analysis Complete!", self.logfile_path)