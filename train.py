import itertools
import gc
import os
import configparser

import tensorflow as tf
import numpy as np

#from train_functions import SRTrainer
from train_functions_gpu import SRTrainerGPU
#from test_functions import SRTester
from test_functions_gpu import SRTesterGPU

from tqdm import tqdm

if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read("TrainingConfig.ini")
    # Toggle GPU/CPU
    if config['COMMON']['use_gpu'].rstrip().lstrip() == "False":
        # TODO FIX CPU VERSION!
        # SRModelTrain = SRTrainer("TrainingConfig.ini")
        # SRModelTrain.LoadTrainingData()
        # SRModelTrain.BuildModel()
        # #test.SetOptimizer()
        # train_loss, valid_epoch, valid_loss = SRModelTrain.Train()
        # SRModelTrain.TrainingAnalysis()

        # del SRModelTrain;
        # gc.collect()
        # SRModelTest = SRTester("TrainingConfig.ini")
        # SRModelTest.LoadTestData()
        # SRModelTest.LoadModel()
        # SRModelTest.TestAnalysis(α = 0.01)
        print("CPU Version broken!")

    else:
        SRModelTrain = SRTrainerGPU("TrainingConfig.ini")
        SRModelTrain.LoadTrainingData()
        SRModelTrain.BuildModel()
        #test.SetOptimizer()
        SRModelTrain.Train()
        SRModelTrain.TrainingAnalysis()

        del SRModelTrain;
        gc.collect()
        SRModelTest = SRTesterGPU("TrainingConfig.ini")
        SRModelTest.LoadTestData()
        SRModelTest.LoadModel()
        SRModelTest.TestAnalysis(α = 0.01)
