import itertools
import gc
import os

import tensorflow as tf
import numpy as np

from train_functions import SRTrainer
from test_functions import SRTester

from tqdm import tqdm

if __name__ == "__main__":
    SRModelTrain = SRTrainer("TrainingConfig.ini")
    SRModelTrain.LoadTrainingData()
    SRModelTrain.BuildModel()
    #test.SetOptimizer()
    train_loss, valid_epoch, valid_loss = SRModelTrain.Train()
    SRModelTrain.TrainingAnalysis()

    del SRModelTrain;
    gc.collect()
    SRModelTest = SRTester("TrainingConfig.ini")
    SRModelTest.LoadTestData()
    SRModelTest.LoadModel()
    SRModelTest.TestAnalysis(α = 0.01)