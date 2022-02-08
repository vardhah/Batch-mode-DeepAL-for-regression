import argparse
import os
import glob
import pathlib
import numpy as np
from trainer import model_trainer
from utils import *
import itertools
from sklearn.model_selection import train_test_split
import pandas as pd
from sim_engine import *
import matplotlib.pyplot as plt
from lp import load_N_predict 
from model_training import train_net
from mpl_toolkits.mplot3d import Axes3D
Axes3D = Axes3D  # pycharm auto import
import logging


total_samples=100000


if __name__ == "__main__":
    
    if not torch.cuda.is_available():
        device = 'cpu'
    
    samples=random_sampling(total_samples)
    #samples= np.loadtxt("./data/pool_data_extended.csv", delimiter=",",skiprows=0, dtype=np.float32)
    f=opensim(samples)
    sim_data=np.concatenate((samples,f.reshape(-1,1)),axis=1)
    named='./data/pool_data_search'+'.csv'
    np.savetxt(named, sim_data, delimiter=",")
      
