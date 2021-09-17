import argparse
import os
import glob
import pathlib
import numpy as np
import torch
import torch.nn as nn
import torch.multiprocessing as _mp
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from model import Net
from trainer import model_trainer
from utils import *
import itertools
from sklearn.model_selection import train_test_split
import pandas as pd
from sim_engine import *
import matplotlib.pyplot as plt
from lp import load_N_predict 
from model_training import train_net
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from mpl_toolkits.mplot3d import Axes3D
Axes3D = Axes3D  # pycharm auto import


input_size=2                             # input size may change if integer/ordinal type variable and represented by one-hot encoding
num_variable = 2                         # number of variables  both real & int type 
output_size=1                            # number of output 
num_iteration=100                         # Number of iteration of sampling

budget_samples=10                        # Number of samples-our budget
ranges=[-10,0,-6.5,0]                    # ranges in form of [low1,high1,low2,high2,...]
mask=['real','real']                     # datatype ['dtype1','dtype2']
random_gridmesh=False                    # (not using now) if state space is pretty big, it is not possible to create a big mesh of samples, in such case for each iteration, we randomly create a biggest possible mesh.       
categories=[[None],[None]]               #categories for ordinal variables, 'None' for real variables 
lbtm_size=1000
#here database container are just numpy array called => sim_data,train_data, test_data,lbtm (they contain unscaled/unnormalised data(i/p-o/p))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Population Based Training")
    parser.add_argument("-d","--device", type=str, default='cuda',
                        help="")
    parser.add_argument("-b","--batch_size", type=int, default=32,
                        help="")
    parser.add_argument("-l","--load", type=bool, default=False,
                        help="")
    parser.add_argument("-e","--epoch", type=int, default=20,
                        help="")

    
    args = parser.parse_args()
    pathlib.Path('./models/student').mkdir(exist_ok=True)
    pathlib.Path('./models/teacher').mkdir(exist_ok=True) 
    device = args.device
    if not torch.cuda.is_available():
        device = 'cpu'
    population= args.population_size
    batch_size = args.batch_size
    max_epoch = args.epoch
    load_e=args.load
    
    ############# Intial sampling for startup & its evaluation
    samples=draw_samples(mask,init_samples,np.array(ranges))
    f,c=mishra(samples[:,0],samples[:,1])
    sim_data=np.concatenate((samples,f.reshape(-1,1)),axis=1)
    print('Samples size:',samples.shape,'f shape:',f.shape)
    #########################################################

    ############# create data holder for train and test
    train_data,test_data= data_split(sim_data,proportion=0.2)
    #########################################################
    
    ############Data preperation for neural net training ( 2 processes: 1. for real: scale data between 0 & 1, for int : one hot encoding ) 
    copied_train_data=np.copy(train_data)
    copied_test_data=np.copy(test_data)
    fitted_train_data= data_preperation(copied_train_data,mask,np.array(ranges),categories)
    fitted_test_data= data_preperation(copied_test_samples,mask,np.array(ranges),categories)
    #################################################################################################
 

    #Loop for training
    for itr in range(num_iteration):
    
     print('==> Train data shape is:',train_data.shape,'Test data sample shape is:',test_data.shape)
     #Training of student model
     train_net(fitted_train_data,batch_size,max_epoch,device,input_size,output_size,'S') 
     
     #prediction on test data on student model  
     lnp=load_N_predict(fitted_test_data[,input_size,output_size,"./models/student/",'S')
     stest_pred=lnp.run() 
 
     #check and label test data which have failed & passed
     lbtm_data=label_data(test_data,stest_pred)

     #updating limited buffer test memory 
     lbtm_data= np.concatenate(test_data,.reshape(-1,1)),axis=1)
     lbtm=update_lbtm(stest_pred,lbtm_size)
     
     #Data preperation for Neural network training(teacher network)
     copied_lbtm_data=np.copy(lbtm)
     fitted_lbtm_data= data_preperation(copied_lbtm_data,mask,np.array(ranges),categories)
     
     #Train Teacher network 
     train_net(fitted_lbtm_data,batch_size,max_epoch,device,input_size,output_size,'T') 

     #procedure for selecting samples for next stage
     
     copied_grid_samples= np.copy(grid_samples)
     #print('grid samples shape in main:',grid_samples,'std shape',std_pred.shape)
     scaled_mesh_data=scale_data(copied_grid_samples,ranges)
     #print('scaled mesh data:',scaled_mesh_data)
     selected_scaled_point, index_sel_points=find_dis_points_with_index(scaled_mesh_data,std_pred,num_samples_per_iteration,distance)
     selected_point = rescale_data(selected_scaled_point,ranges,mask)
     #simulating on slected samples and concatenating to earlier sim data
     #print('shape of selected points are:',selected_point.shape,'index of selcted points:',index_sel_points.shape)
     
     ###############################
     # simulate on selected samples 
     f,c=mishra(selected_point[:,0],selected_point[:,1])
     ###############################

     print('f is:',f.shape)
     sim_data= np.concatenate((sim_data,np.concatenate((selected_point,f.reshape(-1,1)),axis=1)),axis=0)
     
     #update all the databases
     fitted_selected_point=fitted_grid_mesh[index_sel_points]
     fitted_sim_data=np.concatenate((fitted_selected_point,f.reshape(-1,1)),axis=1)
     train_data=append_data_splitmode(train_data,fitted_sim_data,split_size)
     
 
     if itr%50: 
       print('writing data')
       create_files(sim_data,'sim_data')

