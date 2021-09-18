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


input_size=2                             # input size may change if integer/ordinal type variable and represented by one-hot encoding
num_variable = 2                         # number of variables  both real & int type 
output_size=1                            # number of output 
num_iteration=10                        # Number of iteration of sampling

budget_samples=200                        # Number of samples-our budget
ranges=[-10,0,-6.5,0]                    # ranges in form of [low1,high1,low2,high2,...]
mask=['real','real']                     # datatype ['dtype1','dtype2']
random_gridmesh=False                    # (not using now) if state space is pretty big, it is not possible to create a big mesh of samples, in such case for each iteration, we randomly create a biggest possible mesh.       
categories=[[None],[None]]               #categories for ordinal variables, 'None' for real variables 
lbtm_size=100
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
    batch_size = args.batch_size
    max_epoch = args.epoch
    load_e=args.load
    
    ############# Intial sampling for startup & its evaluation
    samples=draw_samples(mask,budget_samples,np.array(ranges))
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
    fitted_test_data= data_preperation(copied_test_data,mask,np.array(ranges),categories)
    #################################################################################################
 

    #Loop for training
    for itr in range(num_iteration):

      ############Data preperation for neural net training ( 2 processes: 1. for real: scale data between 0 & 1, for int : one hot encoding ) 
      copied_train_data=np.copy(train_data)
      copied_test_data=np.copy(test_data)
      fitted_train_data= data_preperation(copied_train_data,mask,np.array(ranges),categories)
      fitted_test_data= data_preperation(copied_test_data,mask,np.array(ranges),categories)
     #################################################################################################
   
    
      print('==> Train data shape is:',train_data.shape,'Test data sample shape is:',test_data.shape)
      #Training of student model
      train_net(fitted_train_data,batch_size,max_epoch,device,input_size,output_size,'S') 
     
      #prediction on test data on student model  
      lnp=load_N_predict(fitted_test_data[:,:-1],input_size,output_size,"./models/student/",'S')
      stest_pred=lnp.run() 
 
      #check and label test data which have failed & passed
      lbtm_data=label_data(test_data,stest_pred)

      #updating limited buffer test memory 
      lbtm=update_lbtm(lbtm_data,lbtm_size)
     
      #Data preperation for Neural network training(teacher network)
      copied_lbtm_data=np.copy(lbtm)
      fitted_lbtm_data= data_preperation(copied_lbtm_data,mask,np.array(ranges),categories)
     
      #Train Teacher network 
      train_net(fitted_lbtm_data,batch_size,max_epoch,device,input_size,output_size,'T') 

      #procedure for selecting samples for next stage
      samples=draw_samples(mask,budget_samples,np.array(ranges))
      copied_random_sample_data=np.copy(samples)
      fitted_random_sample_data= data_preperation(copied_random_sample_data,mask,np.array(ranges),categories)

      lnp=load_N_predict(copied_random_sample_data,input_size,output_size,"./models/student/",'T')
      stest_pred=lnp.run() 
      selected_samples= choose_samples(samples,stet_pred,probality_of_selection=0.8)


      # simulate on selected samples 
      f,c=mishra(selected_point[:,0],selected_point[:,1])
      ###############################
      print('selected samples are:',selected_samples.shape,'f is:',f.shape)
      
      #update all data bases 
      temp_data= np.concatenate(selected_point,f.reshape(-1,1))
      sim_data= np.concatenate((sim_data,np.concatenate((selected_point,f.reshape(-1,1)),axis=1)),axis=0)
      new_train_data,new_test_data=tempdata_split(temp_data,proportion=0.2)
      train_data=np.concatenate((train_data,new_train_data),axis=0)
      test_data= np.concatenate((test_data,new_test_data),axis=0)

     
 
      if itr%50: 
       print('writing data')
       create_files(sim_data,'sim_data')

