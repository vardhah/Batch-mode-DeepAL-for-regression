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


ch=0.5
input_size=14                             # input size may change if integer/ordinal type variable and represented by one-hot encoding
num_variable = 14                        # number of variables  both real & int type 
output_size=1                            # number of output 
num_iteration=1                        # Number of iteration of sampling
init_samples=10 
budget_samples=10                       # Number of samples-our budget
ranges=[1000,5000,100,1000,1,10,0.1,2,0,1,0.1812,ch,0.2024,ch,0.2196,ch,0.2305,ch,\
          0.2311,ch,0.2173,ch,0.1807,ch,0,1,0,1]                    # ranges in form of [low1,high1,low2,high2,...]
#init_ranges=[-10,-8,-6.5,-5]
mask=['real','real','real','real','real','real','real','real','real','real','real','real','real','real']                     # datatype ['dtype1','dtype2']
random_gridmesh=False                    # (not using now) if state space is pretty big, it is not possible to create a big mesh of samples, in such case for each iteration, we randomly create a biggest possible mesh.       
categories=[[None],[None],[None],[None],[None],[None],[None],[None],[None],[None],[None],[None],\
              [None],[None]]               #categories for ordinal variables, 'None' for real variables 
lbtm_size=20
probability_of_selection=0.5
lbtm=None
distance=0.5
#here database container are just numpy array called => sim_data,train_data, test_data,lbtm (they contain unscaled/unnormalised data(i/p-o/p))

#logging.basicConfig(level=logging.DEBUG)
logging.basicConfig(filename='./data/eaxmple.log',filemode='w')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Population Based Training")
    parser.add_argument("-d","--device", type=str, default='cuda',
                        help="")
    parser.add_argument("-b","--batch_size", type=int, default=8,
                        help="")
    parser.add_argument("-l","--load", type=bool, default=False,
                        help="")
    parser.add_argument("-e","--epoch", type=int, default=1500,
                        help="")


    args = parser.parse_args()
    pathlib.Path('./checkpoints/student').mkdir(exist_ok=True)
    pathlib.Path('./checkpoints/teacher').mkdir(exist_ok=True)
    pathlib.Path('./models/student').mkdir(exist_ok=True)
    pathlib.Path('./models/teacher').mkdir(exist_ok=True) 
    device = args.device
    if not torch.cuda.is_available():
        device = 'cpu'
    batch_size = args.batch_size
    max_epoch = args.epoch
    load_e=args.load
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]="1" 
    
    ############# Intial sampling for startup & its evaluation
    samples=random_sampling(init_samples)
    f=opensim(samples)
    #print('f is:',f)
    sim_data=np.concatenate((samples,f.reshape(-1,1)),axis=1)
    #print('Samples size:',samples.shape,'f shape:',f.shape)
    #########################################################

    ############# create data holder for train and test
    train_data,test_data= data_split(sim_data,proportion=0.3)
    #########################################################
    logging.error('%s num of bughet sample is',str(budget_samples))


      
    for itr in range(num_iteration):
      print('*****itr is :',itr)
      ############Data preperation for neural net training ( 2 processes: 1. for real: scale data between 0 & 1, for int : one hot encoding ) 
      copied_train_data=np.copy(train_data)
      copied_test_data=np.copy(test_data)
      fitted_train_data= data_preperation(copied_train_data,mask,np.array(ranges),categories)
      fitted_test_data= data_preperation(copied_test_data,mask,np.array(ranges),categories)
      
     #################################################################################################
      
      #Training of student model.
      train_net(fitted_train_data,batch_size,max_epoch,device,input_size,output_size,'S') 

      #############################################################################################
      
     
      #prediction on test data on student model  
      #print('----evaluate the test database on student net')
      lnp=load_N_predict(fitted_test_data[:,:-1],input_size,output_size,"./models/student/",'S')
      stest_pred=lnp.run() 

      lnp=load_N_predict(fitted_train_data[:,:-1],input_size,output_size,"./models/student/",'S')
      strain_pred=lnp.run() 
      #########################################################################################
      #####################ground truth on labeling data ######################################
      #check and label test data which have failed & passed
      copied_train_data=np.copy(train_data)
      copied_test_data=np.copy(test_data)
      total_eval_ip_data= np.concatenate((copied_train_data,copied_test_data),axis=0)
      total_eval_op_data= np.concatenate((strain_pred,stest_pred),axis=0)
      #print('actual eff:',total_eval_ip_data[:,-1],'predicted eff',total_eval_op_data)
      lbtm_data=label_data(total_eval_ip_data,total_eval_op_data)
      copied_train_data=np.copy(train_data)
      copied_test_data=np.copy(test_data)
      output_analysis_data= np.concatenate((np.concatenate((copied_train_data,copied_test_data),axis=0)[:,-1].reshape(-1,1),total_eval_op_data,lbtm_data[:,-1].reshape(-1,1)),axis=1)
      #print('lbtm data is:',lbtm_data)
      #total_lbtmdata=np.concatenate((lbtm_data),axis=1) 
      named='./data/lbtmdata'+str(itr)+'.csv'
      np.savetxt(named, lbtm_data, delimiter=",")
      np.savetxt('result_comp.csv', output_analysis_data, delimiter=",")
      #fig=plt.figure(figsize=(9,6))
      #imgname='./fig/test_samples'+str(itr)+'.png'
      index_f = np.where(lbtm_data[:,-1]==1)
      index_p = np.where(lbtm_data[:,-1]==0)
      failed_lbtm= lbtm_data[index_f[0]]
      passed_lbtm=lbtm_data[index_p[0]]
     

     
      #Data preperation for Neural network training(teacher network)
      copied_lbtm_data=np.copy(lbtm_data)
      fitted_lbtm_data= data_preperation(copied_lbtm_data,mask,np.array(ranges),categories)
      #print('fitted lbtm data:',fitted_lbtm_data)
      #Train Teacher network
      #print('-----Training teacher net') 
      train_net(fitted_lbtm_data,batch_size,max_epoch,device,input_size,output_size,'T') 
      
      ####Ground truth created by teacher ###################################################
      #######################################################################################
      dense_samples=random_sampling(100000)
      copied_dense_sample_data=np.copy(dense_samples)
      fitted_dense_sample_data= data_preperation(copied_dense_sample_data,mask,np.array(ranges),categories)

      lnp=load_N_predict(fitted_dense_sample_data,input_size,output_size,"./models/teacher/",'T')
      t_test_pred=lnp.run() 
      selected_dense_samples= choose_samples(dense_samples,t_test_pred,probability_of_selection)
      img_name= './fig/teacher_boundary'+str(itr)+'.png'
      
     
      ########################################################################################
      ########################################################################################

      #procedure for selecting samples for next stage
      samples=random_sampling(budget_samples)
      copied_random_sample_data=np.copy(samples)
      fitted_random_sample_data= data_preperation(copied_random_sample_data,mask,np.array(ranges),categories)
      lnp=load_N_predict(fitted_random_sample_data,input_size,output_size,"./models/teacher/",'T')
      stest_pred=lnp.run() 
      #print('stest prediction is', stest_pred)
      selected_samples= choose_samples(samples,stest_pred,probability_of_selection)
      
      while selected_samples.shape[0]<(budget_samples*distance):
          samples=draw_samples(mask,budget_samples,np.array(ranges))
          copied_random_sample_data=np.copy(samples)
          fitted_random_sample_data= data_preperation(copied_random_sample_data,mask,np.array(ranges),categories)
          lnp=load_N_predict(fitted_random_sample_data,input_size,output_size,"./models/teacher/",'T')
          stest_pred=lnp.run() 
          _samples= choose_samples(samples,stest_pred,probability_of_selection)
          selected_samples= np.concatenate((selected_samples,_samples),axis=0)

      if selected_samples.shape[0]<budget_samples:
          num_of_random_samples= budget_samples-selected_samples.shape[0]
          random_additional_samples= random_sampling(num_of_random_samples)
          total_selected_samples= np.concatenate((selected_samples,random_additional_samples),axis=0)
      else: 
          total_selected_samples= selected_samples[0:budget_samples,:]
      #print('shape of total selected samples:',total_selected_samples.shape)

      # simulate on selected samples 
      f=opensim(total_selected_samples)
      

      #update all data bases 
      temp_data= np.concatenate((total_selected_samples,f.reshape(-1,1)),axis=1)
      #print('shape of temp_data:',temp_data)
      sim_data= np.concatenate((sim_data,temp_data),axis=0)
      size_test=int(temp_data.shape[0]*0.3)
      #new_test_data_x=draw_samples(mask,size_test,np.array(ranges))
      new_test_data_x=random_sampling(size_test)
      f_t=opensim(new_test_data_x)
      new_test_data=np.concatenate((new_test_data_x,f_t.reshape(-1,1)),axis=1)
      
      train_data=np.concatenate((train_data,temp_data),axis=0)
      test_data= np.concatenate((test_data,new_test_data),axis=0)
      print('===> shape of temp_data:',temp_data.shape,'sim_data:',sim_data.shape,'train_data',train_data.shape,'test_data',test_data.shape,'lbtm_shape:',lbtm_data.shape)
      first_run_flag=1

     
 
      if itr%50: 
       #print('writing data')
       create_files(sim_data,'sim_data')

