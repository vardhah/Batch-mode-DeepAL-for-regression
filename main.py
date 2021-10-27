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


input_size=4                             # input size may change if integer/ordinal type variable and represented by one-hot encoding
num_variable = 4                         # number of variables  both real & int type 
output_size=1                            # number of output 
num_iteration=2                        # Number of iteration of sampling

budget_samples=1000                        # Number of samples-our budget
ranges=[1,100,1,100,10,200,10,200]                    # ranges in form of [low1,high1,low2,high2,...]
mask=['real','real','real','real']                     # datatype ['dtype1','dtype2']
random_gridmesh=False                    # (not using now) if state space is pretty big, it is not possible to create a big mesh of samples, in such case for each iteration, we randomly create a biggest possible mesh.       
categories=[[None],[None],[None],[None]]               #categories for ordinal variables, 'None' for real variables 
lbtm_size=600
probability_of_selection=0.8
lbtm=None
#here database container are just numpy array called => sim_data,train_data, test_data,lbtm (they contain unscaled/unnormalised data(i/p-o/p))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Population Based Training")
    parser.add_argument("-d","--device", type=str, default='cuda',
                        help="")
    parser.add_argument("-b","--batch_size", type=int, default=16,
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
    print('device is:',device)
    batch_size = args.batch_size
    max_epoch = args.epoch
    load_e=args.load
    
    ###Load evaluation data ,derived ground_truth and created storage for result  
    eval_mesh= np.loadtxt("./data/mesh_data.txt", delimiter=" ",skiprows=0, dtype=np.float32)
    eval_rand= np.loadtxt("./data/random_data.txt", delimiter=" ",skiprows=0, dtype=np.float32)
    f_ground_eval_mesh= p_v_d(eval_mesh)
    f_ground_eval_rand= p_v_d(eval_rand)
    copied_eval_mesh=np.copy(eval_mesh)
    copied_eval_rand=np.copy(eval_rand)
    fitted_eval_mesh= data_preperation(copied_eval_mesh,mask,np.array(ranges),categories)
    fitted_eval_rand= data_preperation(copied_eval_rand,mask,np.array(ranges),categories)
    storage_mesh=[]; storage_rand=[]

    
    ############# Intial sampling for startup & its evaluation
    samples=draw_samples(mask,budget_samples,np.array(ranges))
    f=p_v_d(samples)
    sim_data=np.concatenate((samples,f.reshape(-1,1)),axis=1)   
    print('Samples size:',samples.shape,'f shape:',f.shape)
    #########################################################

    ############# create data holder for train and test
    train_data,test_data= data_split(sim_data,proportion=0.2)
    #########################################################
    

    #Loop for training
    first_run_flag=0
    for itr in range(num_iteration):

      ############Data preperation for neural net training ( 2 processes: 1. for real: scale data between 0 & 1, for int : one hot encoding ) 
      copied_train_data=np.copy(train_data)
      copied_test_data=np.copy(test_data)
      fitted_train_data= data_preperation(copied_train_data,mask,np.array(ranges),categories)
      fitted_test_data= data_preperation(copied_test_data,mask,np.array(ranges),categories)
     #################################################################################################
   
    
      
      #Training of student model
      print('-----Training student net')
      train_net(fitted_train_data,batch_size,max_epoch,device,input_size,output_size,'S') 
     
      #prediction on test data on student model  
      print('----evaluate the test database on student net')
      lnp=load_N_predict(fitted_test_data[:,:-1],input_size,output_size,"./models/student/",'S')
      stest_pred=lnp.run() 
      
      #prediction on eval data on student model
      lnp_e_m=load_N_predict(fitted_eval_mesh,input_size,output_size,"./models/student/",'S')
      eval_mesh_pred=lnp_e_m.run()
      lnp_e_r=load_N_predict(fitted_eval_rand,input_size,output_size,"./models/student/",'S')
      eval_rand_pred=lnp_e_r.run()
      #print('shape of fg_eval_mesh:',f_ground_eval_mesh.shape, 'shape of predict:',eval_mesh_pred.shape)
      
      
      dev_eval_mesh=(np.sum(np.absolute(f_ground_eval_mesh-eval_mesh_pred.reshape(1,-1)))) 
      absolute_f_ground_eval_mesh= np.sum(np.absolute(f_ground_eval_mesh))
      print('dev_eval mesh:', dev_eval_mesh,' absolute_f_ground_eval_mesh:',absolute_f_ground_eval_mesh)
      metrics1= (dev_eval_mesh*100)/absolute_f_ground_eval_mesh
      exact_prediction_diff_mesh=np.absolute(f_ground_eval_mesh-eval_mesh_pred.reshape(1,-1))
      occurrences_more_than_thr_mesh = exact_prediction_diff_mesh > (0.05*np.absolute(f_ground_eval_mesh))	
      total_mesh = occurrences_more_than_thr_mesh.sum()
      percentage_total_mesh= (total_mesh*100)/eval_mesh_pred.shape[0]
      storage_mesh.append((itr,metrics1,total_mesh,percentage_total_mesh))

      dev_eval_rand=(np.sum(np.absolute(f_ground_eval_rand-eval_rand_pred.reshape(1,-1)))) 
      absolute_f_ground_eval_rand= np.sum(np.absolute(f_ground_eval_rand))
      print('dev_eval rand:', dev_eval_rand,' absolute_f_ground_eval_rand:',absolute_f_ground_eval_rand)
      metrics2= (dev_eval_rand*100)/absolute_f_ground_eval_rand
      exact_prediction_diff_rand =np.absolute(f_ground_eval_rand-eval_rand_pred.reshape(1,-1))
      occurrences_more_than_thr_rand = exact_prediction_diff_rand > (0.05*np.absolute(f_ground_eval_rand))	
      total_rand = occurrences_more_than_thr_rand.sum()
      percentage_total_rand = (total_rand*100)/eval_rand_pred.shape[0]
      storage_rand.append((itr,metrics2,total_rand,percentage_total_rand))


      print('======================================================================================================')
      print('itr is:',itr,' ,metrics1 is:',metrics1, ' ,total_mesh is:',total_mesh,' ,percentage_total_mesh:',percentage_total_mesh)
      print('itr is:',itr,' ,metrics2 is:',metrics2, ' ,total_rand is:',total_rand,' ,percentage_total_rand:',percentage_total_rand)
      
      
      #check and label test data which have failed & passed
      copied_test_data=np.copy(test_data)
      #print('test_data ground truth',test_data)
      lbtm_data=label_data(copied_test_data,stest_pred)
      #print('lbtm data:',lbtm_data,'stest_pred',stest_pred)
      #updating limited buffer test memory 
      if first_run_flag==0: 
        lbtm=lbtm_data
      else: 
        lbtm=np.concatenate((lbtm,lbtm_data),axis=0)
      lbtm= lbtm[-1*lbtm_size:-1,:]
      print('lbtm is:',lbtm.shape) 
     
      #Data preperation for Neural network training(teacher network)
      copied_lbtm_data=np.copy(lbtm)
      fitted_lbtm_data= data_preperation(copied_lbtm_data,mask,np.array(ranges),categories)
  
      #Train Teacher network
      print('-----Training teacher net') 
      train_net(fitted_lbtm_data,batch_size,max_epoch,device,input_size,output_size,'T') 
      

      #procedure for selecting samples for next stage
      samples=draw_samples(mask,budget_samples,np.array(ranges))
      copied_random_sample_data=np.copy(samples)
      fitted_random_sample_data= data_preperation(copied_random_sample_data,mask,np.array(ranges),categories)
      lnp=load_N_predict(fitted_random_sample_data,input_size,output_size,"./models/teacher/",'T')
      stest_pred=lnp.run() 
      #print('stest prediction is', stest_pred)
      selected_samples= choose_samples(samples,stest_pred,probability_of_selection)
      print('size of selected samples are:',selected_samples.shape)
      
      while selected_samples.shape[0]<(budget_samples/2):
          samples=draw_samples(mask,budget_samples,np.array(ranges))
          copied_random_sample_data=np.copy(samples)
          fitted_random_sample_data= data_preperation(copied_random_sample_data,mask,np.array(ranges),categories)
          lnp=load_N_predict(fitted_random_sample_data,input_size,output_size,"./models/teacher/",'T')
          stest_pred=lnp.run() 
          _samples= choose_samples(samples,stest_pred,probability_of_selection)
          selected_samples= np.concatenate((selected_samples,_samples),axis=0)

      print('size of selected samples are:',selected_samples.shape)


      if selected_samples.shape[0]<budget_samples:
          num_of_random_samples= budget_samples-selected_samples.shape[0]
          random_additional_samples= draw_samples(mask,num_of_random_samples,np.array(ranges))
          total_selected_samples= np.concatenate((selected_samples,random_additional_samples),axis=0)



      # simulate on selected samples 
      f=p_v_d(total_selected_samples)
      ###############################
      print('*selected samples from teacher is:',selected_samples.shape[0],'total selected samples(+random) are:',total_selected_samples.shape,'f is:',f.shape)
      
      

      #update all data bases 
      temp_data= np.concatenate((total_selected_samples,f.reshape(-1,1)),axis=1)
      #print('shape of temp_data:',temp_data)
      sim_data= np.concatenate((sim_data,temp_data),axis=0)
      new_train_data,new_test_data=data_split(temp_data,proportion=0.2)
      train_data=np.concatenate((train_data,new_train_data),axis=0)
      test_data= np.concatenate((test_data,new_test_data),axis=0)
      print('===> shape of temp_data:',temp_data.shape,'sim_data:',sim_data.shape,'train_data',train_data.shape,'test_data',test_data.shape,'lbtm_shape:',lbtm.shape)
      first_run_flag=1

     
      print('storage mesh is:',storage_mesh) 
    #if itr%50: 
    print('writing data')
    create_files(sim_data,'sim_data')
    create_files(storage_mesh,'storage_mesh')
    create_files(storage_rand,'storage_rand') 


