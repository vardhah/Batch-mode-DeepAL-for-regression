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
num_iteration=20                        # Number of iteration of sampling
init_samples=50 
budget_samples=50                        # Number of samples-our budget
ranges=[-10,0,-6.5,0]                    # ranges in form of [low1,high1,low2,high2,...]
#init_ranges=[-10,-8,-6.5,-5]
mask=['real','real']                     # datatype ['dtype1','dtype2']
#random_gridmesh=False                    # (not using now) if state space is pretty big, it is not possible to create a big mesh of samples, in such case for each iteration, we randomly create a biggest possible mesh.       
categories=[[None],[None]]               #categories for ordinal variables, 'None' for real variables 
probability_of_selection=0.5
lbtm=None
distance=0.5
#here database container are just numpy array called => sim_data,train_data, test_data,lbtm (they contain unscaled/unnormalised data(i/p-o/p))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Population Based Training")
    parser.add_argument("-d","--device", type=str, default='cuda',
                        help="")
    parser.add_argument("-b","--batch_size", type=int, default=8,
                        help="")
    parser.add_argument("-l","--load", type=bool, default=False,
                        help="")
    parser.add_argument("-e","--epoch", type=int, default=1000,
                        help="")

    f = open("./data/log.txt", "w")
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
    samples=draw_samples(mask,init_samples,np.array(ranges))
    f,c=mishra(samples[:,0],samples[:,1])
    sim_data=np.concatenate((samples,f.reshape(-1,1)),axis=1)
    #print('Samples size:',samples.shape,'f shape:',f.shape)
    #########################################################

    ############# create data holder for train and test
    train_data,test_data= data_split(sim_data,proportion=0.1)
    #########################################################
    dense_samples=random_sampling(10000)
    copied_dense_sample_data=np.copy(dense_samples)
    fitted_dense_sample_data= data_preperation(copied_dense_sample_data,mask,np.array(ranges),categories)
    f_d,c_d=mishra(dense_samples[:,0],dense_samples[:,1])
    total_dense_sample=  np.concatenate((dense_samples,f_d.reshape(-1,1)),axis=1)
    np.savetxt('./data/dense_data.csv', total_dense_sample, delimiter=",")
      
    print('Initial sample size:', str(init_samples), 'Budget for iteration:', str(budget_samples)) 
      

    #Loop for training
    for itr in range(num_iteration):
      print('*****itr is :',itr)
      ############Data preperation for neural net training ( 2 processes: 1. for real: scale data between 0 & 1, for int : one hot encoding ) 
      copied_train_data=np.copy(train_data)
      copied_test_data=np.copy(test_data)
      fitted_train_data= data_preperation(copied_train_data,mask,np.array(ranges),categories)
      fitted_test_data= data_preperation(copied_test_data,mask,np.array(ranges),categories)
      
     #################################################################################################
      
      #Training of student model
      print('-----Training student net')
      train_net(fitted_train_data,batch_size,max_epoch,device,input_size,output_size,'S') 

      #############################################################################################
      ######################ground truth on trained data #########################################
      copied_total_dense_sample=np.copy(total_dense_sample)
      lnp=load_N_predict(fitted_dense_sample_data,input_size,output_size,"./models/student/",'S')
      dtest_pred=lnp.run() 
      ground_truth=label_data(copied_total_dense_sample,dtest_pred)
      
      

      #fig=plt.figure(figsize=(9,6))

      #imgname='./fig/manifold'+str(itr)+'.png'
      index_f = np.where(ground_truth[:,-1]==1)
      index_p = np.where(ground_truth[:,-1]==0)
      
      failed_gt= ground_truth[index_f[0]]
      passed_gt=ground_truth[index_p[0]]
      print('Itr is:',itr,'Number of failed data:',failed_gt.shape[0],'Number of passed data:',passed_gt.shape[0])
      #plt.scatter(failed_gt[:,0],failed_gt[:,1],c='lightskyblue',label='f_stu')
      #plt.scatter(passed_gt[:,0],passed_gt[:,1],c='khaki',label='p_stu')
      
      #plt.xlim([-10,0])
      #plt.ylim([-6.5,0])
      #plt.legend()
      #plt.title('failed and passed ground truth data')
      #plt.show(block=False)
      #plt.pause(5)
      #plt.savefig(imgname)
      #plt.close(fig)

      ############################################################################################
      ############################################################################################
     
      #prediction on test and train data on student model  
      #print('----evaluate the test database on student net')
      lnp=load_N_predict(fitted_test_data[:,:-1],input_size,output_size,"./models/student/",'S')
      stest_pred=lnp.run() 
      
      #print('----evaluate the train database on student net')
      lnp=load_N_predict(fitted_train_data[:,:-1],input_size,output_size,"./models/student/",'S')
      strain_pred=lnp.run() 
      
      #########################################################################################
      #####################ground truth on labeling data ######################################
      #check and label test data which have failed & passed
      copied_train_data=np.copy(train_data)
      copied_test_data=np.copy(test_data)
      total_eval_ip_data= np.concatenate((copied_train_data,copied_test_data),axis=0)
      total_eval_op_data= np.concatenate((strain_pred,stest_pred),axis=0)
      #print('test eval ip size:',total_eval_ip_data.shape,'test eval op size:',total_eval_op_data.shape)
      lbtm_data=label_data(total_eval_ip_data,total_eval_op_data)
  
      named='./data/lbtmdata'+str(itr)+'.csv'
      np.savetxt(named, lbtm_data, delimiter=",")

      index_f = np.where(lbtm_data[:,-1]==1)
      index_p = np.where(lbtm_data[:,-1]==0)
      failed_lbtm= lbtm_data[index_f[0]]
      passed_lbtm=lbtm_data[index_p[0]]
      
      ##################################################################################
      ###################################################################################

     
      #Data preperation for Neural network training(teacher network)
      copied_lbtm_data=np.copy(lbtm_data)
      fitted_lbtm_data= data_preperation(copied_lbtm_data,mask,np.array(ranges),categories)
      #print('fitted lbtm data:',fitted_lbtm_data)
      #Train Teacher network
      print('-----Training teacher net') 
      train_net(fitted_lbtm_data,batch_size,max_epoch,device,input_size,output_size,'T') 
      
      ####Ground truth created by teacher ###################################################
      #######################################################################################
      lnp=load_N_predict(fitted_dense_sample_data,input_size,output_size,"./models/teacher/",'T')
      t_test_pred=lnp.run() 
      selected_dense_samples= choose_samples(dense_samples,t_test_pred,probability_of_selection)
      named='./data/teacher_on_dense'+str(itr)+'.csv'
      np.savetxt(named, t_test_pred, delimiter=",")
      img_name= './fig/teacher_boundary'+str(itr)+'.png'
      
      """
      fig=plt.figure(figsize=(9,6))
      plt.scatter(dense_samples[:,0],dense_samples[:,1],c='tomato',label='rejected')
      plt.scatter(selected_dense_samples[:,0],selected_dense_samples[:,1],c='yellowgreen',label='selected')
      plt.xlim([-10,0])
      plt.ylim([-6.5,0])
      plt.legend()
      plt.title('boundary created by teacher network')
      plt.show(block=False)
      plt.pause(5)
      plt.savefig(img_name)
      plt.close(fig)
      """
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
      num_of_teacher_run=1
      while selected_samples.shape[0]<int(budget_samples*distance):
          samples=draw_samples(mask,budget_samples,np.array(ranges))
          copied_random_sample_data=np.copy(samples)
          fitted_random_sample_data= data_preperation(copied_random_sample_data,mask,np.array(ranges),categories)
          lnp=load_N_predict(fitted_random_sample_data,input_size,output_size,"./models/teacher/",'T')
          stest_pred=lnp.run() 
          _samples= choose_samples(samples,stest_pred,probability_of_selection)
          selected_samples= np.concatenate((selected_samples,_samples),axis=0)
          num_of_teacher_run+=1
      selected_samples= selected_samples[0:int(budget_samples*distance),:]
      print("-->selected samples from teacher:",selected_samples.shape,'num of teacher run:',num_of_teacher_run)

      if selected_samples.shape[0]<budget_samples:
          num_of_random_samples= budget_samples-selected_samples.shape[0]
          random_additional_samples= draw_samples(mask,num_of_random_samples,np.array(ranges))
          total_selected_samples= np.concatenate((selected_samples,random_additional_samples),axis=0)
          print("---->Total selected samples :",total_selected_samples.shape)

      # simulate on selected samples 
      f,c=mishra(total_selected_samples[:,0],total_selected_samples[:,1])
      
      print('===> shape of sim_data:',sim_data.shape,'train_data',train_data.shape,'test_data',test_data.shape,'lbtm_shape:',lbtm_data.shape)
      print('*******Finished iteration :',itr)
      print("===================================================================")
      #update all data bases 
      temp_data= np.concatenate((total_selected_samples,f.reshape(-1,1)),axis=1)
      #print('shape of temp_data:',temp_data)
      sim_data= np.concatenate((sim_data,temp_data),axis=0)
      size_test=int(temp_data.shape[0]*0.1)
      #new_test_data_x=draw_samples(mask,size_test,np.array(ranges))
      new_test_data_x=random_sampling(size_test)
      f_t,c_t=mishra(new_test_data_x[:,0],new_test_data_x[:,1])
      new_test_data=np.concatenate((new_test_data_x,f_t.reshape(-1,1)),axis=1)
      
      train_data=np.concatenate((train_data,temp_data),axis=0)
      test_data= np.concatenate((test_data,new_test_data),axis=0)
      

      create_files(sim_data,'sim_data')

    #close("./data/log.txt")
