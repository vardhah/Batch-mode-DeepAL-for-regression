import argparse
import os
import glob
#import pathlib
import numpy as np
from trainer import model_trainer
from utils import *
import itertools
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
from lp import load_N_predict 
from model_training import train_net
from mpl_toolkits.mplot3d import Axes3D
Axes3D = Axes3D  # pycharm auto import
import logging
import shutil

run=[1,2,3,4,5,6,7,8,9,10]
#run=[1,2]

#type of exploration strategy to counter bias introduced by sampling
# currently testing [e_r_1.0, e_r_0.75, e_r_0.50, e_r_0.25, e_topb,e_greedy,betaw]
strategy= 'betaw'
beta=10



tl=1000; th=50000;
rpml=100; rpmh=1000
sl=1; sh=10;
dl=0.1; dh=2;
c1=0.01 
c2=0.01
c3=0.01
c4=0.01
c5=0.01 
c6=0.01
c7=0.01
c8=0.01
c9=0.01
c10=0.001
ch=0.5




search_itr=5                                      # Number of times search for better model
input_size=14                                     # input size may change if integer/ordinal type variable and represented by one-hot encoding
output_size=1                                     # number of output 
num_iteration=50                                  # Number of iteration of sampling
budget_samples=50                                 # Number of samples-our budget per iteration
ranges=[tl,th,sl,sh,rpml,rpmh,dl,dh,c1,ch,c2,ch,c3,ch,c4,ch,c5,ch,\
          c6,ch,c7,ch,c8,ch,0,1,0,1]              # ranges in form of [low1,high1,low2,high2,...]
#init_ranges=[-10,-8,-6.5,-5]
mask=['real','real','real','real','real','real','real','real','real','real','real','real','real','real']                     # datatype ['dtype1','dtype2']
random_gridmesh=False                             # (not using now) if state space is pretty big, it is not possible to create a big mesh of samples, in such case for each iteration, we randomly create a biggest possible mesh.       
categories=[[None],[None],[None],[None],[None],[None],[None],[None],[None],[None],[None],[None],\
              [None],[None]]                      #categories for ordinal variables, 'None' for real variables 
probability_of_selection=0.5
result=[]
#logging.basicConfig(level=logging.DEBUG)
logging.basicConfig(filename='./data/eaxmple.log',filemode='w')
epsilon=None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Population Based Training")
    parser.add_argument("-d","--device", type=str, default='cuda',
                        help="")
    parser.add_argument("-b","--batch_size", type=int, default=64,
                        help="")
    parser.add_argument("-l","--load", type=bool, default=False,
                        help="")
    parser.add_argument("-e","--epoch", type=int, default=50,
                        help="")

    args = parser.parse_args()
    #pathlib.Path('./models/student').mkdir(exist_ok=True)
    #pathlib.Path('./models/teacher').mkdir(exist_ok=True) 
    device = args.device
    if not torch.cuda.is_available():
        device = 'cpu'
    batch_size = args.batch_size
    max_epoch = args.epoch
    load_e=args.load
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]="1" 
    
    ############# Intial sampling for startup & its evaluation
    

    for runid in run: 
     
     print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
     print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
     print('*****runid is :',runid,'epsilon is',epsilon)
     result_file_name= './data/sim_result_ep_greedy.txt'
     student_path='./models/st/ep_greedy/nns_'+str(runid)+'.pt'
     teacher_path='./models/st/ep_greedy/nnt_'+str(runid)+'.pt'
     sim_data_file_name='sim_data_ep_greedy_'+str(runid) 
     student_last_itr_path='./models/st/ep_greedy/nns_'+str(runid)+'_last_itr'+'.pt'
     
     #############################################################################################################
     ##########################Dont change anything below this line ##############################################
     #############################################################################################################

     result=[]
     if strategy=='e_greedy':
        epsilon=0
     elif strategy=='e_r_0.25':
        epsilon=0.25
     elif strategy=='e_r_0.50':
        epsilon=0.5
     elif strategy=='e_r_0.75':
        epsilon=0.75
     elif strategy=='e_r_1.0':
        epsilon=1.0
     elif strategy=='e_topb':
        epsilon=2.0
     elif strategy=='betaw':
        beta_samples=beta*budget_samples
     earlier_passed_gt=0

     for itr in range(num_iteration):  
      if strategy=='e_greedy':    
         epsilon=epsilon+(1/(num_iteration))
      if itr==0:
           pool_mesh= np.loadtxt("./data/pool_data_openprop_scaledeff_no_fail.csv", delimiter=",",skiprows=0, dtype=np.float32)
           sim_data, eval_data=data_split_size(pool_mesh,budget_samples)
           print('train size:',sim_data.shape,'eval size:',eval_data.shape)
           
      ############Data preperation for neural net training ( 2 processes: 1. for real: scale data between 0 & 1, for int : one hot encoding ) 
      train_data,test_data= data_split(sim_data,proportion=0.1)
      copied_train_data=np.copy(train_data)
      copied_test_data=np.copy(test_data)
      fitted_train_data= data_preperation(copied_train_data,mask,np.array(ranges),categories)
      fitted_test_data= data_preperation(copied_test_data,mask,np.array(ranges),categories)
      
      #procedure for generating ground truth by the trained student
      eval_mesh= eval_data[:,0:input_size]
      f_ground_eval_mesh= eval_data[:,input_size]
      copied_eval_mesh=np.copy(eval_mesh)
      fitted_eval_mesh= data_preperation(copied_eval_mesh,mask,np.array(ranges),categories)
      #################################################################################################   
      #Training of student model.
      if itr!= 0:
       shutil.copy(student_path,student_last_itr_path)
       got_better_model=False
      for _ in range(search_itr):  
       print('-------<<<<<<Training student>>>>---------')
       train_net(fitted_train_data,batch_size,max_epoch,device,input_size,output_size,'S',student_path) 
       ########prediction on eval data on student model
       lnp_e_m=load_N_predict(fitted_eval_mesh,input_size,output_size,student_path,'S')
       eval_mesh_pred=lnp_e_m.run()
       copied_dense_sample_data=np.copy(eval_data)
       ground_truth=label_data(copied_dense_sample_data,eval_mesh_pred)
       index_f = np.where(ground_truth[:,-1]==1)
       index_p = np.where(ground_truth[:,-1]==0)
       failed_gt= ground_truth[index_f[0]]
       passed_gt=ground_truth[index_p[0]]
       if passed_gt.shape[0]>earlier_passed_gt:
         earlier_passed_gt=passed_gt.shape[0]
         got_better_model=True
         print('***************Hip Hip Hurray****<got better model>****************')
         break
       else:
         shutil.copy(student_last_itr_path,student_path)
      if got_better_model==False:
        result.append( earlier_passed_gt)
      else:
        result.append( passed_gt.shape[0])
      #############################################################################################
      
      #prediction on test and train data on student model  
      lnp=load_N_predict(fitted_test_data[:,:-1],input_size,output_size,student_path,'S')
      stest_pred=lnp.run() 

      lnp=load_N_predict(fitted_train_data[:,:-1],input_size,output_size,student_path,'S')
      strain_pred=lnp.run() 
      
      ##############################################################################################################
      #####################ground truth on labeling data ###########################################################
      ########check and label test data which have failed & passed #################################################

      copied_train_data=np.copy(train_data)
      copied_test_data=np.copy(test_data)
      total_eval_ip_data= np.concatenate((copied_train_data,copied_test_data),axis=0)
      total_eval_op_data= np.concatenate((strain_pred,stest_pred),axis=0)
      #print('actual eff:',total_eval_ip_data[:,-1],'predicted eff',total_eval_op_data)
      lbtm_data=label_data(total_eval_ip_data,total_eval_op_data)
     
      #Data preperation for Neural network training(teacher network)
      copied_lbtm_data=np.copy(lbtm_data)
      fitted_lbtm_data= data_preperation(copied_lbtm_data,mask,np.array(ranges),categories)
      print('-----Training teacher net') 
      train_net(fitted_lbtm_data,batch_size,max_epoch,device,input_size,output_size,'T',teacher_path) 
      
      ###################################################################################################################
      ####probability prediction by teacher on rest of pool data(eval data) #############################################
      ###################################################################################################################
   
      copied_eval_mesh=np.copy(eval_mesh)
      fitted_eval_mesh= data_preperation(copied_eval_mesh,mask,np.array(ranges),categories)
      

      lnp=load_N_predict(fitted_eval_mesh,input_size,output_size,teacher_path,'T')
      t_eval_pred=lnp.run() 
      copied_eval_data=np.copy(eval_data)
      

      ########################################################################################################################
      ##############################  selection strategy  ####################################################################
      ########################################################################################################################
      if strategy=='betaw':
       selected_samples,rest_of_pool_data= choose_samples_weighted_diversity(copied_eval_data,t_eval_pred,probability_of_selection,beta_samples,budget_samples)
      elif stategy=='e_topb':
       selected_samples,rest_of_pool_data= choose_topb_samples(copied_eval_data,t_eval_pred,probability_of_selection,budget_samples,epsilon)
      else:
        selected_samples,rest_of_pool_data= choose_samples_epsilon(copied_eval_data,t_eval_pred,probability_of_selection,budget_samples,epsilon)
      
      #########################################################################################################################
      ################################ simulate on selected samples ###########################################################
      ######################################################################################################################### 
     
      #update all data bases 
      sim_data= np.concatenate((sim_data,selected_samples),axis=0)
      eval_data=rest_of_pool_data
      #print('===> shape of selected_samples:',selected_samples.shape,'sim_data:',sim_data.shape,'lbtm_shape:',lbtm_data.shape,'rest_of_pool_data',rest_of_pool_data.shape)
      
      print('***itr is:',itr,'epsilon is',epsilon,'Number of failed data:',failed_gt.shape[0],'Number of passed data:',passed_gt.shape[0])
      print('=================================================================')
 
     
     create_files(sim_data,sim_data_file_name)
     #print('---------> result is :',result)
     if runid==1:
           #print('in 1st runid')
           multi_runresults= np.array(result).reshape(1,-1)
     else: 
           multi_runresults= np.concatenate((multi_runresults,np.array(result).reshape(1,-1)),axis=0)
     print('result is:',multi_runresults)
     np.savetxt(result_file_name,multi_runresults,  delimiter=',')
    
      


      
      
