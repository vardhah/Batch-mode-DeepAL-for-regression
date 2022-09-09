import os
import glob
import pathlib
import numpy as np
import torch
import torch.nn as nn
from student_model import SNet
from sklearn.preprocessing import MinMaxScaler
import torch.optim as optim
from utils import *
#from trainer import model_trainer
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from lp import load_N_predict
import shutil

######change for each run_id
run=[1,2,3,4,5]
#run=[1]
####################
a_l=50; a_h=600;
b_l=1 ; b_h=1850
c_l= 50; c_h=600
d_l=50 ;d_h=200
n_l= 1; n_h= 5
t_l=1 ; t_h= 50




search_itr=5 
input_size=6                           # input size may change if integer/ordinal type variable and represented by one-hot encoding
num_variable = 4                        # number of variables  both real & int type 
output_size=1                           # number of output 
num_iteration=50                        # Number of iteration of sampling
budget_samples=50                       # Number of samples-our budget
ranges=[a_l,a_h,b_l,b_h,c_l,c_h,d_l,d_h,n_l,n_h,t_l,t_h]                # ranges in form of [low1,high1,low2,high2,...]

mask=['real','real','real','real','real','real']                     # datatype ['dtype1','dtype2',...]
categories=[[None],[None],[None],[None],[None],[None]]               #categories for ordinal variables, 'None' for real variables 
result=[]



#########################

max_epoch = 200
at_least_epoch=25
batch_size = 64
device='cuda'
loss_fn=nn.L1Loss()
num_co=[]


N = []; minim=20; maxim=1000; sample=0
while sample<maxim:
   sample+=minim; N.append(sample)
print('Samples to evaluate is:',N)
result_file_name= './data/sim_result_uni_random.txt'
        


       

print('run is:',run)

if __name__ == "__main__":
       for runid in run: 
        print('+++++++++++++++++++++++++++++++++++++++++++++++++++')
        print('++++++++++++++++++++++++++++++++++++++++++++++++++++')
        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"]="1"  # specify which GPU(s) to be used
        ###Load evaluation data ,derived ground_truth and created storage for result  
        earlier_passed_gt=0
        result=[]
        for n in N:
         flag_first=0
         if n==minim:
           print('****n is:',n)
           pool_mesh= np.loadtxt("./data/pool_data.csv", delimiter=",",skiprows=0, dtype=np.float32)
           sim_data, eval_data=data_split_size(pool_mesh,minim)
           traini_data,test_data= data_split(sim_data,proportion=0.1) 
         else:
           print('****n is:',n)  
           t_data, eval_data=data_split_size(eval_data,minim)
           e_traini_data,e_test_data= data_split(t_data,proportion=0.1) 
           traini_data=np.concatenate((traini_data,e_traini_data),axis=0)
           test_data=np.concatenate((test_data,e_test_data),axis=0)
         
         print('train size:',traini_data.shape,'test_data:',test_data.shape,'eval size:',eval_data.shape) 

         eval_mesh= eval_data[:,0:6]
         #print('size of eval_mesh',eval_mesh.shape)
         f_ground_eval_mesh= eval_data[:,6]
         #print('size of f_ground_eval',f_ground_eval_mesh.shape)
         copied_eval_mesh=np.copy(eval_mesh)
         fitted_eval_mesh= data_preperation(copied_eval_mesh,mask,np.array(ranges),categories)

     
         #print('Samples size:',samples.shape,'f shape:',f.shape)
         #########################################################

         ############# create data holder for train and test
         train_data,validation_data= data_split(traini_data,proportion=0.1)
         copied_train_data=np.copy(train_data)
         copied_validation_data=np.copy(validation_data)
         #print('copied_train_data', copied_train_data[0])
         fitted_train_data= data_preperation(copied_train_data,mask,np.array(ranges),categories)
         fitted_validation_data= data_preperation(copied_validation_data,mask,np.array(ranges),categories)
         #########################################################
         #print('fitted_train_data', fitted_train_data[0])

         train_data = SimDataset(fitted_train_data)
         validate_data = SimDataset(fitted_validation_data)
         print('length of train data:',len(train_data))
	 
         path='./models/random_uniform/nn_rand_uni_'+str(runid)+'.pt'
         name= 'nn_rand_uni_'+str(runid)+'.pt'
         directory= './models/random_uniform/'
         last_itr_path='./models/random_uniform/nn_rand_uni_'+str(runid)+'_last_itr.pt'

         if n!= minim:
           shutil.copy(path,last_itr_path)
           got_better_model=False
         for _ in range(search_itr): 
          print('-------------<<<<<<<<Training SNET>>>>>>>>>>>>>-----------')
          neuralNet= SNet(input_size,output_size)
          try: 
           neuralNet.load_state_dict(torch.load(path))       
           print("Loaded earlier trained model successfully")
          except: 
           neuralNet= neuralNet.apply(initialize_weights)
           print('Randomly initialising weights')  
          #neuralNet= SNet(input_size,output_size).apply(initialize_weights)
          model = neuralNet.to(device) 
          optimizer = optim.Adam(model.parameters(), lr=0.001)
          epoch=0; loss_train=[];loss_validate=[]    
 
          while True: 
            #print('training epoch:',epoch)   
            if epoch > max_epoch:
                break    
            try:
                dataloader = DataLoader(train_data, batch_size, True)
                correct = 0
                for x, y in dataloader:
                	y=y.view(-1,1)
                	x, y = x.to(device), y.to(device)
                	output = model(x)
                	loss = loss_fn(output, y)
                	optimizer.zero_grad()
                	loss.backward() 
                	optimizer.step()
                	correct+= loss.item()
                train_loss=correct/len(train_data); loss_train.append(train_loss)
                
                with torch.no_grad(): 
                  dataloader = DataLoader(validate_data, batch_size, True)
                  correct = 0
                  for x, y in dataloader:
                    y=y.view(-1,1)
                    x, y = x.to(device), y.to(device)
                    output = model(x)
                    loss = loss_fn(output, y)
                    correct += loss.item()
                validate_loss= correct/len(dataloader); loss_validate.append(validate_loss) 
                
                if epoch <= at_least_epoch:
                  whichmodel=epoch  
                  torch.save(model.state_dict(), path)
                #if epoch%20==0:
                   #print('epoch is:',epoch)
                if epoch> at_least_epoch:
                 diff_loss=np.absolute(train_loss-validate_loss)
                 if flag_first==0: 
                   torch.save(model.state_dict(), path)
                   whichmodel=epoch 
                   flag_first=1
                   last_diff_loss=diff_loss

                 elif flag_first==1:
                  if last_diff_loss>diff_loss:
                   torch.save(model.state_dict(), path); whichmodel=epoch ;
                   last_diff_loss=diff_loss

            except KeyboardInterrupt:
                break
           
            epoch+=1

          #fig=plt.figure(figsize=(9,6))
          #plt.plot(loss_train,label='training')
          #plt.plot(loss_validate,label='validate')
          #plt.legend()
          #plt.show()

         
          print('--> Saved model is from', whichmodel , ' epoch')
         

         
          #prediction on eval data on student model
          lnp_e_m=load_N_predict(fitted_eval_mesh,input_size,output_size,path,'S')
          eval_mesh_pred=lnp_e_m.run()

          copied_dense_sample_data=np.copy(eval_data)
          ground_truth=label_data(copied_dense_sample_data,eval_mesh_pred)
    
          index_f = np.where(ground_truth[:,-1]==1)
          index_p = np.where(ground_truth[:,-1]==0)
      
          failed_gt= ground_truth[index_f[0]]
          passed_gt=ground_truth[index_p[0]]
          #print('passed gt shape:',passed_gt.shape[0])
          if (passed_gt.shape[0]/(passed_gt.shape[0]+failed_gt.shape[0]))>earlier_passed_gt:
             earlier_passed_gt=(passed_gt.shape[0]/(passed_gt.shape[0]+failed_gt.shape[0]))
             current_passed_gt=earlier_passed_gt
             got_better_model=True
             print('***************Hip Hip Hurray****<got better model>****************')
             break
          else: 
             shutil.copy(last_itr_path,path) 
        
         print('=================================================================')
         if got_better_model==False:
            lnp_e_m=load_N_predict(fitted_eval_mesh,input_size,output_size,path,'S')
            eval_mesh_pred=lnp_e_m.run()
            copied_dense_sample_data=np.copy(eval_data)
            ground_truth=label_data(copied_dense_sample_data,eval_mesh_pred)
            index_f = np.where(ground_truth[:,-1]==1)
            index_p = np.where(ground_truth[:,-1]==0)
            failed_gt= ground_truth[index_f[0]]
            passed_gt=ground_truth[index_p[0]]
            result.append( passed_gt.shape[0]/(passed_gt.shape[0]+failed_gt.shape[0]))
            current_passed_gt= passed_gt.shape[0]/(passed_gt.shape[0]+failed_gt.shape[0])
         else:
           result.append( passed_gt.shape[0]/(passed_gt.shape[0]+failed_gt.shape[0]))
         print('***n is:',n,'Number of failed data:',failed_gt.shape[0],'Number of passed data:',passed_gt.shape[0],'percentage is:',passed_gt.shape[0]/(passed_gt.shape[0]+failed_gt.shape[0]))
         if current_passed_gt>= 0.99:
           break
        
        if runid==1:
           print('in 1st runid')
           multi_runresults= np.array(result).reshape(1,-1)
        else: 
           multi_runresults= np.concatenate((multi_runresults,np.array(result).reshape(1,-1)),axis=0)
        print('result is:',multi_runresults)
        np.savetxt(result_file_name,multi_runresults,  delimiter=',')


      
	



