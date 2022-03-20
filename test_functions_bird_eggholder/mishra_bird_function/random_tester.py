import os
import glob
import pathlib
import numpy as np
import torch
import torch.nn as nn
from student_model import SNet
from sklearn.preprocessing import MinMaxScaler
import torch.optim as optim
from sim_engine import *
from utils import *
#from trainer import model_trainer
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from lp import load_N_predict


######change for each run_id
run=[33]
####################


input_size=2                             # input size may change if integer/ordinal type variable and represented by one-hot encoding
output_size=1                            # number of output 
                
ranges=[-10,0,-6.5,0]                    # ranges in form of [low1,high1,low2,high2,...]
mask=['real','real']                     # datatype ['dtype1','dtype2']
categories=[[None],[None]]               #categories for ordinal variables, 'None' for real variables 


max_epoch = 500
batch_size = 32
device='cpu'
loss_fn=nn.L1Loss()
num_co=[]


N = []; minim=50; maxim=1000; sample=0
while sample<maxim:
   sample+=minim; N.append(sample)
#print('Samples to evaluate is:',N)

        


       
###Load evaluation data ,derived ground_truth and created storage for result  
dense_mesh= np.loadtxt("./data/dense_data.csv", delimiter=",",skiprows=0, dtype=np.float32)
eval_mesh= dense_mesh[:,0:2]
f_ground_eval_mesh= dense_mesh[:,2]
copied_eval_mesh=np.copy(eval_mesh)
fitted_eval_mesh= data_preperation(copied_eval_mesh,mask,np.array(ranges),categories)


result=[]
if __name__ == "__main__":
        
        #os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
        #os.environ["CUDA_VISIBLE_DEVICES"]="1"  # specify which GPU(s) to be used
        dense_mesh= np.loadtxt("./data/dense_data.csv", delimiter=",",skiprows=0, dtype=np.float32)
        eval_mesh= dense_mesh[:,0:2]
        f_ground_eval_mesh= dense_mesh[:,2]
        copied_eval_mesh=np.copy(eval_mesh)
        fitted_eval_mesh= data_preperation(copied_eval_mesh,mask,np.array(ranges),categories)
                    
        storage_urtest_mesh=[];
       
        for n in N:
         flag_first=0
         #print('****n is:',n)
         #samples=lhc_samples(n,2,ranges)
         samples= random_sampling(n)
         f,c= mishra(samples[:,0],samples[:,1])
     
         sim_data=np.concatenate((samples,f.reshape(-1,1)),axis=1)   
         #print('Samples size:',samples.shape,'f shape:',f.shape)
         #########################################################

         ############# create data holder for train and test
         train_data,test_data= data_split(sim_data,proportion=0.1)
         copied_train_data=np.copy(train_data)
         copied_test_data=np.copy(test_data)
         fitted_train_data= data_preperation(copied_train_data,mask,np.array(ranges),categories)
         fitted_test_data= data_preperation(copied_test_data,mask,np.array(ranges),categories)
         #########################################################
	 

         train_data = SimDataset(fitted_train_data)
         validate_data = SimDataset(fitted_test_data)
         #print('length of train data:',len(train_data))
	 
         path='./models/random_uniform/nn_rand_uni'+str(n)+'.pt'
         name= 'nn_rand_uni'+str(n)+'.pt'
         directory= './models/random_uniform/'

         neuralNet= SNet(input_size,output_size)
         try: 
           neuralNet.load_state_dict(torch.load(path))       
           #print("Loaded earlier trained model successfully")
         except: 
          neuralNet= neuralNet.apply(initialize_weights)
          #print('Randomly initialising weights')        
 

         #neuralNet= SNet(input_size,output_size).apply(initialize_weights)
         model = neuralNet.to(device) 
         optimizer = optim.Adam(model.parameters(), lr=0.003)
         epoch=0; loss_train=[];loss_validate=[]
         while True:
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
                
                if epoch <= 300:
                  whichmodel=epoch  
                  torch.save(model.state_dict(), path)

                if epoch> 300:
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
         plt.show()

         
         #print('--> Saved model is from', whichmodel , ' epoch')
         

         
         #prediction on eval data on student model
         lnp_e_m=load_N_predict(fitted_eval_mesh,input_size,output_size,directory,'S',name)
         eval_mesh_pred=lnp_e_m.run_name()

         copied_dense_sample_data=np.copy(dense_mesh)
         ground_truth=label_data(copied_dense_sample_data,eval_mesh_pred)
    
         index_f = np.where(ground_truth[:,-1]==1)
         index_p = np.where(ground_truth[:,-1]==0)
      
         failed_gt= ground_truth[index_f[0]]
         passed_gt=ground_truth[index_p[0]]
         print('***n is:',n,'Number of failed data:',failed_gt.shape[0],'Number of passed data:',passed_gt.shape[0])
         result.append(passed_gt.shape[0])
         #print('=================================================================')
        print('result is:',result)  



      
	



