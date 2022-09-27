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
#run=[1,2,3,4,5]
run=[1]
####################
rhog_safety=15096.9      # calculted as pho=1027, g=9.8, safety factor=1.5 ( from STR excel sheet)

thickness_l=0.01; thickness_h=0.5;

length_l=1.2; length_h=3.6;
depth_l=200; depth_h=6000;
#need to find it
radius_l=0.3; radius_h=1.2
#crush_pressure= rhog_safety*depth*0.000001
cp_l=rhog_safety*depth_l*0.000001
cp_h=rhog_safety*depth_h*0.000001
print('cp low is:',cp_l,'cp high is:',cp_h)


input_size=4                             # input size may change if integer/ordinal type variable and represented by one-hot encoding
output_size=1                            # number of output 
ranges=[thickness_l,thickness_h,radius_l,radius_h,length_l,length_h,cp_l,cp_h]                # ranges in form of [low1,high1,low2,high2,...]

mask=['real','real','real','real']                     # datatype ['dtype1','dtype2']
categories=[[None],[None],[None],[None]]  



batch_size = 64
device='cuda'


       


if __name__ == "__main__":
        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"]="1"  # specify which GPU(s) to be used
        ###Load evaluation data ,derived ground_truth and created storage for result  
        test_data= np.loadtxt("./data/dataware/test_data.txt", delimiter=" ",skiprows=0, dtype=np.float32)
        
        copied_test_data=np.copy(test_data)
        fitted_test_data= data_preperation(copied_test_data,mask,np.array(ranges),categories)
        
        testing_data = SimDataset(fitted_test_data)
        fitted_text_X= fitted_test_data[:,:-1]; fitted_test_y=fitted_test_data[:,-1]
        print('fitted X:',fitted_text_X,'fitted test Y:',fitted_test_y)
        path='./models/st/er1/nns_1.pt'
        neuralNet= SNet(input_size,output_size)
        try: 
           neuralNet.load_state_dict(torch.load(path))       
           print("Loaded earlier trained model successfully")
        except: 
           print('Couldnot find weights of NN')  
         
        model = neuralNet.to(device) 
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        epoch=0; loss_train=[];loss_validate=[]      
        with torch.no_grad(): 
            output = model(torch.from_numpy(fitted_text_X).float())
              

        copied_dense_sample_data=np.copy(test_data)
        output=output.cpu().detach().numpy()
        print('cds:',type(copied_dense_sample_data),'output:',type(output))
        ground_truth=label_data(copied_dense_sample_data,output)
    
        index_f = np.where(ground_truth[:,-1]==1)
        index_p = np.where(ground_truth[:,-1]==0)
      
        failed_gt= ground_truth[index_f[0]]
        passed_gt=ground_truth[index_p[0]]

        result=( passed_gt.shape[0]/(passed_gt.shape[0]+failed_gt.shape[0]))
        print('Accuracy is:', result) 


      
	



