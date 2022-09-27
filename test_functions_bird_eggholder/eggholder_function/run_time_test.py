import os
import glob
import pathlib
import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd
#import seaborn as sns
import matplotlib.pylab as plt
from sim_engine import mishra
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
from student_model import SNet
import time

input_size=2                             # input size may change if integer/ordinal type variable and represented by one-hot encoding
num_variable = 2                         # number of variables  both real & int type 
output_size=1   
ranges=[-10,0,-6.5,0]                    # ranges in form of [low1,high1,low2,high2,...]
mask=['real','real']                     # datatype ['dtype1','dtype2']
categories=[[None],[None]]               #categories for ordinal variables, 'None' for real variables 
device='cuda'
N=[100,200,300,400,500,600,700,800,900,1000]
dense_data= pd.read_csv("./data/dense_data.csv").to_numpy()

if __name__ == "__main__":
       dense_data_input= dense_data[:,0:2]
       dense_data_output= dense_data[:,2]
       print('dense data input:',dense_data_input,'dense data output:',dense_data_output)
       
       print('size of dense data:',dense_data.shape) 
       os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
       os.environ["CUDA_VISIBLE_DEVICES"]="1"  
       copied_dense_data=np.copy(dense_data_input)
       fitted_test_data= data_preperation(copied_dense_data,mask,np.array(ranges),categories)

       #print('length of train data:',len(train_data))
	 
       path='./models1_mishra_100/student/nns.pt'
       neuralNet= SNet(input_size,output_size)
       try: 
           neuralNet.load_state_dict(torch.load(path))       
           print("Loaded earlier trained model successfully")
       except: 
          print('failed to load model')
       start=time.time()        
       with torch.no_grad():
            action_original = neuralNet(torch.from_numpy(test_data).float()).cpu().float().numpy()
       print('total time is:',time.time()-start) 
       #print('action original is:',action_original) 
       
       solution=[]
       start=time.time() 
       i=1
       for i in range(test_data.shape[0]):
       #if i==1:
            eva,s=mishra(test_data[i,0],test_data[i,1])
            solution.append(eva)
       print('total time in loop is:',time.time()-start) 
       #print('solution is:',solution)
       
      
