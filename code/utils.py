# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 16:11:45 2021

@author: HPP
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
device = torch.device("cpu")
from sklearn.preprocessing import OneHotEncoder
import copy 
import matplotlib.pyplot as plt
class utilities():
    def __init__(self):
        self.lr=0

    def set_lr(self, lr):
    	 self.lr = lr
    def set_lr_auto(self):
       self.lr = np.random.choice(np.logspace(-3, 0, base=10))

    def get_optimizer(self,model):
     optimizer_class = optim.Adam  
     print('***learning rate is:',self.lr)
     return optimizer_class(model.parameters(), lr=0.001)

def data_partioning(self,data,population):
      #based on population 
     a_list=np.arange(data.shape[0])
     np.random.shuffle(a_list)
     a_split=np.array_split(a_list,population)
     data_split=[]
     for i in range(len(a_split)):
        data_split.append(data[a_split[i]])
     return data_split

def append_data_splitmode(data,new_data,population):
    snew=np.array_split(new_data, population)
    for i in range(population):
      data[i]=np.append(data[i],snew[i],axis=0)
    return data

def append_data_randomly_splitmode(data,new_data,population):
    sel_pop=np.random.randint(population,size=1)
    print('selected population:',sel_pop,"data size is", len(data) )
    data[sel_pop[0]]=np.append(data[sel_pop[0]],new_data,axis=0)
    return data


def scale_data(data,ranges):
     minimum=[] ; total_range=[]
     for i in range(data.shape[1]):
       minimum.append(ranges[2*i])
       total_range.append((ranges[2*i+1]-ranges[2*i]))
     #print('min is:',minimum,'Range is:',total_range,'data b/f scaling is:',data)
     minimum=np.array(minimum).reshape(1,-1)
     total_range=np.array(total_range).reshape(1,-1)
     data= np.divide((data-minimum),total_range)
     #print('min is:',minimum,'Range is:',total_range,'data a/f scaling is:',data)
     return data

def rescale_data(data,ranges,mask):
    print('data b/f rescaling:',data)
    for i in range(len(mask)):
      print('i is:',i,'mask is:',mask[i])
     
      if mask[i]=='real':
         data[:,i]= (data[:,i]*(ranges[2*i+1]- ranges[2*i]))+ranges[2*i]
      elif mask[i]=='int':
         data[:,i]= (data[:,i]*(ranges[2*i+1]- ranges[2*i]))+ranges[2*i]
         data[:,i]=np.array(data[:,i], dtype=np.int16)
    print('data after rescaling:',data)
    return data

#on a given input prepare data for training
def data_preperation(data,mask,ranges,cat):    
    for i in range(len(mask)):
      print('i is:',i,'mask is:',mask[i])
      if mask[i]=='real':
         data[:,i]= (data[:,i]-ranges[2*i])/(ranges[2*i+1]- ranges[2*i])
      elif mask[i]=='int':
         enc = OneHotEncoder(categories=cat[i])
         data[:,i]=enc.transform(data[:,i]).toarray()
    return data
      
update_lbtm(data,size):
    return data[0:size,:] 

def data_split_size(data,size):
      a_list=np.arange(data.shape[0])
      np.random.shuffle(a_list)
      alist=a_list[0:size]
      train_data=data[alist]
      d=np.arange(data.shape[0])
      leftover=np.delete(d,alist)
      validate_data=data[leftover]
      return train_data,validate_data


def data_split(data,proportion=0.2):
      a_list=np.arange(data.shape[0])
      np.random.shuffle(a_list)
      alist=a_list[0:int(data.shape[0]*(1-proportion))]
      train_data=data[alist]
      d=np.arange(data.shape[0])
      leftover=np.delete(d,alist)
      validate_data=data[leftover]
      return train_data,validate_data

def initialize_weights(m):
     if isinstance(m, nn.Linear):
      #nn.init.kaiming_uniform_(m.weight.data)
      torch.nn.init.xavier_normal_(m.weight.data, gain=1.0)
      torch.nn.init.constant_(m.bias.data, 0)

def create_datafiles(data,test_fraction=0.1):
     a_list=np.arange(data.shape[0])
     np.random.shuffle(a_list)
     alist=a_list[0:int(data.shape[0]*(1-test_fraction))]
     train_data=data[alist]
     d=np.arange(data.shape[0])
     leftover=np.delete(d,alist)
     test_data=data[leftover]
     print('train_data to create file:',train_data,'test data in create file:',test_data)
     np.savetxt('./data/train_data.txt', train_data,  delimiter=' ')
     np.savetxt('./data/test_data.txt', test_data, delimiter=' ')
     return 0

def create_files(data,file_name):  
     name_file= './data/'+file_name+'.txt'
     np.savetxt(name_file, data,  delimiter=' ')
     return 0

class SimDataset(torch.utils.data.Dataset):
  def __init__(self, dataset):
    x_tmp = dataset[:,0:-1]
    y_tmp = dataset[:,-1]
    #print('X_tmp is:',x_tmp,'Y_tmp is:',y_tmp)

    self.x = torch.tensor(x_tmp,
      dtype=torch.float32).to(device)
    self.y = torch.tensor(y_tmp,
      dtype=torch.float32).to(device)

  def __len__(self):
    return len(self.x)  # required

  def __getitem__(self, idx):
    if torch.is_tensor(idx):
      idx = idx.tolist()
    preds = self.x[idx,:]
    pol = self.y[idx]
    sample = \
      { preds, pol }
    return preds,pol


