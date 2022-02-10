import numpy as np
import pandas as pd
from numpy import pi
from pyDOE import *   
import matlab.engine
import logging
logging.basicConfig(filename='./data/pool_data_generation.log',filemode='w')

def opensim(x):
    eng = matlab.engine.start_matlab()
    sim_out=[]
    eng.cd(r'/home/hv/harsh/openProp/OpenProp_v3.3.4/SourceCode')
    print('collecting data', 'shape is:',x.shape[0]) 
    for i in range(x.shape[0]):
       sim_out.append(eng.OpenProp_eval(matlab.double(x[i].tolist())))
       if i%5000==2:
          logging.error('%s num of sample evaluated is',str(i))
          sim_data= np.concatenate((x[0:i+1,:],np.array(sim_out).reshape(-1,1)),axis=1)
          named='./data/pool_data_extended'+'.csv'
          np.savetxt(named, sim_data, delimiter=",")
    eng.quit()
    return np.array(sim_out)    
       

  


def random_sampling(n):
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
  t_s=np.random.randint(tl,th, size=n).reshape(-1,1)
  rpm_s=np.random.randint(rpml,rpmh, size=n).reshape(-1,1)
  s_s=np.random.rand(n,1)*(sh-sl)+sl
  d_s=np.random.rand(n,1)*(dh-dl)+dl
  c1_e=np.random.rand(n,1)*(ch-c1)+c1
  c2_s=np.random.rand(n,1)*(ch-c2)+c2
  c3_s=np.random.rand(n,1)*(ch-c3)+c3
  c4_s=np.random.rand(n,1)*(ch-c4)+c4
  c5_s=np.random.rand(n,1)*(ch-c5)+c5
  c6_s=np.random.rand(n,1)*(ch-c6)+c6
  c7_s=np.random.rand(n,1)*(ch-c7)+c7
  c8_s=np.random.rand(n,1)*(ch-c8)+c8
  c9_e=np.ones((n,1))*c9 
  c10_e=np.ones((n,1))*c10 
  rand_points=np.concatenate((t_s,s_s,rpm_s,d_s,c1_e,c2_s,c3_s,c4_s,c5_s,c6_s,c7_s,c8_s,c9_e,c10_e),axis=1)
  print('shape of samples:',rand_points.shape)
  return rand_points

