import numpy as np
import pandas as pd
from numpy import pi
from pyDOE import *   


#10<=x3,x4<=200;  1<= x1,x2 <=99(integer)
def pressure_vessel_design(x):
    x1=x[:,0];x2=x[:,1];x3=x[:,2];x4=x[:,3]
    z1=0.0625*x1; z2=0.0625*x2
    f= 1.7781*z2*np.power(x3,3)+0.6224*z1*x3*x4+3.1661*np.square(z1)*x4+19.84*np.square(z1)*x3
    return f

# -10<=x<=0 ; -6.5<=y<=0 , fmin@(-3.13,-1.58)= -106.7 
def mishra(x,y):
    f=np.sin(y)* np.exp(np.square(1-np.cos(x)))+ np.cos(x)* np.exp(np.square(1-np.sin(y)))+ np.square(x-y)
    g=25-np.square(x+5)-np.square(y+5)
    return f,g


def draw_samples(mask, num_samples,ranges):
    num_real_var= mask.count('real')
    num_int_var= mask.count('int')
    print('num of real variables:',num_real_var,'num of integer variables:',num_int_var)
    if num_real_var>0: 
      index_real=np.array([[2*i,2*i+1] for i, n in enumerate(mask) if n == 'real' ],dtype=int).flatten().tolist()
      range_real= ranges[index_real]
      real_samples=lhc_samples(num_samples,num_real_var,range_real)
    if num_int_var >0: 
      index_int=np.array([[2*i,2*i+1] for i, n in enumerate(mask) if n == 'int' ],dtype=int).flatten().tolist()
      range_int = ranges[index_int] 
      int_samples = random_integer_sampling(num_samples,num_int_var,range_int)

    if (num_int_var >0) & (num_real_var>0):
        samples= np.concatenate((real_samples,int_samples),axis= 1)
    elif num_int_var >0:
        samples= int_samples
    elif num_real_var>0:
        samples= real_samples
    else:  
        samples= None
    print('shape of samples:',samples.shape)
    return samples
    

#latin hypercube sampling
def lhc_samples(n, dim,ranges): 
    samples=lhs(dim, samples=n, criterion='center')
    for i in range(dim): 
       samples[:,i]=samples[:,i]*(ranges[(2*i+1)]-ranges[2*i]) + ranges[2*i]
    return samples

# random integer sampling
def random_integer_sampling(n,dim,ranges):
    for i in range(dim): 
       one_dim_samples=np.random.random_integers(ranges[2*i],ranges[(2*i+1)], size=(n,1))
       if i==0:
         samples= one_dim_samples
       else: 
         samples= np.concatenate((samples,one_dim_samples),axis=1) 
    return samples



