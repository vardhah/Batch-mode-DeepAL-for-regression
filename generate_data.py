# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 16:11:45 2021

@author: HPP
"""

import numpy as np
from sim_engine import * 


def random_samples(n):
    a=np.random.rand(n,2)*99+1
    #b=np.random.rand(1,100, size=n).reshape(-1,1)
    c=np.random.rand(n,2)*190+10
    rand_points=np.concatenate((a,c),axis=1)
    return rand_points


if __name__ == "__main__":
 data= random_samples(30000000)
 name_file= './data/'+'random_data'+'.txt'
 np.savetxt(name_file, data,  delimiter=' ')
 print('shape of data:',data.shape)








