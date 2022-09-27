import os
import glob
import pathlib
import numpy as np
#import matplotlib.pyplot as plt

#import seaborn as sns
import matplotlib.pylab as plt
from sim_engine import mishra




d1l=-10;d1h=0; d2l=-6.5; d2h=0


n=1600
xx= np.arange(-10,0,0.1)
yy= np.arange(-6.5,0,0.1)
xv, yv = np.meshgrid(xx, yy, sparse=False)

sim_data_ann = np.loadtxt('./data2_mishra_100/sim_data.txt', delimiter=" ",skiprows=0, dtype=np.float32)
fig=plt.figure(figsize=(9,6))

sim_data_ann= sim_data_ann[0:n,:]
print('shape of sim data:',sim_data_ann.shape)

#plt.legend()
#plt.savefig(figname)


z,k= mishra(xv,yv)
h = plt.contourf(xx,yy,z)

s=np.random.rand(n,2)
s[:,0]=(s[:,0]*(d1h-d1l))+d1l
s[:,1]=(s[:,1]*(d2h-d2l))+d2l






plt.scatter(sim_data_ann[:,0],sim_data_ann[:,1],color='r',label='random samples')
title='bird on ann'+str(n)
plt.title(title) 
figname='/fig/'+str(n)+'.png'

plt.show()



"""


dev_data_rand = np.loadtxt('./data/dev_original_random.txt', delimiter=" ",skiprows=0, dtype=np.float32)
dev_data_ann = np.loadtxt('./data/mean_data.txt', delimiter=" ",skiprows=0, dtype=np.float32)

#print('sim_data shape:',sim_data_load.shape)
#sim_data_load_t=sim_data_load[:70,:]
#sim_data_load=sim_data_load[:60,:]

fig=plt.figure(figsize=(9,6))
plt.plot(dev_data_rand,label='random samples')
plt.plot(dev_data_ann,label='ann simulated points')


plt.legend()
#plt.savefig('./fig/60t070.png')
plt.show()



grid_data_load = np.loadtxt('./data/grid_mesh.txt', delimiter=" ",skiprows=0, dtype=np.float32)

g_truth,c= mishra(grid_data_load[:,0],grid_data_load[:,1])
g_t=np.sum(np.absolute(g_truth))


print('g_truth is:',g_t,'total_dev_ann',dev_data_ann,'total_dev_rand:',dev_data_rand)
print('g_truth is:',g_t,'total_dev_ann',(dev_data_ann/g_t)*100,'total_dev_rand:',(dev_data_rand/g_t)*100)

print('size of dev_rand', dev_data_rand.shape,'size of dev_ann',dev_data_ann.shape)

indices_to_select=(np.arange(10)+1)*5-1
print('indices to select is:',indices_to_select)

total_rand=(dev_data_rand[indices_to_select]/g_t)*100
total_ann=(dev_data_ann[indices_to_select]/g_t)*100

print('size of final ann',len(total_ann),'size of rand:',len(total_rand))
print('total_ann',total_ann,'total rand:',total_rand)

index = (np.arange(10)+1)*50
bar_width = 10

fig, ax = plt.subplots()
summer = ax.bar(index,total_ann, bar_width,
                label="ANN assisted sampling")

winter = ax.bar(index+bar_width,total_rand ,bar_width, label="Random sampling")

ax.set_xlabel('Number of samples')
ax.set_ylabel('%age error')                       # absolute(prediction-g_truth)/g_truth
ax.set_title('Number of samples to learn the manifold')
ax.set_xticks(index )
#ax.set_xticklabels(["ASB", "Violence", "Theft", "Public Order", "Drugs"])
ax.legend()
plt.grid(ls='--')
plt.show()



sim_data_rand = np.loadtxt('./data/sim_random_data_till_2500.txt', delimiter=" ",skiprows=0, dtype=np.float32)
sim_data_ann = np.loadtxt('./data/sim_data.txt', delimiter=" ",skiprows=0, dtype=np.float32)
print('sim data rand shape:',sim_data_rand.shape,'sim data ann:',sim_data_ann.shape)
curated_sim_rand=sim_data_rand
curated_sim_ann= sim_data_ann[0:500,0:2]
print('sim data rand shape:',curated_sim_rand.shape,'sim data ann:',curated_sim_ann.shape)
fig, ax = plt.subplots()
ax.scatter(curated_sim_rand[:,0],curated_sim_rand[:,1],c='k',marker='x')
ax.set_xlabel('x')
ax.set_ylabel('y')                       # absolute(prediction-g_truth)/g_truth
ax.set_title('Samples selected for evaluation(uniformly random)')
plt.show()

fig, ax = plt.subplots()
ax.scatter(curated_sim_ann[:,0],curated_sim_ann[:,1],c='k',marker='x')
ax.set_xlabel('x')
ax.set_ylabel('y')                       # absolute(prediction-g_truth)/g_truth
ax.set_title('Samples selected for evaluation using ANN')
plt.show()

"""





import matplotlib.pyplot as plt
import numpy as np
""""
# generate 2 2d grids for the x & y bounds
y, x = np.meshgrid(np.linspace(-3, 3, 100), np.linspace(-3, 3, 100))

z = (1 - x / 2. + x ** 5 + y ** 3) * np.exp(-x ** 2 - y ** 2)
# x and y are bounds, so z should be the value *inside* those bounds.
# Therefore, remove the last value from the z array.
print('z is:',z)
z = z[:-1, :-1]

print('z is now:',z)
z_min, z_max = -np.abs(z).max(), np.abs(z).max()

fig, ax = plt.subplots()

c = ax.pcolormesh(x, y, z, cmap='RdBu', vmin=z_min, vmax=z_max)
ax.set_title('pcolormesh')
# set the limits of the plot to the limits of the data
ax.axis([x.min(), x.max(), y.min(), y.max()])
fig.colorbar(c, ax=ax)

plt.show()

xx= np.arange(-20,20,0.1)
yy= np.arange(-20,20,0.1)
print(xx.shape,'yy shape:',yy.shape)
xv, yv = np.meshgrid(xx, yy, sparse=False)
print('xv shape:',xv.shape,'yv shape:',yv.shape)
data=np.vstack([xv.ravel(), yv.ravel()]).T
print('Shape of data is:',data.shape)
grid_mesh_data= np.loadtxt('./data/grid_mesh.txt', delimiter=" ",skiprows=0, dtype=np.float32)

#uncertainity_data= np.loadtxt('./data/uncertainity_data.txt', delimiter=" ",skiprows=0, dtype=np.float32)
#z_init=uncertainity_data[:,0].reshape(65,100)
f_ground,g=mishra(grid_mesh_data[:,0],grid_mesh_data[:,1])

f_g=f_ground
print('grid mesh shape is:',grid_mesh_data.shape,'f_g shape:',f_g.shape)
mean_data= np.loadtxt('./data/mean_data.txt', delimiter=" ",skiprows=0, dtype=np.float32)
z=f_g-mean_data[:,0]
print('z shape is:',z.shape,'f_g shape:',f_g.shape,'mean_data shapeis:',mean_data[:,0].shape)

f_ground,g=mishra(data[:,0],data[:,1])
from mpl_toolkits.mplot3d import Axes3D
Axes3D = Axes3D  # pycharm auto import
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data[:,0],data[:,1],f_ground,label='Bird function')
plt.legend()
#plt.savefig('./fig/60t070.png')
plt.show()



z=(f_g-mean_data[:,5]).reshape(65,100)
z_min,z_max=-np.abs(z).max(), np.abs(z).max()
x=xv
y=yv
print('x shape:',x.shape,'y.shape:',y.shape)
fig, ax = plt.subplots()

c = ax.pcolormesh(x, y, z, cmap='RdBu', vmin=z_min, vmax=z_max)
ax.set_title('deviation_from_ground map after training 20 samples')
# set the limits of the plot to the limits of the data
ax.axis([x.min(), x.max(), y.min(), y.max()])
fig.colorbar(c, ax=ax)

#plt.savefig('./fig/60_M.png')
plt.show()
"""
