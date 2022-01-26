import os
import pathlib
import numpy as np
import torch
import torch.nn as nn
from student_model import SNet
from teacher_model import TNet
from trainer import model_trainer
from utils import *
import pathlib

ul=utilities()
class Worker():
    def __init__(self, batch_size, max_epoch, train_data,test_data,device,neuraln,net_type,path):
        self.max_epoch = max_epoch
        self.epoch=0
        self.batch_size = batch_size
        self.device = device
        model = neuraln.to(device)
        self.path=path 
        self.flag_first=0

        optimizer = ul.get_optimizer(model)
        loss_func = ul.get_lossfunc(net_type)
        self.trainer = model_trainer(model=model,
                               optimizer=optimizer,
                               loss_fn=loss_func,                    # for regression:nn.L1Loss(), for logistic: Binary cross indicator
                               train_data=train_data,
                               test_data=test_data,
                               batch_size=self.batch_size,
                               device=self.device)


    def run(self):
        # Train
        loss_train=[]; loss_validate=[];diff_loss=[];min_diff_loss=0; threshold_epoch=300
        while True:
            #print('self epoch is:',self.epoch)
           
            if self.epoch > self.max_epoch:
                break    
            #checkpoint_path = "checkpoints/task-%03d.pth" % task['id']
            try:
                loss=self.trainer.train()
                #self.trainer.save_checkpoint(checkpoint_path)
            except KeyboardInterrupt:
                break
            loss_train.append(loss)
            self.epoch+=1
            
            validation_error= self.trainer.eval().cpu()
            loss_validate.append(validation_error)
            
            if self.epoch <= threshold_epoch:
               whichmodel=self.epoch  
               self.save_model(self.path)

            if self.epoch> threshold_epoch:
             diff_loss=np.absolute(loss-validation_error)
             if self.flag_first==0: 
                self.save_model(self.path)
                whichmodel=self.epoch 
                self.flag_first=1
                last_diff_loss=diff_loss

             elif self.flag_first==1:
                if last_diff_loss>diff_loss:
                  self.save_model(self.path); whichmodel=self.epoch ;
                  last_diff_loss=diff_loss
            
               
               
        print('-> Saved model is:', whichmodel,' th epoch')   


        #print('Training loss :',loss,'Validation loss is:',validation_error)
        #print('loss of training:',loss_train)
        fig=plt.figure(figsize=(9,6))
        plt.plot(loss_train,label='train')
        plt.plot(loss_validate,label='validate')
        plt.legend()
        #plt.show(block=False)
        #plt.pause(10)
        #plt.close(fig)
        plt.show()
        return loss,validation_error
        #validate
     
    def save_model(self,path):
        self.trainer.save_model(path)



def save_intermittent_model(model,episode,population):
        print("Saving intermittent model now...")
        model_name='./models/'+str(episode)+'/nn'+str(population)+'.pt'
        torch.save(model.state_dict(), model_name)

def train_net(training_data,batch_size,max_epoch,device,input_size,output_size,net_type):
       ul.set_lr_auto()
       accept=0;   
       train_data,validation_data=data_split(training_data,proportion=0.1)
       train_data = SimDataset(train_data)
       validation_data= SimDataset(validation_data)
          
       if net_type=="T":
         path='./models/teacher/'+'nnt'+'.pt'
         neuralNet= TNet(input_size,output_size)
         try: 
           neuralNet.load_state_dict(torch.load(path))       
           print("Loaded earlier trained model successfully")
         except: 
          neuralNet= neuralNet.apply(initialize_weights)
          print('Randomly initialising weights')

       elif net_type=="S":
         path='./models/student/'+'nns'+'.pt'
         neuralNet= SNet(input_size,output_size)
         try: 
           neuralNet.load_state_dict(torch.load(path))       
           print("Loaded earlier trained model successfully")
         except: 
          neuralNet= neuralNet.apply(initialize_weights)
          print('Randomly initialising weights')

         
       w=Worker(batch_size, max_epoch, train_data,validation_data, device,neuralNet,net_type,path)
       training_loss, validation_loss=w.run()

         


    
