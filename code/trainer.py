import torch
from torch.utils.data import DataLoader
import tqdm

class model_trainer:
    def __init__(self, model,model_id: int, optimizer, loss_fn=None, train_data=None,
                 test_data=None, batch_size=None, device=None):
        """Note: Trainer objects don't know about the database."""

        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.train_data = train_data
        self.test_data = test_data
        self.batch_size = batch_size
        self.task_id = None
        self.device = device
        print('len of train_data:',len(train_data),'test data:',len(test_data))
    

    def save_checkpoint(self, checkpoint_path):
        checkpoint = dict(model_state_dict=self.model.state_dict(),
                          optim_state_dict=self.optimizer.state_dict(),
                          batch_size=self.batch_size)
        torch.save(checkpoint, checkpoint_path)

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optim_state_dict'])
        self.batch_size = checkpoint['batch_size']

    def train(self):
        self.model.train()

        #dataloader = tqdm.tqdm(DataLoader(self.train_data, self.batch_size, True),
        #                       desc='Train (task {})'.format(self.task_id),
        #                       ncols=80, leave=True)
        dataloader = DataLoader(self.train_data, self.batch_size, True)
        #print('loaded data')
        correct = 0
        for x, y in dataloader:
            #x=(x+100)/500
            y=y.view(-1,1)
            #print('x is:',x,'y is:',y)

            #print('shape of x:',x.shape,'shape of y:',y.shape)
            x, y = x.to(self.device), y.to(self.device)
            output = self.model(x)
            loss = self.loss_fn(output, y)
            self.optimizer.zero_grad()
            loss.backward() 
            self.optimizer.step()
            correct+= loss.item()
        return correct/len(dataloader)

    def save_model(self,path):
        print("Saving model now...")
        model_name=path
        torch.save(self.model.state_dict(), model_name)

    def save_intermittent_model(self,episode):
        print("Saving intermittent model now...")
        model_name='./models/'+str(episode)+'/nn'+str(self.model_id)+'.pt'
        torch.save(self.model.state_dict(), model_name)

    def eval(self):
        """Evaluate model on the provided validation or test set."""
        self.model.eval()
        with torch.no_grad(): 
         #dataloader = tqdm.tqdm(DataLoader(self.test_data, self.batch_size, True),
         #                      desc='Eval (task {})'.format(self.task_id),
         #                      ncols=80, leave=True)
         dataloader = DataLoader(self.test_data, self.batch_size, True)
         correct = 0
         for x, y in dataloader:
            #x=(x+100)/500
            y=y.view(-1,1)
            x, y = x.to(self.device), y.to(self.device)
            output = self.model(x)
            #print('shape of x:',x.shape,'shape of y:',y.shape)
            loss = self.loss_fn(output, y)
            correct += loss
        return correct/len(dataloader)

    
