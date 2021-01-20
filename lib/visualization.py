from skorch.callbacks import TensorBoard
from torchvision.utils import make_grid
import torch.nn.functional as F
from torch import tensor
import torch
import numpy as np

def extract_images(net, dataset_train):
    generator = net.module_.generator
    
    X = dataset_train.X[:36]
    X = tensor(X)
         
    fake, latent_i, latent_o = generator(X)

    if len(X.shape) == 3:
        shape = int(np.round(np.sqrt(fake.shape[2])))
        fake = fake[:, 0, :]
        fake = F.pad(input=fake, pad=(0, shape ** 2 - fake.shape[1]), mode='constant', value=0)
        X = X[:, 0, :]
        X = F.pad(input=X, pad=(0, shape ** 2 - X.shape[1]), mode='constant', value=0)
        fake = fake.reshape((-1, 1, shape, shape))
        X = X.reshape((-1, 1, shape, shape))
    
    X = make_grid(X, nrow=6)
    fake = make_grid(fake, nrow=6)
    
    return X, fake

def extract_mean_score(net, dataset_train):
  
    X = dataset_train.X
    X = tensor(X, device = net.device)
    
    scores = net.module_.forward(X)[0].cpu().detach().numpy()
    mean_score = np.mean(scores)

    return mean_score

def rename_tensorboard_key(key):

    if key in ['loss_gen_fra', 'loss_gen_app', 'loss_gen_lat']:
        prefix = 'Generator_Losses/'
    elif key in ['loss_dis', 'loss_gen']:
        prefix = 'Player_Losses/'
    elif key in ['dur']:
        prefix = 'Duration/'
    elif key in ['train_loss']:
        prefix = 'Overall_Loss/'
        
    key = prefix + key 
    
    return key

class GANomalyBoard(TensorBoard):
    
    def on_epoch_end(self, net, dataset_train, **kwargs):
        
        epoch = net.history[-1, 'epoch']
        
        X, fake = extract_images(net, dataset_train)
        mean_score = extract_mean_score(net, dataset_train)
        
        self.writer.add_image('Generator/X', X, global_step=epoch)
        self.writer.add_image('Generator/fake', fake, global_step=epoch)
        
        self.writer.add_scalar('Scores/mean_anomaly_score', mean_score, global_step=epoch)
        
        super().on_epoch_end(net, **kwargs)  # call super last
        
