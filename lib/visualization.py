from skorch.callbacks import TensorBoard
from torchvision.utils import make_grid
from torch import tensor
import numpy as np

def extract_images(net, dataset_train):
    generator = net.module_.generator
    
    X = dataset_train.X[:36]
    X = tensor(X)
         
    fake, latent_i, latent_o = generator(X)
    
    X = make_grid(X, nrow=6)
    fake = make_grid(fake, nrow=6)
    
    return X, fake

def extract_mean_score(net, dataset_train):
  
    X = dataset_train.X
    X = tensor(X)
    
    scores = net.module_.forward(X).detach().numpy()
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
        
