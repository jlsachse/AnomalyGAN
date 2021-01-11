"""GANomaly
"""
# pylint: disable=C0301,E1101,W0622,C0103,R0902,R0915

##
from collections import OrderedDict
import os
import time
import numpy as np
from tqdm import tqdm

from torch.autograd import Variable
from torch import ones_like
import torch.optim as optim
import torch.nn as nn
import torch.utils.data
import torchvision.utils as vutils

from lib.networks import NetG, NetD, weights_init
from lib.loss import l2_loss

from skorch import NeuralNet
from skorch.utils import to_tensor
##
class Ganomaly(nn.Module):
    """GANomaly Class
    """

    @property
    def name(self): return 'Ganomaly'

    def __init__(self, isize, nz, nc, ndf, ngf, ngpu, n_extra_layers=0, w_fra = 1, w_app = 1, w_lat = 1, w_lambda = 0.5):
        super().__init__()
        
        self.isize = isize
        self.nc = nc
        self.nz = nz
        self.ndf = ndf
        self.ngf = ngf
        self.ngpu = ngpu
        self.w_fra = w_fra
        self.w_app = w_app
        self.w_lat = w_lat
        self.n_extra_layers = n_extra_layers
        self.w_lambda = w_lambda

        self.l_fra = nn.BCELoss()
        self.l_app = nn.L1Loss()
        self.l_lat = l2_loss
        self.l_dis = nn.L1Loss()

        self.discriminator = NetD(
            isize=self.isize,
            nz=self.nz,
            nc=self.nc,
            ndf=self.ndf,
            ngpu=self.ngpu,
            n_extra_layers=self.n_extra_layers
        )
        self.discriminator.apply(weights_init)
        self.generator = NetG(
            isize=self.isize,
            nz=self.nz,
            nc=self.nc,
            ngf=self.ngf,
            ngpu=self.ngpu,
            n_extra_layers=self.n_extra_layers
        )
        self.generator.apply(weights_init)


    def forward(self, X, y=None): #lambda could also be placed here
        # general forward method just returns fake images
        # repair loss

        fake, latent_i, latent_o = self.generator(X)

        si = X.size()
        sz = latent_i.size()
        app = (X - fake).view(si[0], si[1] * si[2] * si[3])
        lat = (latent_i - latent_o).view(sz[0], sz[1] * sz[2] * sz[3])
        app = torch.mean(torch.abs(app), dim=1)
        lat = torch.mean(torch.pow(lat, 2), dim=1)
        error = self.w_lambda * app + (1 - self.w_lambda) * lat


        return error.reshape(error.size(0))

class GanomalyNet(NeuralNet):
    def __init__(self, *args, optimizer_gen, optimizer_dis, **kwargs):
        self.optimizer_gen = optimizer_gen
        self.optimizer_dis = optimizer_dis

        super().__init__(*args, **kwargs)

    def initialize_optimizer(self, *_, **__):
        args, kwargs = self.get_params_for_optimizer(
            'optimizer_gen', self.module_.generator.named_parameters())
        self.optimizer_gen_ = self.optimizer_gen(*args, **kwargs)

        args, kwargs = self.get_params_for_optimizer(
            'optimizer_dis', self.module_.discriminator.named_parameters())
        self.optimizer_dis_ = self.optimizer_dis(*args, **kwargs)

        return self
    
    def validation_step(self, Xi, yi, **fit_params):
        raise NotImplementedError
    
    def train_step(self, Xi, yi=None, **fit_params):
        Xi = to_tensor(Xi, device=self.device)
        discriminator = self.module_.discriminator
        generator = self.module_.generator
        
        fake, latent_i, latent_o = generator(Xi)

        self.module_.fake = fake
        self.module_.latent_i = fake
        self.module_.latent_o = fake

        pred_real, feat_real = discriminator(Xi)
        pred_fake, feat_fake = discriminator(fake.detach())

        label_real = ones_like(pred_real, dtype=torch.float32, device=self.device).fill_(1.0)

        # update discriminator
        
        loss_dis = self.module_.l_dis(feat_real, feat_fake)
        
        
        loss_gen_fra = self.module_.l_fra(pred_real, label_real)
        loss_gen_app = self.module_.l_app(Xi, fake)
        loss_gen_lat = self.module_.l_lat(latent_i, latent_o)
        loss_gen = loss_gen_fra * self.module_.w_fra + \
                   loss_gen_app * self.module_.w_app + \
                   loss_gen_lat * self.module_.w_lat

        loss_dis.backward(retain_graph=True)  #solve 
        loss_gen.backward(retain_graph=True)

        self.optimizer_dis_.step()
        self.optimizer_gen_.step()

        self.optimizer_dis_.zero_grad()
        self.optimizer_gen_.zero_grad()

        if loss_dis.item() < 1e-5:
            discriminator.apply(weights_init)
            print('Reloading discriminator')

        
        self.history.record_batch('loss_dis', loss_dis.item())
        self.history.record_batch('loss_gen', loss_gen.item())

        self.history.record_batch('loss_gen_fra', loss_gen_fra.item())
        self.history.record_batch('loss_gen_app', loss_gen_app.item())
        self.history.record_batch('loss_gen_lat', loss_gen_lat.item())
        
        return {
            'y_pred': fake,
            'loss': loss_dis + loss_gen,
        }

    def score(self, X):
        X = to_tensor(X, device=self.device)

        discriminator = self.module_.discriminator
        generator = self.module_.generator

        fake, latent_i, latent_o = generator(X)

        pred_real, feat_real = discriminator(X)
        pred_fake, feat_fake = discriminator(fake.detach())

        label_real = ones_like(pred_real, dtype=torch.float32, device=self.device).fill_(1.0)

        # update discriminator
        
        loss_dis = self.module_.l_dis(feat_real, feat_fake)
        
        
        loss_gen_fra = self.module_.l_fra(pred_real, label_real)
        loss_gen_app = self.module_.l_app(X, fake)
        loss_gen_lat = self.module_.l_lat(latent_i, latent_o)
        loss_gen = loss_gen_fra * self.module_.w_fra + \
                   loss_gen_app * self.module_.w_app + \
                   loss_gen_lat * self.module_.w_lat

        return loss_gen
