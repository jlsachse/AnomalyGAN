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

    def __init__(self, isize, nz, nc, ndf, ngf, ngpu, n_extra_layers=0, w_fra = 1, w_app = 1, w_lat = 1):
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


    def forward(self, X, y=None):
        # general forward method just returns fake images
        return self.generator(X)

class GanomalyNet(NeuralNet):
    def __init__(self, *args, optimizer_gen, optimizer_dis, **kwargs):
        self.optimizer_gen = optimizer_gen
        self.optimizer_dis = optimizer_dis

        self.l_fra = nn.BCELoss()
        self.l_app = nn.L1Loss()
        self.l_lat = l2_loss
        self.l_dis = l2_loss

        self.w_fra = 1
        self.w_app = 1
        self.w_lat = 1

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
        
        print(Xi)
        
        fake, latent_i, latent_o = generator(Xi)

        pred_real, feat_real = discriminator(Xi)
        pred_fake, feat_fake = discriminator(fake.detach())

        label_real = ones_like(pred_real, dtype=torch.float32, device=self.device).fill_(1.0)

        # update discriminator
        discriminator.zero_grad()
        loss_dis = self.l_dis(feat_real, feat_fake)
        loss_dis.backward(retain_graph=True)  #solve 
        self.optimizer_dis_.step()

        generator.zero_grad()
        loss_gen_fra = self.l_fra(pred_real, label_real)
        loss_gen_app = self.l_app(Xi, fake)
        loss_gen_lat = self.l_lat(latent_i, latent_o)
        loss_gen = loss_gen_fra * self.w_fra + \
                   loss_gen_app * self.w_app + \
                   loss_gen_lat * self.w_lat
        loss_gen.backward(retain_graph=True)
        self.optimizer_gen_.step()


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
