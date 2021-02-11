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

from lib.networks import GeneratorNet1d, GeneratorNet2d, DiscriminatorNet1d, DiscriminatorNet2d, GeneratorNetFE, DiscriminatorNetFE, weights_init
from lib.loss import l2_loss

from skorch import NeuralNet
from skorch.utils import to_tensor
from skorch.utils import to_numpy

from skorch.callbacks import EpochTimer
from skorch.callbacks import PrintLog
from skorch.callbacks import PassthroughScoring


class Ganomaly1d(nn.Module):
    """GANomaly Class
    """

    @property
    def name(self): return 'Ganomaly'

    def __init__(self, input_size, n_z, n_channels, n_fm_discriminator, n_fm_generator, n_gpus, fraud_weight=1, appearant_weight=1, latent_weight=1, lambda_weight=0.5):
        super().__init__()

        self.input_size = input_size
        self.n_channels = n_channels
        self.n_z = n_z
        self.n_fm_discriminator = n_fm_discriminator
        self.n_fm_generator = n_fm_generator
        self.n_gpus = n_gpus
        self.fraud_weight = fraud_weight
        self.appearant_weight = appearant_weight
        self.latent_weight = latent_weight
        self.lambda_weight = lambda_weight

        self.discriminator_loss = nn.BCELoss()

        self.fraud_loss = l2_loss
        self.appearant_loss = nn.L1Loss()
        self.latent_loss = l2_loss

        self.discriminator = DiscriminatorNet1d(
            input_size=self.input_size,
            n_z=self.n_z,
            n_channels=self.n_channels,
            n_feature_maps=self.n_fm_discriminator,
            n_gpus=self.n_gpus
        )
        self.discriminator.apply(weights_init)
        self.generator = GeneratorNet1d(
            input_size=self.input_size,
            n_z=self.n_z,
            n_channels=self.n_channels,
            n_feature_maps=self.n_fm_generator,
            n_gpus=self.n_gpus
        )
        self.generator.apply(weights_init)

    def forward(self, X, y=None):

        fake, latent_real, latent_fake = self.generator(X)

        appearant_differences = (X - fake).view(fake.size()[0], -1)
        latent_differences = (
            latent_real - latent_fake).view(latent_real.size()[0], -1)

        appearant_loss = torch.mean(appearant_differences.abs(), dim=1)
        latent_loss = torch.mean(torch.pow(latent_differences, 2), dim=1)

        error = self.lambda_weight * appearant_loss + \
            (1 - self.lambda_weight) * latent_loss

        return error.reshape(error.size(0)), X, fake, latent_real, latent_fake


class Ganomaly2d(nn.Module):
    """GANomaly Class
    """

    @property
    def name(self): return 'Ganomaly'

    def __init__(self, input_size, n_z, n_channels, n_fm_discriminator, n_fm_generator, n_gpus, fraud_weight=1, appearant_weight=1, latent_weight=1, lambda_weight=0.5):
        super().__init__()

        self.input_size = input_size
        self.n_channels = n_channels
        self.n_z = n_z
        self.n_fm_discriminator = n_fm_discriminator
        self.n_fm_generator = n_fm_generator
        self.n_gpus = n_gpus
        self.fraud_weight = fraud_weight
        self.appearant_weight = appearant_weight
        self.latent_weight = latent_weight
        self.lambda_weight = lambda_weight

        self.discriminator_loss = nn.BCELoss()

        self.fraud_loss = l2_loss
        self.appearant_loss = nn.L1Loss()
        self.latent_loss = l2_loss

        self.discriminator = DiscriminatorNet2d(
            input_size=self.input_size,
            n_z=self.n_z,
            n_channels=self.n_channels,
            n_feature_maps=self.n_fm_discriminator,
            n_gpus=self.n_gpus
        )
        self.discriminator.apply(weights_init)
        self.generator = GeneratorNet2d(
            input_size=self.input_size,
            n_z=self.n_z,
            n_channels=self.n_channels,
            n_feature_maps=self.n_fm_generator,
            n_gpus=self.n_gpus
        )
        self.generator.apply(weights_init)

    def forward(self, X, y=None):

        fake, latent_real, latent_fake = self.generator(X)

        appearant_differences = (X - fake).view(fake.size()[0], -1)
        latent_differences = (
            latent_real - latent_fake).view(latent_real.size()[0], -1)

        appearant_loss = torch.mean(appearant_differences.abs(), dim=1)
        latent_loss = torch.mean(torch.pow(latent_differences, 2), dim=1)

        error = self.lambda_weight * appearant_loss + \
            (1 - self.lambda_weight) * latent_loss

        return error.reshape(error.size(0)), X, fake, latent_real, latent_fake


class GanomalyFE(nn.Module):
    """GANomaly Class
    """

    @property
    def name(self): return 'GanomalyFE'

    def __init__(self, input_size, n_gpus, fraud_weight=1, appearant_weight=1, latent_weight=1, lambda_weight=0.5):
        super().__init__()

        self.input_size = input_size
        self.n_gpus = n_gpus
        self.fraud_weight = fraud_weight
        self.appearant_weight = appearant_weight
        self.latent_weight = latent_weight
        self.lambda_weight = lambda_weight

        self.discriminator_loss = nn.BCELoss()

        self.fraud_loss = l2_loss
        self.appearant_loss = nn.L1Loss()
        self.latent_loss = l2_loss

        self.discriminator = DiscriminatorNetFE(
            input_size=self.input_size,
            n_gpus=self.n_gpus
        )
        self.discriminator.apply(weights_init)
        self.generator = GeneratorNetFE(
            input_size=self.input_size,
            n_gpus=self.n_gpus
        )
        self.generator.apply(weights_init)

    def forward(self, X, y=None):

        fake, latent_real, latent_fake = self.generator(X)

        appearant_differences = (X - fake).view(fake.size()[0], -1)
        latent_differences = (
            latent_real - latent_fake).view(latent_real.size()[0], -1)

        appearant_loss = torch.mean(appearant_differences.abs(), dim=1)
        latent_loss = torch.mean(torch.pow(latent_differences, 2), dim=1)

        error = self.lambda_weight * appearant_loss + \
            (1 - self.lambda_weight) * latent_loss

        return error.reshape(error.size(0)), X, fake, latent_real, latent_fake


class GanomalyNet(NeuralNet):
    def __init__(self, *args, generator_optimizer, discriminator_optimizer, **kwargs):
        self.generator_optimizer = generator_optimizer
        self.discriminator_optimizer = discriminator_optimizer

        super().__init__(*args, **kwargs)

    def initialize_optimizer(self, *_, **__):
        args, kwargs = self.get_params_for_optimizer(
            'generator_optimizer', self.module_.generator.named_parameters())
        self.generator_optimizer_ = self.generator_optimizer(*args, **kwargs)

        args, kwargs = self.get_params_for_optimizer(
            'discriminator_optimizer', self.module_.discriminator.named_parameters())
        self.discriminator_optimizer_ = self.discriminator_optimizer(
            *args, **kwargs)

        return self

    def validation_step(self, Xi, yi, **fit_params):
        raise NotImplementedError

    @property
    def _default_callbacks(self):
        return [
            ('epoch_timer', EpochTimer()),
            ('train_loss', PassthroughScoring(
                name='train_loss',
                on_train=True,
            )),
            ('discriminator_loss', PassthroughScoring(
                name='discriminator_loss',
                on_train=True
            )),
            ('generator_loss', PassthroughScoring(
                name='generator_loss',
                on_train=True
            )),
            ('fraud_loss', PassthroughScoring(
                name='fraud_loss',
                on_train=True
            )),
            ('appearant_loss', PassthroughScoring(
                name='appearant_loss',
                on_train=True
            )),
            ('latent_loss', PassthroughScoring(
                name='latent_loss',
                on_train=True
            )),
            ('print_log', PrintLog()),
        ]

    def train_step(self, Xi, yi=None, **fit_params):

        # turn input data into tensor
        Xi = to_tensor(Xi, device=self.device)

        # create local variables for the generator and discriminator
        discriminator = self.module_.discriminator
        generator = self.module_.generator

        # forward the generator and obtain it's data
        fake, latent_Xi, latent_fake = generator(Xi)

        # evaluate real and fake data with the discriminator
        prediction_real, features_real = discriminator(Xi)
        prediction_fake, features_fake = discriminator(fake.detach())

        # create a tensor of ones
        # this is used for the discriminator
        labels_real = ones_like(
            prediction_real, dtype=torch.float32, device=self.device).fill_(1.0)

        # calculate generator loss
        fraud_loss = self.module_.fraud_loss(features_real, features_fake)
        appearant_loss = self.module_.appearant_loss(Xi, fake)
        latent_loss = self.module_.latent_loss(latent_Xi, latent_fake)
        generator_loss = self.module_.fraud_weight * fraud_loss + \
            self.module_.appearant_weight * appearant_loss + \
            self.module_.latent_weight * latent_loss

        discriminator_loss = self.module_.discriminator_loss(
            prediction_real, labels_real)
        # calculate discriminator loss

        # set gradient of generator optimizer to zero and update generator weights
        self.generator_optimizer_.zero_grad()
        generator_loss.backward(retain_graph=True)
        self.generator_optimizer_.step()

        # set gradient of discriminator optimizer to zero and update discriminator weights
        self.discriminator_optimizer_.zero_grad()
        discriminator_loss.backward()
        self.discriminator_optimizer_.step()

        # record the different loss values in the models training history
        self.history.record_batch('generator_loss', generator_loss.item())
        self.history.record_batch('fraud_loss', fraud_loss.item())
        self.history.record_batch('appearant_loss', appearant_loss.item())
        self.history.record_batch('latent_loss', latent_loss.item())

        self.history.record_batch(
            'discriminator_loss', discriminator_loss.item())

        # return loss for skorch
        return {'loss': generator_loss + discriminator_loss}

    def score(self, X, y=None):
        X = to_tensor(X, device=self.device)

        # create local variables for the generator and discriminator
        discriminator = self.module_.discriminator
        generator = self.module_.generator

        # forward the generator and obtain it's data
        fake, latent_X, latent_fake = generator(X)

        # evaluate real and fake data with the discriminator
        prediction_real, features_real = discriminator(X)
        prediction_fake, features_fake = discriminator(fake.detach())

        # create a tensor of ones
        # this is used for the discriminator
        labels_real = ones_like(
            prediction_real, dtype=torch.float32, device=self.device).fill_(1.0)

        # calculate generator loss
        fraud_loss = self.module_.fraud_loss(prediction_real, labels_real)
        appearant_loss = self.module_.appearant_loss(X, fake)
        latent_loss = self.module_.latent_loss(latent_X, latent_fake)
        generator_loss = self.module_.fraud_weight * fraud_loss + \
            self.module_.appearant_weight * appearant_loss + \
            self.module_.latent_weight * latent_loss

        generator_loss = generator_loss / \
            (self.module_.fraud_weight +
             self.module_.appearant_weight + self.module_.latent_weight)

        # calculate discriminator loss
        discriminator_loss = self.module_.discriminator_loss(
            features_real, features_fake)

        # calculate train loss
        train_loss = generator_loss + discriminator_loss

        # make scores negative
        # GridSearchCV then takes the lowest value
        generator_loss = -1 * generator_loss.item()
        train_loss = -1 * train_loss.item()

        # return scores as dictionary
        return {'generator_loss': generator_loss, 'train_loss': train_loss}

    def predict_proba(self, X):

        nonlin = self._get_predict_nonlinearity()
        y_probas = []
        for yp in self.forward_iter(X, training=False):
            yp = yp if isinstance(yp, tuple) else yp
            yp = nonlin(yp)
            y_probas.append(to_numpy(yp))
        stacked = list(zip(*y_probas))
        y_proba = [np.concatenate(array) for array in stacked]

        return y_proba

    def predict(self, X):
        return self.predict_proba(X)[0]
