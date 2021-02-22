from numpy import concatenate
from torch import ones_like, zeros_like
import torch.nn as nn
import torch

from skorch import NeuralNet
from skorch.utils import to_tensor, to_numpy
from skorch.callbacks import EpochTimer, PrintLog, PassthroughScoring

from lib.loss import l2_loss
import lib.networks as nets

# Ganomaly1d, Ganomaly2d and GanomalyFE
# are constructed as follows:


class Ganomaly1d(nn.Module):

    def __init__(self, input_size, n_z, n_channels, n_fm_discriminator, n_fm_generator, n_gpus, adversarial_weight=1, contextual_weight=1, encoder_weight=1, lambda_weight=0.5):
        super().__init__()

        self.input_size = input_size
        self.n_channels = n_channels
        self.n_z = n_z
        self.n_fm_discriminator = n_fm_discriminator
        self.n_fm_generator = n_fm_generator
        self.n_gpus = n_gpus
        self.adversarial_weight = adversarial_weight
        self.contextual_weight = contextual_weight
        self.encoder_weight = encoder_weight
        self.lambda_weight = lambda_weight

        # loss functions for this model
        self.discriminator_loss = nn.BCELoss()

        self.adversarial_loss = l2_loss
        self.contextual_loss = nn.L1Loss()
        self.encoder_loss = l2_loss

        # initialize discriminator and generator
        # and weights
        self.discriminator = nets.DiscriminatorNet1d(
            input_size=self.input_size,
            n_z=self.n_z,
            n_channels=self.n_channels,
            n_feature_maps=self.n_fm_discriminator,
            n_gpus=self.n_gpus
        )
        self.discriminator.apply(nets.weights_init)
        self.generator = nets.GeneratorNet1d(
            input_size=self.input_size,
            n_z=self.n_z,
            n_channels=self.n_channels,
            n_feature_maps=self.n_fm_generator,
            n_gpus=self.n_gpus
        )
        self.generator.apply(nets.weights_init)

    # forward returns the anomaly score
    def forward(self, X, y=None):

        fake, latent_real, latent_fake = self.generator(X)

        appearant_differences = (X - fake).view(fake.size()[0], -1)
        latent_differences = (
            latent_real - latent_fake).view(latent_real.size()[0], -1)

        contextual_loss = torch.mean(appearant_differences.abs(), dim=1)
        encoder_loss = torch.mean(torch.pow(latent_differences, 2), dim=1)

        error = self.lambda_weight * contextual_loss + \
            (1 - self.lambda_weight) * encoder_loss

        return error.reshape(error.size(0)), X, fake, latent_real, latent_fake


# The Ganomaly 2d class
# is the same as the 1d constructor
# except for different subnets
class Ganomaly2d(nn.Module):

    def __init__(self, input_size, n_z, n_channels, n_fm_discriminator, n_fm_generator, n_gpus, adversarial_weight=1, contextual_weight=1, encoder_weight=1, lambda_weight=0.5):
        super().__init__()

        self.input_size = input_size
        self.n_channels = n_channels
        self.n_z = n_z
        self.n_fm_discriminator = n_fm_discriminator
        self.n_fm_generator = n_fm_generator
        self.n_gpus = n_gpus
        self.adversarial_weight = adversarial_weight
        self.contextual_weight = contextual_weight
        self.encoder_weight = encoder_weight
        self.lambda_weight = lambda_weight

        self.discriminator_loss = nn.BCELoss()

        self.adversarial_loss = l2_loss
        self.contextual_loss = nn.L1Loss()
        self.encoder_loss = l2_loss

        self.discriminator = nets.DiscriminatorNet2d(
            input_size=self.input_size,
            n_z=self.n_z,
            n_channels=self.n_channels,
            n_feature_maps=self.n_fm_discriminator,
            n_gpus=self.n_gpus
        )
        self.discriminator.apply(nets.weights_init)
        self.generator = nets.GeneratorNet2d(
            input_size=self.input_size,
            n_z=self.n_z,
            n_channels=self.n_channels,
            n_feature_maps=self.n_fm_generator,
            n_gpus=self.n_gpus
        )
        self.generator.apply(nets.weights_init)

    def forward(self, X, y=None):

        fake, latent_real, latent_fake = self.generator(X)

        appearant_differences = (X - fake).view(fake.size()[0], -1)
        latent_differences = (
            latent_real - latent_fake).view(latent_real.size()[0], -1)

        contextual_loss = torch.mean(appearant_differences.abs(), dim=1)
        encoder_loss = torch.mean(torch.pow(latent_differences, 2), dim=1)

        error = self.lambda_weight * contextual_loss + \
            (1 - self.lambda_weight) * encoder_loss

        return error.reshape(error.size(0)), X, fake, latent_real, latent_fake


# The Ganomaly 2d class
# is the same as the 1d constructor
# except for different subnets
class GanomalyFE(nn.Module):

    def __init__(self, input_size, n_gpus, adversarial_weight=1, contextual_weight=1, encoder_weight=1, lambda_weight=0.5):
        super().__init__()

        self.input_size = input_size
        self.n_gpus = n_gpus
        self.adversarial_weight = adversarial_weight
        self.contextual_weight = contextual_weight
        self.encoder_weight = encoder_weight
        self.lambda_weight = lambda_weight

        self.discriminator_loss = nn.BCELoss()

        self.adversarial_loss = l2_loss
        self.contextual_loss = nn.L1Loss()
        self.encoder_loss = l2_loss

        self.discriminator = nets.DiscriminatorNetFE(
            input_size=self.input_size,
            n_gpus=self.n_gpus
        )
        self.discriminator.apply(nets.weights_init)
        self.generator = nets.GeneratorNetFE(
            input_size=self.input_size,
            n_gpus=self.n_gpus
        )
        self.generator.apply(nets.weights_init)

    def forward(self, X, y=None):

        fake, latent_real, latent_fake = self.generator(X)

        appearant_differences = (X - fake).view(fake.size()[0], -1)
        latent_differences = (
            latent_real - latent_fake).view(latent_real.size()[0], -1)

        contextual_loss = torch.mean(appearant_differences.abs(), dim=1)
        encoder_loss = torch.mean(torch.pow(latent_differences, 2), dim=1)

        error = self.lambda_weight * contextual_loss + \
            (1 - self.lambda_weight) * encoder_loss

        return error.reshape(error.size(0)), X, fake, latent_real, latent_fake


class GanomalyNet(NeuralNet):
    def __init__(self, *args, generator_optimizer, discriminator_optimizer, **kwargs):
        self.generator_optimizer = generator_optimizer
        self.discriminator_optimizer = discriminator_optimizer

        super().__init__(*args, **kwargs)

    # initialize both optimizers
    def initialize_optimizer(self, *_, **__):
        args, kwargs = self.get_params_for_optimizer(
            'generator_optimizer', self.module_.generator.named_parameters())
        self.generator_optimizer_ = self.generator_optimizer(*args, **kwargs)

        args, kwargs = self.get_params_for_optimizer(
            'discriminator_optimizer', self.module_.discriminator.named_parameters())
        self.discriminator_optimizer_ = self.discriminator_optimizer(
            *args, **kwargs)

        return self

    # the validation step is not implemented
    def validation_step(self, Xi, yi, **fit_params):
        raise NotImplementedError

    # default callbacks for logging
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
            ('adversarial_loss', PassthroughScoring(
                name='adversarial_loss',
                on_train=True
            )),
            ('contextual_loss', PassthroughScoring(
                name='contextual_loss',
                on_train=True
            )),
            ('encoder_loss', PassthroughScoring(
                name='encoder_loss',
                on_train=True
            )),
            ('print_log', PrintLog()),
        ]

    # training step
    def train_step(self, Xi, yi=None, **fit_params):

        # turn input data into tensor
        Xi = to_tensor(Xi, device=self.device)

        # create local variables for the generator and discriminator
        discriminator = self.module_.discriminator
        generator = self.module_.generator

        # forward the generator and obtain its data
        fake, latent_Xi, latent_fake = generator(Xi)

        # evaluate real and fake data with the discriminator
        prediction_real, features_real = discriminator(Xi)
        prediction_fake, features_fake = discriminator(fake.detach())

        # create a tensor of ones
        # this is used for the discriminator
        labels_real = ones_like(
            prediction_real, dtype=torch.float32, device=self.device)
        labels_fake = zeros_like(
            prediction_fake, dtype=torch.float32, device=self.device)

        # calculate generator loss
        adversarial_loss = self.module_.adversarial_loss(
            features_real, features_fake)
        contextual_loss = self.module_.contextual_loss(Xi, fake)
        encoder_loss = self.module_.encoder_loss(latent_Xi, latent_fake)
        generator_loss = self.module_.adversarial_weight * adversarial_loss + \
            self.module_.contextual_weight * contextual_loss + \
            self.module_.encoder_weight * encoder_loss

        # calculate discriminator loss
        discriminator_loss_real = self.module_.discriminator_loss(
            prediction_real, labels_real)
        discriminator_loss_fake = self.module_.discriminator_loss(
            prediction_fake, labels_fake)
        discriminator_loss = (discriminator_loss_real +
                              discriminator_loss_fake) * 0.5

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
        self.history.record_batch('adversarial_loss', adversarial_loss.item())
        self.history.record_batch('contextual_loss', contextual_loss.item())
        self.history.record_batch('encoder_loss', encoder_loss.item())
        self.history.record_batch(
            'discriminator_loss', discriminator_loss.item())

        # return train loss for skorch
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
            prediction_real, dtype=torch.float32, device=self.device)
        labels_fake = zeros_like(
            prediction_fake, dtype=torch.float32, device=self.device)

        # calculate generator loss
        adversarial_loss = self.module_.adversarial_loss(
            features_real, features_fake)
        contextual_loss = self.module_.contextual_loss(X, fake)
        encoder_loss = self.module_.encoder_loss(latent_X, latent_fake)

        generator_loss = self.module_.adversarial_weight * adversarial_loss + \
            self.module_.contextual_weight * contextual_loss + \
            self.module_.encoder_weight * encoder_loss

        generator_loss = generator_loss / \
            (self.module_.adversarial_weight +
             self.module_.contextual_weight + self.module_.encoder_weight)

        # calculate discriminator loss
        discriminator_loss_real = self.module_.discriminator_loss(
            prediction_real, labels_real)

        discriminator_loss_fake = self.module_.discriminator_loss(
            prediction_fake, labels_fake)

        discriminator_loss = (discriminator_loss_real +
                              discriminator_loss_fake) * 0.5

        # calculate train loss
        train_loss = generator_loss + discriminator_loss

        # make scores negative
        # GridSearchCV then takes the lowest value
        generator_loss = -1 * generator_loss.item()
        train_loss = -1 * train_loss.item()

        if discriminator_loss.item() < 1e-5:
            discriminator.apply(nets.weights_init)

        # return scores as dictionary
        return {'generator_loss': generator_loss, 'train_loss': train_loss}

    # proba also returns inner tensors of models
    # for visualization purposes
    def predict_proba(self, X):

        nonlin = self._get_predict_nonlinearity()
        y_probas = []
        for yp in self.forward_iter(X, training=False):
            yp = yp if isinstance(yp, tuple) else yp
            yp = nonlin(yp)
            y_probas.append(to_numpy(yp))
        stacked = list(zip(*y_probas))
        y_proba = [concatenate(array) for array in stacked]

        return y_proba

    # predict only returns the anomaly scores
    def predict(self, X):
        return self.predict_proba(X)[0]
