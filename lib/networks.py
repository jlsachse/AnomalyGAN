import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F

# Only GeneratorNet1d, DiscriminatorNet2d, Encoder1d and Decoder1d
# are explained as the other classes are just modified copies


def weights_init(mod):
    """
    Custom weights initialization called on netG, netD and netE
    :param m:
    :return:
    """
    classname = mod.__class__.__name__

    if classname.find('Conv') != -1:
        mod.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        mod.weight.data.normal_(1.0, 0.02)
        mod.bias.data.fill_(0)


class Encoder1d(nn.Module):
    def __init__(self, input_size, n_z, n_channels, n_feature_maps, n_gpus):
        super().__init__()

        # n_gpus is later needed for the forward-method
        self.n_gpus = n_gpus

        # set stride and padding to 8/4 in the first layer
        # if the input_size equals 1568 (case for frequency spectra)
        stride = 8 if input_size == 1568 else 4
        padding = 4 if input_size == 1568 else 6

        n_layers = 0

        # calculate layers based on input size
        # divide input size by 4 as each layer
        # changes its input by the same factor
        while input_size >= 12:
            n_layers += 1
            input_size //= 4

        # calculate layers created by loop
        # three layers are created outside
        # of the loop
        n_loop_layers = n_layers - 3 if n_layers >= 3 else 0

        # initialize sequential container for the layers
        main = nn.Sequential()

        # initialize the first layer with LeakyReLU as activation
        # this layer takes the image with n_channels creates a new
        # representation with 1/4 the size and n_feature_maps
        main.add_module('initial-{0}-{1}-convt'.format(n_channels, n_feature_maps),
                        nn.Conv1d(n_channels, n_feature_maps, 16, stride=stride, padding=padding, bias=False))
        main.add_module('initial-relu-{0}'.format(n_feature_maps),
                        nn.LeakyReLU(0.2, inplace=True))

        # create stacks of one convolutional, one batchnorm
        # and one LeakyReLU layer; each time the amount of
        # feature maps is multiplied by 2
        for _ in range(n_loop_layers):
            n_feature_maps_in = n_feature_maps
            n_feature_maps_out = n_feature_maps * 2
            main.add_module('pyramid-{0}-{1}-convt'.format(n_feature_maps_in, n_feature_maps_out),
                            nn.Conv1d(n_feature_maps_in, n_feature_maps_out, 16, stride=4, padding=6, bias=False))
            main.add_module('pyramid-{0}-batchnorm'.format(n_feature_maps_out),
                            nn.BatchNorm1d(n_feature_maps_out))
            main.add_module('pyramid-relu-{0}'.format(n_feature_maps_out),
                            nn.LeakyReLU(0.2, inplace=True))

            n_feature_maps *= 2

        # add last intermediate layer, this time the padding
        # is set to 0
        n_feature_maps_in = n_feature_maps
        n_feature_maps_out = n_feature_maps * 2

        main.add_module('pyramid-{0}-{1}-convt'.format(n_feature_maps_in, n_feature_maps_out),
                        nn.Conv1d(n_feature_maps_in, n_feature_maps_out, 16, stride=4, padding=0, bias=False))
        main.add_module('pyramid-{0}-batchnorm'.format(n_feature_maps_out),
                        nn.BatchNorm1d(n_feature_maps_out))
        main.add_module('pyramid-relu-{0}'.format(n_feature_maps_out),
                        nn.LeakyReLU(0.2, inplace=True))

        # add last layer which has the same filter size as
        # its input and thus a stride of 1; this layer
        # converts the 2d representation into the 1d latent vector
        n_feature_maps *= 2
        main.add_module('final-{0}-{1}-convt'.format(n_feature_maps, 1),
                        nn.Conv1d(n_feature_maps, n_z, 9, stride=1, padding=0, bias=False))

        # assign container to class attribute
        self.main = main

    # forward function calculates the output of the
    # sequential container; there is one or multiple gpus
    # use data_parallel otherwise use the cpu
    def forward(self, input):
        if self.n_gpus >= 1:
            output = nn.parallel.data_parallel(
                self.main, input, range(self.n_gpus))
        else:
            output = self.main(input)

        return output


class Decoder1d(nn.Module):
    def __init__(self, input_size, n_z, n_channels, n_feature_maps, n_gpus):
        super().__init__()

        # n_gpus is later needed for the forward-method
        self.n_gpus = n_gpus

        # set stride and padding to 8/4 in the first layer
        # if the input_size equals 1568 (case for frequency spectra)
        stride = 8 if input_size == 1568 else 4
        padding = 4 if input_size == 1568 else 6

        n_layers = 0

        # calculate layers based on input size
        # divide input size by 4 as each layer
        # changes its input by the same factor
        while input_size >= 12:
            n_layers += 1
            input_size //= 4

        # calculate layers created by loop
        # three layers are created outside
        # of the loop
        # also calculate the amount of feature maps
        n_loop_layers = n_layers - 3 if n_layers >= 3 else 0
        n_feature_maps = n_feature_maps * 2 ** (n_loop_layers + 1)

        # initialize sequential container for the layers
        main = nn.Sequential()

        # initialize the first layer with ReLU as activation
        # this layer takes the latent vector and converts it
        # into a vector of length 9 with n_feature_maps
        main.add_module('initial-{0}-{1}-convt'.format(n_z, n_feature_maps),
                        nn.ConvTranspose1d(n_z, n_feature_maps, 9, stride=1, bias=False))
        main.add_module('pyramid-{0}-batchnorm'.format(n_feature_maps),
                        nn.BatchNorm1d(n_feature_maps))
        main.add_module('initial-{0}-relu'.format(n_feature_maps),
                        nn.ReLU(True))

        # calculate feature maps for second layer
        n_feature_maps_in = n_feature_maps
        n_feature_maps_out = n_feature_maps // 2

        # create second layer, which has a padding of 0
        # and output padding of 1 to create the same dimensions
        # as in encoder
        main.add_module('pyramid-{0}-{1}-convt'.format(n_feature_maps_in, n_feature_maps_out),
                        nn.ConvTranspose1d(n_feature_maps_in, n_feature_maps_out, 16, stride=4, padding=0, output_padding=1, bias=False))
        main.add_module('pyramid-{0}-batchnorm'.format(n_feature_maps_out),
                        nn.BatchNorm1d(n_feature_maps_out))
        main.add_module('pyramid-{0}-relu'.format(n_feature_maps_out),
                        nn.ReLU(True))

        n_feature_maps //= 2

        # create stacks of one transposed convolutional, one batchnorm
        # and one ReLU layer; each time the amount of
        # feature maps is divided by 2
        for _ in range(n_loop_layers):
            n_feature_maps_in = n_feature_maps
            n_feature_maps_out = n_feature_maps // 2

            main.add_module('pyramid-{0}-{1}-convt'.format(n_feature_maps_in, n_feature_maps_out),
                            nn.ConvTranspose1d(n_feature_maps_in, n_feature_maps_out, 16, stride=4, padding=6,  bias=False))
            main.add_module('pyramid-{0}-batchnorm'.format(n_feature_maps_out),
                            nn.BatchNorm1d(n_feature_maps_out))
            main.add_module('pyramid-{0}-relu'.format(n_feature_maps_out),
                            nn.ReLU(True))

            n_feature_maps //= 2

        # add last layer with tanh as activation
        # after this layer the output has the same
        # dimensions as the input of the encoder
        main.add_module('final-{0}-{1}-convt'.format(n_feature_maps, n_channels),
                        nn.ConvTranspose1d(n_feature_maps, n_channels, 16, stride=stride, padding=padding, bias=False))
        main.add_module('final-{0}-tanh'.format(n_channels),
                        nn.Tanh())

        self.main = main

    # forward function calculates the output of the
    # sequential container; there is one or multiple gpus
    # use data_parallel otherwise use the cpu
    def forward(self, input):
        if self.n_gpus >= 1:
            output = nn.parallel.data_parallel(
                self.main, input, range(self.n_gpus))
        else:
            output = self.main(input)
        return output


class Encoder2d(nn.Module):
    def __init__(self, input_size, n_z, n_channels, n_feature_maps, n_gpus):
        super().__init__()
        self.n_gpus = n_gpus

        n_layers = 0

        while input_size >= 3:
            n_layers += 1
            input_size //= 2

        n_intermediate_layers = n_layers - 2 if n_layers >= 2 else 0

        main = nn.Sequential()

        main.add_module('initial-conv-{0}-{1}'.format(n_channels, n_feature_maps),
                        nn.Conv2d(n_channels, n_feature_maps, 4, 2, 1, bias=False))
        main.add_module('initial-relu-{0}'.format(n_feature_maps),
                        nn.LeakyReLU(0.2, inplace=True))

        for _ in range(n_intermediate_layers):
            n_feature_maps_in = n_feature_maps
            n_feature_maps_out = n_feature_maps * 2

            main.add_module('pyramid-{0}-{1}-conv'.format(n_feature_maps_in, n_feature_maps_out),
                            nn.Conv2d(n_feature_maps_in, n_feature_maps_out, 4, 2, 1, bias=False))
            main.add_module('pyramid-{0}-batchnorm'.format(n_feature_maps_out),
                            nn.BatchNorm2d(n_feature_maps_out))
            main.add_module('pyramid-{0}-relu'.format(n_feature_maps_out),
                            nn.LeakyReLU(0.2, inplace=True))

            n_feature_maps *= 2

        main.add_module('final-{0}-{1}-conv'.format(n_feature_maps, 1),
                        nn.Conv2d(n_feature_maps, n_z, 3, 1, 0, bias=False))

        self.main = main

    def forward(self, input):
        if self.n_gpus >= 1:
            output = nn.parallel.data_parallel(
                self.main, input, range(self.n_gpus))
        else:
            output = self.main(input)
        return output


class Decoder2d(nn.Module):
    def __init__(self, input_size, n_z, n_channels, n_feature_maps, n_gpus):
        super().__init__()

        self.n_gpus = n_gpus

        n_layers = 0

        while input_size >= 3:
            n_layers += 1
            input_size = input_size // 2

        n_intermediate_layers = n_layers - 3 if n_layers >= 3 else 0
        n_feature_maps = n_feature_maps * 2 ** (n_intermediate_layers + 1)

        main = nn.Sequential()

        # input is Z, going into a convolution
        main.add_module('initial-{0}-{1}-convt'.format(n_z, n_feature_maps),
                        nn.ConvTranspose2d(n_z, n_feature_maps, 3, 1, 0, bias=False))
        main.add_module('initial-{0}-batchnorm'.format(n_feature_maps),
                        nn.BatchNorm2d(n_feature_maps))
        main.add_module('initial-{0}-relu'.format(n_feature_maps),
                        nn.ReLU(True))

        n_feature_maps_in = n_feature_maps
        n_feature_maps_out = n_feature_maps // 2

        main.add_module('pyramid-{0}-{1}-convt'.format(n_feature_maps_in, n_feature_maps_out),
                        nn.ConvTranspose2d(n_feature_maps_in, n_feature_maps_out, 4, 2, 1, output_padding=1, bias=False))
        main.add_module('pyramid-{0}-batchnorm'.format(n_feature_maps_out),
                        nn.BatchNorm2d(n_feature_maps_out))
        main.add_module('pyramid-{0}-relu'.format(n_feature_maps_out),
                        nn.ReLU(True))

        n_feature_maps //= 2

        for _ in range(n_intermediate_layers):
            n_feature_maps_in = n_feature_maps
            n_feature_maps_out = n_feature_maps // 2

            main.add_module('pyramid-{0}-{1}-convt'.format(n_feature_maps_in, n_feature_maps_out),
                            nn.ConvTranspose2d(n_feature_maps_in, n_feature_maps_out, 4, 2, 1, bias=False))
            main.add_module('pyramid-{0}-batchnorm'.format(n_feature_maps_out),
                            nn.BatchNorm2d(n_feature_maps_out))
            main.add_module('pyramid-{0}-relu'.format(n_feature_maps_out),
                            nn.ReLU(True))

            n_feature_maps //= 2

        main.add_module('final-{0}-{1}-convt'.format(n_feature_maps, n_channels),
                        nn.ConvTranspose2d(n_feature_maps, n_channels, 4, 2, 1, bias=False))
        main.add_module('final-{0}-tanh'.format(n_channels),
                        nn.Tanh())

        self.main = main

    def forward(self, input):
        if self.n_gpus >= 1:
            output = nn.parallel.data_parallel(
                self.main, input, range(self.n_gpus))
        else:
            output = self.main(input)

        return output


class EncoderFE(nn.Module):
    def __init__(self, input_size, n_gpus):
        super().__init__()
        self.n_gpus = n_gpus

        main = nn.Sequential()

        main.add_module('initial-conv-1-4',
                        nn.Conv2d(1, 4, 2, 1, bias=False))

        main.add_module('initial-4-batchnorm',
                        nn.BatchNorm2d(4))

        main.add_module('initial-4-relu',
                        nn.LeakyReLU(0.2, inplace=True))

        main.add_module('final-conv-4-8',
                        nn.Conv2d(4, 8, 2, 1, bias=False))

        self.main = main

    def forward(self, input):
        if self.n_gpus >= 1:
            output = nn.parallel.data_parallel(
                self.main, input, range(self.n_gpus))
        else:
            output = self.main(input)
        return output


class DecoderFE(nn.Module):
    def __init__(self, input_size, n_gpus):
        super().__init__()

        self.n_gpus = n_gpus

        main = nn.Sequential()

        main.add_module('initial-8-4-convt',
                        nn.ConvTranspose2d(8, 4, 2, 1, bias=False))

        main.add_module('initial-4-relu',
                        nn.ReLU(True))

        main.add_module('final-4-1-convt',
                        nn.ConvTranspose2d(4, 1, 2, 1, bias=False))

        main.add_module('final-1-tanh',
                        nn.Tanh())

        self.main = main

    def forward(self, input):
        if self.n_gpus >= 1:
            output = nn.parallel.data_parallel(
                self.main, input, range(self.n_gpus))
        else:
            output = self.main(input)

        return output


class DiscriminatorNet1d(nn.Module):
    """
    DISCRIMINATOR NETWORK
    """

    def __init__(self, input_size, n_z, n_channels, n_feature_maps, n_gpus):
        super().__init__()
        model = Encoder1d(input_size, n_z, n_channels, n_feature_maps, n_gpus)
        layers = list(model.main.children())

        self.features = nn.Sequential(*layers[:-1])
        self.classifier = nn.Sequential(layers[-1])
        self.classifier.add_module('Sigmoid', nn.Sigmoid())

    def forward(self, x):
        features = self.features(x)
        prediction = self.classifier(features)
        prediction = prediction.view(-1, 1).squeeze(1)
        return prediction, features


class GeneratorNet1d(nn.Module):
    """
    GENERATOR NETWORK
    """

    def __init__(self, input_size, n_z, n_channels, n_feature_maps, n_gpus):
        super().__init__()
        # initialize the encoder for real data
        self.real_encoder = Encoder1d(
            input_size, n_z, n_channels, n_feature_maps, n_gpus)

        # initialize the decoder
        self.decoder = Decoder1d(
            input_size, n_z, n_channels, n_feature_maps, n_gpus)

        # initialize the encoder for fake data
        self.fake_encoder = Encoder1d(
            input_size, n_z, n_channels, n_feature_maps, n_gpus)

    def forward(self, x):
        # obtain latent representation of the autoencoder
        latent_real = self.real_encoder(x)

        # obtain reconstructed (fake) data
        fake = self.decoder(latent_real)

        # obtain latent representation of fake data
        latent_fake = self.fake_encoder(fake)
        return fake, latent_real, latent_fake


class DiscriminatorNet2d(nn.Module):
    """
    DISCRIMINATOR NETWORK
    """

    def __init__(self, input_size, n_z, n_channels, n_feature_maps, n_gpus):
        super().__init__()
        # initialize the encoder
        model = Encoder2d(input_size, n_z, n_channels, n_feature_maps, n_gpus)

        # list layers of the encoder
        layers = list(model.main.children())

        # all but last layer for feature matching loss
        self.features = nn.Sequential(*layers[:-1])

        # create classifier with sigmoid
        self.classifier = nn.Sequential(layers[-1])
        self.classifier.add_module('Sigmoid', nn.Sigmoid())

    def forward(self, x):
        # get features for x
        features = self.features(x)

        # get prediction for x
        prediction = self.classifier(features)
        prediction = prediction.view(-1, 1).squeeze(1)
        return prediction, features


class GeneratorNet2d(nn.Module):
    """
    GENERATOR NETWORK
    """

    def __init__(self, input_size, n_z, n_channels, n_feature_maps, n_gpus):
        super().__init__()
        self.real_encoder = Encoder2d(
            input_size, n_z, n_channels, n_feature_maps, n_gpus)
        self.decoder = Decoder2d(
            input_size, n_z, n_channels, n_feature_maps, n_gpus)
        self.fake_encoder = Encoder2d(
            input_size, n_z, n_channels, n_feature_maps, n_gpus)

    def forward(self, x):
        latent_real = self.real_encoder(x)
        fake = self.decoder(latent_real)
        latent_fake = self.fake_encoder(fake)
        return fake, latent_real, latent_fake


class DiscriminatorNetFE(nn.Module):
    """
    DISCRIMINATOR NETWORK
    """

    def __init__(self, input_size, n_gpus):
        super().__init__()
        model = EncoderFE(input_size, n_gpus)
        self.features = model.main

        self.features.add_module('pyramid-8-batchnorm',
                                 nn.BatchNorm2d(8))

        self.features.add_module('pyramid-8-relu',
                                 nn.LeakyReLU(0.2, inplace=True))

        self.classifier = nn.Sequential()
        self.classifier.add_module('Sigmoid', nn.Sigmoid())

    def forward(self, x):
        features = self.features(x)
        prediction = self.classifier(features)
        prediction = prediction.view(-1, 1).squeeze(1)

        return prediction, features


class GeneratorNetFE(nn.Module):
    """
    GENERATOR NETWORK
    """

    def __init__(self, input_size, n_gpus):
        super().__init__()
        self.real_encoder = EncoderFE(input_size, n_gpus)
        self.decoder = DecoderFE(input_size, n_gpus)
        self.fake_encoder = EncoderFE(input_size, n_gpus)

    def forward(self, x):
        latent_real = self.real_encoder(x)
        fake = self.decoder(latent_real)
        latent_fake = self.fake_encoder(fake)
        return fake, latent_real, latent_fake
