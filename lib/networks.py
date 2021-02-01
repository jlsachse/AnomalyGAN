""" Network architectures.
"""
# pylint: disable=W0221,W0622,C0103,R0913

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F

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
    def __init__(self, input_size, nz, nc, ndf, ngpu):
        super().__init__()

        self.ngpu = ngpu

        stride = 8 if input_size == 1568 else 4
        padding = 4 if input_size == 1568 else 6

        n_layers = 0

        while input_size >= 12:
            n_layers += 1
            input_size = input_size // 4

        n_intermediate_layers = n_layers - 3 if n_layers >= 3 else 0


        main = nn.Sequential()



        main.add_module('initial-{0}-{1}-convt'.format(nc, ndf),
                        nn.Conv1d(nc, ndf, 16, stride=stride, padding = padding, bias = False))
        main.add_module('initial-relu-{0}'.format(ndf),
                        nn.LeakyReLU(0.2, inplace=True))


        for _ in range(n_intermediate_layers):
            in_feat = ndf
            out_feat = ndf * 2
            main.add_module('pyramid-{0}-{1}-convt'.format(in_feat, out_feat),
                            nn.Conv1d(in_feat, out_feat, 16, stride=4, padding = 6, bias = False))
            main.add_module('pyramid-{0}-batchnorm'.format(out_feat),
                            nn.BatchNorm1d(out_feat))
            main.add_module('pyramid-relu-{0}'.format(out_feat),
                            nn.LeakyReLU(0.2, inplace=True))

            ndf = ndf * 2

        in_feat = ndf
        out_feat = ndf * 2

        main.add_module('pyramid-{0}-{1}-convt'.format(in_feat, out_feat),
                        nn.Conv1d(in_feat, out_feat, 16, stride=4, padding = 0, bias = False))
        main.add_module('pyramid-{0}-batchnorm'.format(out_feat),
                        nn.BatchNorm1d(out_feat))
        main.add_module('pyramid-relu-{0}'.format(out_feat),
                    nn.LeakyReLU(0.2, inplace=True))

        
        ndf = ndf * 2

        main.add_module('final-{0}-{1}-convt'.format(ndf, 1),
                        nn.Conv1d(ndf, nz, 9, stride=1, padding = 0, bias = False))

        self.main = main


    def forward(self, input):
        if self.ngpu >= 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output



class Decoder1d(nn.Module):
    def __init__(self, input_size, nz, nc, ngf, ngpu):
        super().__init__()

        self.ngpu = ngpu

        stride = 8 if input_size == 1568 else 4
        padding = 4 if input_size == 1568 else 6

        n_layers = 0

        while input_size >= 12:
            n_layers += 1
            input_size = input_size // 4

        n_intermediate_layers = n_layers - 3 if n_layers >= 3 else 0

        cngf = ngf * 2 ** (n_intermediate_layers + 1)

        
        main = nn.Sequential()

        main.add_module('initial-{0}-{1}-convt'.format(nz, cngf),
                        nn.ConvTranspose1d(nz, cngf, 9, stride=1, bias = False))
        main.add_module('pyramid-{0}-batchnorm'.format(cngf),
                        nn.BatchNorm1d(cngf))
        main.add_module('initial-{0}-relu'.format(cngf),
                        nn.ReLU(True))


        main.add_module('pyramid-{0}-{1}-convt'.format(cngf, cngf // 2),
                        nn.ConvTranspose1d(cngf, cngf // 2, 16, stride=4, padding = 0, output_padding=1, bias = False))
        main.add_module('pyramid-{0}-batchnorm'.format(cngf // 2),
                        nn.BatchNorm1d(cngf // 2))
        main.add_module('pyramid-{0}-relu'.format(cngf // 2),
                        nn.ReLU(True))


        cngf = cngf // 2

        for _ in range(n_intermediate_layers):
            main.add_module('pyramid-{0}-{1}-convt'.format(cngf, cngf // 2),
                            nn.ConvTranspose1d(cngf, cngf // 2, 16, stride=4, padding = 6,  bias = False))
            main.add_module('pyramid-{0}-batchnorm'.format(cngf // 2),
                            nn.BatchNorm1d(cngf // 2))
            main.add_module('pyramid-{0}-relu'.format(cngf // 2),
                            nn.ReLU(True))


            cngf = cngf // 2

        main.add_module('final-{0}-{1}-convt'.format(cngf, nc),
                            nn.ConvTranspose1d(cngf, nc, 16, stride=stride, padding = padding, bias = False))
        main.add_module('final-{0}-tanh'.format(nc),
                            nn.Tanh())

        self.main = main


    def forward(self, input):
        if self.ngpu >= 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output



class Encoder2d(nn.Module):
    def __init__(self, input_size, nz, nc, ndf, ngpu):
        super().__init__()
        self.ngpu = ngpu


        n_layers = 0

        while input_size >= 3:
            n_layers += 1
    
            input_size = input_size // 2

        n_intermediate_layers = n_layers - 2 if n_layers >= 2 else 0

        main = nn.Sequential()

        main.add_module('initial-conv-{0}-{1}'.format(nc, ndf),
                        nn.Conv2d(nc, ndf, 4, 2, 1, bias=False))
        main.add_module('initial-relu-{0}'.format(ndf),
                        nn.LeakyReLU(0.2, inplace=True))

        for _ in range(n_intermediate_layers):
            in_feat = ndf
            out_feat = ndf * 2

            main.add_module('pyramid-{0}-{1}-conv'.format(in_feat, out_feat),
                            nn.Conv2d(in_feat, out_feat, 4, 2, 1, bias=False))
            main.add_module('pyramid-{0}-batchnorm'.format(out_feat),
                            nn.BatchNorm2d(out_feat))
            main.add_module('pyramid-{0}-relu'.format(out_feat),
                            nn.LeakyReLU(0.2, inplace=True))


            ndf = ndf * 2
        
        main.add_module('final-{0}-{1}-conv'.format(ndf, 1),
                        nn.Conv2d(ndf, nz, 3, 1, 0, bias=False))

        self.main = main


    def forward(self, input):
        if self.ngpu >= 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output

        

class Decoder2d(nn.Module):
    def __init__(self, input_size, nz, nc, ngf, ngpu):
        super().__init__()
        
        self.ngpu = ngpu

        n_layers = 0

        while input_size >= 3:
            n_layers += 1
    
            input_size = input_size // 2

        n_intermediate_layers = n_layers - 3 if n_layers >= 3 else 0

        cngf = ngf * 2 ** (n_intermediate_layers + 1)
        
        main = nn.Sequential()
 
        # input is Z, going into a convolution
        main.add_module('initial-{0}-{1}-convt'.format(nz, cngf),
                        nn.ConvTranspose2d(nz, cngf, 3, 1, 0, bias=False))
        main.add_module('initial-{0}-batchnorm'.format(cngf),
                        nn.BatchNorm2d(cngf))
        main.add_module('initial-{0}-relu'.format(cngf),
                        nn.ReLU(True))


        main.add_module('initial-{0}-{1}-convt'.format(cngf, cngf // 2),
                        nn.ConvTranspose2d(cngf, cngf // 2, 4, 2, 1, output_padding = 1, bias=False))
        main.add_module('initial-{0}-batchnorm'.format(cngf // 2),
                        nn.BatchNorm2d(cngf // 2))
        main.add_module('initial-{0}-relu'.format(cngf // 2),
                        nn.ReLU(True))

        
        cngf = cngf // 2

        for _ in range(n_intermediate_layers):
            main.add_module('pyramid-{0}-{1}-convt'.format(cngf, cngf // 2),
                            nn.ConvTranspose2d(cngf, cngf // 2, 4, 2, 1, bias=False))
            main.add_module('pyramid-{0}-batchnorm'.format(cngf // 2),
                            nn.BatchNorm2d(cngf // 2))
            main.add_module('pyramid-{0}-relu'.format(cngf // 2),
                            nn.ReLU(True))


            cngf = cngf // 2

        main.add_module('final-{0}-{1}-convt'.format(cngf, nc),
                        nn.ConvTranspose2d(cngf, nc, 4, 2, 1, bias=False))
        main.add_module('final-{0}-tanh'.format(nc),
                        nn.Tanh())

        self.main = main


    def forward(self, input):
        if self.ngpu >= 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output


class EncoderFE(nn.Module):
    def __init__(self, isize, ngpu):
        super().__init__()
        self.ngpu = ngpu

        main = nn.Sequential()

        main.add_module('initial-conv-1-4',
                        nn.Conv2d(1, 4, 2, 1, bias=False))

        main.add_module('pyramid-4-batchnorm',
                        nn.BatchNorm2d(4))

        main.add_module('pyramid-4-relu',
                        nn.LeakyReLU(0.2, inplace=True))
   


        main.add_module('final-conv-4-8',
                        nn.Conv2d(4, 8, 2, 1, bias=False))

        self.main = main

    def forward(self, input):
        if self.ngpu >= 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output



class DecoderFE(nn.Module):
    def __init__(self, isize, ngpu):
        super().__init__()
        
        self.ngpu = ngpu
        
        main = nn.Sequential()

        main.add_module('initial-8-{1}-convt',
                        nn.ConvTranspose2d(8, 4, 2, 1, bias=False))

        main.add_module('initial-4-relu',
                        nn.ReLU(True))

        main.add_module('final-4-1-convt',
                        nn.ConvTranspose2d(4, 1, 2, 1, bias=False))

        main.add_module('final-1-tanh',
                        nn.Tanh())

        self.main = main


    def forward(self, input):
        if self.ngpu >= 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output

class DiscriminatorNet1d(nn.Module):
    """
    DISCRIMINATOR NETWORK
    """

    def __init__(self, isize, nz, nc, ndf, ngpu):
        super().__init__()
        model = Encoder1d(isize, nz, nc, ndf, ngpu)
        layers = list(model.main.children())

        self.features = nn.Sequential(*layers[:-1])
        self.classifier = nn.Sequential(layers[-1])
        self.classifier.add_module('Sigmoid', nn.Sigmoid())

    def forward(self, x):
        features = self.features(x)
        features = features
        classifier = self.classifier(features)
        classifier = classifier.view(-1, 1).squeeze(1)

        return classifier, features



class GeneratorNet1d(nn.Module):
    """
    GENERATOR NETWORK
    """

    def __init__(self, isize, nz, nc, ngf, ngpu):
        super().__init__()
        self.encoder1 = Encoder1d(isize, nz, nc, ngf, ngpu)
        self.decoder = Decoder1d(isize, nz, nc, ngf, ngpu)
        self.encoder2 = Encoder1d(isize, nz, nc, ngf, ngpu)

    def forward(self, x):
        latent_i = self.encoder1(x)
        gen_imag = self.decoder(latent_i)
        latent_o = self.encoder2(gen_imag)
        return gen_imag, latent_i, latent_o



class DiscriminatorNet2d(nn.Module):
    """
    DISCRIMINATOR NETWORK
    """

    def __init__(self, isize, nz, nc, ndf, ngpu):
        super().__init__()
        model = Encoder2d(isize, nz, nc, ndf, ngpu)
        layers = list(model.main.children())

        self.features = nn.Sequential(*layers[:-1])
        self.classifier = nn.Sequential(layers[-1])
        self.classifier.add_module('Sigmoid', nn.Sigmoid())

    def forward(self, x):
        features = self.features(x)
        features = features
        classifier = self.classifier(features)
        classifier = classifier.view(-1, 1).squeeze(1)

        return classifier, features



class GeneratorNet2d(nn.Module):
    """
    GENERATOR NETWORK
    """

    def __init__(self, isize, nz, nc, ngf, ngpu):
        super().__init__()
        self.encoder1 = Encoder2d(isize, nz, nc, ngf, ngpu)
        self.decoder = Decoder2d(isize, nz, nc, ngf, ngpu)
        self.encoder2 = Encoder2d(isize, nz, nc, ngf, ngpu)

    def forward(self, x):
        latent_i = self.encoder1(x)
        gen_imag = self.decoder(latent_i)
        latent_o = self.encoder2(gen_imag)
        return gen_imag, latent_i, latent_o



class DiscriminatorNetFE(nn.Module):
    """
    DISCRIMINATOR NETWORK
    """

    def __init__(self, isize, ngpu):
        super().__init__()
        model = EncoderFE(isize, ngpu)
        self.features = model.main

        self.features.add_module('pyramid-8-batchnorm',
                        nn.BatchNorm2d(8))


        self.features.add_module('pyramid-8-relu',
                        nn.LeakyReLU(0.2, inplace=True))


        self.classifier = nn.Sequential()
        self.classifier.add_module('Sigmoid', nn.Sigmoid())

    def forward(self, x):
        features = self.features(x)
        features = features
        classifier = self.classifier(features)
        classifier = classifier.view(-1, 1).squeeze(1)

        return classifier, features



class GeneratorNetFE(nn.Module):
    """
    GENERATOR NETWORK
    """

    def __init__(self, isize, ngpu):
        super().__init__()
        self.encoder1 = EncoderFE(isize, ngpu)
        self.decoder = DecoderFE(isize, ngpu)
        self.encoder2 = EncoderFE(isize, ngpu)

    def forward(self, x):
        latent_i = self.encoder1(x)
        gen_imag = self.decoder(latent_i)
        latent_o = self.encoder2(gen_imag)
        return gen_imag, latent_i, latent_o