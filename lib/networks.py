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
    def __init__(self, isize, nz, nc, ndf, ngpu):
        super().__init__()

        self.ngpu = ngpu

        assert isize in [28, 56]

        main = nn.Sequential()

        main.add_module('initial-{0}-{1}-convt'.format(nc, ndf),
                        nn.Conv1d(nc, ndf, 16, stride=4, padding = 7, bias = False))
        main.add_module('initial-relu-{0}'.format(ndf),
                        nn.LeakyReLU(0.2, inplace=True))
        csize, cndf = isize / 2, ndf

        while csize >= 14:
            in_feat = cndf
            out_feat = cndf * 2
            main.add_module('pyramid-{0}-{1}-convt'.format(in_feat, out_feat),
                            nn.Conv1d(in_feat, out_feat, 16, stride=4, padding = 7, bias = False))
            main.add_module('pyramid-{0}-batchnorm'.format(out_feat),
                            nn.BatchNorm1d(out_feat))
            main.add_module('pyramid-relu-{0}'.format(out_feat),
                        nn.LeakyReLU(0.2, inplace=True))
            cndf = cndf * 2
            csize = csize / 2

        main.add_module('final-{0}-{1}-convt'.format(cndf, 1),
                        nn.Conv1d(cndf, nz, 16, stride=2, padding = 0, bias = False))

        self.main = main


    def forward(self, input):
        if self.ngpu >= 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output



class Decoder1d(nn.Module):
    def __init__(self, isize, nz, nc, ngf, ngpu):
        super().__init__()

        self.ngpu = ngpu

        assert isize in [28, 56]

        cngf = ngf // 2
        csize = isize

        while csize >= 7:
            cngf = cngf * 2
            csize = csize / 2
        
        main = nn.Sequential()

        main.add_module('initial-{0}-{1}-convt'.format(nz, cngf),
                        nn.ConvTranspose1d(nz, cngf, 16, stride=2, output_padding = 1, bias = False))
        main.add_module('pyramid-{0}-batchnorm'.format(cngf),
                        nn.BatchNorm1d(cngf))
        main.add_module('initial-{0}-relu'.format(cngf),
                        nn.ReLU(True))

        csize, _ = 14, cngf
        while csize <= isize // 2:
            main.add_module('pyramid-{0}-{1}-convt'.format(cngf, cngf // 2),
                            nn.ConvTranspose1d(cngf, cngf // 2, 16, stride=4, padding = 7, output_padding = 2, bias = False))
            main.add_module('pyramid-{0}-batchnorm'.format(cngf // 2),
                            nn.BatchNorm1d(cngf // 2))
            main.add_module('pyramid-{0}-relu'.format(cngf // 2),
                            nn.ReLU(True))
            cngf = cngf // 2
            csize = csize * 2

        main.add_module('final-{0}-{1}-convt'.format(cngf, nc),
                            nn.ConvTranspose1d(cngf, nc, 16, stride=4, padding = 7, output_padding = 2, bias = False))
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
    def __init__(self, isize, nz, nc, ndf, ngpu):
        super().__init__()
        self.ngpu = ngpu

        assert isize in [28, 56]

        main = nn.Sequential()

        main.add_module('initial-conv-{0}-{1}'.format(nc, ndf),
                        nn.Conv2d(nc, ndf, 4, 2, 1, bias=False))
        main.add_module('initial-relu-{0}'.format(ndf),
                        nn.LeakyReLU(0.2, inplace=True))
        csize, cndf = isize / 2, ndf

        while csize >= 14:
            in_feat = cndf
            out_feat = cndf * 2
            main.add_module('pyramid-{0}-{1}-conv'.format(in_feat, out_feat),
                            nn.Conv2d(in_feat, out_feat, 4, 2, 1, bias=False))
            main.add_module('pyramid-{0}-batchnorm'.format(out_feat),
                            nn.BatchNorm2d(out_feat))
            main.add_module('pyramid-{0}-relu'.format(out_feat),
                            nn.LeakyReLU(0.2, inplace=True))
            cndf = cndf * 2
            csize = csize / 2
        
        main.add_module('final-{0}-{1}-conv'.format(cndf, 1),
                        nn.Conv2d(cndf, nz, 4, 1, 0, bias=False))

        self.main = main


    def forward(self, input):
        if self.ngpu >= 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output

        

class Decoder2d(nn.Module):
    def __init__(self, isize, nz, nc, ngf, ngpu):
        super().__init__()

        assert isize in [28, 56]
        
        self.ngpu = ngpu

        cngf = ngf // 2
        csize = isize

        while csize >= 7:
            cngf = cngf * 2
            csize = csize / 2
        
        main = nn.Sequential()
 
        # input is Z, going into a convolution
        main.add_module('initial-{0}-{1}-convt'.format(nz, cngf),
                        nn.ConvTranspose2d(nz, cngf, 4, 1, 0, bias=False))
        main.add_module('initial-{0}-batchnorm'.format(cngf),
                        nn.BatchNorm2d(cngf))
        main.add_module('initial-{0}-relu'.format(cngf),
                        nn.ReLU(True))

        csize, _ = 14, cngf
        while csize <= isize // 2:
            main.add_module('pyramid-{0}-{1}-convt'.format(cngf, cngf // 2),
                            nn.ConvTranspose2d(cngf, cngf // 2, 4, 2, 1, bias=False))
            main.add_module('pyramid-{0}-batchnorm'.format(cngf // 2),
                            nn.BatchNorm2d(cngf // 2))
            main.add_module('pyramid-{0}-relu'.format(cngf // 2),
                            nn.ReLU(True))
            cngf = cngf // 2
            csize = csize * 2

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

        assert isize == 4

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

        assert isize == 4
        
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