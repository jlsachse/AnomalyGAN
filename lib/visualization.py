from skorch.callbacks import TensorBoard
from torchvision.utils import make_grid
import torch.nn.functional as F
from torch import tensor
import torch
import numpy as np


import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.gridspec as gridspec


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

    def __init__(self, *args, plot_type, plot_shape, n_samples, plot_latent_shape, **kwargs):
        self.plot_type = plot_type
        self.plot_shape = plot_shape
        self.n_samples = n_samples
        self.plot_latent_shape = plot_latent_shape

        super().__init__(*args, **kwargs)
    
    def on_epoch_end(self, net, dataset_train, **kwargs):
        
        epoch = net.history[-1, 'epoch']
        
        X, fake, latent_i, latent_o = self._extract_images(net, dataset_train)
        mean_score = extract_mean_score(net, dataset_train)
        
        if self.plot_type == 'image':
            self.writer.add_image('Real and Fake/X', X, global_step=epoch)
            self.writer.add_image('Real and Fake/fake', fake, global_step=epoch)
        else:
            self.writer.add_figure('Real and Fake/X', X, global_step=epoch)
            self.writer.add_figure('Real and Fake/fake', fake, global_step=epoch)

        self.writer.add_figure('Latent/in', latent_i, global_step=epoch)
        self.writer.add_figure('Latent/out', latent_o, global_step=epoch)

        self.writer.add_scalar('Scores/mean_anomaly_score', mean_score, global_step=epoch)
        
        super().on_epoch_end(net, **kwargs)  # call super last


    def _extract_images(self, net, dataset_train):
        generator = net.module_.generator
        
        real = dataset_train.X
        real_tensor = tensor(real)
        fake_tensor, latent_i, latent_o = generator(real_tensor)

        real_tensor = real_tensor[:self.n_samples]
        fake_tensor = fake_tensor[:self.n_samples]

        real = real_tensor.cpu().detach().numpy()
        fake = fake_tensor.cpu().detach().numpy()

        latent_i = latent_i.cpu().detach().numpy()
        latent_o = latent_o.cpu().detach().numpy()


        if self.plot_type == 'lineplot':

            sns.set_style('whitegrid')
            
            real = real.reshape((-1, self.plot_shape))
            fake = fake.reshape((-1, self.plot_shape))

            real_figure = plt.figure(figsize=(10,8))
            fake_figure = plt.figure(figsize=(10,8))

            for index in range(len(real)):
                real_axis = real_figure.add_subplot(4, 1, index + 1)
                fake_axis = fake_figure.add_subplot(4, 1, index + 1)
                real_plot = sns.lineplot(data=real[index], color='black', linewidth=1, ax = real_axis)
                fake_plot = sns.lineplot(data=fake[index], color='black', linewidth=1, ax = fake_axis)
                real_plot.set(xticklabels=[])
                fake_plot.set(xticklabels=[])
                real_plot.set(ylim=(-0.05, 1.05))
                fake_plot.set(ylim=(-0.05, 1.05))


        if self.plot_type == 'barplot':

            sns.set_style('whitegrid')

            real = real.reshape((-1, self.plot_shape))
            fake = fake.reshape((-1, self.plot_shape))

            real_figure = plt.figure(figsize=(10,8))
            fake_figure = plt.figure(figsize=(10,8))

            for index in range(len(real)):
                real_axis = real_figure.add_subplot(4, 1, index + 1)
                fake_axis = fake_figure.add_subplot(4, 1, index + 1)
                real_plot = sns.barplot(y=real[index], x = [x for x in range(self.plot_shape)], color='black', ax = real_axis)
                fake_plot = sns.barplot(y=fake[index], x = [x for x in range(self.plot_shape)], color='black', ax = fake_axis)
                real_plot.set(xticklabels=[])
                fake_plot.set(xticklabels=[])
                real_plot.set(ylim=(-0.05, 1.05))
                fake_plot.set(ylim=(-0.05, 1.05))


        if self.plot_type == 'image':
            fake = fake.reshape((-1, 1, self.plot_shape, self.plot_shape))
            real = real.reshape((-1, 1, self.plot_shape, self.plot_shape))
            real_figure = make_grid(real_tensor, nrow=int(np.sqrt(self.n_samples)))
            fake_figure = make_grid(fake_tensor, nrow=int(np.sqrt(self.n_samples)))


        latent_i = latent_i.reshape((-1, self.plot_latent_shape))
        latent_o = latent_o.reshape((-1, self.plot_latent_shape))

        latent_i_figure = plt.figure(figsize=(10,8))
        latent_o_figure = plt.figure(figsize=(10,8))

        for index in range(4):
            latent_i_axis = latent_i_figure.add_subplot(4, 1, index + 1)
            latent_o_axis = latent_o_figure.add_subplot(4, 1, index + 1)
            latent_i_plot = sns.lineplot(data=latent_i[index], color='black', linewidth=1, ax = latent_i_axis)
            latent_o_plot = sns.lineplot(data=latent_o[index], color='black', linewidth=1, ax = latent_o_axis)
            latent_i_plot.set(xticklabels=[])
            latent_o_plot.set(xticklabels=[])
        
        return real_figure, fake_figure, latent_i_figure, latent_o_figure


def lineplot_comparison(result, first_line, second_line, title, xlabel, ylabel):

    sns.set_style('darkgrid')
    sns.set(rc={'figure.figsize':(20, 10)})

    sns.set_context('notebook')


    conditionFigure = plt.figure(constrained_layout=True)
    cfSpec = gridspec.GridSpec(ncols=1, nrows=4, figure=conditionFigure)
    conditionFigureAxis0 = conditionFigure.add_subplot(cfSpec[0, 0])
    conditionFigureAxis1 = conditionFigure.add_subplot(cfSpec[1, 0])
    conditionFigureAxis2 = conditionFigure.add_subplot(cfSpec[2, 0])
    conditionFigureAxis3 = conditionFigure.add_subplot(cfSpec[3, 0])


    ball_fault =\
    pd.DataFrame(
        [result[result['condition'] == 'Ball Fault'].iloc[0, :][first_line],
         result[result['condition'] == 'Ball Fault'].iloc[0, :][second_line]]
    ).T.rename({0: first_line, 1: second_line}, axis = 1)
    p0 = sns.lineplot(data = ball_fault, ax = conditionFigureAxis0)
    p0.set(xticklabels=[]) 

    inner_race_fault =\
    pd.DataFrame(
        [result[result['condition'] == 'Inner Race Fault'].iloc[0, :][first_line],
         result[result['condition'] == 'Inner Race Fault'].iloc[0, :][second_line]]
    ).T.rename({0: first_line, 1: second_line}, axis = 1)
    p1 = sns.lineplot(data = inner_race_fault, ax = conditionFigureAxis1)
    p1.set(xticklabels=[]) 

    outer_race_fault =\
    pd.DataFrame(
        [result[result['condition'] == 'Outer Race Fault'].iloc[0, :][first_line],
         result[result['condition'] == 'Outer Race Fault'].iloc[0, :][second_line]]
    ).T.rename({0: first_line, 1: second_line}, axis = 1)
    p2 = sns.lineplot(data = outer_race_fault, ax = conditionFigureAxis2)
    p2.set(xticklabels=[]) 

    normal_baseline =\
    pd.DataFrame(
        [result[result['condition'] == 'Normal Baseline'].iloc[0, :][first_line],
         result[result['condition'] == 'Normal Baseline'].iloc[0, :][second_line]]
    ).T.rename({0: first_line, 1: second_line}, axis = 1)
    sns.lineplot(data = normal_baseline, ax = conditionFigureAxis3);

    conditionFigureAxis3.set_xlabel('.', color=(0, 0, 0, 0))
    conditionFigureAxis3.set_ylabel('.', color=(0, 0, 0, 0))

    conditionFigure.text(0.5, 0, xlabel, ha='center')
    conditionFigure.text(0, 0.5, ylabel, va='center', rotation='vertical')
    conditionFigure.suptitle(title);