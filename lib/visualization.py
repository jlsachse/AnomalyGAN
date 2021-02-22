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
    X = tensor(X, device=net.device)

    scores = net.module_.forward(X)[0].cpu().detach().numpy()
    mean_score = np.mean(scores)

    return mean_score


def rename_tensorboard_key(key):

    prefix = 'Not_Specified/'

    if key in ['adversarial_loss', 'contextual_loss', 'encoder_loss']:
        prefix = 'Generator_Losses/'
    elif key in ['discriminator_loss', 'generator_loss']:
        prefix = 'Player_Losses/'
    elif key in ['dur']:
        prefix = 'Duration/'
    elif key in ['train_loss']:
        prefix = 'Overall_Loss/'

    key = prefix + key

    return key

# custom tensorboard calback


class GANomalyBoard(TensorBoard):

    # save parameters for visualizations
    def __init__(self, *args, plot_type, plot_shape, n_samples, plot_latent_shape, **kwargs):

        self.plot_type = plot_type
        self.plot_shape = plot_shape
        self.n_samples = n_samples
        self.plot_latent_shape = plot_latent_shape

        super().__init__(*args, **kwargs)

    # create plots on each epoch end
    def on_epoch_end(self, net, dataset_train, **kwargs):

        epoch = net.history[-1, 'epoch']

        plots = self._extract_images(net, dataset_train)
        mean_score = extract_mean_score(net, dataset_train)

        if self.plot_type == 'image':
            self.writer.add_image('Tensors/X', plots[0], global_step=epoch)
            self.writer.add_image('Tensors/fake',
                                  plots[1], global_step=epoch)
            self.writer.add_figure(
                'Tensors/latent', plots[2], global_step=epoch)
        else:
            self.writer.add_figure('Tensors/real_fake',
                                   plots[0], global_step=epoch)
            self.writer.add_figure('Tensors/latent',
                                   plots[1], global_step=epoch)

        self.writer.add_scalar('Scores/mean_anomaly_score',
                               mean_score, global_step=epoch)

        super().on_epoch_end(net, **kwargs)

    # extract images from model and create plots
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

        latent_i = latent_i.reshape((-1, self.plot_latent_shape))
        latent_o = latent_o.reshape((-1, self.plot_latent_shape))

        latent_figure = plt.figure(figsize=(10, 8))

        for index in range(4):
            latent_axis = latent_figure.add_subplot(4, 1, index + 1)
            latent_axis = sns.lineplot(
                data=latent_i[index], color='black', linewidth=1, ax=latent_axis, alpha=0.7)
            latent_axis = sns.lineplot(
                data=latent_o[index], color='orange', linewidth=1, ax=latent_axis, alpha=0.7)
            latent_axis.set(xticklabels=[])

        if self.plot_type == 'lineplot':

            sns.set_style('whitegrid')

            real = real.reshape((-1, self.plot_shape))
            fake = fake.reshape((-1, self.plot_shape))

            figure = plt.figure(figsize=(10, 8))

            for index in range(len(real)):
                axis = figure.add_subplot(4, 1, index + 1)
                axis = sns.lineplot(
                    data=real[index], color='black', linewidth=1, ax=axis, alpha=0.7)
                axis = sns.lineplot(
                    data=fake[index], color='orange', linewidth=1, ax=axis, alpha=0.7)
                axis.set(xticklabels=[])
            return figure, latent_figure

        elif self.plot_type == 'barplot':

            sns.set_style('whitegrid')

            real = real.reshape((-1, self.plot_shape))
            fake = fake.reshape((-1, self.plot_shape))

            figure = plt.figure(figsize=(10, 8))

            for index in range(len(real)):
                axis = figure.add_subplot(4, 1, index + 1)
                axis = sns.barplot(y=real[index], x=[x for x in range(
                    self.plot_shape)], color='black', ax=axis, alpha=0.7)
                axis = sns.barplot(y=fake[index], x=[x for x in range(
                    self.plot_shape)], color='orange', ax=axis, alpha=0.7)
                axis.set(xticklabels=[])
            return figure, latent_figure

        elif self.plot_type == 'image':
            fake = fake.reshape((-1, 1, self.plot_shape, self.plot_shape))
            real = real.reshape((-1, 1, self.plot_shape, self.plot_shape))
            real_figure = make_grid(
                real_tensor, nrow=int(np.sqrt(self.n_samples)))
            fake_figure = make_grid(
                fake_tensor, nrow=int(np.sqrt(self.n_samples)))
            return real_figure, fake_figure, latent_figure


# this is needed for visualizations within the notebooks
# and is not used by the tenorboard
def lineplot_comparison(result, first_line, second_line, title, xlabel, ylabel):

    sns.set(rc={'figure.figsize': (12, 10)}, style='darkgrid')
    sns.set_context('notebook')

    line_figure = plt.figure(constrained_layout=True)
    cfSpec = gridspec.GridSpec(ncols=1, nrows=4, figure=line_figure)
    line_figure_axis0 = line_figure.add_subplot(cfSpec[0, 0])
    line_figure_axis1 = line_figure.add_subplot(cfSpec[1, 0])
    line_figure_axis2 = line_figure.add_subplot(cfSpec[2, 0])
    line_figure_axis3 = line_figure.add_subplot(cfSpec[3, 0])

    ball_fault =\
        pd.DataFrame(
            [result[result['condition'] == 'Ball Fault'].iloc[0, :][first_line],
             result[result['condition'] == 'Ball Fault'].iloc[0, :][second_line]]
        ).T.rename({0: first_line, 1: second_line}, axis=1)
    sns.lineplot(data=ball_fault, ax=line_figure_axis0, linewidth=1)
    line_figure_axis0.set(xticklabels=[])
    line_figure_axis0.legend(loc='upper left')
    line_figure_axis0.annotate('Ball Fault', xy=(0.99, 0.95), xycoords='axes fraction',
                               horizontalalignment='right', verticalalignment='top')

    inner_race_fault =\
        pd.DataFrame(
            [result[result['condition'] == 'Inner Race Fault'].iloc[0, :][first_line],
             result[result['condition'] == 'Inner Race Fault'].iloc[0, :][second_line]]
        ).T.rename({0: first_line, 1: second_line}, axis=1)
    sns.lineplot(data=inner_race_fault, ax=line_figure_axis1, linewidth=1)
    line_figure_axis1.set(xticklabels=[])
    line_figure_axis1.legend([], [], frameon=False)
    line_figure_axis1.annotate('Inner Race Fault', xy=(0.99, 0.95), xycoords='axes fraction',
                               horizontalalignment='right', verticalalignment='top')

    outer_race_fault =\
        pd.DataFrame(
            [result[result['condition'] == 'Outer Race Fault'].iloc[0, :][first_line],
             result[result['condition'] == 'Outer Race Fault'].iloc[0, :][second_line]]
        ).T.rename({0: first_line, 1: second_line}, axis=1)
    sns.lineplot(data=outer_race_fault, ax=line_figure_axis2, linewidth=1)
    line_figure_axis2.set(xticklabels=[])
    line_figure_axis2.legend([], [], frameon=False)
    line_figure_axis2.annotate('Outer Race Fault', xy=(0.99, 0.95), xycoords='axes fraction',
                               horizontalalignment='right', verticalalignment='top')

    normal_baseline =\
        pd.DataFrame(
            [result[result['condition'] == 'Normal Baseline'].iloc[0, :][first_line],
             result[result['condition'] == 'Normal Baseline'].iloc[0, :][second_line]]
        ).T.rename({0: first_line, 1: second_line}, axis=1)
    sns.lineplot(data=normal_baseline, ax=line_figure_axis3, linewidth=1)
    line_figure_axis3.annotate('Normal Baseline', xy=(0.99, 0.95), xycoords='axes fraction',
                               horizontalalignment='right', verticalalignment='top')

    line_figure_axis3.legend([], [], frameon=False)
    line_figure_axis3.set_xlabel('.', color=(0, 0, 0, 0))
    line_figure_axis3.set_ylabel('.', color=(0, 0, 0, 0))

    line_figure.text(0.5, 0, xlabel, ha='center')
    line_figure.text(0, 0.5, ylabel, va='center', rotation='vertical')
    line_figure.suptitle(title)

    return line_figure
