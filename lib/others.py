from lib.models import Ganomaly1d, Ganomaly2d, GanomalyFE, GanomalyNet
from lib.visualization import GANomalyBoard, rename_tensorboard_key
import torch

from skorch.callbacks import PassthroughScoring, ProgressBar
from torch.utils.tensorboard import SummaryWriter


def create_dataset(data, feature_columns, label_columns, sample_length=3136, **column_values):

    for column, values in column_values.items():
        data = data[data[column].isin(values)]

    features = data.loc[:, feature_columns]
    labels = data.loc[:, label_columns]

    features = features.dropna()

    chunked_features = features.applymap(
        lambda df: list(chunk(df, sample_length, False)))
    stacked_features = chunked_features.stack().explode()

    stacked_features = stacked_features.reset_index(level=[1])
    stacked_features = stacked_features.rename(
        {0: 'vibrationData', 'level_1': 'vibrationOrigin'}, axis=1)

    stacked_features = stacked_features.loc[:, [
        'vibrationData', 'vibrationOrigin']]

    dataset = stacked_features.join(labels, how='left')

    dataset = dataset.reset_index()

    features = dataset['vibrationData']
    labels = dataset.loc[:, label_columns + ['vibrationOrigin', 'index']]

    return features, labels


def chunk(array, chunk_size, keep_rest):

    for position in range(0, len(array), chunk_size):
        result = array[position:position + chunk_size]

        if keep_rest:
            yield result
        else:
            if (len(result) == chunk_size):
                yield result


def build_model(model, device, max_epochs, batch_size, lr, beta1, beta2, workers, plot_type='lineplot', plot_shape=3136, n_samples=4, plot_latent_shape=600, suffix='', callbacks=[], verbose=0, **kwargs):

    module_kwargs = ['input_size', 'n_z', 'n_channels',
                     'n_gpus', 'n_fm_discriminator',
                     'n_fm_generator', 'adversarial_weight', 'contextual_weight',
                     'encoder_weight', 'lambda_weight']

    module_kwargs = {'module__' + key: value for key,
                     value in kwargs.items() if key in module_kwargs}

    if suffix:

        summary_writer = SummaryWriter(log_dir='runs/' + suffix)

        ganomaly_board = \
            GANomalyBoard(
                summary_writer,
                key_mapper=rename_tensorboard_key,
                close_after_train=False,
                plot_type=plot_type,
                plot_shape=plot_shape,
                n_samples=n_samples,
                plot_latent_shape=plot_latent_shape
            )

        callbacks.append(ganomaly_board)

    output_model = GanomalyNet(
        model,
        **module_kwargs,

        verbose=verbose,
        device=device,

        criterion=torch.nn.BCELoss,

        generator_optimizer=torch.optim.Adam,
        generator_optimizer__lr=lr,
        generator_optimizer__betas=(beta1, beta2),

        discriminator_optimizer=torch.optim.Adam,
        discriminator_optimizer__lr=lr,
        discriminator_optimizer__betas=(beta1, beta2),

        batch_size=batch_size,
        max_epochs=max_epochs,

        train_split=False,  # not implemented
        iterator_train__shuffle=True,
        iterator_train__num_workers=workers,
        iterator_valid__num_workers=workers,
    )

    return output_model
