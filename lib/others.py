from lib.models import Ganomaly1d, Ganomaly2d, GanomalyFE, GanomalyNet
from lib.visualization import GANomalyBoard, rename_tensorboard_key
import torch

from skorch.callbacks import PassthroughScoring, ProgressBar
from torch.utils.tensorboard import SummaryWriter

def create_dataset(data, feature_columns, label_columns, sample_length = 3136, **column_values): 
    

    for column, values in column_values.items():
        data = data[data[column].isin(values)]
        
    features = data.loc[:, feature_columns]
    labels = data.loc[:, label_columns]
    
    features = features.dropna()
    
    chunked_features = features.applymap(lambda df: list(chunk(df, sample_length, False)))
    stacked_features = chunked_features.stack().explode()

    stacked_features = stacked_features.reset_index(level=[1])
    stacked_features = stacked_features.rename({0: 'vibrationData', 'level_1': 'vibrationOrigin'}, axis = 1)
    
    stacked_features = stacked_features.loc[:, ['vibrationData', 'vibrationOrigin']]
    
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

def build_model(model, isize, max_epochs, directory, needs_feature_engineering = False, ngpu = 0, nz = 600, ndf = 64, ngf = 64, nc = 1, batch_size = 16, lr = 0.0001, beta1 = 0.5, beta2 = 0.999, workers = 2):
    
    if not needs_feature_engineering:
        output_model = GanomalyNet(
            model,
            module__isize = isize,
            module__nz=nz,
            module__ndf=ndf,
            module__ngf=ngf,
            module__nc=nc,
            module__ngpu=ngpu,
            module__w_app = 30,
            module__w_lambda = 30/31,

            device = torch.device("cuda:0") if torch.cuda.is_available() else 'cpu',

            criterion=torch.nn.BCELoss,

            optimizer_gen=torch.optim.Adam,
            optimizer_gen__lr=lr,
            optimizer_gen__betas=(beta1, beta2),

            optimizer_dis=torch.optim.Adam,
            optimizer_dis__lr=lr,
            optimizer_dis__betas=(beta1, beta2),

            batch_size=batch_size,
            max_epochs=max_epochs,

            train_split=False,  # not implemented
            iterator_train__shuffle=True,
            iterator_train__num_workers=workers,
            iterator_valid__num_workers=workers,

            callbacks=[
                PassthroughScoring('loss_dis', on_train=True),
                PassthroughScoring('loss_gen', on_train=True),
                PassthroughScoring('loss_gen_fra', on_train=True),
                PassthroughScoring('loss_gen_app', on_train=True),
                PassthroughScoring('loss_gen_lat', on_train=True),
                GANomalyBoard(SummaryWriter(log_dir= 'runs/' + directory), key_mapper = rename_tensorboard_key, close_after_train = False)
            ]
        )
    else:
            output_model = GanomalyNet(
            model,
            module__isize = isize,
            module__ngpu=ngpu,

            device = torch.device("cuda:0") if torch.cuda.is_available() else 'cpu',

            criterion=torch.nn.BCELoss,

            optimizer_gen=torch.optim.Adam,
            optimizer_gen__lr=lr,
            optimizer_gen__betas=(beta1, beta2),

            optimizer_dis=torch.optim.Adam,
            optimizer_dis__lr=lr,
            optimizer_dis__betas=(beta1, beta2),

            batch_size=batch_size,
            max_epochs=max_epochs,

            train_split=False,  # not implemented
            iterator_train__shuffle=True,
            iterator_train__num_workers=workers,
            iterator_valid__num_workers=workers,

            callbacks=[
                PassthroughScoring('loss_dis', on_train=True),
                PassthroughScoring('loss_gen', on_train=True),
                PassthroughScoring('loss_gen_fra', on_train=True),
                PassthroughScoring('loss_gen_app', on_train=True),
                PassthroughScoring('loss_gen_lat', on_train=True),
                GANomalyBoard(SummaryWriter(log_dir= 'runs/' + directory), key_mapper = rename_tensorboard_key, close_after_train = False)
            ]
        )
    
    return output_model