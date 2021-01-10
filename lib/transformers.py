from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.utils.random import sample_without_replacement
import numpy as np

class DataSelector(TransformerMixin, BaseEstimator):

    def __init__(self, columns = 'all', **column_values):
        self.columns = columns
        self.column_values = column_values


    def fit(self, X, y=None):
        
        # Return the transformer
        return self


    def transform(self, X):
        
        X_ = X.copy()
        
        if self.columns == 'all':
            columns = slice(None)
        else:
            columns = self.columns
    
        for column, values in self.column_values.items():
            X_ = X_[X_[column].isin(values)]
    
        X_ = X_.loc[:, columns]
        X_ = X_.to_numpy()
        
        return X_

class ArrayFlattener(TransformerMixin, BaseEstimator):

    def fit(self, X, y=None):
        
        # Return the transformer
        return self


    def transform(self, X):
        
        X_ = X.copy()
    
        return np.concatenate(X_)
    
class DataTypeFilter(TransformerMixin, BaseEstimator):
    
    def __init__(self, data_type):
        self.data_type = data_type

    def fit(self, X, y=None):
        
        # Return the transformer
        return self


    def transform(self, X):
        
        X_ = X.copy()
        
        X_ = [array for array in X_ if isinstance(array, self.data_type)]
        X_ = np.array(X_)
        
        return X_
    
    
class ArrayChunker(TransformerMixin, BaseEstimator):
    
    def __init__(self, chunk_size, keep_rest = False):
        self.chunk_size = chunk_size
        self.keep_rest = keep_rest

    def fit(self, X, y=None):
        
        # Return the transformer
        return self


    def transform(self, X):
        
        X_ = X.copy()
        
        X_ = [np.array(list(self._chunk(array, self.chunk_size, self.keep_rest))) for array in X_]
        X_ = np.array(X_)
    
        return X_  

    def _chunk(self, array, chunk_size, keep_rest):
        
        for position in range(0, len(array), chunk_size):
            result = array[position:position + chunk_size]
            
            if keep_rest:
                yield result
            else:
                if (len(result) == chunk_size):
                    yield result
                    
class FeatureExtractor(TransformerMixin, BaseEstimator):

    def __init__(self, axis):
        self.axis = axis

    def fit(self, X, y=None):
        
        # Return the transformer
        return self


    def transform(self, X):
        
        X_ = X.copy()
        
        # The absolute value is used for two different features
        X_abs = np.abs(X_)
        
        # Features from (Jiang et. al, 2019)
        maximum_value = np.max(X_, axis = self.axis)
        mean_value = np.mean(X_, axis = self.axis)
        minimum_value = np.min(X_, axis = self.axis)
        standard_value = np.sqrt(np.mean((X_.T - maximum_value).T ** 2, axis = self.axis))
        peak_to_peak_value = maximum_value - minimum_value
        mean_amplitude = np.max(X_abs, axis = self.axis)
        root_mean_square_value = np.sqrt(np.mean(X_**2, axis = self.axis))
        skewness_value = np.mean((X_**3), axis = self.axis)
        kurtosis_value = np.mean((X_**4), axis = self.axis)
        waveform_indicator = root_mean_square_value / mean_amplitude
        pulse_indicator = np.array(maximum_value / mean_amplitude)
        kurtosis_index = root_mean_square_value / root_mean_square_value
        peak_index = kurtosis_value / root_mean_square_value 
        square_root_amplitude = np.mean(np.sqrt(X_abs)**2, axis = self.axis)
        margin_indicator = maximum_value / square_root_amplitude
        skewness_indicator = skewness_value / (root_mean_square_value ** 4)
        
        X_ = np.array([maximum_value, mean_value, minimum_value, standard_value,
                       peak_to_peak_value, mean_amplitude, root_mean_square_value, skewness_value,
                       kurtosis_value, waveform_indicator, pulse_indicator, kurtosis_index,
                       peak_index, square_root_amplitude, margin_indicator, skewness_indicator])
        X_ = X_.T
        
        return X_
    
    
class ArrayReshaper(TransformerMixin, BaseEstimator):

    def __init__(self, shape):
        self.shape = shape

    def fit(self, X, y=None):
        
        # Return the transformer
        return self


    def transform(self, X):
        
        X_ = X.copy()
        
        X_ = [array.reshape(self.shape) for array in X_]
        X_ = np.array(X_)
        
        return X_


class RandomArraySampler(TransformerMixin, BaseEstimator):

    def __init__(self, n_samples, random_state = None):
        self.n_samples = n_samples
        self.random_state = random_state

    def fit(self, X, y=None):
        
        # Return the transformer
        return self


    def transform(self, X):
        
        X_ = X.copy()

        indeces = sample_without_replacement(len(X_), self.n_samples, random_state = self.random_state)
        X_ = X_[indeces]
        
        return X_

class ArrayEqualizer(TransformerMixin, BaseEstimator):

    def fit(self, X, y=None):
        
        # Return the transformer
        return self


    def transform(self, X):
        
        X_ = X.copy()
        
        shortest_length = np.min([len(array) for array in X_])
        X_ = [array[:shortest_length] for array in X_]
        
        X_ = np.array(X_)
        
        return X_