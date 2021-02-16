from sklearn.base import TransformerMixin
from sklearn.utils.random import sample_without_replacement
import numpy as np
from scipy import signal
from PIL import Image
import PIL


class FeatureExtractor(TransformerMixin):
    """
    scikit-learn implementation of the feature extractor from (Jiang et. al, 2019)
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        # the absolute value is used for two different features
        X_abs = np.abs(X)

        # calculate features from (Jiang et. al, 2019)
        maximum_value = np.max(X, axis=1)
        mean_value = np.mean(X, axis=1)
        minimum_value = np.min(X, axis=1)
        standard_value = np.sqrt(np.mean((X.T - maximum_value).T ** 2, axis=1))
        peak_to_peak_value = maximum_value - minimum_value
        mean_amplitude = np.max(X_abs, axis=1)
        root_mean_square_value = np.sqrt(np.mean(X**2, axis=1))
        skewness_value = np.mean((X**3), axis=1)
        kurtosis_value = np.mean((X**4), axis=1)
        waveform_indicator = root_mean_square_value / mean_amplitude
        pulse_indicator = maximum_value / mean_amplitude
        kurtosis_index = kurtosis_value / root_mean_square_value
        peak_index = maximum_value / root_mean_square_value
        square_root_amplitude = np.mean(np.sqrt(X_abs), axis=1)**2
        margin_indicator = maximum_value / square_root_amplitude
        skewness_indicator = skewness_value / (root_mean_square_value ** 4)

        # create list of calculated values
        X = [
            maximum_value,
            mean_value,
            minimum_value,
            standard_value,
            peak_to_peak_value,
            mean_amplitude,
            root_mean_square_value,
            skewness_value,
            kurtosis_value,
            waveform_indicator,
            pulse_indicator, kurtosis_index,
            peak_index,
            square_root_amplitude,
            margin_indicator,
            skewness_indicator
        ]

        # transform list into array and
        X = np.array(X)

        # transpose array
        X = X.T

        return X


class ArrayReshaper(TransformerMixin):

    def __init__(self, shape):
        # set shape for  the arrays
        self.shape = shape

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        shape = self.shape

        # reshape each array
        X = [array.reshape(shape) for array in X]
        X = np.array(X)

        return X


class ArrayRetyper(TransformerMixin):

    def __init__(self, dtype):
        # set dtype for array
        self.dtype = dtype

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        dtype = self.dtype

        # change dtype of array
        X = np.array(X, dtype=dtype)

        return X


class ArraySTFT(TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        # calculate spectrogram for each sample
        X = [self._calculate_spectrogram(array) for array in X]
        X = np.array(X)

        return X

    def _calculate_spectrogram(self, array):

        # calculate the dimension of the spectrogram
        # by taking the square root of the signal length
        spectrogram_dim = int(np.sqrt(len(array)))

        # calculate the spectrogram
        # the window length is set to two times the spectrogram
        # dimension to create a square spectrogram

        # data about frequency and time is not needed and discarded
        _, _, spectrogram = signal.stft(array, nperseg=(spectrogram_dim * 2))

        # take amplitude and normalize the spectrogram
        spectrogram = np.abs(spectrogram) / len(spectrogram)
        spectrogram = np.array(Image.fromarray(spectrogram).resize(
            (spectrogram_dim, spectrogram_dim), resample=PIL.Image.BILINEAR))

        return spectrogram


class ArrayFFT(TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        # calculate frequency spectrum for each sample
        X = [self._calculate_frequency_spectrum(array) for array in X]
        X = np.array(X)

        return X

    def _calculate_frequency_spectrum(self, array):

        # calculate the length of the frequency spectrum
        frequency_spectrum_len = len(array) // 2

        # calculate the frequency spectrum
        frequency_spectrum = np.fft.fft(array)

        # only keep first half of frequency spectrum
        frequency_spectrum = frequency_spectrum[:frequency_spectrum_len]

        # take amplitude and normalize the frequency spectrum
        frequency_spectrum = np.abs(
            frequency_spectrum) / len(frequency_spectrum)

        return frequency_spectrum


class ArrayMinMaxScaler(TransformerMixin):

    def fit(self, X, y=None):
        self.X_min = X.min()
        self.X_max = X.max()

        return self

    def transform(self, X):

        X_min = self.X_min
        X_max = self.X_max

        # scale each sample between zero and one
        X = [-1 + 2*((x - x.min()) / (x.max() - x.min())) for x in X]

        return X
