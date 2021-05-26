from aubio import source as aubio_source
from wavio import read as wavio_read
from wave import Error as WaveError
from scipy.io.wavfile import read as scipy_read
import numpy as np


class AbstractSource:
    def __init__(self, hop_size: int):
        self.hop_size = hop_size
        self.hop = 0

    def __call__(self):
        raise NotImplementedError("Abstract method implementation missing")

    def get_next_from_data(self, data):
        start = self.hop * self.hop_size
        self.hop += 1
        end = self.hop * self.hop_size
        if end > data.shape[0]:
            end = data.shape[0]
            result = np.zeros(self.hop_size, dtype='float32')
            read = end - start
            result[0:read] = data[start:end]
            return result, read
        return data[start:start + self.hop_size], self.hop_size


class AubioSource(AbstractSource):
    def __init__(self, path: str, hop_size: int):
        super().__init__(hop_size)
        self.src = aubio_source(path, hop_size=self.hop_size)
        self.samplerate = self.src.samplerate
        self.duration_s = self.src.duration / self.samplerate

    def __call__(self):
        return self.src()


class WavioSource(AbstractSource):
    def __init__(self, path: str, hop_size: int):
        super().__init__(hop_size)
        src = wavio_read(path)
        self.samplerate = src.rate
        self.data = (src.data / (2 ** (8 * src.sampwidth - 1) - 1)).astype('float32')

        if len(self.data.shape) > 1:  # merge multiple channels
            self.data = self.data.mean(axis=1)

        self.duration_s = self.data.shape[0] / self.samplerate

    def __call__(self):
        return self.get_next_from_data(self.data)


class ScipySource(AbstractSource):
    def __init__(self, path: str, hop_size: int):
        super().__init__(hop_size)
        self.samplerate, self.data = scipy_read(path)
        self.duration_s = self.data.shape[0] / self.samplerate
        # todo: convert to float32 (?)

    def __call__(self):
        return self.get_next_from_data(self.data)


def create_source(path: str, hop_size: int, verbose: bool = True):
        try:
            return AubioSource(path, hop_size)
        except RuntimeError as e1:
            if verbose:
                print("Aubio can't read this file, switch to wavio")
            try:
                return WavioSource(path, hop_size)
            except WaveError as e2:
                if verbose:
                    print("Wavio can't read this file, switch to scipy.io")
                try:
                    return ScipySource(path, hop_size)
                except ValueError as e3:
                    if verbose:
                        print("Scipy.io can't read this file either -- abort")
                    raise ValueError("File is unreadable:\n" + e1 + "\n" + e2 + "\n" + e3)
