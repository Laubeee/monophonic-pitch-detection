import aubio
import wavio
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
        self.src = aubio.source(path, hop_size=self.hop_size)
        self.samplerate = self.src.samplerate

    def __call__(self):
        return self.src()


class WavioSource(AbstractSource):
    def __init__(self, path: str, hop_size: int):
        super().__init__(hop_size)
        src = wavio.read(path)
        self.samplerate = src.rate
        self.data = (src.data / (2 ** (8 * src.sampwidth - 1) - 1)).astype('float32')
        # todo: merge multiple channels (?)

    def __call__(self):
        self.get_next_from_data(self.data)


class ScipySource(AbstractSource):
    def __init__(self, path: str, hop_size: int):
        super().__init__(hop_size)
        self.samplerate, self.data = scipy_read(path)
        # todo: merge multiple channels (?)  convert to float32?

    def __call__(self):
        self.get_next_from_data(self.data)
