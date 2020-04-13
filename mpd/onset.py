from math import ceil
from collections import deque
import numpy as np

import aubio
import madmom


class AbstractOnsetDetector:
    def __init__(self, hop_size: int, frame_size: int, minioi_ms: int):
        self.hop_size = hop_size  # should be the same as in pitch detection!
        self.buf_size = frame_size
        self.minioi_ms = minioi_ms

        self.onset = None

    def create_detector(self, samplerate):
        raise NotImplementedError("Abstract method implementation missing")

    def process_next(self, samples) -> int:
        raise NotImplementedError("Abstract method implementation missing")


class AubioOnsetDetector(AbstractOnsetDetector):
    def __init__(self, method: str, hop_size: int, frame_size: int=1024, minioi_ms: int=50):
        super().__init__(hop_size, frame_size, minioi_ms)
        self.method = method
        self.threshold = {'hfc': 0.3, 'energy': 0.9, 'specflux': 0.95}[self.method]
        self.compression = 2.0  # default 10.0
        self.silence = -54  # default -70

    def create_detector(self, samplerate):
        self.onset = aubio.onset(self.method, self.buf_size, self.hop_size, samplerate)
        self.onset.set_minioi_ms(self.minioi_ms)
        self.onset.set_threshold(self.threshold)
        self.onset.set_compression(self.compression)
        self.onset.set_silence(self.silence)
        #self.onset.set_awhitening()

    def process_next(self, samples) -> int:
        o = self.onset(samples)
        if o[0] != 0:
            return self.onset.get_last()  # round(self.onset.get_last() / self.onset.samplerate * 1000)
        return 0


class MadmomOnsetDetector(AbstractOnsetDetector):
    def __init__(self, hop_size: int, frame_size: int=2048, minioi_ms: int=50, fps=100, pre_avg=0.15, pre_max=0.01):
        super().__init__(hop_size, frame_size, minioi_ms)
        self.processed_samples = 0
        self.frame_idx = 0
        self.sample_rate = 0

    def create_detector(self, sample_rate):
        self.processed_samples = 0
        self.sample_rate = sample_rate

    def process_next(self, samples) -> int:
        self.processed_samples += len(samples)

        o = self.onset(samples)
        if len(o) > 0 and o[0] > 0:
            return int(o[0] * self.sample_rate)
        return 0


class MadmomFeatureOnsetDetector(MadmomOnsetDetector):
    def __init__(self, method: str, hop_size: int, frame_size: int=2048, minioi_ms: int=50, num_bands=24, log=False):
        super().__init__(hop_size, frame_size, minioi_ms)
        self.method = method
        self.threshold = {'superflux': 0.9, 'complex_flux': .7, 'high_frequency_content': .3, 'spectral_diff': 1.0,
                          'spectral_flux': .6, 'modified_kullback_leibler': 1.0, 'phase_deviation': 1.0,
                          'weighted_phase_deviation': 1.0, 'normalized_weighted_phase_deviation': 1.0,
                          'complex_domain': 1.0, 'rectified_complex_domain': 1.0}[self.method]
        self.num_bands = num_bands
        if self.method in ['superflux'] or log:
            self.fb = madmom.audio.filters.LogarithmicFilterbank
            self.log = np.log10
        else:
            self.fb = None
            self.log = None
        self.buffer = None

    def create_detector(self, sample_rate):
        super().create_detector(sample_rate)
        odf = madmom.features.SpectralOnsetProcessor('superflux', num_bands=self.num_bands, sample_rate=sample_rate,
                                                     log=self.log, filterbank=self.fb, frame_size=self.buf_size,
                                                     hop_size=self.hop_size)
        peak = madmom.features.onsets.OnsetPeakPickingProcessor(threshold=self.threshold, combine=self.minioi_ms / 1000,
                                                                online=True, fps=sample_rate / self.hop_size)
        offset = int(self.buf_size / self.hop_size * 1.25)  # 5, 10
        index = int(offset / 2) + 1  # 3, 6
        self.buffer = np.zeros(offset*self.hop_size)
        self.onset = lambda samples: self.process_onset(samples, odf, peak, index)

    def process_onset(self, samples, odf, peak, odf_index):
        # update buffer with new samples
        self.buffer = np.roll(self.buffer, -len(samples), axis=0)
        self.buffer[-len(samples):] = samples
        return peak(odf(self.buffer)[odf_index:odf_index + 1], reset=False)


class MadmomRNNOnsetDetector(MadmomOnsetDetector):
    def __init__(self, hop_size: int, frame_size: int=2048, minioi_ms: int=50, fps=100):
        super().__init__(hop_size, frame_size, minioi_ms)
        self.threshold = 0.6

        self.fps = fps
        self.pre_avg = 0
        self.pre_max = 0

    def create_detector(self, samplerate):
        super().create_detector(samplerate)
        f = madmom.features.RNNOnsetProcessor(online=True, origin='online')  # hop_size=self.hop_size
        peak = madmom.features.OnsetPeakPickingProcessor(threshold=self.threshold, combine=self.minioi_ms/1000)
                                                         # pre_avg=self.pre_avg, fps=self.fps, online=True,
                                                         # pre_max=self.pre_max, reset=False)
        self.onset = lambda samples: peak(f(samples, reset=False))
