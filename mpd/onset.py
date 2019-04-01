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

    def process_next(self, samples, last_onset) -> int:
        raise NotImplementedError("Abstract method implementation missing")


class AubioOnsetDetector(AbstractOnsetDetector):
    def __init__(self, method: str, hop_size: int, frame_size: int=1024, minioi_ms: int=50):
        super().__init__(hop_size, frame_size, minioi_ms)
        self.method = method
        self.threshold = {'hfc': 0.3, 'energy': 0.9, 'specflux': 0.9}[self.method]

    def create_detector(self, samplerate):
        self.onset = aubio.onset(self.method, self.buf_size, self.hop_size, samplerate)
        self.onset.set_minioi_ms(self.minioi_ms)
        self.onset.set_threshold(self.threshold)

    def process_next(self, samples, last_onset) -> int:
        o = self.onset(samples)
        if o[0] != 0:
            last_onset = self.onset.get_last()  # round(self.onset.get_last() / self.onset.samplerate * 1000)
        return last_onset  # return as n sample


class MadmomOnsetDetector(AbstractOnsetDetector):
    def __init__(self, hop_size: int, frame_size: int=2048, minioi_ms: int=50, fps=100, pre_avg=0.15, pre_max=0.01):
        super().__init__(hop_size, frame_size, minioi_ms)
        self.buffer = None
        self.processed_samples = 0
        self.samplerate = 0

    def create_detector(self, samplerate):
        self.processed_samples = 0
        self.samplerate = samplerate
        self.buffer = deque(maxlen=ceil(self.buf_size / self.hop_size))
        for i in range(self.buffer.maxlen):
            self.buffer.append(np.zeros(self.hop_size, dtype='int16'))

    def process_next(self, samples, last_onset) -> int:
        self.processed_samples += len(samples)
        samples = np.round(np.array(samples) * np.iinfo('int16').max).astype('int16')
        self.buffer.append(samples)
        for p in self.onset(np.hstack(self.buffer)):
            onset_sample = int((self.processed_samples - len(self.buffer) * self.hop_size) + p * self.samplerate)
            if onset_sample - self.minioi_ms * self.samplerate / 1000 > last_onset:
                last_onset = onset_sample
        return last_onset  # return as n sample


class MadmomFeatureOnsetDetector(MadmomOnsetDetector):
    def __init__(self, method: str, hop_size: int, frame_size: int=2048, minioi_ms: int=50,
                 num_bands=24, log=False, fps=100):
        super().__init__(hop_size, frame_size, minioi_ms)
        self.method = method
        self.threshold = {'superflux': 3, 'complex_flux': 7, 'high_frequency_content': .3, 'spectral_diff': 1.0,
                          'spectral_flux': 6, 'modified_kullback_leibler': 1.0, 'phase_deviation': 1.0,
                          'weighted_phase_deviation': 1.0, 'normalized_weighted_phase_deviation': 1.0,
                          'complex_domain': 1.0, 'rectified_complex_domain': 1.0}[self.method]
        self.num_bands = num_bands
        self.fps = fps
        self.pre_avg = 0
        self.pre_max = 0
        if self.method in ['superflux'] or log:
            self.fb = madmom.audio.filters.LogarithmicFilterbank
            self.log = np.log10
        else:
            self.fb = None
            self.log = None

    def create_detector(self, samplerate):
        super().create_detector(samplerate)
        f = madmom.features.SpectralOnsetProcessor(self.method, fps=self.fps, log=self.log, sample_rate=samplerate,
                                                   filterbank=self.fb, num_bands=self.num_bands)
        peak = madmom.features.onsets.OnsetPeakPickingProcessor(fps=self.fps, threshold=self.threshold,
                                                                pre_avg=self.pre_avg, combine=self.minioi_ms/1000,
                                                                pre_max=self.pre_max, online=True, reset=False)
        self.onset = lambda samples: peak(f(samples, reset=False))


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
