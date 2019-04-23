from collections import deque
import operator

import numpy as np

import aubio
from .utils import sigmoid


class AbstractPitchDetector:
    def __init__(self, hop_size: int, frame_size: int):
        self.hop_size = hop_size
        self.frame_size = frame_size

        self.pitch = None

    def create_detector(self, samplerate):
        raise NotImplementedError("Abstract method implementation missing")

    def process_next(self, samples):
        raise NotImplementedError("Abstract method implementation missing")


class AubioPitchDetector(AbstractPitchDetector):
    """ Onset- and Pitch detection using the aubio library directly
    (aubio onset and aubio pitch)
    """

    def __init__(self, method: str, hop_size: int, frame_size=4096, history_length: int=8):
        super().__init__(hop_size, frame_size)
        self.method = method  # yinfft, yin, mcomb

        hops = min(history_length, int(frame_size / hop_size) - 1)
        self.p_weights = np.ones(history_length)
        self.p_weights[:hops] = [sigmoid(x) for x in (np.arange(-4, 4, 8 / hops) - 0.5)]  # .1,.2,.3,.4,.75,1,1.2,1.3
        self.history = deque(maxlen=len(self.p_weights))

    def create_detector(self, samplerate):
        self.pitch = aubio.pitch(self.method, self.frame_size, self.hop_size, samplerate)
        self.pitch.set_unit('midi')
        # self.pitch.set_tolerance() 0.15 yin 0.85 yinfft

    def process_next(self, samples):
        pitch = int(round(self.pitch(samples)[0]))
        self.history.append(pitch)

        pitch_candidates = {}
        for i in range(len(self.history)):
            pitch_candidates[self.history[i]] = pitch_candidates.get(self.history[i], 0) + self.p_weights[i]
        p_max = max(pitch_candidates.items(), key=operator.itemgetter(1))[0]
        return pitch if pitch_candidates[pitch] >= pitch_candidates[p_max] else p_max  # take last if two are equal
