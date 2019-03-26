from collections import deque
import operator

import numpy as np

import aubio
from .utils import sigmoid


class AbstractPitchDetector:
    def __init__(self, time_limit_s: float=0.1, onset_minioi_ms: int=50):
        self.time_limit_s = time_limit_s
        self.onset_minioi_ms = onset_minioi_ms

    def get_pitches(self, path: str):
        raise NotImplementedError("Abstract method needs implementation")


class AubioPitchDetector(AbstractPitchDetector):
    def __init__(self, time_limit_s: float=0.1, onset_minioi_ms: int=50):
        super().__init__(time_limit_s, onset_minioi_ms)

        self.hop_size = 512  # [samples]
        n_values = 8  # history length

        self.onset_method = 'energy'  # energy: local energy, hfc: high frequency content (default), phase/wphase
        self.onset_buf_size = 1024  # [samples]
        self.onset_threshold = 0.9  # default 0.3 -- testing showed that 0.9, 1.0 and 1.1 yield the same top results

        self.pitch_method = 'yinfft'  # yin, mcomb
        self.pitch_buf_size = 4096  # [samples]

        self.p_weights = [sigmoid(x) for x in (np.arange(-4, 4, 8 / n_values) - 0.5)]  # [.1,.2,.3,.4,.75,1.0,1.2,1.3]

    def get_pitches(self, path: str):
        src = aubio.source(path, hop_size=self.hop_size)
        sample_limit = src.samplerate * self.time_limit_s
        o = aubio.onset(self.onset_method, self.onset_buf_size, self.hop_size, src.samplerate)
        o.set_threshold(self.onset_threshold)
        o.set_minioi_ms(self.onset_minioi_ms)

        p = aubio.pitch(self.pitch_method, self.pitch_buf_size, self.hop_size, src.samplerate)
        p.set_unit('midi')
        # p.set_tolerance() 0.15 yin 0.85 yinfft

        print(path.split('\\')[-1], src.samplerate, src.channels, src.duration)

        pitches = {}  # onset[ms]:pitch[midi]
        pitches_history = deque(maxlen=len(self.p_weights))
        total_read = 0
        last_onset = -2 * p.buf_size
        while True:
            samples, read = src()
            total_read += read

            onset = o(samples)
            if onset[0] != 0:
                last_onset = o.get_last()

            midi = int(round(p(samples)[0]))
            pitches_history.append(midi)

            samples_past_100ms_after_onset = total_read - sample_limit - last_onset
            if samples_past_100ms_after_onset < 0 < samples_past_100ms_after_onset + p.hop_size:
                p_cand = {}
                for i in range(len(pitches_history)):
                    ph = pitches_history[i]
                    p_cand[ph] = p_cand.get(ph, 0) + self.p_weights[i]
                p_max = max(p_cand.items(), key=operator.itemgetter(1))[0]
                p_max = midi if p_cand[midi] >= p_cand[p_max] else p_max
                if p_max > 0:
                    pitches[round(last_onset / src.samplerate * 1000)] = p_max

            if read < src.hop_size:
                break
        return pitches
