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
    """ Onset- and Pitch detection using the aubio library directly
    (aubio onset and aubio pitch)
    """

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


class AubioNotesDetector(AbstractPitchDetector):
    """ onset and pitch detection using aubio notes which internally uses aubio pitch and aubio onset

    For some reason it isn't possible to set the onset method and other parameters that are available in the bin
    """

    def __init__(self, time_limit_s: float=0.1, onset_minioi_ms: int=50):
        super().__init__(time_limit_s, onset_minioi_ms)

        self.window_size = 4096
        self.hop_size = 1024
        self.min_velocity = 75

    def get_pitches(self, path: str):
        src = aubio.source(path, hop_size=self.hop_size)
        notes_o = aubio.notes("default", self.window_size, self.hop_size, src.samplerate)
        notes_o.set_minioi_ms(self.onset_minioi_ms)

        # total number of frames read
        pitches = {}  # onset[ms]:pitch[midi]
        total_read = 0
        while True:
            samples, read = src()
            total_read += read

            new_note = notes_o(samples)
            if new_note[0] != 0 and new_note[1] >= self.min_velocity:
                pitches[round((total_read - self.window_size) * 1000 / src.samplerate)] = new_note[0]
                # print("%.6f" % (total_read / float(src.samplerate)), new_note)

            if read < self.hop_size:
                break
        print(pitches)
        return pitches


class PitchDetector(AbstractPitchDetector):
    """ This variant uses a low-pass filter and YIN, just like the app "Pitch" does. """

    def __init__(self, time_limit_s: float=0.1, onset_minioi_ms: int=50):
        super().__init__(time_limit_s, onset_minioi_ms)

        self.hop_size = 512  # [samples]
        n_values = 8  # history length

        self.onset_method = 'energy'  # energy: local energy, hfc: high frequency content (default), phase/wphase
        self.onset_buf_size = 1024  # [samples]
        self.onset_threshold = 0.9  # default 0.3 -- testing showed that 0.9, 1.0 and 1.1 yield the same top results

        self.pitch_method = 'yin'
        self.pitch_buf_size = 4096  # [samples]

        self.p_weights = [sigmoid(x) for x in (np.arange(-4, 4, 8 / n_values) - 0.5)]  # [.1,.2,.3,.4,.75,1.0,1.2,1.3]

    def get_pitches(self, path: str):
        src = aubio.source(path, hop_size=self.hop_size)
        sample_limit = src.samplerate * self.time_limit_s

        f = aubio.digital_filter(order=3)  # 7 for A-Filter, 5 for C-Filter, 3 for biquad
        # coeficients generated on https://www.earlevel.com/main/2013/10/13/biquad-calculator-v2/
        if src.samplerate == 44100:
            f.set_biquad(.07909669122050075, .1581933824410015, .07909669122050075, -1.1486877651747005, .4650745300567037)  # q=0.85, f=4700
        elif src.samplerate == 48000:
            f.set_biquad(.06844301311767674, .13688602623535348, .06844301311767674, -1.2193255395824403, .4930975920531473)  # q=0.85, f=4700
            #f.set_biquad(0.01801576198494065, 0.0360315239698813,  0.01801576198494065, -1.4631087710168378, 0.5351718189566004)  # q=0.5, f=2350

        o = aubio.onset(self.onset_method, self.onset_buf_size, self.hop_size, src.samplerate)
        o.set_threshold(self.onset_threshold)
        o.set_minioi_ms(self.onset_minioi_ms)

        p = aubio.pitch(self.pitch_method, self.pitch_buf_size, self.hop_size, src.samplerate)
        p.set_unit('midi')
        # p.set_tolerance() 0.15 yin 0.85 yinfft

        pitches = {}  # onset[ms]:pitch[midi]
        pitches_history = deque(maxlen=len(self.p_weights))
        total_read = 0
        last_onset = -2 * p.buf_size
        while True:
            samples, read = src()
            total_read += read
            samples = f(samples)

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
