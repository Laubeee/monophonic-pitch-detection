import os
import sys

import numpy as np
import pyaudio

module_dir = os.path.realpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
if module_dir not in sys.path:
    sys.path.append(module_dir)

from mpd.pitch import AubioPitchDetector
from mpd.utils import midi2char

# config / arguments

if __name__ == '__main__':
    FORMAT = pyaudio.paFloat32
    CHANNELS = 1
    RECORD_SECONDS = 5

    hop_size = 512
    pitch_method = 'yinfft'
    pitch_frame_size = 2048
    history_length = 8

    filter_method = None  # 'lowpass'
    pd = AubioPitchDetector(pitch_method, hop_size, pitch_frame_size, history_length)

    # init mic
    sr = 44100

    # get onset+pitches of mic input
    pd.create_detector(sr)

    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=sr, input=True, frames_per_buffer=hop_size)

    while True:
        buffer = stream.read(hop_size)  # get next hop_size of samples
        samples = np.fromstring(buffer, dtype=np.float32)
        pitch = pd.process_next(samples)

        print("\t\t\t", pitch, "\t", midi2char(pitch), "\t\t", end="\r")  # confidence = pitch_obj.get_confidence()

        if pitch == 44:
            break

    stream.stop_stream()
    stream.close()
    p.terminate()
