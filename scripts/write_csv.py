import os
import sys
import csv

import aubio

module_dir = os.path.realpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
if module_dir not in sys.path:
    sys.path.append(module_dir)

from mpd.source import create_source
from mpd.onset import AubioOnsetDetector, MadmomFeatureOnsetDetector, MadmomRNNOnsetDetector
from mpd.pitch import AubioPitchDetector

# config / arguments
folder = r'C:\Projects\MusicTranscription\MAB-TonyGame\recordings\benchmarks\7'
folder = r'C:\Projects\_Studiprojekte\19HS-P5 Tony Game\Abgabe\network_code\our_recordings\guitar\0'
folder = r'C:\Projects\MusicTranscription\MAB-TonyGame\recordings\soundtesting_pipeline'
folder = r'C:\Projects\MusicTranscription\MAB-TonyGame\recordings\benchmarks\3'
# destination folder for mock CSV
dest = r'C:\Projects\MusicTranscription\MAB-TonyGame\recordings\\soundtesting_pipeline\3'

# C:\Projects\MusicTranscription\MAB-TonyGame\recordings\soundtesting_pipeline

if __name__ == '__main__':
    hop_size = 512
    onset_method = 'specflux'
    onset_buf_size = 2048
    onset_minioi_ms = 150

    pitch_method = 'yinfft'
    pitch_frame_size = 4096
    history_length = 8

    filter_method = 'lowpass'

    # onset method: energy, hfc, specflux
    od = AubioOnsetDetector(onset_method, hop_size, onset_buf_size, onset_minioi_ms)
    # od = MadmomFeatureOnsetDetector('superflux', hop_size, onset_buf_size, onset_minioi_ms, num_bands=60)
    # od = MadmomRNNOnsetDetector(hop_size, 4096, onset_minioi_ms, fps=100)
    pd = AubioPitchDetector(pitch_method, hop_size, pitch_frame_size, history_length)

    sample_limit = (history_length + 1) * hop_size
    for subdir, dirs, files in os.walk(folder):
        for file in files:
            if len(sys.argv) > 2 and file != sys.argv[2]:
                continue
            path = os.path.join(subdir, file)
            if path.endswith('.wav'):
                # get onset+pitches of the wav file
                src = create_source(path, hop_size=hop_size, verbose=False)

                od.create_detector(src.samplerate)
                pd.create_detector(src.samplerate)

                af = aubio.digital_filter(order=3)  # 7 for A-Filter, 5 for C-Filter, 3 for biquad
                if filter_method == 'lowpass':
                    if src.samplerate == 44100:
                        af.set_biquad(.07909669122050075, .1581933824410015, .07909669122050075, -1.1486877651747005, .4650745300567037)  # q=0.85, f=4700
                    elif src.samplerate == 48000:
                        af.set_biquad(.06844301311767674, .13688602623535348, .06844301311767674, -1.2193255395824403, .4930975920531473)  # q=0.85, f=4700
                        # self.filter_obj.set_biquad(0.01801576198494065, 0.0360315239698813,  0.01801576198494065, -1.4631087710168378, 0.5351718189566004)  # q=0.5, f=2350

                pitches = {}  # onset[ms]:pitch[midi]
                total_read = 0
                last_onset = -sample_limit
                while True:
                    samples, read = src()
                    total_read += read

                    onset = od.process_next(samples)
                    last_onset = max(onset, last_onset)
                    pitch = pd.process_next(af(samples))

                    samples_past_limit_after_onset = total_read - sample_limit - last_onset
                    if samples_past_limit_after_onset < 0 < samples_past_limit_after_onset + hop_size:  # and pitch > 0:
                        pitches[round(last_onset / src.samplerate * 1000)] = pitch

                    if read < src.hop_size:
                        break

                path_csv = (path.replace(folder, dest) if dest else folder)[:-3] + "csv"
                if not os.path.exists(path_csv):
                    if not os.path.exists(os.path.dirname(path_csv)):
                        os.mkdir(os.path.dirname(path_csv))
                    with open(path_csv, mode='w', newline="\n") as f:
                        writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                        for onset in pitches:
                            writer.writerow([onset, pitches[onset]])
                else:
                    print(f"Did not create file {path_csv}, file already exists!")
