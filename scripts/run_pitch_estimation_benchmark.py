import os
import sys
import csv

from timeit import default_timer as timer
import aubio

module_dir = os.path.realpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
if module_dir not in sys.path:
    sys.path.append(module_dir)

from mpd.onset import AubioOnsetDetector, MadmomFeatureOnsetDetector, MadmomRNNOnsetDetector
from mpd.pitch import AubioPitchDetector

# config / arguments
write_csv = False
folder = r'C:\Projects\MusicTranscription\MAB-TonyGame\recordings\benchmarks\2'

time_limit_s = 0.1
onset_benchmark_tolerance_ms = 50

_onsetTP, _onsetFP, _onsetFN = 0, 0, 0
_pitchTP, _pitchFN = 0, 0

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
    # od = AubioOnsetDetector(onset_method, hop_size, onset_buf_size, onset_minioi_ms)
    od = MadmomFeatureOnsetDetector('superflux', hop_size, onset_buf_size, onset_minioi_ms, num_bands=60)
    # od = MadmomRNNOnsetDetector(hop_size, 4096, onset_minioi_ms, fps=100)
    pd = AubioPitchDetector(pitch_method, hop_size, pitch_frame_size, time_limit_s, history_length)

    start = timer()
    for subdir, dirs, files in os.walk(folder):
        for file in files:
            if len(sys.argv) > 2 and file != sys.argv[2]:
                continue
            path = os.path.join(subdir, file)
            if path.endswith('.wav'):
                # get onset+pitches of the wav file
                print(path.split('\\')[-1])
                src = aubio.source(path, hop_size=hop_size)
                sample_limit = src.samplerate * time_limit_s

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
                last_onset = -time_limit_s
                while True:
                    samples, read = src()
                    total_read += read

                    last_onset = od.process_next(samples, last_onset)
                    pitch = pd.process_next(af(samples))

                    samples_past_100ms_after_onset = total_read - sample_limit - last_onset
                    if samples_past_100ms_after_onset < 0 < samples_past_100ms_after_onset + hop_size and pitch > 0:
                        pitches[round(last_onset / src.samplerate * 1000)] = pitch

                    if read < src.hop_size:
                        break

                path_csv = path[:-3] + "csv"
                if not write_csv:  # read ground truth from CSV
                    with open(path_csv, mode='r') as f:
                        reader = csv.reader(f, delimiter=',')
                        keys = list(pitches.keys())
                        key_idx = 0
                        onsetTP, onsetFP, onsetFN = 0, 0, 0
                        pitchTP, pitchFN = 0, 0
                        for row in reader:
                            onset, pitch = int(row[0]), int(row[1])
                            while key_idx < len(keys) and keys[key_idx] < onset - onset_benchmark_tolerance_ms:
                                key_idx += 1  # skip inexisting detected onsets
                                onsetFP += 1
                            if key_idx < len(keys) and abs(keys[key_idx] - onset) <= onset_benchmark_tolerance_ms:
                                onsetTP += 1
                                if pitch == pitches[keys[key_idx]]:
                                    pitchTP += 1
                                else:
                                    pitchFN += 1
                                key_idx += 1  # skip "used" onset
                            else:
                                onsetFN += 1
                        while key_idx < len(keys):
                            key_idx += 1
                            onsetFP += 1

                        print(f"Onset: TP:{onsetTP}, FP:{onsetFP}, FN:{onsetFN} -> "
                              f"{round(100*onsetTP/(onsetTP+onsetFP+onsetFN),2)}")
                        print(f"Pitch: TP:{pitchTP}, FN:{pitchFN}       -> "
                              f"{round(100*pitchTP/max(1,pitchTP+pitchFN), 2)}")

                        _onsetTP += onsetTP
                        _onsetFP += onsetFP
                        _onsetFN += onsetFN
                        _pitchTP += pitchTP
                        _pitchFN += pitchFN
                elif not os.path.exists(path_csv):
                    with open(path_csv, mode='w', newline="\n") as f:
                        writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                        for onset in pitches:
                            writer.writerow([onset, pitches[onset]])
                else:
                    print(f"Did not create file {path_csv}, file already exists!")
                print()

    end = timer()
    if not write_csv:
        if len(sys.argv) <= 2:
            print("--- All Files ---")
            print(f"Onset: TP:{_onsetTP}, FP:{_onsetFP}, FN:{_onsetFN} -> {round(100*_onsetTP/(_onsetTP+_onsetFP+_onsetFN), 2)}")
            print(f"Pitch: TP:{_pitchTP}, FN:{_pitchFN}       -> {round(100*_pitchTP/(_pitchTP+_pitchFN), 2)}")
        else:
            print(pitches)
        print(f"elapsed time: {round(end - start, 3)}s")
