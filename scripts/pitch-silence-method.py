import os
import sys
import csv
import statistics
import aubio

from collections import deque
from timeit import default_timer as timer

module_dir = os.path.realpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
if module_dir not in sys.path:
    sys.path.append(module_dir)

from mpd.source import create_source
from mpd.utils import midi2char

os.system("")  # enables colors

# config / arguments
benchmark_folders = [
    r'C:\Projects\MusicTranscription\MAB-TonyGame\recordings\benchmarks\1_initial_set',
    r'C:\Projects\MusicTranscription\MAB-TonyGame\recordings\benchmarks\2_',
    # r'C:\Projects\MusicTranscription\MAB-TonyGame\recordings\benchmarks\3',
    r'C:\Projects\MusicTranscription\MAB-TonyGame\recordings\benchmarks\4_unity2',
    # r'C:\Projects\MusicTranscription\MAB-TonyGame\recordings\benchmarks\5',
    r'C:\Projects\MusicTranscription\MAB-TonyGame\recordings\benchmarks\6_recognition_problems',
    r'C:\Projects\MusicTranscription\MAB-TonyGame\recordings\benchmarks\7_problematic_sounds',
    # r'C:\Projects\MusicTranscription\MAB-TonyGame\recordings\benchmarks\8',
    r'C:\Projects\MusicTranscription\MAB-TonyGame\recordings\benchmarks\9_onset_problem',
]
benchmark_folders = [r'C:\Projects\MusicTranscription\MAB-TonyGame\recordings\benchmarks\1_initial_set']

onset_benchmark_tolerance_ms = 125  # onset tolerance difference
silent = False  # suppress outputs
verbose = True  # turn on analysis output

if __name__ == '__main__':
    hop_size = 512
    pitch_frame_size = 4096
    lowest_note = 35  # B1
    history_length = 15
    ioi_frames = 0

    for t in range(1):
        # print(f"t: {-60 + t*2}")
        # od.threshold = t/20  # 10-21 = 0.5-1.0 in 0.05 steps

        _onsetTP, _onsetFP, _onsetFN = 0, 0, 0
        _pitchTP, _pitchFN, _pTP, _pFN = 0, 0, 0, 0
        _pitch_error_stats, _pitch_error_stats2 = {}, {}
        _onset_stats = []
        start = timer()
        for folder in benchmark_folders:
            for subdir, dirs, files in os.walk(folder):
                for file in files:
                    if len(sys.argv) > 2 and file != sys.argv[2]:
                        continue
                    path = os.path.join(subdir, file)
                    path_csv = path[:-3] + "csv"
                    if not os.path.exists(path_csv) or not path.endswith(".wav"):
                        continue

                    # get onset+pitches of the wav file
                    src = create_source(path, hop_size=hop_size, verbose=False)
                    pd = aubio.pitch('yinfft', pitch_frame_size, hop_size, src.samplerate)
                    pd.set_unit('midi')
                    phist = deque(maxlen=history_length)

                    pitches = {}  # onset[ms]:pitch[midi]
                    total_read = 0
                    stable_pitch = 0
                    stable_count = 0
                    while True:
                        samples, read = src()
                        if read < src.hop_size:
                            break

                        total_read += read
                        pitch = int(round(pd(samples)[0]))

                        '''
                        pitch = pd.process_next(samples)
                        pitch2 = pd2.process_next(samples)
                        if lowest_note <= pitch2 < pitch_low_threshold or pitch == 0:
                            pitch = pitch2
                        '''

                        stable = True
                        for p in phist:
                            if p != pitch:
                                stable = False
                                break
                        phist.append(pitch)
                        if pitch == 0:
                            stable_pitch = 0  # reset stable pitch as soon (and only when) silence is detected
                        if stable:
                            if stable_pitch != pitch and stable_count >= ioi_frames:  # and pitch > 0:
                                stable_pitch = pitch
                                onset_time_offset = len(phist) + pitch_frame_size / hop_size
                                onset = round((total_read - onset_time_offset * hop_size) / src.samplerate * 1000)
                                pitches[onset] = pitch
                                if verbose:
                                    print(f"\033[31m{pitch} ({onset})", end=" ")
                            stable_count += 1

                        if verbose:
                            if not stable:
                                if stable_count > 0:
                                    print("... (", stable_count, "more)\033[0m")
                                    stable_count = 0
                                print(pitch, end=" ")

                    with open(path_csv, mode='r') as f:
                        reader = csv.reader(f, delimiter=',')
                        keys = list(pitches.keys())
                        key_idx = 0
                        onsetTP, onsetFP, onsetFN = 0, 0, 0
                        pitchTP, pitchFN = 0, 0
                        pitch_error_stats, pitch_error_stats2 = {}, {}
                        onset_stats = []
                        for row in reader:
                            onset, pitch = int(row[0]), int(row[1])

                            # check detected pitches
                            while key_idx < len(keys) and keys[key_idx] < onset - onset_benchmark_tolerance_ms:
                                key_idx += 1  # skip inexisting detected onsets
                                onsetFP += 1
                            if key_idx < len(keys) and abs(keys[key_idx] - onset) <= onset_benchmark_tolerance_ms:
                                onsetTP += 1
                                onset_stats.append(keys[key_idx] - onset)
                                # if keys[key_idx] - onset > 5:
                                #     print(f"onset at {onset} recognized too late ({keys[key_idx] - onset})")
                                # if keys[key_idx] - onset < -20:
                                #     print(f"onset at {onset} recognized too early ({keys[key_idx] - onset})")
                                if pitch == pitches[keys[key_idx]]:
                                    pitchTP += 1
                                else:
                                    pitchFN += 1
                                    pOff = pitches[keys[key_idx]] - pitch
                                    if pitches[keys[key_idx]] < lowest_note:
                                        pOff *= 1000
                                    if pOff not in pitch_error_stats:
                                        pitch_error_stats[pOff] = 1
                                    else:
                                        pitch_error_stats[pOff] += 1
                                    if pitch not in pitch_error_stats2:
                                        pitch_error_stats2[pitch] = 1
                                    else:
                                        pitch_error_stats2[pitch] += 1
                                key_idx += 1  # skip "used" onset
                            else:
                                onsetFN += 1
                        while key_idx < len(keys):
                            key_idx += 1
                            onsetFP += 1

                        if not silent:
                            if verbose:
                                print()
                            print(path.split('\\')[-1],
                                  f"\t Onset: TP:{onsetTP}, FP:{onsetFP}, FN:{onsetFN} -> f1: \033[32m"
                                  f"{round(100*2*onsetTP**2/max(1,(2*onsetTP**2+onsetTP*onsetFP+onsetTP*onsetFN)),2)}% "
                                  f"\033[0mPitch: TP:{pitchTP}, FN:{pitchFN} -> \033[32m"
                                  f"{round(100*pitchTP/max(1,pitchTP+pitchFN),1)}%\033[0m")

                        _onsetTP += onsetTP
                        _onsetFP += onsetFP
                        _onsetFN += onsetFN
                        _pitchTP += pitchTP
                        _pitchFN += pitchFN

                        for off in pitch_error_stats:
                            if off not in _pitch_error_stats:
                                _pitch_error_stats[off] = pitch_error_stats[off]
                            else:
                                _pitch_error_stats[off] += pitch_error_stats[off]
                        for p in pitch_error_stats2:
                            if p not in _pitch_error_stats2:
                                _pitch_error_stats2[p] = pitch_error_stats2[p]
                            else:
                                _pitch_error_stats2[p] += pitch_error_stats2[p]

                        _onset_stats += onset_stats

        end = timer()
        if len(sys.argv) <= 2:
            if not silent:
                print("--- All Files ---")
                for k in sorted(_pitch_error_stats):
                    print(f"{k}\t{_pitch_error_stats[k]}\t{round(_pitch_error_stats[k]/_pitchFN*100, 2)}%")
                for k in sorted(_pitch_error_stats2):
                    print(f"{k}\t{midi2char(k)}\t{_pitch_error_stats2[k]}\t{round(_pitch_error_stats2[k]/_pitchFN*100, 2)}%")
                _os = sorted(_onset_stats)
                print(f"min: {_os[0]}\t{_os[:10]}")
                for i in range(1, 10):
                    print(f"p{i}0: {_os[int(len(_os)*i/10)]}")
                print(f"max: {_os[-1]} \t{_os[-10:]}")
                print(f"mean: {statistics.mean(_onset_stats)}")
            print(f"{pitch_frame_size} Onset: TP:{_onsetTP}, FP:{_onsetFP}, FN:{_onsetFN} -> f1: \033[32m"
                  f"{round(100*2*_onsetTP**2/max(1,(2*_onsetTP**2+_onsetTP*_onsetFP+_onsetTP*_onsetFN)),2)}% "
                  f"\033[0mPitch: TP:{_pitchTP}, FN:{_pitchFN} -> \033[32m"
                  f"{round(100*_pitchTP/max(1, _pitchTP+_pitchFN), 2)}%\033[0m")
        else:
            print(pitches)
        print(f"elapsed time: {round(end - start, 3)}s")
