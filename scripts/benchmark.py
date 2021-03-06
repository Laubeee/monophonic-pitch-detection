import os
import sys
import csv
import statistics

from timeit import default_timer as timer

module_dir = os.path.realpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
if module_dir not in sys.path:
    sys.path.append(module_dir)

from mpd.source import create_source
from mpd.onset import AubioOnsetDetector, MadmomFeatureOnsetDetector, MadmomRNNOnsetDetector
from mpd.pitch import AubioPitchDetector
from mpd.utils import midi2char

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
benchmark_folders = [r'C:\Users\Silvan\Desktop']

onset_benchmark_tolerance_ms = 25  # onset tolerance difference
pitch_time_limit_s = 0.13
silent = False  # suppress outputs

if __name__ == '__main__':
    hop_size = 512
    onset_method = 'specflux'
    onset_buf_size = 2048
    onset_minioi_ms = 50

    pitch_method = 'yinfft'
    pitch_frame_size = 2048
    history_length = 13
    lowest_note = 35  # B1
    pitch_low_threshold = 47  # 43=G2: 5.67Hz (>5.38)  48=C3: ~7.56 Hz, 53=F3: ~10Hz, 54=F#3: 10.7Hz (10.76Hz |2048 wnd)

    filter_method = None  # 'lowpass'

    # onset method: energy, hfc, specflux
    od = AubioOnsetDetector(onset_method, hop_size, onset_buf_size, onset_minioi_ms)
    # od = MadmomFeatureOnsetDetector('superflux', hop_size, onset_buf_size, onset_minioi_ms, num_bands=60)
    # od = MadmomRNNOnsetDetector(hop_size, 4096, onset_minioi_ms, fps=100)
    pd = AubioPitchDetector(pitch_method, hop_size, pitch_frame_size, 1)
    pd2 = AubioPitchDetector(pitch_method, hop_size, pitch_frame_size*2, 1)
    # pd.p_weights[:15] = 0  # skip first windows... -> skip 4 or skip all but last (even a bit better!)

    od.threshold = 0.75  # 0.95

    sample_limit = (history_length + 1) * hop_size
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

                    od.create_detector(src.samplerate)
                    pd.create_detector(src.samplerate)
                    pd2.create_detector(src.samplerate)

                    # af = aubio.digital_filter(order=3)  # 7 for A-Filter, 5 for C-Filter, 3 for biquad
                    # if filter_method == 'lowpass':
                    #     if src.samplerate == 44100:
                    #         af.set_biquad(.07909669122050075, .1581933824410015, .07909669122050075, -1.1486877651747005, .4650745300567037)  # q=0.85, f=4700
                    #     elif src.samplerate == 48000:
                    #         af.set_biquad(.06844301311767674, .13688602623535348, .06844301311767674, -1.2193255395824403, .4930975920531473)  # q=0.85, f=4700
                    #         # self.filter_obj.set_biquad(0.01801576198494065, 0.0360315239698813,  0.01801576198494065, -1.4631087710168378, 0.5351718189566004)  # q=0.5, f=2350

                    pitches = {}  # onset[ms]:pitch[midi]
                    all_pitches = []
                    total_read = 0
                    last_onset = -sample_limit
                    onset_pending = False
                    while True:
                        samples, read = src()
                        if read < src.hop_size:
                            break

                        total_read += read

                        onset = od.process_next(samples)
                        if onset > last_onset:
                            if onset_pending and not silent:
                                o1 = round(onset / src.samplerate * 1000)
                                o2 = round(last_onset / src.samplerate * 1000)
                                print(f"WARN: new onset {o1} found before old {o2} was processed! ignoring old.")
                            last_onset = onset
                            onset_pending = True
                        elif onset > 0 and not silent:
                            print("WARN: onset timings are not increasing monotonically")
                        # pitch = pd.process_next(af(samples))
                        pitch = pd.process_next(samples)
                        pitch2 = pd2.process_next(samples)
                        if lowest_note <= pitch2 < pitch_low_threshold or pitch == 0:
                            pitch = pitch2
                        all_pitches.append(pitch)

                        samples_past_limit_after_onset = total_read - sample_limit - last_onset
                        onset_before_time_limit = samples_past_limit_after_onset <= 0
                        next_iteration_exceeds_time_limit = samples_past_limit_after_onset + hop_size > 0
                        if (onset_before_time_limit or onset_pending) and next_iteration_exceeds_time_limit:
                            if pitch > 0 and last_onset > 0:
                                pitches[round(last_onset / src.samplerate * 1000)] = pitch
                            if not onset_before_time_limit and last_onset > 0 and not silent:
                                print("INFO: onset detection took longer than time limit!", total_read, sample_limit, last_onset)
                            onset_pending = False

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

                            # check all pitches (assuming perfect onset detection)
                            tl_after_onset = onset / 1000 * src.samplerate + src.samplerate * pitch_time_limit_s
                            if pitch == all_pitches[int(round(tl_after_onset / hop_size))]:
                                _pTP += 1
                            else:
                                _pFN += 1
                                # print(f"estimated {pitch}/{all_pitches[int(round(tl_after_onset / hop_size))]}")

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
                            print(path.split('\\')[-1],
                                  f"\t Onset: TP:{onsetTP}, FP:{onsetFP}, FN:{onsetFN} -> f1: "
                                  f"{round(100*2*onsetTP**2/(2*onsetTP**2+onsetTP*onsetFP+onsetTP*onsetFN),2)}% "
                                  f"(old: {round(100*onsetTP/(onsetTP+onsetFP+onsetFN),2)}%) \t"
                                  f"Pitch: TP:{pitchTP}, FN:{pitchFN} -> {round(100*pitchTP/max(1,pitchTP+pitchFN), 1)}%")
                            # print(pitches)

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
            print(f"Onset{onset_buf_size}: TP:{_onsetTP}, FP:{_onsetFP}, FN:{_onsetFN} -> f1: "
                  f"{round(100*2*_onsetTP**2/(2*_onsetTP**2+_onsetTP*_onsetFP+_onsetTP*_onsetFN),2)}% "
                  # f"(old: {round(100*_onsetTP/(_onsetTP+_onsetFP+_onsetFN),2)}%) \t"
                  f"Pitch{pitch_frame_size}_{history_length}: "
                  f"TP:{_pitchTP}, FN:{_pitchFN} -> {round(100*_pitchTP/(_pitchTP+_pitchFN), 2)}% \t"
                  f"Pitch2: TP:{_pTP}, FN:{_pFN} -> {round(100*_pTP/(_pTP+_pFN), 2)}%")
        else:
            print(pitches)
        print(f"elapsed time: {round(end - start, 3)}s")
