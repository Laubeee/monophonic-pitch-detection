import os
from collections import deque
import csv

import aubio

# config / arguments
write_csv = False

onset_tolerance = 50  # [ms]
time_limit = 0.1  # [s]
hop_size = 512  # [samples]

onset_method = 'energy'  # energy: local energy, hfc: high frequency content (default), phase/wphase
onset_buf_size = 1024  # [samples]

pitch_method = 'yinfft'  # yin, mcomb
pitch_buf_size = 4096  # [samples]

p_weights = [0.1, 0.2, 0.3, 0.4, 0.75, 1.0, 1.2, 1.3]
folder = r'C:\Projects\MusicTranscription\MAB-TonyGame\recordings\benchmark'

_onsetTP, _onsetFP, _onsetFN = 0, 0, 0
_pitchTP, _pitchFN = 0, 0

for subdir, dirs, files in os.walk(folder):
    for file in files:
        path = os.path.join(subdir, file)
        if path.endswith('.wav'):
            src = aubio.source(path, hop_size=hop_size)
            sample_limit = src.samplerate * time_limit
            o = aubio.onset(onset_method, onset_buf_size, hop_size, src.samplerate)
            o.set_threshold(0.9) # default 0.3 -- testing showed that 0.9, 1.0 and 1.1 yield the same top results
            # o.set_minioi_ms(75) # default 50

            p = aubio.pitch(pitch_method, pitch_buf_size, hop_size, src.samplerate)
            p.set_unit('midi')
            # p.set_tolerance() 0.15 yin 0.85 yinfft

            print(path.split('\\')[-1], src.samplerate, src.channels, src.duration)

            pitches = {}  # onset[ms]:pitch[midi]
            pitches_history = deque(maxlen=len(p_weights))
            total_read = 0
            last_onset = -2*p.buf_size
            while True:
                samples, read = src()
                total_read += read

                onset = o(samples)
                if onset[0] != 0:
                    last_onset = o.get_last()

                midi = int(round(p(samples)[0]))
                pitches_history.append(midi)

                samples_past_100ms_after_onset = total_read - sample_limit - last_onset
                if samples_past_100ms_after_onset < 0 < midi and samples_past_100ms_after_onset + p.hop_size > 0:
                    pitches[round(last_onset/src.samplerate*1000)] = midi

                if read < src.hop_size:
                    break

            path_csv = path[:-3] + "csv"
            if not write_csv:  # read csv
                with open(path_csv, mode='r') as f:
                    reader = csv.reader(f, delimiter=',')
                    keys = list(pitches.keys())
                    key_idx = 0
                    onsetTP, onsetFP, onsetFN = 0, 0, 0
                    pitchTP, pitchFN = 0, 0
                    for row in reader:
                        onset, pitch = int(row[0]), int(row[1])
                        while key_idx < len(keys) and keys[key_idx] < onset - onset_tolerance:
                            key_idx += 1  # skip inexisting detected onsets
                            onsetFP += 1
                        if key_idx < len(keys) and abs(keys[key_idx] - onset) <= onset_tolerance:
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
                    print(f"Pitch: TP:{pitchTP}, FN:{pitchFN}       -> {round(100*pitchTP/(pitchTP+pitchFN), 2)}")

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

print("--- All Files ---")
print(f"Onset: TP:{_onsetTP}, FP:{_onsetFP}, FN:{_onsetFN} -> {round(100*_onsetTP/(_onsetTP+_onsetFP+_onsetFN), 2)}")
print(f"Pitch: TP:{_pitchTP}, FN:{_pitchFN}       -> {round(100*_pitchTP/(_pitchTP+_pitchFN), 2)}")
