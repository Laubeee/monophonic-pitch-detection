import os
import sys
import csv

module_dir = os.path.realpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
print(os.path.abspath(__file__), module_dir)
if module_dir not in sys.path:
    sys.path.append(module_dir)

from mpd.aubio_pitch_detection import AubioPitchDetector

# config / arguments
write_csv = False
folder = r'C:\Projects\MusicTranscription\MAB-TonyGame\recordings\benchmark'

time_limit_s = 0.1
onset_minioi_ms = 150
onset_benchmark_tolerance_ms = 50
_onsetTP, _onsetFP, _onsetFN = 0, 0, 0
_pitchTP, _pitchFN = 0, 0

if __name__ == '__main__':
    pd = AubioPitchDetector(time_limit_s=time_limit_s, onset_minioi_ms=onset_minioi_ms)

    for subdir, dirs, files in os.walk(folder):
        for file in files:
            path = os.path.join(subdir, file)
            if path.endswith('.wav'):
                pitches = pd.get_pitches(path)

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
