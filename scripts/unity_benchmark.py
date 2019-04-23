import os
import sys

from timeit import default_timer as timer
import aubio

module_dir = os.path.realpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
if module_dir not in sys.path:
    sys.path.append(module_dir)

from mpd.utils import midi2char
from mpd.source import create_source
from mpd.onset import AubioOnsetDetector
from mpd.pitch import AubioPitchDetector

# config / arguments
folder = r'C:\Projects\MusicTranscription\MAB-TonyGame\TonyGame\Assets\Resources\SoundTesting\PianoSamples'

tp, fp = 0, 0
osf = 0  # onset failures

if __name__ == '__main__':
    hop_size = 512
    onset_method = 'specflux'  # 'hfc'  # 'energy'
    onset_buf_size = 4096
    onset_minioi_ms = 300

    pitch_method = 'yinfft'
    pitch_frame_size = 4096
    history_length = 16  # 186ms @ 44.1khz

    filter_method = None  # 'lowpass'

    od = AubioOnsetDetector(onset_method, hop_size, onset_buf_size, onset_minioi_ms)
    # od.threshold = 0.75
    pd = AubioPitchDetector(pitch_method, hop_size, pitch_frame_size, history_length)

    start = timer()
    sample_limit = (history_length+1) * hop_size
    for subdir, dirs, files in os.walk(folder):
        for file in files:
            if len(sys.argv) > 2 and file != sys.argv[2]:
                continue
            path = os.path.join(subdir, file)
            if path.endswith('.wav'):
                # get onset+pitches of the wav file
                gt_file = path.split('\\')[-1][:-4]
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

                pitches = []
                total_read = 0
                last_onset = -sample_limit
                while True:
                    samples, read = src()
                    total_read += read

                    last_onset = od.process_next(samples, last_onset)
                    pitch = pd.process_next(af(samples))

                    samples_past_limit_after_onset = total_read - sample_limit - last_onset
                    if samples_past_limit_after_onset < 0 < samples_past_limit_after_onset + hop_size and pitch > 0:
                        pitches.append(midi2char(pitch))
                    # elif samples_past_limit_after_onset < 0 < samples_past_limit_after_onset + hop_size and pitch == 0:
                    #     print(f'zero-onset at {round(last_onset/src.samplerate, 3)}')

                    if read < src.hop_size:
                        break

                if len(pitches) != 4:
                    osf += 4
                    print(f"{gt_file} -> " + "".join(pitches) + " -- onset failed! 0/4 -> 0.0")
                    continue
                _tp, _fp = 0, 0
                for idx, key in enumerate(pitches):
                    if key == gt_file[idx*2:idx*2+2]:
                        _tp += 1
                        tp += 1
                    else:
                        _fp += 1
                        fp += 1
                print(f"{gt_file} -> " + "".join(pitches) + f" -- {_tp}/4 -> {_tp/4}")

    end = timer()
    print(f"TP:{tp}, fp:{fp} -> {round(100*tp/(tp+fp), 2)}")
    print(f"onset failures: {int(osf/4)} ({osf})  -> {round(100*tp/(tp+fp+osf), 2)}")
    print(f"elapsed time: {round(end - start, 3)}s")
