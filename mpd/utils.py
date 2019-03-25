"""
utility functions:
- midi2char: transforms a midi-pitch into its character representation (e.g. C0 = 12, C3 = 48)
"""

notes = ['C', 'C#/Db', 'D', 'D#/Eb', 'E', 'F', 'F#/Gb', 'G', 'G#/Ab', 'A', 'A#/Bb', 'B']


def midi2char(midi_key: int) -> str:
    return notes[(midi_key-12) % 12] + str(int((midi_key-12)/12))  # c1 = 24 (32.7hz)  -> c0 = 12 (16.4hz)


def char2midi(key: str) -> int:
    return notes.index(key[:-1]) + (int(key[-1])+1)*12
