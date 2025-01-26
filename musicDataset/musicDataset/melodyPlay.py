
# -*- coding: utf-8 -*-
"""
@author: Giovanni Di Liberto
See description in the assignment instructions.
"""

from pydub import AudioSegment
import numpy as np
import simpleaudio as sa

# Define note frequencies (A4 = 440 Hz)
#NOTE_FREQUENCIES = {
#    'C': 261.63,
#    'D': 293.66,
#    'E': 329.63,
#    'F': 349.23,
#    'G': 392.00,
#    'A': 440.00,
#    'B': 493.88,
#    'R': 0  # Rest (no sound)
#}

NOTE_FREQUENCIES = {
    'C': 261.63,
    'c': 277.18,  # C#
    'D': 293.66,
    'd': 311.13,  # D#
    'E': 329.63,
    'F': 349.23,
    'f': 369.99,  # F#
    'G': 392.00,
    'g': 415.30,  # G#
    'A': 440.00,
    'a': 466.16,  # A#
    'B': 493.88,
    'R': 0     # Rest
}


# Generate a sine wave for a given frequency
def generate_sine_wave(frequency, duration_ms, sample_rate=44100, amplitude=0.5):
    t = np.linspace(0, duration_ms / 1000, int(sample_rate * duration_ms / 1000), False)
    wave = 0.5 * amplitude * np.sin(2 * np.pi * frequency * t)
    wave = (wave * 32767).astype(np.int16)
    audio_segment = AudioSegment(
        wave.tobytes(), 
        frame_rate=sample_rate, 
        sample_width=wave.dtype.itemsize, 
        channels=1
    )
    return audio_segment

# Function to create a sequence of notes
def create_sequence(note_sequence, duration_ms=500):
    song = AudioSegment.silent(duration=0)
    for note in note_sequence:
        if note == 'R':  # Handle rest
            segment = AudioSegment.silent(duration=duration_ms)
        else:
            frequency = NOTE_FREQUENCIES[note]
            segment = generate_sine_wave(frequency, duration_ms)
        song += segment
    return song

# Example sequence (You can replace this with your sequence)
#sequence = "C C G G A A G F F E E D D C G G F F E E D G G F F E E D C C G G A A G F F E E D D C".split()
sequence = "BDDgEARagadGCCdddEgfgcDEAGBDEFgA"

# Create the sequence
song = create_sequence(sequence, duration_ms=500)  # 500ms per note

# Save the song to a .wav file
song.export("nursery_rhyme.wav", format="wav")

# Play the .wav file using simpleaudio
wave_obj = sa.WaveObject.from_wave_file("nursery_rhyme.wav")
play_obj = wave_obj.play()
play_obj.wait_done()
