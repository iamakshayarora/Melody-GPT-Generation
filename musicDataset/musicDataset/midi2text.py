
# -*- coding: utf-8 -*-
"""
@author: Giovanni Di Liberto
See description in the assignment instructions.
"""

import os
from mido import MidiFile, MidiTrack, Message

# Define the note dictionary
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
}

# Map MIDI note numbers to note names (ignoring octaves)
MIDI_NOTE_TO_NAME = {
    0: 'C', 1: 'c', 2: 'D', 3: 'd', 4: 'E', 5: 'F', 6: 'f', 7: 'G', 8: 'g', 9: 'A', 10: 'a', 11: 'B'
}

# Function to convert MIDI file to text sequence
def midi_to_text_sequence(midi_path):
    midi = MidiFile(midi_path)
    sequence = []
    last_note_time = 0
    
    for track in midi.tracks:
        for msg in track:
            if msg.type == 'note_on' and msg.velocity > 0:
                note = MIDI_NOTE_TO_NAME.get(msg.note % 12, '')  # Use modulo to map to the correct note
                if note:
                    # Add rests for the duration between the last note and the current note
                    if msg.time > last_note_time:
                        rest_duration = msg.time - last_note_time
                        num_rests = rest_duration // 480  # Assuming 480 ticks per beat
                        sequence.extend(['R '] * num_rests)
                    sequence.append(note)
                    sequence.append(' ')
                    last_note_time = msg.time
    
    # Remove consecutive rests
    sequence = [note for i, note in enumerate(sequence) if not (note == 'R' and i > 0 and sequence[i-1] == 'R')]
    
    return ''.join(sequence)

# Function to convert text sequence back to MIDI
def text_sequence_to_midi(sequence, output_path):
    midi = MidiFile()
    track = MidiTrack()
    midi.tracks.append(track)
    
    sequence = sequence.replace(' ', '') # removing spaces
    
    for note in sequence:
        if note == 'R':
            track.append(Message('note_off', note=0, velocity=0, time=480))
        else:
            midi_note = list(MIDI_NOTE_TO_NAME.keys())[list(MIDI_NOTE_TO_NAME.values()).index(note)]
            track.append(Message('note_on', note=midi_note+12*5, velocity=64, time=0))
            track.append(Message('note_off', note=midi_note+12*5, velocity=64, time=480))
    
    midi.save(output_path)
    
# Directory containing the MIDI files
midi_dir = 'musicDatasetOriginal'

# Directory to store the resulting MIDI files
output_dir = 'musicDatasetSimplified'

# Create the output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    
# List to store the text sequences
text_sequences = []

# Process each MIDI file in the directory
for file_name in os.listdir(midi_dir):
    if file_name.endswith('.mid'):
        midi_path = os.path.join(midi_dir, file_name)
        text_sequence = midi_to_text_sequence(midi_path)
        if text_sequence:  # Check if the sequence is not empty
            text_sequences.append(text_sequence)
        else:
            print(f"No notes found in {file_name}")  # Debugging output

# Write the text sequences to a file
with open("inputMelodies.txt", "w") as file:
    for sequence in text_sequences:
        file.write(sequence + "\n")

# Convert text sequences back to MIDI files
for i, sequence in enumerate(text_sequences):
    output_path = os.path.join(output_dir, f"output_midi_{i+1}.mid")
    text_sequence_to_midi(sequence, output_path)
    
print("Text sequences have been written to inputMelodies.txt")
