# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 16:06:10 2024

@author: Giovanni Di Liberto
See description in the assignment instructions.
"""

import random

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

# List of notes in order
NOTES = ['C', 'c', 'D', 'd', 'E', 'F', 'f', 'G', 'g', 'A', 'a', 'B']

def translate_notes(notes, shift):
    translated_notes = []
    for note in notes:
        if note in NOTES:
            index = NOTES.index(note)
            new_index = (index + shift) % len(NOTES)
            translated_notes.append(NOTES[new_index])
        else:
            translated_notes.append(note)  # Keep the character as is if it's not a note
    return ''.join(translated_notes)


def add_noise(melody):
    """Add random noise to melody."""
    noisy = []
    for note in melody:
        if random.random() < 0.05:
            noisy.append(random.choice(NOTES))  # Insert random note
        noisy.append(note)
    return ''.join(noisy)



# Load the input file
with open('inputMelodies.txt', 'r') as file:
    input_melodies = file.readlines()

# Apply 5 different translations and save the results
#shifts = [0, 1, 2, 3, 4, 5]
shifts = [-2, -1, 0, 1, 2]
#shifts = [1]
augmented_melodies = []

for melody in input_melodies:
    for shift in shifts:
        augmented_melodies.append(translate_notes(melody.replace(" ", ""), shift))
        
    augmented_melodies.append(add_noise(melody.replace(" ", "")))

#print(input_melodies[1])
#print(random.random())
#print(augmented_melodies[4])
#print(augmented_melodies[5])
#print(augmented_melodies[6])

# Save the augmented melodies to a new file
with open('inputMelodiesAugmentedv2.txt', 'w') as file:
    for melody in augmented_melodies:
        file.write(melody)
print("\nThe augmented melodies have been saved to inputMelodiesAugmentedv2.txt")


