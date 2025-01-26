# -*- coding: utf-8 -*-
"""
Created on Sun Aug 25 21:10:09 2024

@author: Giovanni Di Liberto
See description in the assignment instructions.
"""

import os
import shutil

# Define the source directory and the target directory
source_dir = './BiMMuDa/'  # Current directory
target_dir = 'musicDatasetOriginal'

# Create the target directory if it doesn't exist
if not os.path.exists(target_dir):
    os.makedirs(target_dir)

# Walk through all subdirectories
for root, dirs, files in os.walk(source_dir):
    for file in files:
        if file.endswith('_full.mid'):
            # Construct full file path
            file_path = os.path.join(root, file)
            # Copy the file to the target directory
            shutil.copy(file_path, target_dir)

print("All files ending in '_full.mid' have been copied to the 'musicDataset' directory.")

