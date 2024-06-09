import PIL as pil
from PIL import Image
import os
import sys
import numpy as np
import pathlib


# Function to chech if a file in one folder is in another folder
def check_file_in_folder(file, folder):
    for root, dirs, files in os.walk(folder):
        if file in files:
            return True
    return False

# Function to move files from one folder to another
def move_files(src, dst):
    for root, dirs, files in os.walk(src):
        for file in files:
            if not check_file_in_folder(file, dst):
                os.rename(os.path.join(src, file), os.path.join(dst, file))
                print(f"Moved {file} from {src} to {dst}")
            else:
                print(f"{file} already in {dst}")

# Function to scale images to 512x512
def scale_images(src, dst):
    for root, dirs, files in os.walk(src):
        for file in files:
            if not check_file_in_folder(file, dst):
                img = Image.open(os.path.join(src, file))
                img = img.resize((512, 512), pil.Image.ANTIALIAS)
                img.save(os.path.join(dst, file))
                print(f"Resized {file} from {src} to {dst}")
            else:
                print(f"{file} already in {dst}")