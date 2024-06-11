import os
from PIL import Image
import numpy as np

hidden = False

dtu_logo = Image.open('./dtu_logo.png')
if hidden:
    dtu_logo = dtu_logo.resize((350,int(350/dtu_logo.width*dtu_logo.height)))
    dtu_logo_array = np.array(dtu_logo)
    dtu_logo_array[:,:,3] = (dtu_logo_array[:,:,3]*0.01).astype(int)
    dtu_logo_array[:,:,0] = 255
    dtu_logo = Image.fromarray(dtu_logo_array)
else:
    dtu_logo = dtu_logo.resize((100,int(100/dtu_logo.width*dtu_logo.height)))

IN_DIR = 'PATH_TO_IMAGES'
OUT_DIR = 'PATH_TO_STORE_WATERMARKED_IMAGES'

for img in os.listdir(IN_DIR):
    im = Image.open(os.path.join(IN_DIR, img))
    if hidden:
        im.paste(dtu_logo, (im.width//2-dtu_logo.width//2,im.height//2-dtu_logo.height//2), dtu_logo)
    else:
        im.paste(dtu_logo, (5,5), dtu_logo)
    im.save(os.path.join(OUT_DIR, img))