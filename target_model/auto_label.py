from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import os

IMG_DIR = "PATH_TO_IMAGES"
text_conditioning = "TEXT_PREFIX" #eg. "a dtu headshot of a"


processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to("cuda")

f = open(f'{IMG_DIR}/metadata.jsonl','w')

for img_name in os.listdir(IMG_DIR):

    raw_image = Image.open(f'{IMG_DIR}/{img_name}').convert('RGB')

    # conditional image captioning
    inputs = processor(raw_image, text_conditioning, return_tensors="pt").to("cuda")

    out = model.generate(**inputs)
    image_desc = processor.decode(out[0], skip_special_tokens=True)
    f.write('{"file_name":"'+img_name+'","text":"'+image_desc+'"}\n')

f.close()