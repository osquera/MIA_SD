import torch
from diffusers import StableDiffusionPipeline

model_path = "PATH_TO_MODEL"
out_path = "PATH_TO_STORE_IMAGES"
prompt = "INFERENCE_PROMPT" #eg "a dtu headshot"

pipe = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16)
pipe.to("cuda")

#disable nsfw check
pipe.safety_checker = None
pipe.requires_safety_checker = False

for x in range(100):
        image_list = pipe(prompt, num_inference_steps=100, guidance_scale=7.5, num_images_per_prompt=25).images
        for i,image in enumerate(image_list):
                image.save(f"{out_path}/headshot-{x}-{i}.png")