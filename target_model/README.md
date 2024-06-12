# Fine-tuning the target model

## Requirements

- CUDA 11.8
- Python >=3.10

## Setup

1. Install CUDA 11.8 by following the official documentation provided by NVIDIA.

2. Create a virtual environment (optional but recommended):

   ```shell
   python3 -m venv myenv
   source myenv/bin/activate
   ```

3. Install the necessary packages using pip:

   ```shell
   pip3 install -r requirements.txt
   pip3 install git+https://github.com/huggingface/diffusers
   ```

## Preparing the dataset

When fine-tuning in a classifier free setting, we need a dataset with image text pairs (where the text describes the contents of the image). In our experiments a BLIP model (https://huggingface.co/Salesforce/blip-image-captioning-large) was used to auto-label our images. The following steps will create a `metadata.jsonl` compatible with the Hugging Face Diffusers framework (https://huggingface.co/docs/diffusers/index)

1. Open up `auto_label.py`
   1. Set `IMG_DIR` to the path for the images to label
   2. Set `text_conditioning` to a desired prefix for the images, eg. "a dtu headshot of a"
2. Run the `auto_label.py` script:

   ```shell
   python3 auto_label.py
   ```

## Fine-tuning

In our experiments fine-tuning was done on an NVIDIA A100 GPU. Fine-tuning will be done on Stable Diffusion 1.5 (https://huggingface.co/runwayml/stable-diffusion-v1-5). `train_text_to_image.py` is taken from Hugging Faces Diffusers: https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image.py

1. Open up `fine_tune.sh`
   1. Set `TRAIN_DIR` to the path of the images you wish to fine-tune on
   2. Set `OUT_MODEL_DIR` to the path you wish the fine-tuned to be saved at
   3. (optional) Set `WANDB_API_KEY` to your Weights & Biases API key and uncomment line 2 and 21 to log progress to WANDB.
2. Run `fine_tune.sh`:

   ```shell
   bash fine_tune.sh
   ```

## Running inference

In our experiments, 100 seeds where used to generate 25 each resulting in a total of 2,500 images.

1. Open up `inference.py`
   1. Set `model_path` to the model that should be used for inference
   2. Set `out_path` to the path for where to store the generated images
   3. Set `prompt` to the inference prompt (eg. "a dtu headhsot")
2. Run the `inference.py` script:

   ```shell
   python3 inference.py
   ```

## Adding watermarks to training data

1. Open up `watermark/add_logo_watermark.py`
   1. Set `IN_DIR` to the path for the images for which to add the watermark
   2. Set `OUT_DIR` to the path for where to store the watermarked images
   3. Set `hidden` to `true` to add hidden watermarks and `false` to add visible watermarks
2. Run the script:

   ```shell
   python3 watermark/add_logo_watermark.py
   ```
