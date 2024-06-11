# Fine-tuning target model

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
