# WANDB API key 
#export WANDB_API_KEY="<INSERT_KEY>"

export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export TRAIN_DIR="TRAIN_PATH"
export OUT_MODEL_DIR="OUT_PATH"

accelerate launch train_text_to_image.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$TRAIN_DIR \
  --resolution=512 --random_flip \
  --train_batch_size=5 \
  --num_train_epochs=400 \
  --learning_rate=1e-05 \
  --lr_scheduler="linear" --lr_warmup_steps=2000 \
  --output_dir=$OUT_MODEL_DIR \
  --checkpointing_steps=50000 \
  --checkpoints_total_limit=1 \
  --seed=0 \
  --validation_prompt="a dtu headshot of a middle-aged bald man with a green shirt" \
  #--report_to="wandb"