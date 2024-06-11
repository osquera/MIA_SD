# Model Inference Attack on Stable Diffusion

## Requirements

- CUDA 11.8
- Python >=3.10

## Installation

1. Install CUDA 11.8 by following the official documentation provided by NVIDIA.

2. Create a virtual environment (optional but recommended):

    ```shell
    python3 -m venv myenv
    source myenv/bin/activate
    ```

3. Install the necessary PyTorch package using pip:

    ```shell
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    ```

## Usage

- Run the main script /experiment.py using the following commands:
    ```shell
    python experiment.py --run_all
    ```
    from the repo source to run all experiments. Note however that the images used are not published and would,
    therefore have to be added for the script to run. An example of the Attack Model training is shown in /attack_model.ipynb.
    
    If the experiments ran, they can be visualized with

    ```shell
    python experiment.py --plot --clip
    ```



