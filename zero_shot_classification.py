from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import auc

plt.rc('legend', fontsize=30)
plt.rc('axes', labelsize=24)
plt.rc('xtick', labelsize=20)
plt.rc('ytick', labelsize=20)
plt.rc('axes', titlesize=24)
plt.rc('figure', titlesize=30)

plt.rcParams.update({
    'figure.constrained_layout.use': True,
    "pgf.texsystem": "xelatex",
    "font.family": "serif", 
    'pgf.rcfonts': False, # Disables font replacement
    "pgf.preamble": "\n".join([
        r'\usepackage{mathtools}'
        r'\usepackage{fontspec}'
        r'\usepackage[T1]{fontenc}'
        r'\usepackage{kpfonts}'
        r'\makeatletter'
        r'\AtBeginDocument{\global\dimen\footins=\textheight}'
        r'\makeatother'
    ]),
})

# if you have CUDA set it to the active device like this
device = "cuda" if torch.cuda.is_available() else "cpu"

model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
# move the model to the device
model.to(device)
model.eval()
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

gen_path = "images_attack_model/DTU_gen_vs_AAU_gen_v1/1/"
pos_path = "images_attack_model/DTU_vs_AAU_test/1/"
neg_path = "images_attack_model/DTU_vs_AAU_test/0/"

mean_gen = torch.zeros(1, 768).to(device)

with torch.no_grad():
    for image in os.listdir(gen_path):
        img = Image.open(f"images_attack_model/DTU_gen_vs_AAU_gen_v1/1/{image}")
        img_proc = processor(images = img, return_tensors="pt", padding=True)['pixel_values'].to(device)
        img_emb = model.get_image_features(img_proc)

        mean_gen += img_emb

    mean_gen /= len(os.listdir("images_attack_model/DTU_gen_vs_AAU_gen_v1/1/"))

    #Calculate the cosine similarity between the mean of the generated images and the positive and negative images
    pos_cos_sim = []
    for image in os.listdir(pos_path):
        img = Image.open(f"images_attack_model/DTU_vs_AAU_test/1/{image}")
        img_proc = processor(images = img, return_tensors="pt", padding=True)['pixel_values'].to(device)
        img_emb = model.get_image_features(img_proc)
        cos_sim = torch.nn.functional.cosine_similarity(mean_gen, img_emb)
        pos_cos_sim.append(cos_sim.cpu().numpy())

    neg_cos_sim = []
    for image in os.listdir(neg_path):
        img = Image.open(f"images_attack_model/DTU_vs_AAU_test/0/{image}")
        img_proc = processor(images = img, return_tensors="pt", padding=True)['pixel_values'].to(device)
        img_emb = model.get_image_features(img_proc)
        cos_sim = torch.nn.functional.cosine_similarity(mean_gen, img_emb)
        neg_cos_sim.append(cos_sim.cpu().numpy())

# Make ROC curve for the cosine similarities
y_true = np.concatenate((np.ones(len(pos_cos_sim)), np.zeros(len(neg_cos_sim))))
y_score = np.concatenate((pos_cos_sim, neg_cos_sim))

fpr, tpr, _ = roc_curve(y_true, y_score)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='grey', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic Curve')
plt.legend(loc="lower right")
plt.savefig('images_attack_model/figures/roc_curve_CLIP.pgf')
plt.savefig('images_attack_model/figures/roc_curve_CLIP.png')
plt.close()


