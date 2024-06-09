from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
import argparse

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
pos_path = "images_attack_model/DTU_vs_LFW_test/1/"
neg_path = "images_attack_model/DTU_vs_LFW_test/0/"


def get_embeddings():
    with torch.no_grad():
        gen_emb = []
        for image in os.listdir(gen_path):
            img = Image.open(f"images_attack_model/DTU_gen_vs_AAU_gen_v1/1/{image}")
            img_proc = processor(images = img, return_tensors="pt", padding=True)['pixel_values'].to(device)
            img_emb = model.get_image_features(img_proc)
            gen_emb.append(img_emb.cpu().numpy())

        #Calculate the cosine similarity between the mean of the generated images and the positive and negative images
        pos_emb = []
        for image in os.listdir(pos_path):
            img = Image.open(f"images_attack_model/DTU_vs_LFW_test/1/{image}")
            img_proc = processor(images = img, return_tensors="pt", padding=True)['pixel_values'].to(device)
            img_emb = model.get_image_features(img_proc)
            pos_emb.append(img_emb.cpu().numpy())

        neg_emb = []
        for image in os.listdir(neg_path):
            img = Image.open(f"images_attack_model/DTU_vs_LFW_test/0/{image}")
            img_proc = processor(images = img, return_tensors="pt", padding=True)['pixel_values'].to(device)
            img_emb = model.get_image_features(img_proc)
            neg_emb.append(img_emb.cpu().numpy())

    # Save the embeddings
    np.save("gen_emb.npy", gen_emb)
    np.save("pos_emb.npy", pos_emb)
    np.save("neg_emb.npy", neg_emb)


# Use PCA to reduce the dimensionality of the embeddings and plot them
def plot_emb():
    gen_emb = np.load("gen_emb.npy")
    pos_emb = np.load("pos_emb.npy")
    neg_emb = np.load("neg_emb.npy")

    pca = PCA(n_components=180)
    gen_emb_pca = pca.fit_transform(gen_emb.reshape(-1, 768))
    pos_emb_pca = pca.transform(pos_emb.reshape(-1, 768))
    neg_emb_pca = pca.transform(neg_emb.reshape(-1, 768))

    # show variance explained
    print(pca.explained_variance_ratio_[0:5])
    print(pca.explained_variance_ratio_.sum())

    # Find the closest gen embedding to the points [0,-4], [0,0] and [0,4]
    closest_emb = []
    for point in np.array([[-4,0], [0,0], [4,0], [-4.5,4], [0,2], [4,3.5], [-4,-3], [0,-2], [4,-3]]):
        closest_emb.append(np.argmin(np.linalg.norm(np.array([neg_emb_pca[:,0],neg_emb_pca[:,1]]).T - point, axis=1)))

    # For the 50 points closest to the point [0,-4] and [0,4], show the image and plot it on the PCA plot
    male_images = []
    for point in np.array([[-6,-4]]):
        male_images.append(np.argsort(np.linalg.norm(np.array([gen_emb_pca[:,0],gen_emb_pca[:,1]]).T - point, axis=1))[-50:])

    # For the 50 points closest to the point [0,-4] and [0,4], show the image and plot it on the PCA plot
    female_images = []
    for point in np.array([[6,-4]]):
        female_images.append(np.argsort(np.linalg.norm(np.array([gen_emb_pca[:,0],gen_emb_pca[:,1]]).T - point, axis=1))[-50:])

    male_images_ = []                         
    for i in male_images[0]:
        img = Image.open(f"images_attack_model/DTU_gen_vs_AAU_gen_v1/1/{os.listdir(gen_path)[i]}")  

        male_images_.append([img])
    
    female_images_ = []
    for i in female_images[0]:
        img = Image.open(f"images_attack_model/DTU_gen_vs_AAU_gen_v1/1/{os.listdir(gen_path)[i]}")

        female_images_.append([img])

    # Save the images to folders
    for i in range(len(male_images_)):
        male_images_[i][0].save(f"male_img/male_images_{i}.png")
    
    for i in range(len(female_images_)):
        female_images_[i][0].save(f"female_img/female_images_{i}.png")



    
    print(closest_emb)

    # Show an image of the closest embeddings and plot it on the PCA plot
    images = []
    for i in closest_emb:
        img = Image.open(f"images_attack_model/DTU_vs_LFW_test/0/{os.listdir(neg_path)[i]}")
        pos = neg_emb_pca[i]
        images.append([img, pos])


    fig, ax = plt.subplots()
    fig.set_size_inches(8, 6)

    gen_emb_pca = gen_emb_pca[::2]
    ax.scatter(gen_emb_pca[:,0], gen_emb_pca[:,1], label="Generated", color="blue",zorder=0)
    pos_emb_pca = pos_emb_pca[::2]
    ax.scatter(pos_emb_pca[:,0], pos_emb_pca[:,1], label="Positive", color="green",zorder=1)
    # Only keep 1/10 of the negative embeddings to avoid clutter
    neg_emb_pca = neg_emb_pca[::5]
    ax.scatter(neg_emb_pca[:,0], neg_emb_pca[:,1], label="Negative", color="red",zorder=2)

    for i in range(len(images)):
        ax.imshow(np.asarray(images[i][0]), extent=[images[i][1][0]-0.75, images[i][1][0]+0.75, images[i][1][1]-0.6, images[i][1][1]+0.75], zorder=10+i*5)

    ax.set_xlim(-6,8)
    ax.set_ylim(-8,8)

    ax.legend()
    ax.set(xlabel='PCA 1', ylabel='PCA 2',
        title='PCA of CLIP embeddings')
    ax.legend(loc='lower right', shadow=True, fontsize='x-large')
    fig.savefig("pca_clip_lfw.pgf")
    fig.savefig("pca_clip_lfw.png")
    plt.close()




parser = argparse.ArgumentParser(description='Get or plot embeddings')
parser.add_argument('--get_emb', action='store_true', help='Get embeddings')
parser.add_argument('--plot', action='store_true', help='Plot embeddings')

args = parser.parse_args()

if args.get_emb:
    get_embeddings()
elif args.plot:
    plot_emb()
else:
    print("Please specify either --get_emb or --plot")
    exit(1)

if __name__ == '__main__':
    plot_emb()




