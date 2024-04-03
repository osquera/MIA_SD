
import os
import cv2
import torch
import numpy as np
import PIL as pil

from facexlib.utils import align_crop_face_landmarks, paste_face_back
from facexlib.detection import init_detection_model
from facexlib.utils.face_restoration_helper import get_largest_face
from facexlib.visualization import visualize_detection

def get_centered_face_from_img(img_path_in: str, img_path_out:str ):

    # initialize model
    det_net = init_detection_model('retinaface_resnet50', half=False)

    for root, dirs, files in os.walk(img_folder_path):
        for file in files:
            img_path = os.path.join(root, file)
            img_name = os.path.basename(img_path).split('.')[0]

            img_ori = cv2.imread(img_path)
            h, w = img_ori.shape[0:2]
            # if larger than 800, scale it
            scale = max(h / 800, w / 800)
            if scale > 1:
                img = cv2.resize(img_ori, (int(w / scale), int(h / scale)), interpolation=cv2.INTER_LINEAR)
            else:
                img = img_ori

            with torch.no_grad():
                bboxes = det_net.detect_faces(img, 0.97)
            if scale > 1:
                bboxes *= scale  # the score is incorrect
            try:
                bboxes = get_largest_face(bboxes, h, w)[0]
                #visualize_detection(img_ori, [bboxes], f'{temp_path}/{img_name}_detection.png')
            except:
                print(f'No face detected in {img_name}')
                continue

            landmarks = np.array([[bboxes[i], bboxes[i + 1]] for i in range(5, 15, 2)])

            cropped_face, _ = align_crop_face_landmarks(
                img_ori,
                landmarks,
                output_size=512,
                transform_size=None,
                enable_padding=True,
                return_inverse_affine=False,
                shrink_ratio=(1, 1))

            cv2.imwrite(f'{img_path_out}/{img_name}_face.png', cropped_face)

# A more naive function which finds the face and the crops the image to 512x512
def get_face_from_img(img_folder_path: str, img_path_out:str):
    # initialize model
    det_net = init_detection_model('retinaface_resnet50', half=False)

    for root, dirs, files in os.walk(img_folder_path):
        for file in files:
            img_path = os.path.join(root, file)
            img_name = os.path.basename(img_path).split('.')[0]

            img_ori = cv2.imread(img_path)
            h, w = img_ori.shape[0:2]
            # if larger than 800, scale it
            scale = max(h / 800, w / 800)
            if scale > 1:
                img = cv2.resize(img_ori, (int(w / scale), int(h / scale)), interpolation=cv2.INTER_LINEAR)
            else:
                img = img_ori

            with torch.no_grad():
                bboxes = det_net.detect_faces(img, 0.97)
            if scale > 1:
                bboxes *= scale
            try:
                bboxes = get_largest_face(bboxes, h, w)[0]
            except:
                print(f'No face detected in {img_name}')
                continue

            landmarks = np.array([[bboxes[i], bboxes[i + 1]] for i in range(5, 15, 2)])

            pil_img = pil.Image.open(img_path)

            # Crop to 512x512 while making sure that the face is still in the image
            left = max(0, bboxes[0] - 100)
            right = min(w, bboxes[2] + 100)
            top = max(0, bboxes[1] - 100)
            bottom = min(h, bboxes[3] + 100)
            pil_img = pil_img.crop((left, top, right, bottom))
            pil_img = pil_img.resize((512, 512))

            

            print('stop')






if __name__ == '__main__':

    #img_folder_path = 'C:\\DTU - KID\\6. Semester\\MIA_SD\\DTU_img_upscaled3\\restored_imgs'
    img_folder_path = 'C:\\DTU - KID\\6. Semester\\MIA_SD\\test'
    #img_path_out = 'C:\\DTU - KID\\6. Semester\\MIA_SD\\DTU_img_upscaled_centered'
    img_path_out = 'C:\\DTU - KID\\6. Semester\\MIA_SD\\test_out'
    #get_centered_face_from_img(img_folder_path, img_path_out)

    get_face_from_img(img_folder_path, img_path_out)
   
