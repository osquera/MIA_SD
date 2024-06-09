
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
            scale = min(h / 800, w / 800)
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
            # if larger than 512, scale it
            scale = min(h / 512, w / 512)
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

            eye_left = landmarks[0]
            eye_right = landmarks[1]
        
            eye_middle = (eye_left + eye_right) / 2

            # Crop the image above 512/2 and under 512/2 and to the sides of eye_middle:
            # image = pil.Image.fromarray(img)
            # image = image.crop((eye_middle[0]-256, eye_middle[1]-256, eye_middle[0]+256, eye_middle[1]+256))

            # Use cv2 instead of PIL
            left = int(eye_middle[0]-256)
            right = int(eye_middle[0]+256)
            top = int(eye_middle[1]-256)
            bottom = int(eye_middle[1]+256)

            w = img.shape[1]
            h = img.shape[0]

            if left < 0:
                right = right - left
                left = 0
            if right > w:
                left = 0
                right = 512
            if bottom > h:
                top = 0
                bottom = 512
            if top < 0:
                bottom = bottom - top
                top = 0

            image = img[top:bottom, left:right]

            # Save the image
            cv2.imwrite(f'{img_path_out}/{img_name}_face.png', image)



if __name__ == '__main__':

    # img_folder_path = 'C:\\DTU - KID\\6. Semester\\MIA_SD\\DTU_img_upscaled3\\restored_imgs'
    # img_path_out = 'C:\\DTU - KID\\6. Semester\\MIA_SD\\DTU_img_upscaled_centered_eyes'
    
    img_folder_path = 'C:\\DTU - KID\\6. Semester\\MIA_SD\\AAU_img_upscaled\\restored_imgs'
    img_path_out = 'C:\\DTU - KID\\6. Semester\\MIA_SD\\AAU_img_upscaled_centered_eyes'

    # get_centered_face_from_img(img_folder_path, img_path_out)

    get_face_from_img(img_folder_path, img_path_out)
   
