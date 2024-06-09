


# The purpose of this script is to take a picture of some size X x Y and turn it into a quadratic image of size Z x Z.
# Z should be 512
# The images should first be downscaled to 512 x Y or X x 512, then the image should cropped to 512 x 512

import cv2
import os

def get_quadratic_image(image_path: str):
    image = cv2.imread(image_path)
    height, width, _ = image.shape
    if height > width:
        # resize to 512 x Y
        image = cv2.resize(image, (512, int(512 * height / width)))
        # crop to 512 x 512
        image = image[:512, :]
    else:
        # resize to X x 512
        image = cv2.resize(image, (int(512 * width / height), 512))
        # crop to 512 x 512
        image = image[:, :512]
    return image

def turn_all_photos_to_quadratic_images(path_out: str, path_in: str = 'DTU_img_upscaled3/restored_imgs'):
    for file in os.listdir(path_in):
        image = get_quadratic_image(path_in + '/' + file)
        cv2.imwrite(path_out + '/' + file, image)



if __name__ == '__main__':
    turn_all_photos_to_quadratic_images(path_out='images_attack_model/DTU_gen_vs_AAU_v1/aau', path_in='images_attack_model/DTU_gen_vs_AAU_v1/aau')

