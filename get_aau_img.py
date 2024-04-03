import requests
from bs4 import BeautifulSoup
import regex as re
import cv2
import numpy as np
import os

def get_aau_img():
    """
    This function returns all images of the DTU staff from the DTU website.
    It uses pagination to get all the images.
    
    """
    url = 'https://vbn.aau.dk/da/persons/?ordering=lastNameFirstName&descending=true'
    
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    images = soup.find_all('img', {'class': 'image'})

    while soup.find('a', {'class': 'nextLink'}):
        next_page = soup.find('a', {'class': 'nextLink'}).get('href')
        response = requests.get('https://vbn.aau.dk' + next_page)
        soup = BeautifulSoup(response.text, 'html.parser')
        images += soup.find_all('img', {'class': 'image'})

    return images

def download_images(images, path):
    """
    This function downloads the images from the DTU website.
    """
    for i, img in enumerate(images):
        img_url = img.get('src')
        img_url = re.sub(r'w=\d+', 'width=4000', img_url)
        img_data = requests.get('https://vbn.aau.dk' + img_url).content
        with open(f'{path}/img_{2500+i}.jpg', 'wb') as handler:
            handler.write(img_data)      

def remove_small_images(path):
    """
    This function removes images that are smaller than 100 bytes.
    """
    count = 0

    for img in os.listdir(path):
        if os.path.getsize(f'{path}/{img}') < 100:
            os.remove(f'{path}/{img}')
            count += 1
    
    print(f'{count} images removed')

def remove_image_equal_to_stock_image(path, stock_img_path):
    """
    This function removes images that are equal to a stock image.
    """
    count = 0
    stock_img = cv2.imread(stock_img_path)
    for img in os.listdir(path):
        img_path = f'{path}/{img}'
        img = cv2.imread(img_path)
        if np.array_equal(stock_img, img):
            os.remove(img_path)
            count += 1
    
    print(f'{count} images removed')

if __name__ == '__main__':
    images = get_aau_img()
    download_images(images, 'AAU_img') # 
    print(f'{len(images)} images downloaded')
    remove_small_images('AAU_img') # 
    print(f'{len(os.listdir("AAU_img"))} images left')
    remove_image_equal_to_stock_image('AAU_img', 'no_image_aau.jpg') 
    print(f'{len(os.listdir("AAU_img"))} images left')
    remove_image_equal_to_stock_image('AAU_img', 'no_image.jpg')
    print(f'{len(os.listdir("AAU_img"))} images left')
    remove_image_equal_to_stock_image('AAU_img', 'no_image_aau2.jpg')
    print(f'{len(os.listdir("AAU_img"))} images left')
