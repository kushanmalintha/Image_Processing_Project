import cv2
import os
import numpy as np

from tqdm import tqdm

os.makedirs('../gaussian_blurred', exist_ok=True)

src_dir = 'C:/Users/kusha/OneDrive/Desktop/COM/Sem_5/CO543-Image Processing/Labs/Project/ImageProcessingMiniProject/sharp'
images = os.listdir(src_dir)
dst_dir = '../gaussian_blurred'

for i, img in tqdm(enumerate(images), total=len(images)):
    img = cv2.imread(f"{src_dir}/{images[i]}", cv2.IMREAD_COLOR)
    # add gaussian blurring
    blur = cv2.GaussianBlur(img, (31, 31), 0)
    cv2.imwrite(f"{dst_dir}/{images[i]}", blur)
print('DONE')
