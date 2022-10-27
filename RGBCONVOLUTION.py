import cv2 as cv
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
from keras.layers import Conv2D
from scipy import signal
import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


# b,g,r=cv.split(img)
# cv.imshow("Red",r)
# cv.imshow("Green",g)
# cv.imshow("Blue",b)
# cv.waitKey(0)

sharpen = np.array((
	[0, -1, 0],
	[-1, 5, -1],
	[0, -1, 0]), dtype="int")

# ใช้งานได้ดี
largeBlur = np.ones((21, 21), dtype="float") * (1.0 / (21 * 21))

bur = np.array([
    [0.0204, 0.0204, 0.0204],
    [0.0204, 0.0204, 0.0204],
    [0.0204, 0.0204, 0.0204]
])

outline = np.array([
    [-1, -1, -1],
    [-1,  8, -1],
    [-1, -1, -1]
])

identity = np.array([
    [0, 0, 0],
    [0, 1, 0],
    [0, 0, 0]
])

image = cv.imread("00000001.jpg")
img_RGB = cv.cvtColor(src=image, code=cv.COLOR_BGR2RGB)
# plt.figure(figsize=(10,8))


# signal.convolve2d( 2d_image, kernel_mantrix, mode = same(ให้ทำ padding), ...)
kernals = [1]

# ทำทุก ch สำหรับ ดรสะำพบางตัวที่มีผลลัพออกมาเหมาะสมแบบ เบลอได้
for i in range(img_RGB.shape[2]):
    img_RGB[:,:,i] = signal.convolve2d(img_RGB[:,:,i], largeBlur, mode='same', boundary='fill', fillvalue=0)

# การทำแค่บาง ch เพื่อให้ผลลัพออกมาแต่สีไม่เพี้ยน
for i in range(1):
    img_RGB[:,:,i] = signal.convolve2d(img_RGB[:,:,i], sharpen, mode='same', boundary='fill', fillvalue=0)

img_RGB[:,:,1] = signal.convolve2d(img_RGB[:,:,1], sharpen, mode='same', boundary='fill', fillvalue=0)


img_RGB = cv.cvtColor(src=img_RGB, code=cv.COLOR_RGB2BGR)
# cv.imwrite('RGB1.jpg', img_RGB)
plt.figure(figsize=(10,8))
map = 'gray'
plt.subplot(2,2,1); plt.imshow(img_RGB)
plt.title('RGB')
plt.subplot(2,2,2); plt.imshow(img_RGB[:,:,0], cmap = map)
plt.title('R')
plt.subplot(2,2,3); plt.imshow(img_RGB[:,:,1], cmap = map)
plt.title('G')
plt.subplot(2,2,4); plt.imshow(img_RGB[:,:,2], cmap = map)
plt.title('B')
plt.show()