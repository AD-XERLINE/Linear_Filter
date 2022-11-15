import cv2 as cv
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
from keras.layers import Conv2D
from scipy import signal
import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


# --------------------------Filter------------------------------
sharpen = np.array((
	[0, -1, 0],
	[-1, 5, -1],
	[0, -1, 0]), dtype="int")

smallBlur = np.ones((7, 7), dtype="float") * (1.0 / (7 * 7))

largeBlur = np.ones((21, 21), dtype="float") * (1.0 / (21 * 21))

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

emboss = np.array([
    [-2, -1, 0],
    [-1, 1, 1],
    [0, 1, 2]
])
# ----------------------------------------------------------


# -----------------input before loop -------------------------
inp = input("choose number of picture: ")
image = cv.imread(inp+".jpg")
img_RGB = cv.cvtColor(src=image, code=cv.COLOR_BGR2RGB)
# -----------------input before loop -------------------------

# --------------------------อยู่ใน loop ใช้สำหรับเลือกแต่ละโหมด----------------------------
# ทำทุก ch สำหรับ ดรสะำพบางตัวที่มีผลลัพออกมาเหมาะสมแบบ เบลอ(smallbur,largebur,identity)
for i in range(img_RGB.shape[2]):
    img_RGB[:,:,i] = signal.convolve2d(img_RGB[:,:,i], sharpen, mode='same', boundary='fill', fillvalue=0)

# # การทำแค่ 2 ch เพื่อให้ผลลัพออกมาแต่สีไม่เพี้ยน (sharpen,emboss)
Temp0 = [0,1]
for i in Temp0:
    img_RGB[:,:,i] = signal.convolve2d(img_RGB[:,:,i], sharpen, mode='same', boundary='fill', fillvalue=0)

# เหมาะการทำเป็นสีเดียวเช่น outline
Temp1 = 1
img_RGB[:,:,Temp1] = signal.convolve2d(img_RGB[:,:,1], sharpen, mode='same', boundary='fill', fillvalue=0)
# ------------------------------------------------------------------------------------

# ---------------------------ส่วนของเวฤรูปแล้วคืนค่าสี--------------------------------------
img_RGB = cv.cvtColor(src=img_RGB, code=cv.COLOR_RGB2BGR)
cv.imwrite('RGB1.jpg', img_RGB)
# ------------------------------------------------------------------------------------



