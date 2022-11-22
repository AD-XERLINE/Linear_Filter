import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy import signal
from scipy import spatial
import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


def calculate_similarity(vector1, vector2):
    sim_cos = spatial.distance.cosine(vector1, vector2)
    return sim_cos

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

Bsable = np.array([
    [-1, -2, -1],
    [0, 0, 0],
    [1, 2, 1]
])

Tsable = np.array([
    [1, 2, 1],
    [0, 0, 0],
    [-1, -2, -1]
])

prewitt = np.array([
    [-1, 0, 1],
    [-1, 0, 1],
    [-1, 0, 1]
])
# ----------------------------------------------------------




# -----------------input before loop -------------------------
inp = input("Choose number of picture: ")

image = cv.imread("PICTURE\\" +inp+ ".jpg")
img_RGB = image

img_RGB = cv.cvtColor(src=image, code=cv.COLOR_BGR2RGB)
# ------------------------------------------------------------


# ----------------------loop game -----------------------------------------
InMode = int(input("Choose filter do you want [1]sharpen [2]smallBlur [3]largeBlur [4]outline [5]identity [6]emboss [7]Bsable [8]Tsable : "))

if InMode == 1:

    CHshapen = int(input("Choose sharpen mode do you want [3]3 layers [2]2 layers : "))

    if CHshapen == 3:
        for i in range(img_RGB.shape[2]):
            img_RGB[:,:,i] = signal.convolve2d(img_RGB[:,:,i], sharpen, mode='same', boundary='fill', fillvalue=0)
        img_RGB = cv.cvtColor(src=img_RGB, code=cv.COLOR_RGB2BGR)

    if CHshapen == 2:

        CHRGB = input("Choose channel of picture (example: R G): ")

        if CHRGB == 'R G' or CHRGB == 'G R':
            Temp0 = [0,1]
            for i in Temp0:
                img_RGB[:,:,i] = signal.convolve2d(img_RGB[:,:,i], sharpen, mode='same', boundary='fill', fillvalue=0)
            img_RGB = cv.cvtColor(src=img_RGB, code=cv.COLOR_RGB2BGR) 

        elif CHRGB == 'R B' or CHRGB == 'B R':
            Temp0 = [0,2]
            for i in Temp0:
                img_RGB[:,:,i] = signal.convolve2d(img_RGB[:,:,i], sharpen, mode='same', boundary='fill', fillvalue=0)
            img_RGB = cv.cvtColor(src=img_RGB, code=cv.COLOR_RGB2BGR) 

        elif CHRGB == 'G B' or CHRGB == 'B G':
            Temp0 = [1,2]
            for i in Temp0:
                img_RGB[:,:,i] = signal.convolve2d(img_RGB[:,:,i], sharpen, mode='same', boundary='fill', fillvalue=0)
            img_RGB = cv.cvtColor(src=img_RGB, code=cv.COLOR_RGB2BGR) 
            

elif InMode == 2:
    # ทำทุก ch สำหรับ ดรสะำพบางตัวที่มีผลลัพออกมาเหมาะสมแบบ เบลอ(smallbur,largebur,identity,Tsable,Bsable)
    for i in range(img_RGB.shape[2]):
        img_RGB[:,:,i] = signal.convolve2d(img_RGB[:,:,i], smallBlur, mode='same', boundary='fill', fillvalue=0)
    img_RGB = cv.cvtColor(src=img_RGB, code=cv.COLOR_RGB2BGR)

elif InMode == 3:
    for i in range(img_RGB.shape[2]):
        img_RGB[:,:,i] = signal.convolve2d(img_RGB[:,:,i], largeBlur, mode='same', boundary='fill', fillvalue=0)
    img_RGB = cv.cvtColor(src=img_RGB, code=cv.COLOR_RGB2BGR)

elif InMode == 4:
    image = cv.cvtColor(src=image, code=cv.COLOR_RGB2BGR)
    image = cv.cvtColor(src=image, code=cv.COLOR_BGR2GRAY)
    img_RGB = signal.convolve2d(image ,outline, mode='same', boundary='fill', fillvalue=0)

elif InMode == 5:
    for i in range(img_RGB.shape[2]):
        img_RGB[:,:,i] = signal.convolve2d(img_RGB[:,:,i], identity, mode='same', boundary='fill', fillvalue=0)
    img_RGB = cv.cvtColor(src=img_RGB, code=cv.COLOR_RGB2BGR)

elif InMode == 6:

    CHemboss = int(input("Choose emboss mode do you want [3]3 layers [2]2 layers : "))

    if CHemboss == 3:
        for i in range(img_RGB.shape[2]):
            img_RGB[:,:,i] = signal.convolve2d(img_RGB[:,:,i], emboss, mode='same', boundary='fill', fillvalue=0)
        img_RGB = cv.cvtColor(src=img_RGB, code=cv.COLOR_RGB2BGR) 

    if CHemboss == 2:

        CHRGB = input("Choose channel of picture (example: R G): ")

        if CHRGB == 'R G' or CHRGB == 'G R':
            Temp0 = [0,1]
            for i in Temp0:
                img_RGB[:,:,i] = signal.convolve2d(img_RGB[:,:,i], emboss, mode='same', boundary='fill', fillvalue=0)
            img_RGB = cv.cvtColor(src=img_RGB, code=cv.COLOR_RGB2BGR) 

        elif CHRGB == 'R B' or CHRGB == 'B R':
            Temp0 = [0,2]
            for i in Temp0:
                img_RGB[:,:,i] = signal.convolve2d(img_RGB[:,:,i], emboss, mode='same', boundary='fill', fillvalue=0)
            img_RGB = cv.cvtColor(src=img_RGB, code=cv.COLOR_RGB2BGR) 

        elif CHRGB == 'G B' or CHRGB == 'B G':
            Temp0 = [1,2]
            for i in Temp0:
                img_RGB[:,:,i] = signal.convolve2d(img_RGB[:,:,i], emboss, mode='same', boundary='fill', fillvalue=0)
            img_RGB = cv.cvtColor(src=img_RGB, code=cv.COLOR_RGB2BGR) 

elif InMode == 7:
    CHBsable = int(input("Choose Bsable mode do you want [3]3 layers [2]2 layers : "))

    if CHBsable == 3:
        for i in range(img_RGB.shape[2]):
            img_RGB[:,:,i] = signal.convolve2d(img_RGB[:,:,i], emboss, mode='same', boundary='fill', fillvalue=0)
        img_RGB = cv.cvtColor(src=img_RGB, code=cv.COLOR_RGB2BGR) 

    if CHBsable == 2:

        CHRGB = input("Choose channel of picture (example: R G): ")

        if CHRGB == 'R G' or CHRGB == 'G R':
            Temp0 = [0,1]
            for i in Temp0:
                img_RGB[:,:,i] = signal.convolve2d(img_RGB[:,:,i], Bsable, mode='same', boundary='fill', fillvalue=0)
            img_RGB = cv.cvtColor(src=img_RGB, code=cv.COLOR_RGB2BGR) 

        elif CHRGB == 'R B' or CHRGB == 'B R':
            Temp0 = [0,2]
            for i in Temp0:
                img_RGB[:,:,i] = signal.convolve2d(img_RGB[:,:,i], Bsable, mode='same', boundary='fill', fillvalue=0)
            img_RGB = cv.cvtColor(src=img_RGB, code=cv.COLOR_RGB2BGR) 

        elif CHRGB == 'G B' or CHRGB == 'B G':
            Temp0 = [1,2]
            for i in Temp0:
                img_RGB[:,:,i] = signal.convolve2d(img_RGB[:,:,i], Bsable, mode='same', boundary='fill', fillvalue=0)
            img_RGB = cv.cvtColor(src=img_RGB, code=cv.COLOR_RGB2BGR)

elif InMode == 8:
    CHTsable = int(input("Choose Tsable mode do you want [3]3 layers [2]2 layers : "))

    if CHTsable == 3:
        for i in range(img_RGB.shape[2]):
            img_RGB[:,:,i] = signal.convolve2d(img_RGB[:,:,i], Tsable, mode='same', boundary='fill', fillvalue=0)
        img_RGB = cv.cvtColor(src=img_RGB, code=cv.COLOR_RGB2BGR) 

    if CHTsable == 2:

        CHRGB = input("Choose channel of picture (example: R G): ")

        if CHRGB == 'R G':
            Temp0 = [0,1]
            for i in Temp0:
                img_RGB[:,:,i] = signal.convolve2d(img_RGB[:,:,i], Tsable, mode='same', boundary='fill', fillvalue=0)
            img_RGB = cv.cvtColor(src=img_RGB, code=cv.COLOR_RGB2BGR) 

        elif CHRGB == 'R B':
            Temp0 = [0,2]
            for i in Temp0:
                img_RGB[:,:,i] = signal.convolve2d(img_RGB[:,:,i], Tsable, mode='same', boundary='fill', fillvalue=0)
            img_RGB = cv.cvtColor(src=img_RGB, code=cv.COLOR_RGB2BGR) 

        elif CHRGB == 'G B':
            Temp0 = [1,2]
            for i in Temp0:
                img_RGB[:,:,i] = signal.convolve2d(img_RGB[:,:,i], Tsable, mode='same', boundary='fill', fillvalue=0)
            img_RGB = cv.cvtColor(src=img_RGB, code=cv.COLOR_RGB2BGR)
# -------------------------------------------------------------------------------


# ---------------------------ส่วนของเวฤรูปแล้วคืนค่าสี--------------------------------------
# img_RGB = cv.cvtColor(src=img_RGB, code=cv.COLOR_RGB2BGR)
cv.imwrite('RGB1.jpg', img_RGB)
# ------------------------------------------------------------------------------------


# ----------------------หาความเหมือนระหว่่าง 2 รูป----------------------------------------
image = cv.imread("PICTURE\\" +inp+ ".jpg")
img_RGB = cv.imread("RGB1.jpg")

A = image.flatten()
B = img_RGB.flatten()


print("")
print("Difference value :", calculate_similarity(A,B))
print("")
# ------------------------------------------------------------------------------------


# ---------------------------------การแสดงผล------------------------------------------
img = mpimg.imread('RGB1.jpg')
img2 = mpimg.imread("PICTURE\\" +inp+ ".jpg")
plt.figure(figsize=(10, 8))
plt.subplot(2,2,1); plt.imshow(img2)
plt.title('OLD IMAGE')
plt.subplot(2,2,2); plt.imshow(img)
plt.title('NEW IMAGE')
plt.show()
# -----------------------------------------------------------------------------------
