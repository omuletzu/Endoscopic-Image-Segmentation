import cv2

img = cv2.imread('tissue-type.png')

height, width, _ = img.shape

for i in range(height):
    for j in range(width):
        if img[i, j] == (40, 41, 41):
            img[i, j] = (127, 127, 127)

        if img[i, j] == (66, 82, 224):
            img[i, j] = (127, 127, 127)

        if img[i, j] == (120, 34, 207):
            img[i, j] = (127, 127, 127)

        if img[i, j] == (28, 223, 245):
            img[i, j] = (127, 127, 127)

        if img[i, j] == (10, 231, 239):
            img[i, j] = (127, 127, 127)

        if img[i, j] == (0, 22, 255):
            img[i, j] = (127, 127, 127)

        if img[i, j] == (0, 23, 255):
            img[i, j] = (127, 127, 127)

        if img[i, j] == (0, 28, 252):
            img[i, j] = (127, 127, 127)

        if img[i, j] == (58, 255, 212):
            img[i, j] = (127, 127, 127)

        if img[i, j] == (169, 127, 248):
            img[i, j] = (127, 127, 127)

        if img[i, j] == ():
            img[i, j] = (127, 127, 127)

        if img[i, j] == (40, 41, 41):
            img[i, j] = (127, 127, 127)

        if img[i, j] == (40, 41, 41):
            img[i, j] = (127, 127, 127)

        if img[i, j] == (40, 41, 41):
            img[i, j] = (127, 127, 127)


cv2.imshow('img', img)
cv2.waitKey(0)