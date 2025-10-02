import os
import cv2
import numpy as np
from glob import glob
from tensorflow.keras.models import load_model


def preprocess_original(path):
    img = cv2.imread(path)
    img = img.astype('float32') / 255.0
    return img


def read_images_from_folder(folder_path):
    image_list = sorted(glob(os.path.join(folder_path, "*.bmp")))
    image_list = [img.replace('\\', '/') for img in image_list]
    return image_list


image_path = '../IMAGES/segmentation_images/'
prediction_path = '../IMAGES/prediction_images/'
image_list = read_images_from_folder(image_path)

unet_model = load_model('unet-model.h5')

for img in image_list:
    processed_img = np.array(preprocess_original(img))

    processed_img = np.expand_dims(processed_img, axis=0)

    prediction = unet_model.predict(processed_img)

    prediction_class_id = np.argmax(prediction, axis=-1)

    prediction_class_id = np.squeeze(prediction_class_id, axis=0)

    img_filename = os.path.basename(img)
    cv2.imwrite(os.path.join(prediction_path, img_filename), prediction_class_id.astype('uint8'))




