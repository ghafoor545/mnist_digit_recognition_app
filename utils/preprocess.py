# Optional utility if you want to handle uploaded images too
import numpy as np
from PIL import Image
import cv2

def preprocess_image(image: Image.Image):
    image = image.convert('L')
    img_array = np.array(image)

    # Threshold and center crop similar to canvas
    _, img_thresh = cv2.threshold(img_array, 10, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(img_thresh)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        digit = img_array[y:y+h, x:x+w]
        aspect_ratio = w / h
        if aspect_ratio > 1:
            resized_digit = cv2.resize(digit, (20, int(20 / aspect_ratio)))
        else:
            resized_digit = cv2.resize(digit, (int(20 * aspect_ratio), 20))

        pad_top = (28 - resized_digit.shape[0]) // 2
        pad_bottom = 28 - resized_digit.shape[0] - pad_top
        pad_left = (28 - resized_digit.shape[1]) // 2
        pad_right = 28 - resized_digit.shape[1] - pad_left
        img_resized = np.pad(resized_digit, ((pad_top, pad_bottom), (pad_left, pad_right)), 'constant', constant_values=0)
    else:
        img_resized = cv2.resize(img_array, (28, 28))

    if np.mean(img_resized) > 127:
        img_resized = 255 - img_resized

    img_resized = img_resized.astype('float32') / 255.0
    return img_resized.reshape(1, 28, 28, 1)
