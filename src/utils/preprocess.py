import cv2

def preprocess_image(image_path, target_size=(160, 160)):
    image = cv2.imread(image_path)
    image = cv2.resize(image, target_size)
    return image