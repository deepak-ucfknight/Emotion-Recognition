import cv2
from keras.models import load_model
import numpy as np
from src.utils.datasets import get_labels
from src.utils.inference import draw_text
from src.utils.inference import draw_bounding_box
from src.utils.inference import apply_offsets
from src.utils.preprocessor import preprocess_input
import dlib
from src.utils.datasets import Base_path
import os

# parameters for loading data and images
image_path = 'test_image.jpg' # path of the image
directory_path = os.path.basename(image_path)

emotion_model_path = Base_path + '/trained_models/model A_ferplus.hdf5'
emotion_labels = get_labels('ferplus')
font = cv2.FONT_HERSHEY_SIMPLEX

# hyper-parameters for bounding boxes shape
emotion_offsets = (40, 40)


# loading models
emotion_classifier = load_model(emotion_model_path, compile=False)
# gender_classifier = load_model(gender_model_path, compile=False)

# getting input model shapes for inference
emotion_target_size = emotion_classifier.input_shape[1:3]
# gender_target_size = gender_classifier.input_shape[1:3]

# loading images
rgb_image = cv2.imread(image_path)
gray_image = cv2.cvtColor(rgb_image,cv2.COLOR_BGR2GRAY)
gray_image = np.squeeze(gray_image)
gray_image = gray_image.astype('uint8')

detector = dlib.get_frontal_face_detector()
faces = detector(gray_image,1)

for face_coordinates in faces:

    x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
    gray_face = gray_image[y1:y2, x1:x2]

    gray_face = cv2.resize(gray_face, (emotion_target_size))
    # cv2.imshow('face', gray_face)

    gray_face = preprocess_input(gray_face, True)
    gray_face = np.expand_dims(gray_face, 0)
    gray_face = np.expand_dims(gray_face, -1)
    emotion_label_arg = np.argmax(emotion_classifier.predict(gray_face))
    emotion_text = emotion_labels[emotion_label_arg]

    color = (0, 0, 255)

    draw_bounding_box(face_coordinates, rgb_image, color)
    draw_text(face_coordinates, rgb_image, emotion_text, color, 0, 0, 1, 1)

cv2.imwrite(directory_path + '/predicted_result.png', rgb_image)
