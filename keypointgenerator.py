from imutils import face_utils
import numpy as np
import argparse
import dlib
import cv2
import glob
import os
import pathlib

filepath = "Path of the folder which contains the images"
save = "Path of the folder where we want to save the keypoint generated images"
# vaal = os.path.isfile("/Users/deez/dlibs/PrivateTest/0.jpg")
files = glob.glob(filepath)
# construct the argument parser and parse the arguments

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
detector = dlib.get_frontal_face_detector()
# detector = dlib.cnn_face_detection_model_v1('mmod_human_face_detector.dat')
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

for img in files:
    filename = os.path.basename(img)
    image = cv2.imread(img)
    raw_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(raw_gray,(48,48))
    rects = detector(gray, 1)
    for (i, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        for (x, y) in shape:
            cv2.circle(gray, (x, y), 1, (0, 0, 0), -1)
    cv2.imwrite(save + filename,gray)

