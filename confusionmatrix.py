import sys
import os
import cv2
from keras.models import load_model
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


from src.utils.datasets import get_labels
from src.utils.inference import detect_faces
from src.utils.inference import draw_text
from src.utils.inference import draw_bounding_box
from src.utils.inference import apply_offsets
from src.utils.inference import load_detection_model
from src.utils.inference import load_image
from src.utils.preprocessor import preprocess_input
from src.utils.datasets import DataManager
from src.utils.datasets import split_data
from sklearn.metrics import confusion_matrix,accuracy_score


emotion_model_path = 'fc_key.hdf5'

# val = os.path.isfile(emotion_model_path)

emotion_classifier = load_model(emotion_model_path, compile=False)
emotion_target_size = emotion_classifier.input_shape[1:3]
emotion_classifier.summary()
dataset_name = "ferpluskey"
input_shape = (64, 64, 1)
validation_split = .1


data_loader = DataManager(dataset_name, image_size=input_shape[:2])
faces, emotions = data_loader.get_data()
faces = preprocess_input(faces)
num_samples, num_classes = emotions.shape
train_data, val_data = split_data(faces, emotions, validation_split)
train_faces, train_emotions = train_data
val_faces, val_emotions = val_data

labels = ['neutral','happiness','surprise','sadness','anger','disgust','fear','contempt','unknown', 'NF']

pred = emotion_classifier.predict(val_faces)
results = np.zeros_like(pred)
results[np.arange(len(pred)),pred.argmax(1)] = 1
# results = results.argmax(axis=1)
cm = confusion_matrix(val_emotions.argmax(axis=1),results.argmax(axis=1))

cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

accuracy = accuracy_score(val_emotions.argmax(axis=1),results.argmax(axis=1))

df_cm = pd.DataFrame(cm, index=labels, columns=labels)
fig = plt.figure(figsize=(15,15))
plt.title('Accuracy :' + str(round(accuracy,2)))
heatmap = sns.heatmap(df_cm, annot=True, fmt=".2f")
heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=14)
heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=14)
plt.ylabel('Predicted label')
plt.xlabel('True label')

plt.show()


