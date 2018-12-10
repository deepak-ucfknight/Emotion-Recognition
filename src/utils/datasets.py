import pandas as pd
import numpy as np
import os
import cv2
from glob import iglob
import glob

#   All path information is related to the system environment being used, Modify them as per the needs

Base_path = '/home/deez/Emotion_classification'

class DataManager(object):
    """Class for loading fer2013 emotion classification dataset or
        imdb gender classification dataset."""
    def __init__(self, dataset_name='imdb',
                 dataset_path=None, image_size=(48, 48)):

        self.dataset_name = dataset_name
        self.dataset_path = dataset_path
        self.image_size = image_size
        if self.dataset_path is not None:
            self.dataset_path = dataset_path
        elif self.dataset_name == 'fer2013':
            self.dataset_path = (Base_path + '/datasets/fer2013/fer2013/fer2013.csv')
        elif self.dataset_name == 'fer2013key':
            self.dataset_path = (Base_path + '/datasets/fer2013/fer2013key/')
        elif self.dataset_name == 'ckplus':
            self.dataset_path = (Base_path + '/datasets/ckplus/')
        elif self.dataset_name == "ckpluskey":
            self.dataset_path = (Base_path +'/datasets/ckplus/ckpluskey/')
        elif self.dataset_name == 'ferpluskey':
            self.dataset_path = (Base_path + '/datasets/fer2013/fer2013key/')
        elif self.dataset_name == 'ferplus':
            self.dataset_path = (Base_path + '/datasets/fer2013/fer2013/fer2013.csv')


    def get_data(self):
        if self.dataset_name == 'fer2013':
            ground_truth_data = self._load_fer2013()
        elif self.dataset_name == 'fer2013key':
            ground_truth_data = self._load_fer2013key()
        if self.dataset_name == 'ferplus':
            ground_truth_data = self._load_ferplus()
        elif self.dataset_name == 'ferpluskey':
            ground_truth_data = self._load_ferpluskey()
        elif self.dataset_name == 'ckplus':
            ground_truth_data = self._load_ckplus()
        elif self.dataset_name == 'ckpluskey':
            ground_truth_data = self._load_ckpluskey()
        return ground_truth_data

    def _load_ferplus(self):
        data = pd.read_csv(self.dataset_path)
        pixels = data['pixels'].tolist()
        width, height = 48, 48
        faces = []
        for pixel_sequence in pixels:
            face = [int(pixel) for pixel in pixel_sequence.split(' ')]
            face = np.asarray(face).reshape(width, height)
            face = cv2.resize(face.astype('uint8'), self.image_size)
            faces.append(face.astype('float32'))
        faces = np.asarray(faces)
        faces = np.expand_dims(faces, -1)
        fer_plus_emotions = np.load(Base_path + '/datasets/ferplus/ferpluslabels.npy')
        return faces, fer_plus_emotions

    def _load_ferpluskey(self):
        num_classes = 10
        count = 0
        y_size, x_size = self.image_size
        train_directory = (self.dataset_path + '/Training/*.*')
        validation_directory = (self.dataset_path + '/PrivateTest/*.*')
        test_directory = (self.dataset_path + '/PublicTest/*.*')
        file_list = [f for f in iglob(train_directory, recursive=True) if os.path.isfile(f)]
        num_faces = len(file_list) - 1
        train_x = np.zeros(shape=(num_faces, y_size, x_size))
        train_y = np.zeros(shape=(num_faces, num_classes))
        for image_file in file_list:
            if image_file.lower().endswith('.jpg'):
                filename = os.path.basename(image_file)
                idx = int(os.path.splitext(filename)[0])
                image_array = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
                image_array = cv2.resize(image_array, (y_size, x_size))
                face = image_array.astype('float32')
                train_x[idx] = face


        file_list = [f for f in iglob(validation_directory, recursive=True) if os.path.isfile(f)]
        num_faces = len(file_list) - 1
        val_x = np.zeros(shape=(num_faces, y_size, x_size))
        val_y = np.zeros(shape=(num_faces, num_classes))
        for image_file in file_list:
            if image_file.lower().endswith('.jpg'):
                filename = os.path.basename(image_file)
                idx = int(os.path.splitext(filename)[0])
                image_array = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
                image_array = cv2.resize(image_array, (y_size, x_size))
                face = image_array.astype('float32')
                val_x[idx] = face


        file_list = [f for f in iglob(test_directory, recursive=True) if os.path.isfile(f)]
        num_faces = len(file_list) - 1
        test_x = np.zeros(shape=(num_faces, y_size, x_size))
        test_y = np.zeros(shape=(num_faces, num_classes))
        for image_file in file_list:
            if image_file.lower().endswith('.jpg'):
                filename = os.path.basename(image_file)
                idx = int(os.path.splitext(filename)[0])
                image_array = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
                image_array = cv2.resize(image_array, (y_size, x_size))
                face = image_array.astype('float32')
                test_x[idx] = face

        faces = np.concatenate((train_x, test_x, val_x), axis=0)
        faces = np.expand_dims(faces, -1)
        fer_plus_emotions = np.load(Base_path + '/datasets/ferplus/ferpluslabels.npy')
        return faces, fer_plus_emotions

    def _load_ckpluskey(self):
        files = glob.glob(self.dataset_path + '*.png')
        num_faces = len(files)
        y_size, x_size = self.image_size
        faces = np.zeros(shape=(num_faces, y_size, x_size))
        for file_arg, file_path in enumerate(files):
            image_array = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            image_array = cv2.resize(image_array, (y_size, x_size))
            face = image_array.astype('float32')
            faces[file_arg] = face
        faces = np.expand_dims(faces, -1)
        emotions = np.load((self.dataset_path + '/labels.npy'))
        return faces, emotions

    def _load_ckplus(self):

        num_classes = 8

        emotion_file_list = []
        emotion_file_name = []
        image_file_list = []
        count = 0
        root_dir = (Base_path + '/datasets/ckplus/***/**/*')
        file_list = [f for f in iglob(root_dir, recursive=True) if os.path.isfile(f)]

        for file in file_list:
            if file.lower().endswith('.txt'):
                path, file_name = os.path.split(file)
                emotion_file_list.append(file)
                emotion_file_name.append(file_name[:-12])
                count += 1

        i_count = 0

        for image_file in file_list:
            if image_file.lower().endswith('.png'):
                path, image_name = os.path.split(image_file)
                image_name_without_extension = os.path.splitext(image_name)[0]
                if image_name_without_extension in emotion_file_name:
                    image_file_list.append(image_file)
                    i_count += 1


        num_faces = len(image_file_list)
        y_size, x_size = self.image_size
        faces = np.zeros(shape=(num_faces, y_size, x_size))
        emotions = np.zeros(shape=(num_faces, num_classes))

        for file_arg, file_path in enumerate(image_file_list):
            image_array = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            image_array = cv2.resize(image_array, (y_size, x_size))
            face = image_array.astype('float32')
            faces[file_arg] = image_array

        for file_arg, file_path in enumerate(emotion_file_list):
            with open(file_path, "r") as myfile:
                data = myfile.readlines()
                label = int(os.path.splitext(data[0])[0])
                emotions[file_arg, label] = 1

        faces = np.expand_dims(faces, -1)

        return faces, emotions

    def _load_fer2013(self):
        data = pd.read_csv(self.dataset_path)
        pixels = data['pixels'].tolist()
        width, height = 48, 48
        faces = []
        for pixel_sequence in pixels:
            face = [int(pixel) for pixel in pixel_sequence.split(' ')]
            face = np.asarray(face).reshape(width, height)
            face = cv2.resize(face.astype('uint8'), self.image_size)
            faces.append(face.astype('float32'))
        faces = np.asarray(faces)
        faces = np.expand_dims(faces, -1)
        emotions = pd.get_dummies(data['emotion']).as_matrix()

        return faces, emotions

    def _load_fer2013key(self):
        num_classes = 7
        count = 0
        y_size, x_size = self.image_size
        train_directory = (self.dataset_path + '/Training/*.*')
        validation_directory = (self.dataset_path + '/PrivateTest/*.*')
        test_directory = (self.dataset_path + '/PublicTest/*.*')
        file_list = [f for f in iglob(train_directory, recursive=True) if os.path.isfile(f)]
        num_faces = len(file_list) - 1
        train_x = np.zeros(shape=(num_faces, y_size, x_size))
        train_y = np.zeros(shape=(num_faces, num_classes))
        for image_file in file_list:
            if image_file.lower().endswith('.jpg'):
                filename = os.path.basename(image_file)
                idx = int(os.path.splitext(filename)[0])
                image_array = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
                image_array = cv2.resize(image_array, (y_size, x_size))
                face = image_array.astype('float32')
                train_x[idx] = face

            if image_file.lower().endswith('.npy'):
                path = os.path.dirname(image_file)
                train_y = np.load(path + '/labels.npy')
        file_list = [f for f in iglob(validation_directory, recursive=True) if os.path.isfile(f)]
        num_faces = len(file_list) - 1
        val_x = np.zeros(shape=(num_faces, y_size, x_size))
        val_y = np.zeros(shape=(num_faces, num_classes))
        for image_file in file_list:
            if image_file.lower().endswith('.jpg'):
                filename = os.path.basename(image_file)
                idx = int(os.path.splitext(filename)[0])
                image_array = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
                image_array = cv2.resize(image_array, (y_size, x_size))
                face = image_array.astype('float32')
                val_x[idx] = face
            if image_file.lower().endswith('.npy'):
                path = os.path.dirname(image_file)
                val_y = np.load(path + '/labels.npy')
        file_list = [f for f in iglob(test_directory, recursive=True) if os.path.isfile(f)]
        num_faces = len(file_list) - 1
        test_x = np.zeros(shape=(num_faces, y_size, x_size))
        test_y = np.zeros(shape=(num_faces, num_classes))
        for image_file in file_list:
            if image_file.lower().endswith('.jpg'):
                filename = os.path.basename(image_file)
                idx = int(os.path.splitext(filename)[0])
                image_array = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
                image_array = cv2.resize(image_array, (y_size, x_size))
                face = image_array.astype('float32')
                test_x[idx] = face
            if image_file.lower().endswith('.npy'):
                path = os.path.dirname(image_file)
                test_y = np.load(path + '/labels.npy')
        faces = np.concatenate((train_x, test_x, val_x), axis=0)
        faces = np.expand_dims(faces, -1)
        emotions = np.concatenate((train_y, val_y, test_y), axis=0)
        return  faces, emotions


def get_labels(dataset_name):
    if dataset_name == 'fer2013' or dataset_name == 'fer2013key':
        return {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy',
                4: 'sad', 5: 'surprise', 6: 'neutral'}
    elif dataset_name == 'ckplus' or dataset_name == 'ckpluskey':
        return {0: 'neutral', 1: 'anger', 2: 'contempt', 3: 'disgust', 4: 'fear', 5: 'happy', 6: 'sadness', 7: 'surprise'}
    elif dataset_name == 'ferplus' or dataset_name == 'ferpluskey':
        return {0: 'neutral', 1: 'happiness', 2: 'surprise', 3: 'sadness', 4: 'anger', 5: 'disgust', 6: 'fear',
                7: 'contempt', 8:'unknown', 9:'NF'}
    else:
        raise Exception('Invalid dataset name')


def split_data(x, y, validation_split=.2):
    num_samples = len(x)
    num_train_samples = int((1 - validation_split)*num_samples)
    train_x = x[:num_train_samples]
    train_y = y[:num_train_samples]
    val_x = x[num_train_samples:]
    val_y = y[num_train_samples:]
    train_data = (train_x, train_y)
    val_data = (val_x, val_y)
    return train_data, val_data
