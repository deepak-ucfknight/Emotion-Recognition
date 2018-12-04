from scipy.io import loadmat
import pandas as pd
import numpy as np
from random import shuffle
import os
import cv2
from glob import iglob

import glob

Base_path = '/home/deez/face_classification-master'

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
        elif self.dataset_name == 'imdb':
            self.dataset_path = (Base_path + '/datasets/imdb_crop/imdb.mat')
        elif self.dataset_name == 'fer2013':
            self.dataset_path = (Base_path + '/datasets/fer2013/fer2013/fer2013.csv')
        elif self.dataset_name == 'KDEF':
            self.dataset_path = (Base_path + '/datasets/KDEF/')
        elif self.dataset_name == 'ck':
            self.dataset_path = (Base_path + '/datasets/ck/')
        elif self.dataset_name == 'ckplus':
            self.dataset_path = (Base_path + '/datasets/ckplus/')
        elif self.dataset_name == 'ferpluskey':
            self.dataset_path = (Base_path + '/datasets/fer2013/')
        elif self.dataset_name == 'ferplus':
            self.dataset_path = (Base_path + '/datasets/fer2013/fer2013/fer2013.csv')
        else:
            raise Exception(
                    'Incorrect dataset name, please input imdb or fer2013')

    def get_data(self):
        if self.dataset_name == 'imdb':
            ground_truth_data = self._load_imdb()
        elif self.dataset_name == 'ferplus':
            ground_truth_data = self._load_fer2013()
        elif self.dataset_name == 'KDEF':
            ground_truth_data = self._load_KDEF()
        elif self.dataset_name == 'ck':
            ground_truth_data = self._load_ck()
        elif self.dataset_name == 'ckplus':
            ground_truth_data = self._load_ckplus()
        elif self.dataset_name == 'ferpluskey':
            ground_truth_data = self._load_ferpluskey()
        return ground_truth_data

    def _load_ck(self):
        faces = np.zeros(shape=(10, 64, 64))
        emotions = np.zeros(shape=(10, 7))
        return faces,emotions


    def _load_ferpluskey(self):
        num_classes = 7

        count = 0
        y_size, x_size = self.image_size



        train_directory = (Base_path + '/datasets/fer2013/Training/*.*')
        validation_directory = (Base_path + '/datasets/fer2013/PrivateTest/*.*')
        test_directory = (Base_path + '/datasets/fer2013//PublicTest/*.*')
        label_directory = (Base_path + '/datasets/fer2013/')
        t = os.path.isfile(label_directory + '/ferpluslabels.npy')
        fer_plus_emotions = np.load(label_directory + '/ferpluslabels.npy')
        file_list = [f for f in iglob(train_directory, recursive=True) if os.path.isfile(f)]
        num_faces = len(file_list)-1
        train_x = np.zeros(shape=(num_faces, y_size, x_size))
        train_y = np.zeros(shape=(num_faces, num_classes))

        for  image_file in file_list:
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

        num_faces = len(file_list)-1
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

        num_faces = len(file_list)-1
        test_x = np.zeros(shape=(num_faces, y_size, x_size))
        test_y = np.zeros(shape=(num_faces, num_classes))

        for  image_file in file_list:
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

        faces = np.concatenate((train_x,test_x,val_x),axis=0)
        faces = np.expand_dims(faces, -1)
        emotions = np.concatenate((train_y,val_y,test_y),axis=0)



        return faces, fer_plus_emotions
    def _load_ckplus(self):
        faces = np.load((Base_path + '/datasets/ckplus/CKkeyfaces.npy'))
        faces = np.expand_dims(faces, -1)
        emotions = np.load((Base_path + './datasets/ckplus/labels.npy'))
        return faces, emotions

    def _load_ckplus1(self):

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

    def _load_imdb(self):
        face_score_treshold = 3
        dataset = loadmat(self.dataset_path)
        image_names_array = dataset['imdb']['full_path'][0, 0][0]
        gender_classes = dataset['imdb']['gender'][0, 0][0]
        face_score = dataset['imdb']['face_score'][0, 0][0]
        second_face_score = dataset['imdb']['second_face_score'][0, 0][0]
        face_score_mask = face_score > face_score_treshold
        second_face_score_mask = np.isnan(second_face_score)
        unknown_gender_mask = np.logical_not(np.isnan(gender_classes))
        mask = np.logical_and(face_score_mask, second_face_score_mask)
        mask = np.logical_and(mask, unknown_gender_mask)
        image_names_array = image_names_array[mask]
        gender_classes = gender_classes[mask].tolist()
        image_names = []
        for image_name_arg in range(image_names_array.shape[0]):
            image_name = image_names_array[image_name_arg][0]
            image_names.append(image_name)
        return dict(zip(image_names, gender_classes))

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
        label_directory = (Base_path + '/datasets/fer2013/')
        t = os.path.isfile(label_directory + '/ferpluslabels.npy')
        fer_plus_emotions = np.load(label_directory + '/ferpluslabels.npy')
        return faces, fer_plus_emotions

    def _load_KDEF(self):
        class_to_arg = get_class_to_arg(self.dataset_name)
        num_classes = len(class_to_arg)

        file_paths = []
        for folder, subfolders, filenames in os.walk(self.dataset_path):
            for filename in filenames:
                if filename.lower().endswith(('.jpg')):
                    file_paths.append(os.path.join(folder, filename))

        num_faces = len(file_paths)
        y_size, x_size = self.image_size
        faces = np.zeros(shape=(num_faces, y_size, x_size))
        emotions = np.zeros(shape=(num_faces, num_classes))
        for file_arg, file_path in enumerate(file_paths):
            image_array = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            image_array = cv2.resize(image_array, (y_size, x_size))
            faces[file_arg] = image_array
            file_basename = os.path.basename(file_path)
            file_emotion = file_basename[4:6]
            # there are two file names in the dataset
            # that don't match the given classes
            try:
                emotion_arg = class_to_arg[file_emotion]
            except:
                continue
            emotions[file_arg, emotion_arg] = 1
        faces = np.expand_dims(faces, -1)
        return faces, emotions


def get_labels(dataset_name):
    if dataset_name == 'fer2013':
        return {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy',
                4: 'sad', 5: 'surprise', 6: 'neutral'}
    elif dataset_name == 'imdb':
        return {0: 'woman', 1: 'man'}
    elif dataset_name == 'KDEF':
        return {0: 'AN', 1: 'DI', 2: 'AF', 3: 'HA', 4: 'SA', 5: 'SU', 6: 'NE'}
    elif dataset_name == 'ckplus':
        return {0: 'neutral', 1: 'anger', 2: 'contempt', 3: 'disgust', 4: 'fear', 5: 'happy', 6: 'sadness', 7: 'surprise'}
    elif dataset_name == 'ferplus':
        return {0: 'neutral', 1: 'happiness', 2: 'surprise', 3: 'sadness', 4: 'anger', 5: 'disgust', 6: 'fear',
                7: 'contempt', 8:'unknown', 9:'NF'}
    else:
        raise Exception('Invalid dataset name')


def get_class_to_arg(dataset_name='fer2013'):
    if dataset_name == 'fer2013':
        return {'angry': 0, 'disgust': 1, 'fear': 2, 'happy': 3, 'sad': 4,
                'surprise': 5, 'neutral': 6}
    elif dataset_name == 'imdb':
        return {'woman': 0, 'man': 1}
    elif dataset_name == 'KDEF':
        return {'AN': 0, 'DI': 1, 'AF': 2, 'HA': 3, 'SA': 4, 'SU': 5, 'NE': 6}
    else:
        raise Exception('Invalid dataset name')


def split_imdb_data(ground_truth_data, validation_split=.2, do_shuffle=False):
    ground_truth_keys = sorted(ground_truth_data.keys())
    if do_shuffle is not False:
        shuffle(ground_truth_keys)
    training_split = 1 - validation_split
    num_train = int(training_split * len(ground_truth_keys))
    train_keys = ground_truth_keys[:num_train]
    validation_keys = ground_truth_keys[num_train:]
    return train_keys, validation_keys


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
