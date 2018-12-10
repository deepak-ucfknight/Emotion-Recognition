To Train datasets.

1. Clone the repository
2. Create a folder called datasets in the root directory fo the repository
3. To train on FER13 dataset , create folder called fer2013 in the datasets folder and place the files in the folder
4. To train on CK+ dataset, create folder called CKPlus in the datasets folder and place the files
5. Finally run the train_emotion_classifer.py file. (Double check on the paths set in the code)


To see a demo,


1. give the name of the model to load in demo.py file
2. place the image to be tested in images directory
3. run the demo.py file
4. new file called ‘predicted_test_image’ wil be generate containing the prediction

To generate key-points


1. place all images for which keypoints has to be generated in a folder
2. create a folder where keypoint generated images has to be saved
3. set the path variables in keypointgenerator.py file to the folder where the images are present
4. run the keypointgenerator.py file

Datasets,

CK and CK+: http://www.pitt.edu/~emotion/ck-spread.htm
FER2013: https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data
FER-plus: https://github.com/Microsoft/FERPlus

Environment Setup

1. Install miniconda or Anaconda
2. In the command prompt type, conda env create -f environment.yml
3. this will create all the necessary setup related for the project
4. activate the environment by typing conda activate gpu_env in command prompt
5. deactivae the environment by typing conda deactivate

Acknowledgements,

We sincerely thank Octavio Arriaga et al., for their codes posted in GitHub at https://github.com/oarriaga/face_classification
