To Train datasets.

Clone the repository
Create a folder called datasets in the root directory fo the repository
To train on FER13 dataset , create folder called fer2013 in the datasets folder and place the files in the folder
To train on CK+ dataset, create folder called CKPlus in the datasets folder and place the files
Finally run the train_emotion_classifer.py file. (Double check on the paths set in the code)


To see a demo,
give the name of the model to load in demo.py file
place the image to be tested in images directory
run the demo.py file

To generate key-points
place all images for which keypoints has to be generated in a folder
create a folder where keypoint generated images has to be saved
set the path variables in keypointgenerator.py file to the folder where the images are present
run the keypointgenerator.py file

Acknowledgements,
We sincerely thank Octavio Arriaga et al., for their codes posted in GitHub at https://github.com/oarriaga/face_classification

