# face-recognition-training-testing
It is performed over Yalefaces datasets

# Basically the entire process can be summarised in few steps:
1> training the recognizer model using the LBPH feature vector.

2> the above training model will create the "train.xml" file.

3> Now this XML file can be used directly in test.py file to match for faces.

# Theory:
1> first we try to extract the faces from the original images by a face detection algorithm making use of Haar cascade classifiers.

2> Now that we have extracted the faces, We also assign the corresponding labels to them mentioned in the name of images.

3> Using the local binary pattern histogram(LBPH) feature vector  we train our model using any suitable machine learning algorithm and then  apply then predict the matching confidence of our images.

