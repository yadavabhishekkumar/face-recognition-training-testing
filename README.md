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

# Requirements in terms of environment

The above project was performed in python language over compiler of version 2.7.12 in 64 bit mode. The most important component of this project was open source tool "OpenCV" version 2.4.13. You will also need to import Pillow, numpy,OS.

# Possible errors

There might be some technical difficulties (like in the installation of numpy or python) that you all may face so just mail the error(how so ever trivial error may be) to my mail id: "ay70415" +"@gmail.com" . Just join them (Did this for security purpose from web crawlers)
