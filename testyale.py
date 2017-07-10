import numpy as np 
import os
import cv2

from PIL import Image



faceCascade = cv2.CascadeClassifier("C:\\opencv\\build\\etc\\haarcascades\\haarcascade_frontalface_default.xml");
recognizer = cv2.createLBPHFaceRecognizer()

path ="C:\\Users\\DELL\\codes\\Image\\FACE\\method3\\yalefaces" 
recognizer.load("C:\\Users\\DELL\\codes\\Image\\FACE\\method3\\train.xml")	 

image_paths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.sad')] 

for image_path in image_paths: 
	predict_image_pil = Image.open(image_path).convert('L')
	predict_image = np.array(predict_image_pil, 'uint8')
	faces = faceCascade.detectMultiScale(predict_image) 
	for (x, y, w, h) in faces:
	 nbr_predicted, conf = recognizer.predict(predict_image[y: y + h, x: x + w]) 
	 nbr_actual = int(os.path.split(image_path)[1].split(".")[0].replace("subject", ""))
	 if nbr_actual == nbr_predicted: 
	   print (" {} is Correctly Recognized with confidence {} ").format(nbr_actual, conf) 
	 else:
	   print ("{} is Incorrectly Recognized as {} with confidence {}").format(nbr_actual, nbr_predicted,conf) 
	 cv2.imshow("Recognizing Face", predict_image[y: y + h, x: x + w])
	 cv2.waitKey(10)