#importing the libraries
import os 
import cv2
import numpy as np
import matplotlib.pyplot as plt
#setting the path to the directory containing the pics
path = r'C:\Users\HP\Documents\Py Programs\dina'
#appending the pics to the training data list
training_data = []
for img in os.listdir(path):
    pic = cv2.imread(os.path.join(path,img))
    pic = cv2.resize(pic,(80,80))
    training_data.append([pic])
print(training_data)
#converting the list to numpy array and saving it to a file using #numpy.save
np.save(os.path.join(path,'train.npy'),np.array(training_data))