#importing the libraries
import os 
import cv2
import numpy as np
import matplotlib.pyplot as plt
#setting the path to the directory containing the pics
path = r'C:\Users\HP\Documents\Py Programs\FAJLOVI ISPIT IIS\seminarski 2\Medin'
#appending the pics to the training data list
training_data = []
for img in os.listdir(path):
    pic = cv2.imread(os.path.join(path,img))
    pic = cv2.cvtColor(pic,cv2.COLOR_BGR2RGB)
    pic = cv2.resize(pic,(80,80))
    training_data.append([pic])
#converting the list to numpy array and saving it to a file using #numpy.save
np.save(os.path.join(path,'features.npy'),np.array(training_data))
#loading the saved file once again
saved = np.load(os.path.join(path,'features.npy'))
#plt.imshow(saved[0].reshape(80,80,3))
#plt.imshow(np.array(training_data[0]).reshape(80,80,3))
#plt.show()