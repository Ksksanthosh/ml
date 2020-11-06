from google.colab import drive
drive.mount('/content/drive')
#importing this to use files in drive

#required packages
import numpy as np
import cv2 
import os
import random
import matplotlib.pyplot as plt
import pickle

#declaring mu path and the categories
DIRECTORY= r'/content/drive/My Drive/Colab Notebooks/IMAGE_CLASSIFICATION/train' #mention the directory where your training files are
CATEGORIES =['cat', 'dog']

IMG_SIZE=120 #sixe of the image

data=[]

#loop categorize the images

for category in CATEGORIES:
        folder=os.path.join(DIRECTORY,category)
        label = CATEGORIES.index(category)
        for img in os.listdir(folder):
            img_path= os.path.join(folder,img)
            img_arr= cv2.imread(img_path)
            img_arr=cv2.resize(img_arr, (IMG_SIZE,IMG_SIZE))
            data.append([img_arr,label])

random.shuffle(data) #the data will be organised

#separating the data into x and y
x=[]
y=[]


for features, labels in data:
        x.append(features)
        y.append(labels)

#converting it into the array
x= np.array(x)
y= np.array(y)

pickle.dump(x, open('x.pkl','wb'))
pickle.dump(y, open('y.pkl','wb'))

#x contains the values of pixels from 0 to 255, for easy calucation dividing it by 255 so the values will be 0 to 1
x=x/255

x.shape

#importing required libraries for building our model
from keras.models import Sequential
from keras.layers import Conv2D , MaxPooling2D , Flatten , Dense

#building the model
model = Sequential()

model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D((2,2)))

model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D((2,2)))

model.add(Flatten())

model.add(Dense(128,input_shape=x.shape[1:], activation='relu'))

model.add(Dense(2, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

results=model.fit(x,y, epochs=5, validation_split=0.1)

model.save('cat.h5')

#ploting the accurary
plt.plot(results.history['accuracy'])

