#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import os
from matplotlib import pyplot as plt
import cv2
import random
import pickle
from tqdm import tqdm
from PIL import Image


# In[2]:


import tensorflow as tf 
from tensorflow.keras.models import Sequential,model_from_json
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D,MaxPool2D,InputLayer, BatchNormalization
#from keras.models import model_from_json
#from keras.models import load_model
from keras.utils import np_utils
from keras.optimizers import Adam


# In[3]:


if not tf.test.gpu_device_name():
    print("no gpu")
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


# In[4]:


DATADIR = r"C:\D\Graduation Project 2\datasets\model 1\train"
# Loadin Train Data
CATEGORIES = ['not car', 'car']

IMG_SIZE = 128
training_data = []
def create_training_data():
    for category in tqdm(CATEGORIES) :
        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try :
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                img_array = cv2.resize(img_array, (IMG_SIZE,IMG_SIZE))
                training_data.append([img_array, class_num])
            except Exception as e:
                pass

create_training_data()

random.shuffle(training_data)
print('Training Data Loaded', len(training_data))


# In[5]:


test_DATADIR = r"C:\D\Graduation Project 2\datasets\model 1\test"
# Loadin Train Data
CATEGORIES = ['not car', 'car']

IMG_SIZE = 128
testing_data = []
def create_testing_data():
    for category in tqdm(CATEGORIES) :
        path = os.path.join(test_DATADIR, category)
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try :
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                img_array = cv2.resize(img_array, (IMG_SIZE,IMG_SIZE))
                testing_data.append([img_array, class_num])
            except Exception as e:
                pass

create_testing_data()

random.shuffle(testing_data)
print('Testing Data Loaded', len(testing_data))


# In[9]:


a = 1010
plt.imshow(training_data[a][0],cmap='Greys_r')
print(training_data[a][1])
print(training_data[a][0].shape)
print(len(training_data))


# In[10]:



plt.imshow(testing_data[a][0],cmap='Greys_r')
print(testing_data[a][1])
print(testing_data[a][0].shape)
print(len(testing_data))


# In[11]:


X_train = [] 
Y_train = [] 
X_test = [] 
Y_test = [] 

for features, label in training_data:
    X_train.append(features)
    Y_train.append(label)

for features, label in testing_data:
    X_test.append(features)
    Y_test.append(label)


X_train = np.array(X_train).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
X_test = np.array(X_test).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
Y_train=np.array(Y_train)
Y_test=np.array(Y_test)


# In[12]:


print('X_train:',X_train.shape)
print('Y_train:',Y_train.shape)
print('X_test:',X_test.shape)
print('Y_test:',Y_test.shape)
print(X_train.shape[1:])
print(X_test.shape[1:])


# In[13]:


#X = X/255.0
#y_train=np_utils.to_categorical(y_train)
#y_test=np_utils.to_categorical(y_test)


# In[14]:


print('Y_train:',Y_train.shape)
print('Y_test:',Y_test.shape)
i=1010
print(Y_train[i])
print(Y_test[i])
plt.imshow(X_train[i],cmap='Greys_r')


# In[79]:


from time import time
from tensorflow.python.keras.callbacks import TensorBoard
import datetime


# In[93]:


model = Sequential()

# model.add(InputLayer(input_shape=X_train.shape[1:]))

model.add(Conv2D(16, (3, 3), input_shape = X_train.shape[1:], padding='same', activation='relu',strides=(2,2)))
model.add(Conv2D(16, (3, 3), padding='same', activation='relu',strides=(2,2)))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32, (3, 3), padding='same', activation='relu',strides=(2,2)))
model.add(Conv2D(32, (3, 3), padding='same', activation='relu',strides=(2,2)))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(16))
model.add(Activation("relu"))

model.add(Dense(16))
model.add(Activation("relu"))

model.add(Dense(8))
model.add(Activation("relu"))

model.add(Dense(8))
model.add(Activation("relu"))


model.add(Dense(2))
model.add(Activation("softmax"))
# model.add(Activation("softmax"))


# In[64]:


cnn1 = Sequential([
    Conv2D(16, kernel_size=(3, 3), activation='relu',padding='same',input_shape=X_train.shape[1:]),
    Conv2D(16, kernel_size=(3, 3), activation='relu',padding='same'),
    MaxPooling2D(pool_size=(2, 2),strides=2),
    Conv2D(32, kernel_size=(3, 3), activation='relu'),
    Conv2D(32, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2),strides=2),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(10, activation='softmax')
])


# In[94]:


t = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard = TensorBoard(log_dir="logs/{}".format(t))


# In[95]:


# model.compile(loss="sparse_categorical_crossentropy",optimizer="adam",metrics=["accuracy"], )
# cnn1.compile(Adam(lr=.0001),loss="sparse_categorical_crossentropy",metrics=["accuracy"], )
model.compile(Adam(lr=.0001),loss="sparse_categorical_crossentropy",metrics=["accuracy"], )


# In[96]:


carDetection = model.fit(X_train, Y_train, batch_size=32, epochs=20, validation_data=(X_test,Y_test), callbacks=[tensorboard])
# carDetection = cnn1.fit(X_train, Y_train, batch_size=32, epochs=20, validation_data=(X_test,Y_test), callbacks=[tensorboard])


# In[97]:



def prepare(path, IMG_SIZE = 64):
    kernel = np.ones((5,5),np.float32)/70
    
    img_array = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    #new_array = cv2.filter2D(new_array,-1,kernel)
    #new_array = cv2.medianBlur(new_array,5)
    #new_array = cv2.bilateralFilter(new_array,9,75,75)
    #new_array = cv2.GaussianBlur(new_array,(5,5),0)
    #new_array = cv2.Sobel(new_array,cv2.CV_64F,0,1,ksize=5)
    plt.imshow(new_array,cmap='Greys_r')

    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)


# In[54]:


p = r"C:\D\Graduation Project 2\datasets\Model 3 Datasets\side\BMW_Z4_2011_47_17_250_30_6_70_50_167_18_RWD_2_2_Convertible_AMB.jpg"
image = prepare(r"C:\D\test cars\m.jpg",IMG_SIZE=128)
#image = prepare(p)
prediction = cnn1.predict([image])
f=prediction[0][0]*100
r=prediction[0][1]*100
s=prediction[0][2]*100
print('%.2f'%f,'%',' Front')
print('%.2f'%r,'%',' Rear')
print('%.2f'%s,'%',' Side')


# In[100]:


p = r"C:\D\Graduation Project 2\datasets\Model 3 Datasets\side\BMW_Z4_2011_47_17_250_30_6_70_50_167_18_RWD_2_2_Convertible_AMB.jpg"
image = prepare(r"C:\D\test cars\m1.jpg",IMG_SIZE=128)
#image = prepare(p)
prediction = model.predict([image])
f=prediction[0][0]*100
r=prediction[0][1]*100
print('%.2f'%f,'%',' Not a Car')
print('%.2f'%r,'%',' Car')


# In[77]:


# model.summary()
model.summary()


# In[30]:


test_DATADIR = r"C:\D\test cars"
# Loadin Train Data
CATEGORIES = ['front', 'rear', 'side']

IMG_SIZE = 128
validation_data = []
def create_validating_data():
    for category in tqdm(CATEGORIES) :
        path = os.path.join(test_DATADIR, category)
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try :
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                img_array = cv2.resize(img_array, (IMG_SIZE,IMG_SIZE))
                validation_data.append([img_array, class_num])
            except Exception as e:
                pass

create_validating_data()

random.shuffle(validation_data)
print('Testing Data Loaded', len(validation_data))

X_validate = [] 
Y_validate = [] 

for features, label in validation_data:
    X_validate.append(features)
    Y_validate.append(label)


X_validate = np.array(X_validate).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
Y_validate=np.array(Y_validate)


# In[31]:


loss, accuracy = model.evaluate(X_validate, Y_validate)
print (loss*100, accuracy*100)


# In[ ]:





# In[126]:


os.mkdir(r"C:\D\Graduation Project 2\models\Side Models\code v2\{}".format(t))
model_json = model.to_json()
with open(r"C:\D\Graduation Project 2\models\Side Models\code v2\{}\model.json".format(t), "w") as json_file :
	json_file.write(model_json)

model.save_weights(r"C:\D\Graduation Project 2\models\Side Models\code v2\{}\model.h5".format(t))
print("Saved model to disk")

model.save(r'C:\D\Graduation Project 2\models\Side Models\code v2\{}\CNN.model'.format(t))


# In[57]:


img = Image.open(r"C:\D\test cars\dc4.jpg") # image extension *.png,*.jpg
new_width  = 256
new_height = 256
img = img.resize((new_width, new_height), Image.ANTIALIAS)
img.save(r"C:\D\test cars\rdc.jpg")


# In[ ]:




