

# ### Importing All necessary libraries

# In[1]:

import os
import csv
import cv2
import sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from keras.models import Sequential, Model, model_from_json
from keras.layers import Activation, Dense, Dropout, ELU, Flatten, Lambda, SpatialDropout2D
from keras.layers import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


# ### Define Data structures and functions

# In[2]:

raw_samples =   []
car_images =    []
steering_angles=[]


# #### defining image properties

# In[3]:

# define the input size
height = 160
width = 320
depth = 3

# define how images should be cropped
reduction_upper = 70
reduction_lower = 25
reduction_left  = 0
reduction_right = 0

# specify the batch size and training epochs
batch_size = 32
epochs = 5


# #### File-importing function

# In[4]:

def import_samples(csv_filepath, samples):
    with open(csv_filepath) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            samples.append(line)
      
        # samples = samples[1:] # headers removal  
        print("Total number of samples right now = ", len(samples))
   
    return samples


# #### Augmentation Function

# In[5]:

def augmentation(image_path, angle_single):
    
    image_single = cv2.imread(image_path)   
    image_flip = np.fliplr(image_single)    
    angle_negative = (-1.0)*angle_single

    return image_flip, angle_negative  
     


# #### create the generator function

# In[6]:

def generator(samples, batch_size):
    
    no_samples = len(samples)
    
    while 1:
        shuffle(samples)
        
        # loop through each batch of data
        for sample_pos in range(0, no_samples, batch_size):
            sample_batches = samples[sample_pos:sample_pos + batch_size]
            
            center_images = []
            center_angles = []
            left_images = []
            left_angles = []
            right_images=[]
            right_angles=[]
            correction = 0.25   # define the steering correction for left and right cameras.
            
            
            # loop through every single record within a batch of data
            for sample_no in sample_batches:     
                
                source_path_A = sample_no[0]   # specify the path for images taken by central camera.
                center_angle_single = float(sample_no[3])
                center_image_single_flip, angle_negative_center = augmentation(source_path_A,center_angle_single)
                center_image_single = cv2.imread(source_path_A)
                center_images.append(center_image_single)
                center_angles.append(center_angle_single)
                center_images.append(center_image_single_flip)
                center_angles.append(angle_negative_center)
#_____________________________________________________________________________________________________________________                
     
                source_path_B = sample_no[1]   # specify the path for images taken by left camera.        
                left_angle_single = float(sample_no[3])+correction
                left_image_single_flip, angle_negative_left = augmentation(source_path_B,left_angle_single)
                left_image_single = cv2.imread(source_path_B)                         
                center_images.append(left_image_single)
                center_angles.append(left_angle_single)
                center_images.append(left_image_single_flip)
                center_angles.append(angle_negative_left)
#_____________________________________________________________________________________________________________________                  
       
                source_path_C = sample_no[2]  # specify the path for images taken by right camera.
                right_angle_single = float(sample_no[3])-correction                
                right_image_single_flip, angle_negative_right = augmentation(source_path_C,right_angle_single)               
                right_image_single = cv2.imread(source_path_C)            
                center_images.append(right_image_single)
                center_angles.append(right_angle_single)
                center_images.append(right_image_single_flip)
                center_angles.append(angle_negative_right)
#_____________________________________________________________________________________________________________________  
       
            X_train = np.array(center_images)
            y_train = np.array(center_angles)
             
            yield shuffle(X_train, y_train)
                    
                                


# In[7]:

# extracting data from CSV files
current_sample = import_samples('E:/DrivingDataTrackA/driving_log.csv', raw_samples)
current_sample = import_samples('E:/DrivingDataTrackA_enhanced/driving_log.csv', current_sample)


current_sample = import_samples('E:/DrivingDataTrackA_Recovery/driving_log.csv', current_sample)


current_sample = import_samples('E:/DrivingDataTrackB/driving_log.csv', current_sample)
current_sample = import_samples('E:/DrivingData/driving_log.csv', current_sample)


# In[8]:

# train-validation split
train_samples, valid_samples = train_test_split(current_sample, test_size=0.1)


# In[9]:

# define two generators that will be fed into the Nvidia pipeline.
train_generator = generator(train_samples, batch_size)
valid_generator = generator(valid_samples, batch_size)


# #### Use LeNet Architecture

# In[10]:

model = Sequential()
model.add(Cropping2D(cropping=((reduction_upper,reduction_lower),
                               (reduction_left,reduction_right)),input_shape=(height, width, depth)))
model.add(Lambda(lambda x: (x / 255.0) - 0.5))
model.add(Convolution2D(16, (5, 5), strides=(1, 1), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Convolution2D(32, (5, 5), strides=(1, 1), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(32))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.compile(loss='mse',optimizer=Adam(lr = 0.0001),metrics=['mean_absolute_error'])
model.summary()


'''
# NVIDIA MODEL

model = Sequential()
model.add(Cropping2D(cropping=((reduction_upper,reduction_lower),
                               (reduction_left,reduction_right)),input_shape=(height, width, depth)))
model.add(Lambda(lambda x: (x / 255.0) - 0.5))
model.add(Convolution2D(24, (5, 5), strides=(2, 2), activation='relu'))
model.add(Convolution2D(36, (5, 5), strides=(2, 2), activation='relu'))
model.add(Convolution2D(48, (5, 5), strides=(2, 2), activation='relu'))
model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(10, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(1, activation='tanh'))
model.compile(loss='mse', optimizer=Adam(lr=0.0001),metrics=['accuracy'])
model.summary()

'''


# #### Creating a checkpoint

# In[11]:

checkpointer = ModelCheckpoint(
                                    filepath= 'checkpoints/model{epoch:02d}.h5', 
                                    verbose=1, 
                                    save_best_only=False,
                                    save_weights_only=False,
                                    mode = 'auto'
                               
                              )


# In[12]:

# starting the training process 
history_object = model.fit_generator(
                                            generator = train_generator, 
                                            steps_per_epoch=len(train_samples)/batch_size,
                                            epochs = epochs,
                                            verbose = 1,
                                            validation_data=valid_generator,
                                            validation_steps =len(valid_samples)/batch_size,
                                                                           
                                    )

print('Congratulations! Training completed successfully!')
model.save('model_LeNet_two.h5')
print('Model saved successfully!')


# In[13]:


print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()





