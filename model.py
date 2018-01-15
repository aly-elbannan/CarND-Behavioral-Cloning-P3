import csv
import re
import cv2
import numpy as np

datasets = ["recorded_data/sample_data", "recorded_data/track1", "recorded_data/track2"]
model_save_path = "models/nvidia_sample_data_track1_track2_dropout"
num_epochs = 10
start_epoch = 0
random_state = None

dataset_samples = []

for dataset_path in datasets:
    csv_file = open(dataset_path + "/driving_log.csv", 'r')
    driving_log = csv.reader(csv_file)
    for line in driving_log:
        line[0] = dataset_path+"/IMG/"+re.split('/|\\\\', line[0])[-1]
        line[1] = dataset_path+"/IMG/"+re.split('/|\\\\', line[1])[-1]
        line[2] = dataset_path+"/IMG/"+re.split('/|\\\\', line[2])[-1]
        dataset_samples.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(dataset_samples, test_size=0.2, random_state=random_state)

def sample_generator(samples, data_augmentation_enable=False, multiple_cameras_enabled=False, steering_correction=0.2, batch_size=32):
    
    _X_train = []
    _y_train = []
    _X_train_nparray = None
    _y_train_nparray = None
    count_batch = 0
    count_total = 0
    
    total_num_samples = len(samples)
    if data_augmentation_enable: total_num_samples *= 2
    if multiple_cameras_enabled: total_num_samples *= 3
    
    def add_to_batch(image, angle):
        nonlocal count_batch, count_total, _X_train, _y_train
        count_batch += 1
        count_total += 1
        _X_train.append(image)
        _y_train.append(angle)
    
    def is_batch_complete():
        nonlocal count_batch, count_total, _X_train, _y_train, _X_train_nparray, _y_train_nparray
        if count_batch == batch_size or count_total == total_num_samples:
            _X_train_nparray = np.asarray(_X_train)
            _y_train_nparray = np.asarray(_y_train)
            _X_train = []
            _y_train = []
            count_batch = 0
            if count_total == total_num_samples: count_total = 0
            return True
        else:
            return False
    
    while True:
        for line in samples:
            
            # Read image paths
            center_image_path   = line[0]
            left_image_path     = line[1]
            right_image_path    = line[2]
            
            # Read recorded steering angle
            steering_angle = float(line[3])
            
            # Read center image and convert to RGB
            center_image = cv2.imread(center_image_path)
            center_image = cv2.cvtColor(center_image, cv2.COLOR_BGR2RGB)
            
            # Add center image to batch and yield if batch is complete
            add_to_batch(center_image, steering_angle)
            if is_batch_complete(): yield _X_train_nparray, _y_train_nparray
            
            if multiple_cameras_enabled:
                if data_augmentation_enable:
                    
                    # Flip center image
                    flipped_center_image = cv2.flip(center_image, flipCode=1)
                    
                    # Add flipped center image to batch and yield if batch is complete
                    add_to_batch(flipped_center_image, -steering_angle)
                    if is_batch_complete(): yield _X_train_nparray, _y_train_nparray
                
                # Read left image and convert to RGB
                left_image = cv2.imread(left_image_path)    
                left_image = cv2.cvtColor(left_image, cv2.COLOR_BGR2RGB)
                
                # Add left image to batch and yield if batch is complete
                add_to_batch(left_image, steering_angle+steering_correction)
                if is_batch_complete(): yield _X_train_nparray, _y_train_nparray
                
                if data_augmentation_enable:
                    
                    # Flip left image
                    flipped_left_image = cv2.flip(left_image, flipCode=1)
                    
                    # Add flipped left image to batch and yield if batch is complete
                    add_to_batch(flipped_left_image, -(steering_angle+steering_correction))
                    if is_batch_complete(): yield _X_train_nparray, _y_train_nparray
                
                # Read right image and convert to RGB
                right_image = cv2.imread(right_image_path)    
                right_image = cv2.cvtColor(right_image, cv2.COLOR_BGR2RGB)
                
                # Add right image to batch and yield if batch is complete
                add_to_batch(right_image, steering_angle-steering_correction)
                if is_batch_complete(): yield _X_train_nparray, _y_train_nparray
                
                if data_augmentation_enable:
                    
                    # Flip right image
                    flipped_right_image = cv2.flip(right_image, flipCode=1)
                    
                    # Add flipped left image to batch and yield if batch is complete
                    add_to_batch(flipped_right_image, -(steering_angle-steering_correction))
                    if is_batch_complete(): yield _X_train_nparray, _y_train_nparray
            else:            
                if data_augmentation_enable:
                    
                    # Flip center image
                    flipped_center_image = cv2.flip(center_image, flipCode=1)
                    
                    # Add flipped center image to batch and yield if batch is complete
                    add_to_batch(flipped_center_image, -steering_angle)
                    if is_batch_complete(): yield _X_train_nparray, _y_train_nparray

train_sample_generator = sample_generator(train_samples, batch_size=128, data_augmentation_enable=True, multiple_cameras_enabled=True)
validation_sample_generator = sample_generator(validation_samples, batch_size=32, data_augmentation_enable=False, multiple_cameras_enabled=False)


from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout, Conv2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
from keras.callbacks import LambdaCallback    

def LeNet(model):
    model.add(Conv2D(6, 3, 3, activation='relu'))
    model.add(Conv2D(16, 3, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    
    return model


def Nvidia(model, dropout=True):
    model.add(Conv2D(24, 5, 5, activation='relu', subsample=(2, 2)))
    model.add(Conv2D(36, 5, 5, activation='relu', subsample=(2, 2)))
    model.add(Conv2D(48, 5, 5, activation='relu', subsample=(2, 2)))
    model.add(Conv2D(64, 3, 3, activation='relu'))
    model.add(Conv2D(64, 3, 3, activation='relu'))
    model.add(Flatten())
    
    if dropout == True:
        model.add(Dropout(0.25))
    
    model.add(Dense(100))
    
    if dropout == True:
        model.add(Dropout(0.25))
    
    model.add(Dense(50))
    
    if dropout == True:
        model.add(Dropout(0.25))
    
    model.add(Dense(10))
    
    if dropout == True:
        model.add(Dropout(0.25))
    
    return model

def create_model():
    # Create Model
    model = Sequential()
    
    # Normalizaton layers
    ## Cropping to remove horizon and hood
    model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160, 320, 3)))
    ## Normalize pixel values to [-0.5, 0.5]
    model.add(Lambda(lambda x: (x/255.0) - 0.5))
    
    # Choose DNN
    model = Nvidia(model)
    
    # output layer of size 1
    model.add(Dense(1))
    
    return model


from keras.models import load_model
import glob
import os

if start_epoch > 0:
    file_path = glob.glob(model_save_path+"/model_epochs/model_epoch{}*".format(start_epoch-1))
    assert len(file_path) == 1
    print("Loading model from path: {}".format(file_path[0]))
    model = load_model(file_path[0])
else:
    model = create_model()
    model.compile(optimizer='adam', loss='mse')

if not os.path.exists(model_save_path+"/model_epochs"):
    os.makedirs(model_save_path+"/model_epochs")

def on_epoch_end(epoch, logs):
    # Save the model on 
    model.save(model_save_path+"/model_epochs/model_epoch{}_loss_{:.4f}_val_loss_{:.4f}.h5".format(epoch+1, logs['loss'], logs['val_loss']))
      
lambda_cbk = LambdaCallback(on_epoch_begin=None, on_epoch_end=on_epoch_end, on_batch_begin=None, on_batch_end=None, on_train_begin=None, on_train_end=None)

model.fit_generator(train_sample_generator, validation_data=validation_sample_generator, samples_per_epoch=len(train_samples)*6, nb_val_samples=len(validation_samples), initial_epoch=start_epoch, nb_epoch=num_epochs, callbacks=[lambda_cbk])

model.save(model_save_path+"/model.h5")