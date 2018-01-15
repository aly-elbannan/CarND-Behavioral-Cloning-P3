# Behavioral Cloning Project

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

# Submitted Files

1. [model.py](model.py)<br>
Includes the definition of the Network Architecture, and scripts used for training and training/validation sample generation.

1. [drive.py](drive.py)<br>
Script used to communicate with the simulator to drive the car in Autonomous model.

1. [model.h5](model.h5)<br>
Saved output of the trained model to be used for testing.

1. [video.mp4](video.mp4)<br>
Output video of testing the model.

# Network Architecture

![](https://devblogs.nvidia.com/parallelforall/wp-content/uploads/2016/08/cnn-architecture-624x890.png)

The Network Architecture used is based on the architecture published by Nvidia for use in [End-to-End Deep Learning for Self-Driving Cars](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/)
* 3 Convolutional layers with 5x5 Kernel and 2x2 Stride
* 2 Convolutional layers with 3x3 Kernel and 1x1 Stride
* 3 Fully Connected layers with sizes 100, 50 and 10
* ReLU activation function is used for all layers to introduce non-linearity
* Drouput with probability 25% is used for the fully connected layers to reduce overfitting.

The model is implemented in [model.py](./model.py)
```python
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
```

## Input and Output Layers

* Input Image is cropped by 50px from the top to remove the horizon
* Input Image is cropped by 20px from the bottom to remove the hood of the car
* Data normalization is done using Keras Lambda layer so that input image pixel values are in the range [-0.5, 0.5]
* Output layer consists of one fully connected node to control the steering angle

Input and Output layers are implemented in [model.py](./model.py)
```python
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
```

#  Training

## Data Collection

* Sample Training Data provided in project resources. It includes forward and backward laps in track 1
* Manual recording of 2 forward laps in track 1 including recovery scenarios
* Manual recording of 3 forward laps and 3 backward laps in track 2

## Data Augmentation

* Samples from the 3 cameras are used. A steering correction factor (0.2) is added to the recorded steering angle when processing the left image and subtracted when processing the right image
* Images are flipped (left to right) and their steering angle is inverted (*-1)
* After preprocessing all the images are presented to the model as if they are the center image

## Color Space Conversion

* During training, the images are read using opnecv `cv2.imread()` function and converted to RGB format. During inference the model is presented with RGB images.

## Training Parameters

* Loss Functions: Mean Squared Error
* Optimizer: AdamOptimizer to manage the optimization of learning rate
* Number of Epochs: 10

## Training/Validation Sample Generation

Since the dataset is large (~130,000 images), loading the entire dataset in memory for training and validation results in an out of memory exception.

The issue can be resolved in keras by using `model.fit_generator()` function instead of `model.fit()`. This requires the implementation of a [generator](https://wiki.python.org/moin/Generators) that yields a batch set every time it is called instead of requiring the whole training/validation set.

1. First the driving_log.csv is read from each recording session
1. The log lines are merged into one log
1. The samples (lines from the merged log) are split into Training and Validation sets using `sklearn.model_selection.train_test_split()` with ration 20%
1. `train_sample_generator` and `validation_sample_generator` are created to be used with `mode.fit_generator()`

An implementation is provided in [model.py](mode.py)
```python
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
```