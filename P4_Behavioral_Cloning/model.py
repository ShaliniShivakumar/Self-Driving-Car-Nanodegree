from os import path
import csv
import cv2
import numpy as np
import sklearn
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Conv2D, Dropout
import matplotlib.pyplot as plt


def loadDataFromCSV(basePath):
    lines = []  
    with open(path.join(basePath, 'driving_log.csv')) as f:
        content = csv.reader(f)
        next(content)
        for line in content:
            lines.append(line)                                                    
    return lines

    
#Load csv file
#basePath = '/home/workspace/CarND-Behavioral-Cloning-P3/data/'
basePath = './data/'
print("Loading data")
samples = loadDataFromCSV(basePath)

print(samples[0])

images = []
angles = []
for sample in samples:
    for i in range(3):    
        source_path = sample[i]
        filename = source_path.split('/')[-1]         
        current_path = "./data/IMG/"+filename
        if current_path is None:
            print("image is empty")
        image = cv2.imread(current_path)
        new_image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        images.append(new_image)
        angle = float(sample[3])
        
        correction = 0.2
        steering = 0.0
        
        direction = filename.split('_')
        
        if "left" in direction:
            steering = angle + correction
        elif "right" in direction:
            steering = angle - correction
        else:
            steering = angle
                    
        angles.append(steering)
    

#Split data in traning and validation
train_images, validation_images, train_angles, validation_angles = train_test_split(images, angles, test_size=0.2)

print("train_images shape: ", len(train_images))
print("validation_images shape: ", len(validation_images))
total_epochs = 5
batch_size = 64

def geo_transform_image(image, x_pixel, y_pixel):

    '''
    Shift the given image by x_pixel and y_pixel amount
    '''
    rows, cols, dim = image.shape
    M = np.float32([[1,0,x_pixel],[0,1,y_pixel]])
    geo_trans_img = cv2.warpAffine(image,M,(cols,rows))
    return geo_trans_img

def generator(train_images, train_angles, batch_size=32):
    
    num_train_images = len(train_images)
    
    while(1):
        shuffle(train_images, train_angles)
        
        for offset in range(0, num_train_images, batch_size):
           batch_images = train_images[offset:offset+batch_size]
           batch_angles = train_angles[offset:offset+batch_size] 
           augmented_images = []
           augmented_angles = []           
            
           for image, steering_angle in zip(batch_images, batch_angles):
              image = cv2.GaussianBlur(image, (3,3), 0)
              augmented_images.append(image)
              augmented_angles.append(steering_angle)
              #plt.show(image) 
            
              if abs(steering_angle) > 0.43:
                #Flip image vertically in y-axis    
                flipped_image = cv2.flip(image, 1)
                augmented_images.append(flipped_image)
                augmented_angles.append(steering_angle* -1.0)
                #plt.show(flipped_image)
                
                
                #Geo transform the flipped image
                image = geo_transform_image(flipped_image, 10, 10)
                augmented_images.append(image)
                augmented_angles.append(steering_angle* -1.0)
           
           X_train = np.array(augmented_images) 
           y_train = np.array(augmented_angles) 
           
           yield shuffle(X_train, y_train) 
                
                
train_generator = generator(train_images, train_angles, batch_size)
validation_generator = generator(validation_images,validation_angles, batch_size)


model = Sequential()

model.add(Lambda(lambda x: x/255.0 -0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping =((70,25), (0,0))) )

# 5x5 kernel with strides of 2x2, input depth 3 output depth 24
model.add(Conv2D(24,5,5,subsample=(2,2),activation="relu"))

# 5x5 kernel with strides of 2x2, input depth 24 output depth 36
model.add(Conv2D(36,5,5,subsample=(2,2),activation="relu"))

# 5x5 kernel with strides of 2x2, input depth 36 output depth 48
model.add(Conv2D(48,5,5,subsample=(2,2),activation="relu"))

# 3x3 kernel with strides of 1x1, input depth 48 output depth 64
model.add(Conv2D(64,3,3,activation="relu"))

# 3x3 kernel with strides of 1x1, input depth 64 output depth 64
model.add(Conv2D(64,3,3,activation="relu"))

model.add(Flatten())

model.add(Dropout(0.40))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))
model.summary()
adam =Adam(lr=0.001)

model.compile(loss='mse', optimizer=adam)

history_object = model.fit_generator(train_generator, samples_per_epoch = \
                                     len(train_images), \
                                     validation_data=validation_generator, \
                                     nb_val_samples=len(validation_images), \
                                     nb_epoch=total_epochs, verbose = 1)

model.save('model.h5')

   
