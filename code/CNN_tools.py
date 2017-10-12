import cv2
import os
import sys
import numpy as np
from PIL import Image
sys.path.insert(0, '/Workspace-Github/face-recognition/code')
import opencv_tools
import keras
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten

subjects = ["", "YANG MI", "BABY"]

def prepare_training_data(data_folder_path):
    faces,labels = opencv_tools.prepare_training_data(data_folder_path)
    x_train = []
    for face in faces:
        im = Image.fromarray(face)
        imResize = im.resize((128,128), Image.ANTIALIAS)
        x_train.append(np.array(imResize))
    y_train = labels
    return np.array(x_train), np.array(y_train)
    
    
    
def train_CNN(x_train, y_train):
    
    batch_size = 50
    num_classes = 2
    epochs = 20
    
    print(np.shape(x_train))
    x_train = x_train.reshape(-1, 16384,1)
    x_train = x_train.astype('float32')
    x_train /= 255
    print(x_train.shape[0], 'train samples')
    
    y_train = keras.utils.to_categorical(y_train-1, num_classes)
    
    img_rows, img_cols = 128, 128
    x_train = x_train.reshape(x_train.shape[0], img_cols, img_rows, 1) #1 means: grey 1 layer
    
    model = Sequential()
    model.add(Conv2D(64, (5, 5), activation='relu', input_shape=(img_cols, img_rows, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(64, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Flatten(input_shape=model.output_shape[1:])) # input: 64 layers of 4*4, output: =64*4*4=1024
    model.add(Dense(64, activation='relu')) #=128
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))
    model.summary()
    
    model.compile(loss=keras.losses.categorical_crossentropy,
                    optimizer=keras.optimizers.SGD(),
                    metrics=['accuracy'])
    
    # check-points
    filepath="weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]
    
    run = model.fit(x_train, y_train,
                      batch_size=batch_size,
                      epochs=epochs,
                      verbose=1,
                      validation_data=None,
                      callbacks=callbacks_list)
    return model


def predict(test_img, model):
    #make a copy of the image as we don't want to chang original image
    img = test_img.copy()
    #detect face from the image
    face, rect = opencv_tools.detect_face_CV2(img)
    im = Image.fromarray(face)
    imResize = im.resize((128,128), Image.ANTIALIAS)

    y_test = (np.array(imResize)/255).reshape(1,128,128,1).astype('float32')
    #predict the image using our face recognizer 
    y_label = np.argmax(model.predict(y_test, verbose=0))+1
    print(y_label)
    #get name of respective label returned by face recognizer
    label_text = subjects[y_label]
    
    #draw a rectangle around face detected
    opencv_tools.draw_rectangle(img, rect)
    #draw name of predicted person
    opencv_tools.draw_text(img, label_text, rect[0], rect[1]-5)
    
    return img