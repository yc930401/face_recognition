import cv2
import os
import sys
import numpy as np
from PIL import Image
sys.path.insert(0, '/Workspace-Github/face_recognition/code')
import opencv_tools
from sklearn.decomposition import RandomizedPCA
from sklearn.neighbors import KNeighborsClassifier

subjects = ["", "YANG MI", "BABY"]

def prepare_training_data(data_folder_path, n_component):
    #faces,labels = opencv_tools.prepare_training_data(data_folder_path)
    f = open('D:/Workspace-Github/face_recognition/serialized/data_train.file', 'rb')
    data = pickle.load(f)
    faces, labels = data[0], data[1]
    x_train = []
    for face in faces:
        im = Image.fromarray(face)
        imResize = im.resize((128,128), Image.ANTIALIAS)
        x_train.append(np.array(imResize).reshape(1, -1)[0])
    y_train = labels
    pca = RandomizedPCA(n_components=n_component)
    x_train = pca.fit_transform(x_train)
    print(pca.explained_variance_ratio_)
    return np.array(x_train), np.array(y_train),pca
    
    
    
def train_CNN(x_train, y_train):
    
    knn = KNeighborsClassifier(n_neighbors = 3, weights = 'distance')#, weights = 'distance')
    knn.fit(x_train, y_train)
    return knn


def predict(test_img, model, pca):
    #make a copy of the image as we don't want to chang original image
    img = test_img.copy()
    #detect face from the image
    face, rect = opencv_tools.detect_face_CV2(img)
    im = Image.fromarray(face)
    imResize = im.resize((128,128), Image.ANTIALIAS)

    y_test = pca.transform(np.array(imResize).reshape(1, -1))
    #predict the image using our face recognizer 
    y_label = model.predict(y_test)[0]
    print(y_label)
    #get name of respective label returned by face recognizer
    label_text = subjects[y_label]
    
    #draw a rectangle around face detected
    opencv_tools.draw_rectangle(img, rect)
    #draw name of predicted person
    opencv_tools.draw_text(img, label_text, rect[0], rect[1]-5)
    
    return img