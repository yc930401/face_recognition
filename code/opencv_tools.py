import cv2
import os
import dlib
import pickle
import numpy as np
from imutils.face_utils import FaceAligner, rect_to_bb

subjects = ["", "YANG MI", "BABY"]


def draw_rectangle(img, rect):
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
#function to draw text on give image starting from
#passed (x, y) coordinates. 
def draw_text(img, text, x, y):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
    

#function to detect face using OpenCV
def detect_face_CV2(img):
    #convert the test image to gray image as opencv face detector expects gray images
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    #load OpenCV face detector, I am using LBP which is fast and provides better result on my dataset.
    face_cascade = cv2.CascadeClassifier('/Workspace-Github/face_recognition/opencv-files/haarcascade_frontalface_alt.xml')
    #face_cascade = cv2.CascadeClassifier('/Workspace-Github/face-recognition/opencv-files/lbpcascade_frontalface.xml')
    
    #let's detect multiscale (some images may be closer to camera than others) images
    #result is a list of faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    #if no faces are detected then return original img
    if (len(faces) == 0):
        return None, None
    
    #under the assumption that there will be only one face,
    #extract the face area
    rect = (x, y, w, h) = faces[0]
    print(rect)
    predictor = dlib.shape_predictor('/Workspace-Github/face_recognition/opencv-files/shape_predictor_68_face_landmarks.dat')
    fa = FaceAligner(predictor, desiredFaceWidth=128)
    face = dlib.rectangle(left=int(x), top=int(y), right=int(x+w), bottom=int(y+h))
    print(face)
    faceAligned = fa.align(img, gray, face)
    #print(x,y,w,h)
    #return only the face part of the image
    return cv2.cvtColor(faceAligned, cv2.COLOR_BGR2GRAY), rect


#function to detect face using dlib
def detect_face_dlib(img):
    #convert the test image to gray image as opencv face detector expects gray images
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    face_detector = dlib.get_frontal_face_detector()
    
    # Run the HOG face detector on the image data.
    # The result will be the bounding boxes of the faces in our image.
    faces = face_detector(img, 1)
    
    #if no faces are detected then return original img
    if (len(faces) == 0):
        return None, None
    #under the assumption that there will be only one face,
    #extract the face area
    (x, y, w, h) = rect_to_bb(faces[0])
    predictor = dlib.shape_predictor('/Workspace-Github/face_recognition/opencv-files/shape_predictor_68_face_landmarks.dat')
    fa = FaceAligner(predictor, desiredFaceWidth=128)
    faceAligned = fa.align(img, gray, faces[0])
    #print(x,y,w,h)
    #return only the face part of the image
    return cv2.cvtColor(faceAligned, cv2.COLOR_BGR2GRAY), (x, y, w, h) #gray[y:y+w, x:x+h], (x,y,w,h)


def prepare_training_data(data_folder_path):
    
    #get the directories (one directory for each subject) in data folder
    dirs = os.listdir(data_folder_path)
    
    #list to hold all subject faces
    faces = []
    #list to hold labels for all subjects
    labels = []
    
    #let's go through each directory and read images within it
    for dir_name in dirs:
        
        #our subject directories start with letter 's' so
        #ignore any non-relevant directories if any
        if not dir_name.startswith("train"):
            continue;
            
        label = int(dir_name.replace("train", ""))
        subject_dir_path = data_folder_path + "/" + dir_name
        
        #get the images names that are inside the given subject directory
        subject_images_names = os.listdir(subject_dir_path)
        
        for image_name in subject_images_names:
    
            if image_name.startswith("."):
                continue;
            
            image_path = subject_dir_path + "/" + image_name

            #read image
            image = cv2.imread(image_path)
            
            #detect face_
            face, rect = detect_face_CV2(image)
                        
            if face is not None and min(rect) >= 0:
                #add face to list of faces
                faces.append(face)
                #add label for this face
                labels.append(label)
                #display original image
                draw_rectangle(image, rect)
                cv2.imshow("Training on image...", cv2.resize(image, (400, 500)))
                # display aligned image
                cv2.imshow("Aligned", face)
                cv2.waitKey(100)
            else:
                print('Cannot detect face! ')
            
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cv2.destroyAllWindows()
    f = open('D:/Workspace-Github/face_recognition/serialized/data_train.file', 'wb')
    pickle.dump([faces, labels], f, pickle.HIGHEST_PROTOCOL)
    return faces, labels



#this function recognizes the person in image passed
#and draws a rectangle around detected face with name of the 
#subject
def predict(test_img, face_recognizer):
    #make a copy of the image as we don't want to chang original image
    img = test_img.copy()
    #detect face from the image
    face, rect = detect_face_CV2(img)

    #predict the image using our face recognizer 
    label= face_recognizer.predict(face)[0]
    #get name of respective label returned by face recognizer
    label_text = subjects[label]
    
    #draw a rectangle around face detected
    draw_rectangle(img, rect)
    #draw name of predicted person
    draw_text(img, label_text, rect[0], rect[1]-5)
    
    return img






