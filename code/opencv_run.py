import os
import cv2
import sys
import numpy as np
sys.path.insert(0, '/Workspace-Github/face-recognition/code')
import opencv_tools


print("Preparing data...")
faces, labels = opencv_tools.prepare_training_data("/Workspace-Github/face-recognition/training-data")
print("Data prepared")

#print total faces and labels
print("Total faces: ", len(faces))
print("Total labels: ", len(labels))


#create our LBPH face recognizer 
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

#or use EigenFaceRecognizer by replacing above line with 
#face_recognizer = cv2.face.EigenFaceRecognizer_create()

#or use FisherFaceRecognizer by replacing above line with 
#face_recognizer = cv2.face.FisherFaceRecognizer_create()

#train our face recognizer of our training faces
face_recognizer.train(faces, np.array(labels))

print("Predicting images...")

#load test images
dirs = os.listdir('/Workspace-Github/face-recognition/test-data')
    #load test images
for i in range(len(dirs)):
    dir = dirs[i]
    test_img = cv2.imread('/Workspace-Github/face-recognition/test-data/' + dir)
    predicted_img = opencv_tools.predict(test_img, face_recognizer)
    cv2.imshow(str(i), cv2.resize(predicted_img, (400, 500)))
cv2.waitKey(0)
cv2.destroyAllWindows()
print("Prediction complete")