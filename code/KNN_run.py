import os
import cv2
import sys
import numpy as np
sys.path.insert(0, '/Workspace-Github/face_recognition/code')
import opencv_tools
import KNN_tools


print("Preparing data...")
x_train, y_train, pca = KNN_tools.prepare_training_data("/Workspace-Github/face_recognition/training-data", 10)
print("Data prepared")


model = KNN_tools.train_CNN(x_train, y_train)

print("Predicting images...")
#load test images
dirs = os.listdir('/Workspace-Github/face_recognition/test-data')
    #load test images
for i in range(len(dirs)):
    dir = dirs[i]
    test_img = cv2.imread('/Workspace-Github/face_recognition/test-data/' + dir)
    predicted_img = KNN_tools.predict(test_img, model, pca)
    cv2.imshow(str(i), cv2.resize(predicted_img, (400, 500)))
    cv2.imwrite('/Workspace-Github/face_recognition/result/KNN/' + dir, predicted_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
print("Prediction complete")