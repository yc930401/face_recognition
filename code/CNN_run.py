import os
import cv2
import sys
import numpy as np
sys.path.insert(0, '/Workspace-Github/face-recognition/code')
import opencv_tools
import CNN_tools


print("Preparing data...")
x_train, y_train = CNN_tools.prepare_training_data("/Workspace-Github/face-recognition/training-data")
print("Data prepared")


model = CNN_tools.train_CNN(x_train, y_train)

print("Predicting images...")
#load test images
dirs = os.listdir('/Workspace-Github/face-recognition/test-data')
    #load test images
for i in range(len(dirs)):
    dir = dirs[i]
    test_img = cv2.imread('/Workspace-Github/face-recognition/test-data/' + dir)
    predicted_img = CNN_tools.predict(test_img, model)
    cv2.imshow(str(i), cv2.resize(predicted_img, (400, 500)))
cv2.waitKey(0)
cv2.destroyAllWindows()
print("Prediction complete")