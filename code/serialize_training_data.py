import os
import cv2
import sys
import numpy as np
sys.path.insert(0, '/Workspace-Github/face_recognition/code')
import opencv_tools


print("Preparing data...")
opencv_tools.prepare_training_data("/Workspace-Github/face_recognition/training-data", True)
print("Data prepared")

#print total faces and labels
print("Total faces: ", len(faces))
print("Total labels: ", len(labels))
