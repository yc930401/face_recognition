# face_recognition
Face Recognition with OpenCV, CNN and KNN

## Introduction
In this project, I built three models for face recognition: OpenCV LBPHFaceRecognizer, CNN and KNN.<br />
Main libraries used are cv2, dlib, PIL, keras and sklearn.

## Image preprocess
The images are different size RGB images with faces in different directions. Some photos are full-body photos and some are half-body photos.<br />
1. Change RGB images to gray scale images, because color do not help us classify face in this project.
2. Face detection using cv2 or dlib.
3. Resize images to 128*128.
4. Align images so that the eyes in all images are in the same positions.<br />
Images after preprocess are shown below.<br />
![Exmaple result](/result/preprocess_1.jpg)
![Exmaple result](/result/preprocess_2.jpg)

## Models
1. OpenCV LBPHFaceRecognizer
2. CNN model in keras
3. KNN using sklearn

## Result
![Exmaple result](/result/KNN/2.jpg)
![Exmaple result](/result/KNN/3.jpg)
![Exmaple result](/result/KNN/5.jpg)
![Exmaple result](/result/KNN/6.jpg)

## References
https://www.pyimagesearch.com/2017/05/22/face-alignment-with-opencv-and-python/
http://nbviewer.jupyter.org/gist/hernamesbarbara/5768969
https://www.superdatascience.com/opencv-face-recognition/
https://medium.com/@ageitgey/machine-learning-is-fun-part-4-modern-face-recognition-with-deep-learning-c3cffc121d78
