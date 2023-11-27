import torch
import uuid
import os
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt


# Loading model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')


# Try and detect an image
img = 'https://www.telegraph.co.uk/content/dam/news/2023/01/15/TELEMMGLPICT000322063156_trans_NvBQzQNjv4BqpVlberWd9EgFPZtcLiMQf0Rf_Wk3V23H2268P_XkPxc.jpeg?imwidth=1280' 

results = model(img)
print(results)

%matplotlib inline
plt.imshow(np.squeeze(results.render()))
plt.show()


# Real time detection using OpenCV - VideoCapture(0) for camera, otherwise specify video or image
cap = cv2.VideoCapture(0)

while cap.isOpened(): # while capture device is open
    ret, frame = cap.read() # reading capture and unpacking the return value and frame

    results = model(frame) # make detections - passing through frame from video capture to model


    cv2.imshow('YOLO', np.squeeze(results.render())) # squeezing out results from np array and passing it to rendering function

    if cv2.waitKey(10) & 0xFF == ord('q'): # if q is pressed then exit loop
        break

cap.release()
cv2.destroyAllWindows()


# Train a model from scratch with custom images and labels

img_path = os.path.join('data', 'images')
labels = ['awake', 'drowsy']
num_imgs = 20

cap = cv2.VideoCapture(0)

# Loop through the labels
for label in labels:
    print('Collecting images for: {}'.format(label))
    time.sleep(5)

    # Looping through images (num_imgs)
    for image in range(num_imgs):
        print('Collecting images for {}, image number {}'.format(label, image))

        # Webcam on
        ret, frame = cap.read()

        # Image path
        imgname = os.path.join(img_path, label+'.'+str(uuid.uuid1())+'.jpg')

        # writing out img to file
        cv2.imwrite(imgname, frame)

        # Render to screen
        cv2.imshow('Image Collection', frame)

        time.sleep(4) # adding a slight delay so that we can take different shots

    if cv2.waitKey(10) & 0xFF == ord('q'): # if q is pressed then exit loop
        break

      
cap.release()
cv2.destroyAllWindows()

for label in labels:
    print('Collecting images for: {}'.format(label))
    for image in range(num_imgs):
        print('Collecting images for {}, image number {}'.format(label, image))

