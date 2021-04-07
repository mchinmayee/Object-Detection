#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 16:04:29 2021

@author: chinu
"""

import numpy as np
import argparse
import cv2
import matplotlib.pylab as plt


prototextPath = "/Users/myworkspace/InternFolder/Object-Detection-Tutorial-master/MobileNetSSD_deploy.prototxt.txt"

caffeModel = "/Users/myworkspace/InternFolder/Object-Detection-Tutorial-master/MobileNetSSD_deploy.caffemodel"

image = "/Users/myworkspace/InternFolder/Object-Detection-Tutorial-master/images/example_04.jpg"

img_rd = cv2.imread(image)
img_rd.shape
(h, w) = img_rd.shape[:2]

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# load our serialized model from disk

net = cv2.dnn.readNetFromCaffe( prototextPath, caffeModel)


blob = cv2.dnn.blobFromImage(cv2.resize(img_rd, (300, 300)), 0.007843, (300, 300), 127.5)

print("[INFO] computing object detections...")
net.setInput(blob)
output = net.forward()

confidence = 0.1
        
for i in np.arange(0, output.shape[2]):
    confid_prob = output[0, 0, i, 2]
    print('confid_prob',confid_prob)

    if confid_prob > confidence:
       idx = int(output[0, 0, i, 1])
       box = output[0, 0, i, 3:7] * np.array([w, h, w, h])
       (startX, startY, endX, endY) = box.astype("int") 
       label = "{}: {:.2f}%".format(CLASSES[idx], confid_prob * 100)
       print("[INFO] {}".format(label))
       cv2.rectangle(img_rd, (startX, startY), (endX, endY),
  			COLORS[idx], 2)
       y = startY - 15 if startY - 15 > 15 else startY + 15
       cv2.putText(img_rd, label, (startX, y),
  			cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
       

# show the output image

cv2.imshow("Image", img_rd)
cv2.waitKey(0)











