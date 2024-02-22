---
title: Traffic Signs Detection 
categories:
- ML_project
excerpt: |
  The primary objectives for any recognition system include **detection** (ascertaining the location and size of an object within an input image) and **classification** (assigning the detected objects into specific subclasses). Typically, a singular detection/classification model, such as YOLO or SSD, is employed for both tasks, where input images are annotated with bounding boxes and their corresponding classes. However, the labelling and training of such datasets demand considerable time and effort. Consequently, the principal aim of this project is to identify a single main class (signs) and to incorporate a custom-built Convolutional Neural Network for the classification of the detected objects into subclasses (for instance, speed limits, stop signs, etc.). This approach necessitates training a detection model only once to recognize one main class, while allowing for the training of multiple classification models to categorize detected objects based on the specific requirements of the task.
feature_text: |
  ## Detecting Traffic Signs
  Enhancing Autonomous Vehicle Technologies with Cutting-Edge Traffic Sign Detection Using YOLO v3, OpenCV, and Keras
feature_image: "/assets/signals.png"
image: "https://picsum.photos/2560/600?image=733"
---



# Traffic Signs Detection with OpenCV & Keras

Leveraging **YOLO v3**, **OpenCV**, and **Keras** for traffic signs detection offers a cutting-edge solution in computer vision, crucial for enhancing autonomous vehicle technologies. **YOLO v3** excels in real-time object detection, identifying traffic signs quickly and accurately. **OpenCV** aids in processing these images, while **Keras** specializes in classifying the detected signs into their respective categories, such as speed limits and stop signs. This combination delivers a powerful, efficient system for traffic sign detection, crucial for improving the safety and reliability of autonomous navigation systems.

* Initially, a model trained within the Darknet framework utilizes the OpenCV dnn library to **identify Traffic Signs across four distinct categories**.
* Subsequently, another model, developed using Keras, **classifies** the segmented portions of these Traffic Signs into one of **43 different classes**.
* Although the results are currently experimental, they hold potential for future enhancements.


Example of detections on video are shown below. **Trained weights** can be found in the course mentioned above.

![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F3400968%2Fbcdae0b57021d6ac3e86a9aa2e8c4b08%2Fts_detections.gif?generation=1581700736851192&alt=media)

# Importing required libraries


```python
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2
import time
from timeit import default_timer as timer
import matplotlib.pyplot as plt
import pickle

from keras.models import load_model

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
# for dirname, _, filenames in os.walk('../input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

print(os.listdir('archive/'))

# Any results we write to the current directory are saved as output

```

    ['test.pickle', 'data0.pickle', 'data2.pickle', 'std_gray.pickle', 'mean_image_rgb.pickle', 'data6.pickle', 'datasets_preparing.py', 'data8.pickle', 'data4.pickle', 'label_names.csv', 'data1.pickle', 'valid.pickle', 'std_rgb.pickle', 'data3.pickle', 'train.pickle', 'data7.pickle', 'mean_image_gray.pickle', 'data5.pickle', 'labels.pickle']


# 📂 Loading *labels*


```python
# Reading csv file with labels' names
# Loading two columns [0, 1] into Pandas dataFrame
labels = pd.read_csv('../input/traffic-signs-preprocessed/label_names.csv')

# Check point
# Showing first 5 rows from the dataFrame
print(labels.head())
print()

# To locate by class number use one of the following
# ***.iloc[0][1] - returns element on the 0 column and 1 row
print(labels.iloc[0][1])  # Speed limit (20km/h)
# ***['SignName'][1] - returns element on the column with name 'SignName' and 1 row
print(labels['SignName'][1]) # Speed limit (30km/h)

```

       ClassId              SignName
    0        0  Speed limit (20km/h)
    1        1  Speed limit (30km/h)
    2        2  Speed limit (50km/h)
    3        3  Speed limit (60km/h)
    4        4  Speed limit (70km/h)
    
    Speed limit (20km/h)
    Speed limit (30km/h)


# 📂 Loading trained Keras CNN model for Classification


```python
# Loading trained CNN model to use it later when classifying from 4 groups into one of 43 classes
model = load_model('../input/traffic-signs-classification-with-cnn/model-23x23.h5')

# Loading mean image to use for preprocessing further
# Opening file for reading in binary mode
with open('../input/traffic-signs-preprocessed/mean_image_rgb.pickle', 'rb') as f:
    mean = pickle.load(f, encoding='latin1')  # dictionary type
    
print(mean['mean_image_rgb'].shape)  # (3, 32, 32)

```

    (3, 32, 32)


# 💠 Loading YOLO v3 network by OpenCV dnn library

## Loading *trained weights* and *cfg file* into the Network


```python
# Trained weights can be found in the course mentioned above
path_to_weights = '../input/trained-traffic-signs-detector-based-on-yolo-v3/yolov3_ts_train_5000.weights'
path_to_cfg = '../input/traffic-signs-dataset-in-yolo-format/yolov3_ts_test.cfg'

# Loading trained YOLO v3 weights and cfg configuration file by 'dnn' library from OpenCV
network = cv2.dnn.readNetFromDarknet(path_to_cfg, path_to_weights)

# To use with GPU
network.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
network.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL_FP16)

```

## Getting *output layers* where detections are made


```python
# Getting names of all YOLO v3 layers
layers_all = network.getLayerNames()

# Check point
# print(layers_all)

# Getting only detection YOLO v3 layers that are 82, 94 and 106
layers_names_output = [layers_all[i[0] - 1] for i in network.getUnconnectedOutLayers()]

# Check point
print()
print(layers_names_output)  # ['yolo_82', 'yolo_94', 'yolo_106']

```

    
    ['yolo_82', 'yolo_94', 'yolo_106']


## Setting *probability*, *threshold* and *colour* for bounding boxes


```python
# Minimum probability to eliminate weak detections
probability_minimum = 0.2

# Setting threshold to filtering weak bounding boxes by non-maximum suppression
threshold = 0.2

# Generating colours for bounding boxes
# randint(low, high=None, size=None, dtype='l')
colours = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')

# Check point
print(type(colours))  # <class 'numpy.ndarray'>
print(colours.shape)  # (43, 3)
print(colours[0])  # [25  65 200]

```

    <class 'numpy.ndarray'>
    (43, 3)
    [203  49  61]


# 🎬 Reading input video


```python
# Reading video from a file by VideoCapture object
# video = cv2.VideoCapture('../input/traffic-signs-dataset-in-yolo-format/traffic-sign-to-test.mp4')
# video = cv2.VideoCapture('../input/videofortesting/ts_video_1.mp4')
video = cv2.VideoCapture('../input/videofortesting/ts_video_6.mp4')

# Writer that will be used to write processed frames
writer = None

# Variables for spatial dimensions of the frames
h, w = None, None

```

# ➿ Processing frames in the loop


```python
%matplotlib inline

# Setting default size of plots
plt.rcParams['figure.figsize'] = (3, 3)

# Variable for counting total amount of frames
f = 0

# Variable for counting total processing time
t = 0

# Catching frames in the loop
while True:
    # Capturing frames one-by-one
    ret, frame = video.read()

    # If the frame was not retrieved
    if not ret:
        break
       
    # Getting spatial dimensions of the frame for the first time
    if w is None or h is None:
        # Slicing two elements from tuple
        h, w = frame.shape[:2]

    # Blob from current frame
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)

    # Forward pass with blob through output layers
    network.setInput(blob)
    start = time.time()
    output_from_network = network.forward(layers_names_output)
    end = time.time()

    # Increasing counters
    f += 1
    t += end - start

    # Spent time for current frame
    print('Frame number {0} took {1:.5f} seconds'.format(f, end - start))

    # Lists for detected bounding boxes, confidences and class's number
    bounding_boxes = []
    confidences = []
    class_numbers = []

    # Going through all output layers after feed forward pass
    for result in output_from_network:
        # Going through all detections from current output layer
        for detected_objects in result:
            # Getting 80 classes' probabilities for current detected object
            scores = detected_objects[5:]
            # Getting index of the class with the maximum value of probability
            class_current = np.argmax(scores)
            # Getting value of probability for defined class
            confidence_current = scores[class_current]

            # Eliminating weak predictions by minimum probability
            if confidence_current > probability_minimum:
                # Scaling bounding box coordinates to the initial frame size
                box_current = detected_objects[0:4] * np.array([w, h, w, h])

                # Getting top left corner coordinates
                x_center, y_center, box_width, box_height = box_current
                x_min = int(x_center - (box_width / 2))
                y_min = int(y_center - (box_height / 2))

                # Adding results into prepared lists
                bounding_boxes.append([x_min, y_min, int(box_width), int(box_height)])
                confidences.append(float(confidence_current))
                class_numbers.append(class_current)
                

    # Implementing non-maximum suppression of given bounding boxes
    results = cv2.dnn.NMSBoxes(bounding_boxes, confidences, probability_minimum, threshold)

    # Checking if there is any detected object been left
    if len(results) > 0:
        # Going through indexes of results
        for i in results.flatten():
            # Bounding box coordinates, its width and height
            x_min, y_min = bounding_boxes[i][0], bounding_boxes[i][1]
            box_width, box_height = bounding_boxes[i][2], bounding_boxes[i][3]
            
            
            # Cut fragment with Traffic Sign
            c_ts = frame[y_min:y_min+int(box_height), x_min:x_min+int(box_width), :]
            # print(c_ts.shape)
            
            if c_ts.shape[:1] == (0,) or c_ts.shape[1:2] == (0,):
                pass
            else:
                # Getting preprocessed blob with Traffic Sign of needed shape
                blob_ts = cv2.dnn.blobFromImage(c_ts, 1 / 255.0, size=(32, 32), swapRB=True, crop=False)
                blob_ts[0] = blob_ts[0, :, :, :] - mean['mean_image_rgb']
                blob_ts = blob_ts.transpose(0, 2, 3, 1)
                # plt.imshow(blob_ts[0, :, :, :])
                # plt.show()

                # Feeding to the Keras CNN model to get predicted label among 43 classes
                scores = model.predict(blob_ts)

                # Scores is given for image with 43 numbers of predictions for each class
                # Getting only one class with maximum value
                prediction = np.argmax(scores)
                # print(labels['SignName'][prediction])


                # Colour for current bounding box
                colour_box_current = colours[class_numbers[i]].tolist()

                # Drawing bounding box on the original current frame
                cv2.rectangle(frame, (x_min, y_min),
                              (x_min + box_width, y_min + box_height),
                              colour_box_current, 2)

                # Preparing text with label and confidence for current bounding box
                text_box_current = '{}: {:.4f}'.format(labels['SignName'][prediction],
                                                       confidences[i])

                # Putting text with label and confidence on the original image
                cv2.putText(frame, text_box_current, (x_min, y_min - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour_box_current, 2)


    # Initializing writer only once
    if writer is None:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        # Writing current processed frame into the video file
        writer = cv2.VideoWriter('result.mp4', fourcc, 25,
                                 (frame.shape[1], frame.shape[0]), True)

    # Write processed current frame to the file
    writer.write(frame)


# Releasing video reader and writer
video.release()
writer.release()

```

    Frame number 1 took 5.52782 seconds
    Frame number 2 took 0.30848 seconds
    Frame number 3 took 0.28469 seconds
    Frame number 4 took 0.28405 seconds
    Frame number 5 took 0.28431 seconds
    Frame number 6 took 0.28376 seconds
    Frame number 7 took 0.28409 seconds
    Frame number 8 took 0.28311 seconds
    Frame number 9 took 0.28223 seconds
    Frame number 10 took 0.28190 seconds
    Frame number 11 took 0.28196 seconds
    Frame number 12 took 0.28338 seconds
    Frame number 13 took 0.28229 seconds
    Frame number 14 took 0.28356 seconds
    Frame number 15 took 0.28600 seconds
    Frame number 16 took 0.29676 seconds
    Frame number 17 took 0.28499 seconds
    Frame number 18 took 0.28229 seconds
    Frame number 19 took 0.28268 seconds
    Frame number 20 took 0.28224 seconds
    Frame number 21 took 0.28413 seconds
    Frame number 22 took 0.28356 seconds
    Frame number 23 took 0.28463 seconds
    Frame number 24 took 0.28319 seconds
    Frame number 25 took 0.28683 seconds
    Frame number 26 took 0.28725 seconds
    Frame number 27 took 0.28898 seconds
    Frame number 28 took 0.28659 seconds
    Frame number 29 took 0.28389 seconds
    Frame number 30 took 0.28378 seconds
    Frame number 31 took 0.28475 seconds
    Frame number 32 took 0.28251 seconds
    Frame number 33 took 0.28269 seconds
    Frame number 34 took 0.28348 seconds
    Frame number 35 took 0.28382 seconds
    Frame number 36 took 0.28517 seconds
    Frame number 37 took 0.28654 seconds
    Frame number 38 took 0.28655 seconds
    Frame number 39 took 0.28563 seconds
    Frame number 40 took 0.28403 seconds
    Frame number 41 took 0.28611 seconds
    Frame number 42 took 0.28182 seconds
    Frame number 43 took 0.29107 seconds
    Frame number 44 took 0.28453 seconds
    Frame number 45 took 0.28377 seconds
    Frame number 46 took 0.28447 seconds
    Frame number 47 took 0.28513 seconds
    Frame number 48 took 0.28379 seconds
    Frame number 49 took 0.28776 seconds
    Frame number 50 took 0.28735 seconds
    Frame number 51 took 0.28501 seconds
    Frame number 52 took 0.28487 seconds
    Frame number 53 took 0.28414 seconds
    Frame number 54 took 0.28307 seconds
    Frame number 55 took 0.28253 seconds
    Frame number 56 took 0.28466 seconds
    Frame number 57 took 0.28470 seconds
    Frame number 58 took 0.28420 seconds
    Frame number 59 took 0.28372 seconds
    Frame number 60 took 0.28218 seconds
    Frame number 61 took 0.28171 seconds
    Frame number 62 took 0.28202 seconds
    Frame number 63 took 0.28421 seconds
    Frame number 64 took 0.28256 seconds
    Frame number 65 took 0.28365 seconds
    Frame number 66 took 0.28422 seconds
    Frame number 67 took 0.28526 seconds
    Frame number 68 took 0.28542 seconds
    Frame number 69 took 0.28809 seconds
    Frame number 70 took 0.30996 seconds
    Frame number 71 took 0.39311 seconds
    Frame number 72 took 0.29010 seconds
    Frame number 73 took 0.28246 seconds
    Frame number 74 took 0.28550 seconds
    Frame number 75 took 0.28487 seconds
    Frame number 76 took 0.28557 seconds
    Frame number 77 took 0.28503 seconds
    Frame number 78 took 0.28439 seconds
    Frame number 79 took 0.28619 seconds
    Frame number 80 took 0.28490 seconds
    Frame number 81 took 0.28697 seconds
    Frame number 82 took 0.28585 seconds
    Frame number 83 took 0.28697 seconds
    Frame number 84 took 0.28652 seconds
    Frame number 85 took 0.28533 seconds
    Frame number 86 took 0.28449 seconds
    Frame number 87 took 0.28467 seconds
    Frame number 88 took 0.28317 seconds
    Frame number 89 took 0.28570 seconds
    Frame number 90 took 0.28561 seconds
    Frame number 91 took 0.28517 seconds
    Frame number 92 took 0.28556 seconds
    Frame number 93 took 0.28932 seconds
    Frame number 94 took 0.28604 seconds
    Frame number 95 took 0.28409 seconds
    Frame number 96 took 0.29276 seconds
    Frame number 97 took 0.28508 seconds
    Frame number 98 took 0.28641 seconds
    Frame number 99 took 0.28600 seconds
    Frame number 100 took 0.28778 seconds
    Frame number 101 took 0.28482 seconds
    Frame number 102 took 0.28302 seconds
    Frame number 103 took 0.28462 seconds
    Frame number 104 took 0.28550 seconds
    Frame number 105 took 0.28799 seconds
    Frame number 106 took 0.28656 seconds
    Frame number 107 took 0.28632 seconds
    Frame number 108 took 0.28320 seconds
    Frame number 109 took 0.28345 seconds
    Frame number 110 took 0.28199 seconds
    Frame number 111 took 0.28399 seconds
    Frame number 112 took 0.28442 seconds
    Frame number 113 took 0.28563 seconds
    Frame number 114 took 0.28655 seconds
    Frame number 115 took 0.28611 seconds
    Frame number 116 took 0.28686 seconds
    Frame number 117 took 0.28606 seconds
    Frame number 118 took 0.28554 seconds
    Frame number 119 took 0.28580 seconds
    Frame number 120 took 0.28680 seconds
    Frame number 121 took 0.28843 seconds
    Frame number 122 took 0.29228 seconds
    Frame number 123 took 0.29320 seconds
    Frame number 124 took 0.28505 seconds
    Frame number 125 took 0.28440 seconds
    Frame number 126 took 0.28295 seconds
    Frame number 127 took 0.28477 seconds
    Frame number 128 took 0.28561 seconds
    Frame number 129 took 0.28503 seconds
    Frame number 130 took 0.28481 seconds
    Frame number 131 took 0.28668 seconds
    Frame number 132 took 0.28682 seconds
    Frame number 133 took 0.28491 seconds
    Frame number 134 took 0.28558 seconds
    Frame number 135 took 0.28669 seconds
    Frame number 136 took 0.28582 seconds
    Frame number 137 took 0.28552 seconds
    Frame number 138 took 0.28360 seconds
    Frame number 139 took 0.28467 seconds
    Frame number 140 took 0.28590 seconds
    Frame number 141 took 0.28458 seconds
    Frame number 142 took 0.28605 seconds
    Frame number 143 took 0.28407 seconds
    Frame number 144 took 0.28660 seconds
    Frame number 145 took 0.28519 seconds
    Frame number 146 took 0.28497 seconds
    Frame number 147 took 0.28567 seconds
    Frame number 148 took 0.28722 seconds
    Frame number 149 took 0.30203 seconds
    Frame number 150 took 0.39440 seconds
    Frame number 151 took 0.29713 seconds
    Frame number 152 took 0.28998 seconds
    Frame number 153 took 0.28315 seconds
    Frame number 154 took 0.28381 seconds
    Frame number 155 took 0.28484 seconds
    Frame number 156 took 0.28455 seconds
    Frame number 157 took 0.28462 seconds
    Frame number 158 took 0.28355 seconds
    Frame number 159 took 0.28561 seconds
    Frame number 160 took 0.28295 seconds
    Frame number 161 took 0.28152 seconds
    Frame number 162 took 0.28430 seconds
    Frame number 163 took 0.28459 seconds
    Frame number 164 took 0.28397 seconds
    Frame number 165 took 0.28378 seconds
    Frame number 166 took 0.28418 seconds
    Frame number 167 took 0.28647 seconds
    Frame number 168 took 0.28554 seconds
    Frame number 169 took 0.28408 seconds
    Frame number 170 took 0.28628 seconds
    Frame number 171 took 0.28269 seconds
    Frame number 172 took 0.28406 seconds
    Frame number 173 took 0.28388 seconds
    Frame number 174 took 0.28510 seconds
    Frame number 175 took 0.28671 seconds
    Frame number 176 took 0.29481 seconds
    Frame number 177 took 0.28572 seconds
    Frame number 178 took 0.28461 seconds
    Frame number 179 took 0.28354 seconds
    Frame number 180 took 0.28381 seconds
    Frame number 181 took 0.28487 seconds
    Frame number 182 took 0.28755 seconds
    Frame number 183 took 0.28590 seconds
    Frame number 184 took 0.28326 seconds
    Frame number 185 took 0.28332 seconds
    Frame number 186 took 0.28574 seconds
    Frame number 187 took 0.28746 seconds
    Frame number 188 took 0.28684 seconds
    Frame number 189 took 0.28440 seconds
    Frame number 190 took 0.28410 seconds
    Frame number 191 took 0.28283 seconds
    Frame number 192 took 0.28129 seconds
    Frame number 193 took 0.28304 seconds
    Frame number 194 took 0.28127 seconds
    Frame number 195 took 0.28551 seconds
    Frame number 196 took 0.28479 seconds
    Frame number 197 took 0.28270 seconds
    Frame number 198 took 0.28455 seconds
    Frame number 199 took 0.28374 seconds
    Frame number 200 took 0.28207 seconds
    Frame number 201 took 0.28154 seconds
    Frame number 202 took 0.28349 seconds
    Frame number 203 took 0.29308 seconds
    Frame number 204 took 0.28364 seconds
    Frame number 205 took 0.28461 seconds
    Frame number 206 took 0.28661 seconds
    Frame number 207 took 0.28589 seconds
    Frame number 208 took 0.28265 seconds
    Frame number 209 took 0.28450 seconds
    Frame number 210 took 0.28276 seconds
    Frame number 211 took 0.28015 seconds
    Frame number 212 took 0.28183 seconds
    Frame number 213 took 0.28343 seconds
    Frame number 214 took 0.28340 seconds
    Frame number 215 took 0.28382 seconds
    Frame number 216 took 0.28503 seconds
    Frame number 217 took 0.28606 seconds
    Frame number 218 took 0.28351 seconds
    Frame number 219 took 0.28184 seconds
    Frame number 220 took 0.28242 seconds
    Frame number 221 took 0.28519 seconds
    Frame number 222 took 0.28396 seconds
    Frame number 223 took 0.29134 seconds
    Frame number 224 took 0.28385 seconds
    Frame number 225 took 0.28655 seconds
    Frame number 226 took 0.28653 seconds
    Frame number 227 took 0.28642 seconds
    Frame number 228 took 0.29651 seconds
    Frame number 229 took 0.29838 seconds
    Frame number 230 took 0.38873 seconds
    Frame number 231 took 0.28324 seconds
    Frame number 232 took 0.28234 seconds
    Frame number 233 took 0.28309 seconds
    Frame number 234 took 0.28333 seconds
    Frame number 235 took 0.28514 seconds
    Frame number 236 took 0.28152 seconds
    Frame number 237 took 0.28373 seconds
    Frame number 238 took 0.28316 seconds
    Frame number 239 took 0.28187 seconds
    Frame number 240 took 0.28413 seconds
    Frame number 241 took 0.28420 seconds
    Frame number 242 took 0.28467 seconds
    Frame number 243 took 0.28419 seconds
    Frame number 244 took 0.28625 seconds
    Frame number 245 took 0.28635 seconds
    Frame number 246 took 0.28186 seconds
    Frame number 247 took 0.28371 seconds
    Frame number 248 took 0.28258 seconds
    Frame number 249 took 0.28080 seconds
    Frame number 250 took 0.28274 seconds
    Frame number 251 took 0.28412 seconds
    Frame number 252 took 0.28426 seconds
    Frame number 253 took 0.28682 seconds
    Frame number 254 took 0.28167 seconds
    Frame number 255 took 0.28321 seconds
    Frame number 256 took 0.29472 seconds
    Frame number 257 took 0.28530 seconds
    Frame number 258 took 0.28456 seconds
    Frame number 259 took 0.28545 seconds
    Frame number 260 took 0.28461 seconds
    Frame number 261 took 0.28277 seconds
    Frame number 262 took 0.28404 seconds
    Frame number 263 took 0.28606 seconds
    Frame number 264 took 0.28602 seconds
    Frame number 265 took 0.28313 seconds
    Frame number 266 took 0.28395 seconds
    Frame number 267 took 0.28183 seconds
    Frame number 268 took 0.28174 seconds
    Frame number 269 took 0.28410 seconds
    Frame number 270 took 0.28458 seconds
    Frame number 271 took 0.28675 seconds
    Frame number 272 took 0.28688 seconds
    Frame number 273 took 0.28503 seconds
    Frame number 274 took 0.28500 seconds
    Frame number 275 took 0.28310 seconds
    Frame number 276 took 0.28119 seconds
    Frame number 277 took 0.28336 seconds
    Frame number 278 took 0.28553 seconds
    Frame number 279 took 0.28352 seconds
    Frame number 280 took 0.28545 seconds
    Frame number 281 took 0.28195 seconds
    Frame number 282 took 0.28735 seconds
    Frame number 283 took 0.29508 seconds
    Frame number 284 took 0.28249 seconds
    Frame number 285 took 0.28274 seconds
    Frame number 286 took 0.28265 seconds
    Frame number 287 took 0.28077 seconds
    Frame number 288 took 0.28335 seconds
    Frame number 289 took 0.28146 seconds
    Frame number 290 took 0.28254 seconds
    Frame number 291 took 0.28325 seconds
    Frame number 292 took 0.28674 seconds
    Frame number 293 took 0.28403 seconds
    Frame number 294 took 0.28442 seconds
    Frame number 295 took 0.28130 seconds
    Frame number 296 took 0.28412 seconds
    Frame number 297 took 0.28505 seconds
    Frame number 298 took 0.28399 seconds
    Frame number 299 took 0.28320 seconds
    Frame number 300 took 0.28322 seconds
    Frame number 301 took 0.28711 seconds
    Frame number 302 took 0.28616 seconds
    Frame number 303 took 0.28305 seconds
    Frame number 304 took 0.28422 seconds
    Frame number 305 took 0.28305 seconds
    Frame number 306 took 0.28035 seconds
    Frame number 307 took 0.29199 seconds
    Frame number 308 took 0.28817 seconds
    Frame number 309 took 0.29844 seconds
    Frame number 310 took 0.29078 seconds
    Frame number 311 took 0.28241 seconds
    Frame number 312 took 0.28323 seconds
    Frame number 313 took 0.28173 seconds
    Frame number 314 took 0.28147 seconds
    Frame number 315 took 0.28425 seconds
    Frame number 316 took 0.28448 seconds
    Frame number 317 took 0.28600 seconds
    Frame number 318 took 0.28327 seconds
    Frame number 319 took 0.28452 seconds
    Frame number 320 took 0.28637 seconds
    Frame number 321 took 0.28627 seconds
    Frame number 322 took 0.28629 seconds
    Frame number 323 took 0.28339 seconds
    Frame number 324 took 0.28367 seconds
    Frame number 325 took 0.28587 seconds
    Frame number 326 took 0.28509 seconds
    Frame number 327 took 0.28538 seconds
    Frame number 328 took 0.28441 seconds
    Frame number 329 took 0.28467 seconds
    Frame number 330 took 0.28634 seconds
    Frame number 331 took 0.28423 seconds
    Frame number 332 took 0.28498 seconds
    Frame number 333 took 0.28668 seconds
    Frame number 334 took 0.28574 seconds
    Frame number 335 took 0.28656 seconds
    Frame number 336 took 0.29757 seconds
    Frame number 337 took 0.28516 seconds
    Frame number 338 took 0.28614 seconds
    Frame number 339 took 0.28451 seconds
    Frame number 340 took 0.28847 seconds
    Frame number 341 took 0.28675 seconds
    Frame number 342 took 0.28578 seconds
    Frame number 343 took 0.28292 seconds
    Frame number 344 took 0.28233 seconds
    Frame number 345 took 0.28436 seconds
    Frame number 346 took 0.28223 seconds
    Frame number 347 took 0.28319 seconds
    Frame number 348 took 0.28307 seconds
    Frame number 349 took 0.28340 seconds
    Frame number 350 took 0.28312 seconds
    Frame number 351 took 0.28410 seconds
    Frame number 352 took 0.28347 seconds
    Frame number 353 took 0.28253 seconds
    Frame number 354 took 0.28212 seconds
    Frame number 355 took 0.28306 seconds
    Frame number 356 took 0.28282 seconds
    Frame number 357 took 0.28295 seconds
    Frame number 358 took 0.28414 seconds
    Frame number 359 took 0.28654 seconds
    Frame number 360 took 0.28786 seconds
    Frame number 361 took 0.28524 seconds
    Frame number 362 took 0.29588 seconds
    Frame number 363 took 0.29353 seconds
    Frame number 364 took 0.28240 seconds
    Frame number 365 took 0.28102 seconds
    Frame number 366 took 0.28119 seconds
    Frame number 367 took 0.28203 seconds
    Frame number 368 took 0.28123 seconds
    Frame number 369 took 0.28110 seconds
    Frame number 370 took 0.28499 seconds
    Frame number 371 took 0.28321 seconds
    Frame number 372 took 0.28100 seconds
    Frame number 373 took 0.28231 seconds
    Frame number 374 took 0.28319 seconds
    Frame number 375 took 0.28479 seconds
    Frame number 376 took 0.28427 seconds
    Frame number 377 took 0.28608 seconds
    Frame number 378 took 0.28534 seconds
    Frame number 379 took 0.28462 seconds
    Frame number 380 took 0.28592 seconds
    Frame number 381 took 0.28679 seconds
    Frame number 382 took 0.28434 seconds
    Frame number 383 took 0.28003 seconds
    Frame number 384 took 0.28291 seconds
    Frame number 385 took 0.31018 seconds
    Frame number 386 took 0.29572 seconds
    Frame number 387 took 0.29071 seconds
    Frame number 388 took 0.28215 seconds
    Frame number 389 took 0.29660 seconds
    Frame number 390 took 0.29125 seconds
    Frame number 391 took 0.28239 seconds
    Frame number 392 took 0.28199 seconds
    Frame number 393 took 0.28238 seconds
    Frame number 394 took 0.28447 seconds
    Frame number 395 took 0.28208 seconds
    Frame number 396 took 0.28237 seconds
    Frame number 397 took 0.28580 seconds
    Frame number 398 took 0.28642 seconds
    Frame number 399 took 0.28645 seconds
    Frame number 400 took 0.28367 seconds
    Frame number 401 took 0.28445 seconds
    Frame number 402 took 0.28423 seconds
    Frame number 403 took 0.28436 seconds
    Frame number 404 took 0.28554 seconds
    Frame number 405 took 0.28653 seconds
    Frame number 406 took 0.28459 seconds
    Frame number 407 took 0.28579 seconds
    Frame number 408 took 0.28109 seconds
    Frame number 409 took 0.28283 seconds
    Frame number 410 took 0.28278 seconds
    Frame number 411 took 0.28106 seconds
    Frame number 412 took 0.28101 seconds
    Frame number 413 took 0.28220 seconds
    Frame number 414 took 0.28197 seconds
    Frame number 415 took 0.28409 seconds
    Frame number 416 took 0.28440 seconds
    Frame number 417 took 0.29191 seconds
    Frame number 418 took 0.28376 seconds
    Frame number 419 took 0.28163 seconds
    Frame number 420 took 0.28463 seconds
    Frame number 421 took 0.28094 seconds
    Frame number 422 took 0.28145 seconds
    Frame number 423 took 0.28248 seconds
    Frame number 424 took 0.28267 seconds
    Frame number 425 took 0.28153 seconds
    Frame number 426 took 0.28141 seconds
    Frame number 427 took 0.28267 seconds
    Frame number 428 took 0.28160 seconds
    Frame number 429 took 0.28295 seconds
    Frame number 430 took 0.28292 seconds
    Frame number 431 took 0.28278 seconds
    Frame number 432 took 0.28456 seconds
    Frame number 433 took 0.28434 seconds
    Frame number 434 took 0.28545 seconds
    Frame number 435 took 0.28577 seconds
    Frame number 436 took 0.28626 seconds
    Frame number 437 took 0.28582 seconds
    Frame number 438 took 0.28445 seconds
    Frame number 439 took 0.28572 seconds
    Frame number 440 took 0.28439 seconds
    Frame number 441 took 0.28639 seconds
    Frame number 442 took 0.28399 seconds
    Frame number 443 took 0.28124 seconds
    Frame number 444 took 0.29253 seconds
    Frame number 445 took 0.28448 seconds
    Frame number 446 took 0.28451 seconds
    Frame number 447 took 0.28002 seconds
    Frame number 448 took 0.28403 seconds
    Frame number 449 took 0.28084 seconds
    Frame number 450 took 0.28517 seconds
    Frame number 451 took 0.28547 seconds
    Frame number 452 took 0.28482 seconds
    Frame number 453 took 0.28425 seconds
    Frame number 454 took 0.28362 seconds
    Frame number 455 took 0.28363 seconds
    Frame number 456 took 0.28389 seconds
    Frame number 457 took 0.28525 seconds
    Frame number 458 took 0.28030 seconds
    Frame number 459 took 0.28581 seconds
    Frame number 460 took 0.28566 seconds
    Frame number 461 took 0.28429 seconds
    Frame number 462 took 0.28112 seconds
    Frame number 463 took 0.29488 seconds
    Frame number 464 took 0.28477 seconds
    Frame number 465 took 0.28046 seconds
    Frame number 466 took 0.28188 seconds
    Frame number 467 took 0.30430 seconds
    Frame number 468 took 0.29189 seconds
    Frame number 469 took 0.28744 seconds
    Frame number 470 took 0.29079 seconds
    Frame number 471 took 0.29167 seconds
    Frame number 472 took 0.28232 seconds
    Frame number 473 took 0.28225 seconds
    Frame number 474 took 0.28513 seconds
    Frame number 475 took 0.28421 seconds
    Frame number 476 took 0.28111 seconds
    Frame number 477 took 0.28141 seconds
    Frame number 478 took 0.28410 seconds
    Frame number 479 took 0.28299 seconds
    Frame number 480 took 0.28371 seconds
    Frame number 481 took 0.28478 seconds
    Frame number 482 took 0.28366 seconds
    Frame number 483 took 0.28398 seconds
    Frame number 484 took 0.28656 seconds
    Frame number 485 took 0.28441 seconds
    Frame number 486 took 0.28336 seconds
    Frame number 487 took 0.28206 seconds
    Frame number 488 took 0.28605 seconds
    Frame number 489 took 0.28500 seconds
    Frame number 490 took 0.28582 seconds
    Frame number 491 took 0.28510 seconds
    Frame number 492 took 0.28411 seconds
    Frame number 493 took 0.28051 seconds
    Frame number 494 took 0.28275 seconds
    Frame number 495 took 0.28497 seconds
    Frame number 496 took 0.28270 seconds
    Frame number 497 took 0.29587 seconds
    Frame number 498 took 0.28226 seconds
    Frame number 499 took 0.28232 seconds
    Frame number 500 took 0.28405 seconds
    Frame number 501 took 0.28243 seconds
    Frame number 502 took 0.28481 seconds
    Frame number 503 took 0.28029 seconds
    Frame number 504 took 0.27936 seconds
    Frame number 505 took 0.28082 seconds
    Frame number 506 took 0.28182 seconds
    Frame number 507 took 0.28221 seconds
    Frame number 508 took 0.28344 seconds
    Frame number 509 took 0.28629 seconds
    Frame number 510 took 0.28458 seconds
    Frame number 511 took 0.28489 seconds
    Frame number 512 took 0.28251 seconds
    Frame number 513 took 0.28448 seconds
    Frame number 514 took 0.28538 seconds
    Frame number 515 took 0.28534 seconds
    Frame number 516 took 0.28393 seconds
    Frame number 517 took 0.28128 seconds


## 🏁 FPS results


```python
print('Total number of frames', f)
print('Total amount of time {:.5f} seconds'.format(t))
print('FPS:', round((f / t), 1))

```

    Total number of frames 517
    Total amount of time 152.90212 seconds
    FPS: 3.4



```python
# Saving locally without committing
from IPython.display import FileLink

FileLink('result.mp4')

```




<a href='result.mp4' target='_blank'>result.mp4</a><br>



# 🎈 Reading input images


```python
# Reading image with OpenCV library
# In this way image is opened already as numpy array
# WARNING! OpenCV by default reads images in BGR format
# image_BGR = cv2.imread('../input/videofortesting/traffic_sign.jpg')
# image_BGR = cv2.imread('../input/videofortesting/ts_video_1_1.png')
# image_BGR = cv2.imread('../input/videofortesting/ts_video_1_2.png')
# image_BGR = cv2.imread('../input/videofortesting/ts_video_1_3.png')
# image_BGR = cv2.imread('../input/videofortesting/ts_video_1_4.png')
# image_BGR = cv2.imread('../input/videofortesting/ts_video_6_1.png')
# image_BGR = cv2.imread('../input/videofortesting/ts_video_6_2.png')
# image_BGR = cv2.imread('../input/videofortesting/ts_video_6_3.png')
image_BGR = cv2.imread('../input/videofortesting/ts_final_1.png')

# Check point
# Showing image shape
print('Image shape:', image_BGR.shape)  # tuple of (731, 1092, 3)

# Getting spatial dimension of input image
h, w = image_BGR.shape[:2]  # Slicing from tuple only first two elements

# Check point
# Showing height an width of image
print('Image height={0} and width={1}'.format(h, w))  # 731 1092

```

    Image shape: (720, 1280, 3)
    Image height=720 and width=1280


# 🧿 Processing single image


```python
# Variable for counting total processing time
t = 0

# Blob from current frame
blob = cv2.dnn.blobFromImage(image_BGR, 1 / 255.0, (416, 416), swapRB=True, crop=False)

# Forward pass with blob through output layers
network.setInput(blob)
start = time.time()
output_from_network = network.forward(layers_names_output)
end = time.time()

# Time
t += end - start
print('Total amount of time {:.5f} seconds'.format(t))

# Lists for detected bounding boxes, confidences and class's number
bounding_boxes = []
confidences = []
class_numbers = []

# Going through all output layers after feed forward pass
for result in output_from_network:
    # Going through all detections from current output layer
    for detected_objects in result:
        # Getting 80 classes' probabilities for current detected object
        scores = detected_objects[5:]
        # Getting index of the class with the maximum value of probability
        class_current = np.argmax(scores)
        # Getting value of probability for defined class
        confidence_current = scores[class_current]

        # Eliminating weak predictions by minimum probability
        if confidence_current > probability_minimum:
            # Scaling bounding box coordinates to the initial frame size
            box_current = detected_objects[0:4] * np.array([w, h, w, h])

            # Getting top left corner coordinates
            x_center, y_center, box_width, box_height = box_current
            x_min = int(x_center - (box_width / 2))
            y_min = int(y_center - (box_height / 2))

            # Adding results into prepared lists
            bounding_boxes.append([x_min, y_min, int(box_width), int(box_height)])
            confidences.append(float(confidence_current))
            class_numbers.append(class_current)
                

# Implementing non-maximum suppression of given bounding boxes
results = cv2.dnn.NMSBoxes(bounding_boxes, confidences, probability_minimum, threshold)

# Checking if there is any detected object been left
if len(results) > 0:
    # Going through indexes of results
    for i in results.flatten():
        # Bounding box coordinates, its width and height
        x_min, y_min = bounding_boxes[i][0], bounding_boxes[i][1]
        box_width, box_height = bounding_boxes[i][2], bounding_boxes[i][3]
            
            
        # Cut fragment with Traffic Sign
        c_ts = image_BGR[y_min:y_min+int(box_height), x_min:x_min+int(box_width), :]
        # print(c_ts.shape)
            
        if c_ts.shape[:1] == (0,) or c_ts.shape[1:2] == (0,):
            pass
        else:
            # Getting preprocessed blob with Traffic Sign of needed shape
            blob_ts = cv2.dnn.blobFromImage(c_ts, 1 / 255.0, size=(32, 32), swapRB=True, crop=False)
            blob_ts[0] = blob_ts[0, :, :, :] - mean['mean_image_rgb']
            blob_ts = blob_ts.transpose(0, 2, 3, 1)
            # plt.imshow(blob_ts[0, :, :, :])
            # plt.show()

            # Feeding to the Keras CNN model to get predicted label among 43 classes
            scores = model.predict(blob_ts)

            # Scores is given for image with 43 numbers of predictions for each class
            # Getting only one class with maximum value
            prediction = np.argmax(scores)
            print(labels['SignName'][prediction])


            # Colour for current bounding box
            colour_box_current = colours[class_numbers[i]].tolist()
            
            # Green BGR
            colour_box_current = [0, 255, 61]
            
            # Yellow BGR
#             colour_box_current = [0, 255, 255]

            # Drawing bounding box on the original current frame
            cv2.rectangle(image_BGR, (x_min, y_min),
                              (x_min + box_width, y_min + box_height),
                              colour_box_current, 6)

#             # Preparing text with label and confidence for current bounding box
#             text_box_current = '{}: {:.4f}'.format(labels['SignName'][prediction],
#                                                    confidences[i])
            
#             # Putting text with label and confidence on the original image
#             cv2.putText(image_BGR, text_box_current, (x_min, y_min - 15),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.9, colour_box_current, 2)
            
            if prediction == 5:
                # Preparing text with label and confidence for current bounding box
                text_box_current = '{}: {:.4f}'.format('Speed limit 60', confidences[i])

                # Putting text with label and confidence on the original image
                cv2.putText(image_BGR, text_box_current, (x_min - 110, y_min - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, colour_box_current, 2)
                
            elif prediction == 9:            
                # Preparing text with label and confidence for current bounding box
                text_box_current = '{}: {:.4f}'.format('No overtaking', confidences[i])

                # Putting text with label and confidence on the original image
                cv2.putText(image_BGR, text_box_current, (x_min - 110, y_min + box_height + 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, colour_box_current, 2)

#             elif prediction == 17:            
#                 # Preparing text with label and confidence for current bounding box
#                 text_box_current = '{}: {:.4f}'.format('No entry', confidences[i])

#                 # Putting text with label and confidence on the original image
#                 cv2.putText(image_BGR, text_box_current, (x_min - 170, y_min - 15),
#                                 cv2.FONT_HERSHEY_SIMPLEX, 0.9, colour_box_current, 2)
                
                
# Saving image
cv2.imwrite('result.png', image_BGR)

```

    Total amount of time 0.29843 seconds
    Ahead only
    No passing





    True



# 🦞 Showing processed image


```python
%matplotlib inline

plt.rcParams['figure.figsize'] = (35.0, 35.0) # Setting default size of plots

image_BGR = cv2.imread('/kaggle/working/result.png')

# Showing image shape
print('Image shape:', image_BGR.shape)  # tuple of (800, 1360, 3)

# Getting spatial dimension of input image
h, w = image_BGR.shape[:2]  # Slicing from tuple only first two elements

# Showing height an width of image
print('Image height={0} and width={1}'.format(h, w))  # 800 1360

plt.imshow(cv2.cvtColor(image_BGR, cv2.COLOR_BGR2RGB))
plt.axis('off')
# plt.title('Keras Visualization', fontsize=18)

# Showing the plot
plt.show()

plt.close()

```

    Image shape: (720, 1280, 3)
    Image height=720 and width=1280



    
![png](/assets/output_28_1.png)
    



```python
# Saving locally without committing
from IPython.display import FileLink

FileLink('result.png')

```




<a href='result.png' target='_blank'>result.png</a><br>



# 🔎 Example of the result

![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F3400968%2Fa57f58b38e3caab6fbf72169895f5074%2Fresult.gif?generation=1585955236302060&alt=media)
