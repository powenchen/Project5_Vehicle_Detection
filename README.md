**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/01.png
[image2]: ./output_images/02.png
[image3]: ./output_images/03.png
[image4]: ./output_images/04.png
[image5]: ./output_images/05.png
[image6]: ./output_images/06.png
[video1]: ./project_video.mp4

### Histogram of Oriented Gradients (HOG)

#### 1. How I extracted HOG features from the training images.

The code for training is contained in training.py.


I started by reading in all the `vehicle` and `non-vehicle` images.  Here are example images of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I defined the feature extraction process in 'featureExtraction()', it starts with converting the image to HLS color space, after this , it use the hog() function provided by skimage to extract the hog features of an image

Here is an example using the `HLS` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and test the accuracy using 20% of images as test set. Here's the result:
| ColorSpace 	| orientation 	| pix_per_cells | cell_per_block | accuracy(%) 		| 
|:-------------:|:-------------:|:-------------:|:-------------: |:----------------:| 
| RGB			| 9 			| 4				| 2				 | 96.6%			|
| RGB			| 9 			| 8				| 2				 | 96.7%			|
| RGB			| 9 			| 12			| 2				 | 97.5%			|
| RGB			| 10 			| 4				| 2				 | 96.3%			|
| RGB			| 10 			| 8				| 2				 | 96.3%			|
| RGB			| 10 			| 12			| 2				 | 97.2%			|
| RGB			| 11 			| 4				| 2				 | 96.3%			|
| RGB			| 11 			| 8				| 2				 | 96.9%			|
| RGB			| 11 			| 12			| 2				 | 97.6%			|
| YCrCb			| 9 			| 4				| 2				 | 96.9%			|
| YCrCb			| 9 			| 8				| 2				 | 97.0%			|
| YCrCb			| 9 			| 12			| 2				 | 96.3%			|
| YCrCb			| 10 			| 4				| 2				 | 97.8%			|
| YCrCb			| 10			| 8				| 2				 | 96.0%			|
| YCrCb			| 10 			| 12			| 2				 | 97.9%			|
| YCrCb			| 11 			| 4				| 2				 | 97.2%			|
| YCrCb			| 11 			| 8				| 2				 | 98.1%			|
| YCrCb			| 11 			| 12			| 2				 | 95.6%			|
| HLS			| 9 			| 4				| 2				 | 96.7%			|
| HLS			| 9 			| 8				| 2				 | 96.8%			|
| HLS			| 9 			| 12			| 2				 | 98.1%			|
| HLS			| 10 			| 4				| 2				 | 97.2%			|
| HLS			| 10			| 8				| 2				 | 96.2%			|
| HLS			| 10 			| 12			| 2				 | 97.1%			|
| HLS			| 11 			| 4				| 2				 | 96.8%			|
| HLS			| 11 			| 8				| 2				 | 97.5%			|
| HLS			| 11 			| 12			| 2				 | 97.7%			|
| LUV			| 9 			| 4				| 2				 | 96.4%			|
| LUV			| 9 			| 8				| 2				 | 96.3%			|
| LUV			| 9 			| 12			| 2				 | 96.5%			|
| LUV			| 10 			| 4				| 2				 | 97.1%			|
| LUV			| 10			| 8				| 2				 | 97.1%			|
| LUV			| 10 			| 12			| 2				 | 94.4%			|
| LUV			| 11 			| 4				| 2				 | 97.8%			|
| LUV			| 11 			| 8				| 2				 | 97.1%			|
| LUV			| 11 			| 12			| 2				 | 97.5%			|

Based on the accuracy and the size of the feature vector, I chose HLS color space, orientation = 11, pix_per_cells = 12 and cell_per_block = 2. 

#### 3. How I trained a classifier using your selected HOG features (and color features if you used them).

First, I split the data set(shuffled) into training and testing sets with:
```python
X_train, X_test, y_train, y_test = train_test_split(hog_feature_list, answer_vec, test_size=0.2, shuffle=True)
 ```

Second, I trained a linear SVM using LinearSVC() provided by sklearn module
```python
svc = LinearSVC()
svc.fit(X_train, y_train)
 ```

Finally, I test my SVC model with svc.predict():
```python
y_pred = svc.predict(X_test)
acc = accuracy_score(y_pred, y_test)
 ```

### Sliding Window Search

#### 1. How I implemented a sliding window search. 

I create sliding windows with 4 different scales(60x60, 70x70, 80x80 and 90x90) according to their y coordinate in the image since cars are smaller in the camera image when their distance to the camera are longer.

Each sliding window has 75% overlapping with the next window.

Here's an image shows the sliding windows in my system:

![alt text][image3]

#### 2. Some examples of test images.

Ultimately I searched on HLS 3-channel HOG features, which provided a nice result.  Here are some example images:

![alt text][image4]

I also tuned the sliding window size and overlapping value to get a balance between computational complexity and accuracy(if we use more sliding windows, the accuracy will increase as well as the time spent on computation)

---

### Video Implementation

#### 1. A link to my final video output.
Here's a [link to my video result](./project_video_output.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames, their corresponding heatmaps, label images and the resulting bounding boxes:

![alt text][image5]



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The most obvious problem of the implementation in this project is it takes too much time for the processing of each frame while it dosen't take any time in training, which is oppsite to Neural Network approach. This makes it really difiicult to be used on real-time systems.

Besides, it tends to have false positives when there is shadow in the image, appending features from different feature spaces or having a finer sliding window can help a little bit but will greatly increase the computation complexity.

SVM classifier is very easy to implement, however it is not robust in comparison to neural networks. This module would be more robust if we could replace the SVM classifier with some neural network classifier(for example: YOLO).