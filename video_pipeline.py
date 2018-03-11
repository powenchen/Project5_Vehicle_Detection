import numpy as np
import cv2
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip
from os import listdir
import _pickle as pickle
from sklearn.svm import LinearSVC
from training import featureExtraction
from scipy.ndimage.measurements import label
from sklearn.preprocessing import StandardScaler
from functools import reduce

X_scaler_file = open("X_scaler.pickle",'rb')
X_scaler = pickle.load(X_scaler_file)

svc_file = open("svc_model.pickle",'rb')
svc = pickle.load(svc_file)

prev = []

def generate_window_list(x_max,stride=10):
	bboxes = []
	overlap = 0.75

	size = 60
	xs = range(size,x_max,int(size * (1.0-overlap)))
	ys = range(400,500,int(size * (1.0-overlap)))
	for x in xs:
		for y in ys:
			if int(y-size) >= 0  and int(x-size) >= 0 :
				bboxes.append([(int(x-size),int(y-size)),(int(x),int(y))]) 
	size = 70
	xs = range(size,x_max,int(size * (1.0-overlap)))
	ys = range(450,550,int(size * (1.0-overlap)))
	for x in xs:
		for y in ys:
			if int(y-size) >= 0  and int(x-size) >= 0 :
				bboxes.append([(int(x-size),int(y-size)),(int(x),int(y))])

	size = 80
	xs = range(size,x_max,int(size * (1.0-overlap)))
	ys = range(500,600,int(size * (1.0-overlap)))
	for x in xs:
		for y in ys:
			if int(y-size) >= 0  and int(x-size) >= 0 :
				bboxes.append([(int(x-size),int(y-size)),(int(x),int(y))])

	size = 90
	xs = range(size,x_max,int(size * (1.0-overlap))) 
	ys = range(500,650,int(size * (1.0-overlap)))
	for x in xs:
		for y in ys:
			if int(y-size) >= 0  and int(x-size) >= 0 :
				bboxes.append([(int(x-size),int(y-size)),(int(x),int(y))])
	
	return bboxes

windows = generate_window_list(1280)
print("There are {} windows".format(len(windows)))

# Source: Lesson 20, Section 34. Search and Classify
def search_windows(img, windows, clf):
	global X_scaler

	#1) Create an empty list to receive positive detection windows
	on_windows = []
	#2) Iterate over all windows in the list
	for i in range(len(windows)):
		window = windows[i]
		#3) Extract the test window from original image
		img_window = img[window[0][1]:window[1][1], window[0][0]:window[1][0]]
		test_img = cv2.resize( img_window,(64, 64))      
		#4) Extract features for that window using single_img_features()
		feature = featureExtraction(test_img,cv2.COLOR_RGB2YCrCb, 9, 8)
		#5) Scale extracted features to be fed to classifier
		scaled_feature = X_scaler.transform(feature.reshape(1, -1))
		#6) Predict using your classifier
		

		prediction = clf.predict(scaled_feature)
		#7) If positive (prediction == 1) then save the window
		if prediction == 1:
			on_windows.append(window)
	
	#8) Return windows for positive detections
	return on_windows


def add_heat(heatmap, bbox_list):
	# Iterate through list of bboxes
	for box in bbox_list:
		# Add += 1 for all pixels inside each bbox
		# Assuming each "box" takes the form ((x1, y1), (x2, y2))
		heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

	# Return updated heatmap
	return heatmap

def heat_map(test_img,rectangles):
	heatmap_img = np.zeros_like(test_img[:,:,0])
	heatmap_img = add_heat(heatmap_img, rectangles)

	return heatmap_img
	

def apply_threshold(heatmap, threshold):
	# Zero out pixels below the threshold
	heatmap[heatmap <= threshold] = 0
	# Return thresholded map
	return heatmap


def draw_labeled_bboxes(img, labels):
	
	# Iterate through all detected cars
	rects = []
	for car_number in range(1, labels[1]+1):
		# Find pixels with each car_number label value
		nonzero = (labels[0] == car_number).nonzero()
		# Identify x and y values of those pixels
		nonzeroy = np.array(nonzero[0])
		nonzerox = np.array(nonzero[1])
		# Define a bounding box based on min/max x and y
		bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
		rects.append(bbox)
		# Draw the box on the image
		dx,dy = np.abs(bbox[0][1] -bbox[1][1]),np.abs(bbox[0][0] -bbox[1][0])
		if dx * dy > 1600 and dx > 20 and dy > 20: 
			cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
	# Return the image and final rectangles
	return img, rects

def pipeline(img,demo=False,frame_window = 3):
	global prev
	global svc
	
	if len(prev) > frame_window:
		prev = prev[-frame_window:]


	on_windows = search_windows(img, windows, svc)

	heatmap_img = heat_map(img,on_windows)

	if demo:
		plt.figure(figsize=(10,10))
		plt.imshow(heatmap_img, cmap='hot')
		plt.show()

	prev.append(heatmap_img)

	heatmap_img = reduce(lambda x, y: x + y, prev) / len(prev)

	thresh = 2
	thres_img = apply_threshold(heatmap_img,thresh)

	labels = label(thres_img)
	# Draw bounding boxes on a copy of the image
	draw_img, rect = draw_labeled_bboxes(np.copy(img), labels)
	return draw_img




"""
for file in listdir("test_images"):
	if file.endswith(".png") or file.endswith(".jpg"):
		img = cv2.imread("test_images/"+file)
		img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

		draw_img = pipeline(img,True)
		draw_img = cv2.cvtColor(draw_img,cv2.COLOR_BGR2RGB)
		cv2.imshow(file,draw_img)
		cv2.waitKey(0)
"""
#output = "test_video_output.mp4"#"project_video_output.mp4"
#clip1 = VideoFileClip("test_video.mp4")#VideoFileClip("project_video.mp4")


output = "project_video_output3.mp4"
clip1 = VideoFileClip("project_video.mp4")
white_clip = clip1.fl_image(pipeline)
white_clip.write_videofile(output, audio=False)
