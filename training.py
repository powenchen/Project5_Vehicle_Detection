from skimage.feature import hog
import cv2
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
import _pickle as pickle
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

def featureExtraction(image,color_space=cv2.COLOR_BGR2HLS,orient=9,pix_per_cell = 8, cell_per_block=2):
	image = cv2.cvtColor(image, color_space)
	channel1 = image[:,:,0]
	channel2 = image[:,:,1]
	channel3 = image[:,:,2]

	hog_features1 = hog(channel1, orientations=orient, 
						pixels_per_cell=(pix_per_cell, pix_per_cell),
						cells_per_block=(cell_per_block, cell_per_block), 
						transform_sqrt=False, 
						block_norm ='L2-Hys',
						feature_vector=False)
	hog_features2 = hog(channel2, orientations=orient, 
						pixels_per_cell=(pix_per_cell, pix_per_cell),
						cells_per_block=(cell_per_block, cell_per_block), 
						transform_sqrt=False, 
						block_norm ='L2-Hys',
						feature_vector=False)
	hog_features3 = hog(channel3, orientations=orient, 
						pixels_per_cell=(pix_per_cell, pix_per_cell),
						cells_per_block=(cell_per_block, cell_per_block), 
						transform_sqrt=False, 
						block_norm ='L2-Hys',
						feature_vector=False)

	hog_features = np.hstack((np.ravel(hog_features1),np.ravel(hog_features2),np.ravel(hog_features3)))

	return hog_features


"""
	# Plot the examples
	fig = plt.figure()
	plt.subplot(121)
	plt.imshow(image, cmap='gray')
	plt.title('Example Car Image')
	plt.subplot(122)
	plt.imshow(hog_image, cmap='gray')
	plt.title('HOG Visualization')
	plt.show()
"""


def featureList(folder_names,color_space=cv2.COLOR_BGR2HLS,orient=9,pix_per_cell = 8, cell_per_block=2):
	hog_feature_list = []
	for folder_name in folder_names:
		for file in os.listdir(folder_name):
			file = folder_name + "/" + file
			if file.endswith(".png") or file.endswith(".jpg"):
				image = cv2.imread(file)
				hog_feature_vec = featureExtraction(image, color_space,orient,pix_per_cell,cell_per_block)
				hog_feature_list.append(hog_feature_vec)

	return np.asarray(hog_feature_list)


def main():
	color_space,orientation,pix_per_cell = cv2.COLOR_BGR2HLS, 9, 8
	training_folders = []
	for folder in os.listdir("vehicles"):
		training_folders.append("vehicles/"+folder)
	hog_feature_list1 = featureList(training_folders,color_space,orientation,pix_per_cell)

	answer_vec1 = np.ones(len(hog_feature_list1))
	
	training_folders = []
	for folder in os.listdir("non-vehicles"):
		training_folders.append("non-vehicles/"+folder)
	hog_feature_list2 = featureList(training_folders,color_space,orientation,pix_per_cell)
	answer_vec2 = np.zeros(len(hog_feature_list2))

	hog_feature_list = np.vstack((hog_feature_list1,hog_feature_list2))
	answer_vec = np.hstack((answer_vec1,answer_vec2))


	X_train, X_test, y_train, y_test = train_test_split(hog_feature_list, answer_vec, test_size=0.2, shuffle=True)


	svc = LinearSVC()
	svc.fit(X_train, y_train)

	y_pred = svc.predict(X_test)
	acc = accuracy_score(y_pred, y_test)

	print("(HLS ,orientation={},pix_per_cell={},feature size = {})acc: {}%".format(orientation,pix_per_cell,X_train[0].shape[0],100.0*acc))

	svc_file = open("svc_model.pickle",'wb')
	pickle.dump(svc,svc_file)

if __name__ == "__main__":
	main()