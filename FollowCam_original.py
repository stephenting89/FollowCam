# import the necessary packages
from __future__ import print_function
from imutils.object_detection import non_max_suppression
from imutils import paths
import numpy as np
import argparse
import time
import imutils
import cv2

#------------Controller Parameters ------------------
threshold_percent = 0.1
#----------------------------------------------------

#features_number = 0
height_from_floor  = list()


# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                     maxLevel = 2,
                     criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)) 

feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.2,
                       minDistance = 3,
                       blockSize = 7)


def find_features(image, rectangle, features_num):
	# rectangle (x1,x2,y1,y2)

	dist2B = list()
	features = list()
	global features_number

	x1 = rectangle[0]
	x2 = rectangle[2]
	y1 = rectangle[1]
	y2 = rectangle[3]

	cropped_image = image[y1:y2, x1:x2]
	

	#gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
	#corners = cv2.goodFeaturesToTrack(cropped_image,25,0.01,10)
	corners = cv2.goodFeaturesToTrack(cropped_image, mask = None, **feature_params)
	#print ("Corners" + str(corners))
	#corners = np.int0(corners)
	print ("\n")
	#print (type(corners))
	#print(corners)

	for corner in corners:
		#print (corner)
		distance = (y2-y1) - corner[0][1]
		dist2B.append(distance)

		corner[0][0] = rectangle[0] + corner[0][0]
		corner[0][1] = rectangle[1] + corner[0][1]
		features.append(corner)

	for i in range(len(features)):
		d = dist2B.pop(0)
		height_from_floor.append(d)


	#features_number = len(features)
	#print ("num" + str(features_number))
	return features

def main():
	# initialize the HOG descriptor/person detector
	camera = cv2.VideoCapture(0); 
	time.sleep(0.25)
	hog = cv2.HOGDescriptor()
	hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

	Threshold = 0
	features_number = 0

	while True: # main loop

		tracked_features = None

		while True: # detection loop, loop over the images

			unchangedPointsMap = dict()

			# load the image and resize it to (1) reduce detection time
			# and (2) improve detection accuracy
			(grabbed, current_frame) = camera.read()
			current_frame = imutils.resize(current_frame, width = 300)
			current_frame_copy = current_frame.copy()
			current_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
		 
			# detect people in the image
			(rects, weights) = hog.detectMultiScale(current_frame, winStride=(4, 4),
				padding=(8, 8), scale=1.5)
		 
			# draw the original bounding boxes
			
			for i in range(len(rects)):
				x, y, w, h = rects[i]
				rects[i][0] = x + 15
				rects[i][1] = y + 40
				rects[i][2] = w - 30
				rects[i][3] = h - 20

			for (x, y, w, h) in rects:
				cv2.rectangle(current_frame_copy, (x, y), (x + w, y + h), (0, 0, 255), 2)
		 
			# apply non-maxima suppression to the bounding boxes using a
			# fairly large overlap threshold to try to maintain overlapping
			# boxes that are still people
			rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
			pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
		 
			# draw the final bounding boxes
			for (xA, yA, xB, yB) in pick:
				cv2.rectangle(current_frame, (xA, yA), (xB, yB), (0, 255, 0), 2)
			
			print("{} original boxes, {} after suppression".format(len(rects), len(pick)))
			
		# 	if len(rects) > 0:
		# 		features = find_features(current_frame, rects[0], 0)
		# 		print("NUM" + str(features_number))
		# 		break

		# 	# cv2.imshow("HOG", current_frame_copy)
		# 	# key = cv2.waitKey(1) & 0xFF
			
		# 	# if key == ord("w"):
		# 	# 	break

		# features_number = len(features)
		# Threshold = features_number * threshold_percent
			
		# while True: # Tracking loop

		# 	#print ("Threshold" + str(Threshold))
		# 	if features_number < Threshold:
		# 		print ("Features less than threshold")
		# 		break
		# 	else:
		# 		(grabbed, next_frame) = camera.read()
		# 		next_frame = imutils.resize(next_frame, width = 300)
		# 		if not grabbed:
		# 			print ("Camera read failed")
		# 			return
		# 		current_frame_copy = next_frame.copy()
		# 		next_frame = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)

	 # 			#-----------Tracking using LK ---------------------------

	 # 			try:
	 # 				features = np.array(features)
	 # 				#print("Features" + str(features))
	 # 				(tracked_features, status, feature_errors) = cv2.calcOpticalFlowPyrLK(current_frame, next_frame, features, None, **lk_params)
		# 			#print("TEST")
		# 			# print("KEYS" + str(unchangedPointsMap.keys()))
		# 			# for i in range(len(tracked_features[0])):
		# 			# 	f = tracked_features[0][i]
		# 			# 	x = round(f[0])
		# 			# 	y = round(f[1])
		# 			# 	print("x and y" + str((x,y)))
		# 			# 	if (x,y) in unchangedPointsMap.keys():
		# 			# 		unchangedPointsMap[(x,y)] += 1
		# 			# 		print("ADDED" + str(unchangedPointsMap[(x,y)]))
		# 			# 		if unchangedPointsMap[(x,y)] == 30:
		# 			# 			print ("BEFORE" + str(tracked_features[0]))
		# 			# 			tracked_features = np.delete(tracked_features,i,0)
		# 			# 			unchangedPointsMap.pop((x,y))
		# 			# 			print ("AFTER" + str(tracked_features[0]))
		# 			# 	else:
		# 			# 		unchangedPointsMap[(x,y)] = 0

		# 			# print("BEFORE" + str(tracked_features))
		# 			# tracked_features[tracked_features[:,0].argsort()]
		# 			# print("AFTER" + str(tracked_features))

		# 			arr_x = []
		# 			arr_y = []

		# 			for i in range(len(tracked_features)):
		# 				f = tracked_features[i]
		# 				x = f[0][0]
		# 				y = f[0][1]

		# 				arr_x.append(x)
		# 				arr_y.append(y)

		# 			print("X_arr" + str(arr_x))
		# 			print("Y_arr" + str(arr_y))
		# 			print ("X SORTED " + str(sorted(arr_x)))
		# 			print ("Y SORTED " + str(sorted(arr_y)))

		# 			arr_x = sorted(arr_x)
		# 			arr_y = sorted(arr_y)

		# 			mid = len(arr_x)/2
		# 			X = arr_x[mid]
		# 			mid = len(arr_y)/2
		# 			Y = arr_y[mid]

		# 			new_feature_number = 0
		# 			temp_set_number = []
		# 			temp_distance = []
		# 			j = 0
		# 			for i in range(features_number):
		# 				if status[i] == 1:
		# 					new_feature_number += 1
		# 					#temp_set_number.append()
		# 					#temp_distance.append(height_from_floor[i])
		# 					j += 1
					
		# 			#height_from_floor = temp_distance
		# 			features_number = new_feature_number
		# 			#print("Features_num" + str(features_number))
		# 			features = []

		# 			for i in range(features_number):
		# 				features.append(tracked_features[i])

		# 			features = np.array(features)
		# 			tracked_features = []
		# 			current_frame = next_frame.copy()
	 # 			except Exception, e:
	 # 				raise e

	 # 			#-------Showing Points ------------------------
	 # 			for i in range(features_number):
	 # 				# print ("features " + str(features)) 
	 # 				# print ("features0 " + str(features[0]))
	 # 				# print ("features00 " + str(features[0][0]))
	 # 				# print ("features000 " + str(features[0][0][0]))

	 # 				#print ("features " + str(features[i]))
	 # 				cv2.circle(current_frame_copy,
	 # 						   tuple(features[i][0]),
	 # 						   3,
	 # 						   255,
	 # 						   -1)

	 # 			cv2.circle(current_frame_copy,
	 # 						(X,Y),
	 # 						5,
	 # 						(0,0,255),
	 # 						-1)

			# show the output images
			cv2.imshow("HOG", current_frame_copy)
			key = cv2.waitKey(1) & 0xFF
			
			if key == ord("w"):
				break
			
	camera.release()
	cv2.destroyAllWindows()


if __name__ == '__main__':
	main()
