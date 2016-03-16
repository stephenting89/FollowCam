# import the necessary packages
from __future__ import print_function
from imutils.object_detection import non_max_suppression
from imutils import paths
import numpy as np
import argparse
import time
import imutils
import cv2
import smbus
import sys
import RPi.GPIO as gpio
from picamera.array import PiRGBArray
from picamera import PiCamera

#Address for I2C, needs to match arduino
#address = 0x06
#bus = smbus.SMBus(1)

#------------Controller Parameters ------------------
threshold_percent = 0.1
#----------------------------------------------------

#======camera calibration parameters============
fku = 1000;
v0 = 100;
h = 1;

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

def people_floor(features_number, features, height_from_floor):
	sum = 0
	m = 0
	for i in range(features_number):
		sum += features[i][0][1] + height_from_floor[i]
		m += 1

	if m == 0:
		status = False
		return status, 0
	else:
		status = True
		return status, sum/m

def people_floor_test(Y, height_from_floor):	
	sum = Y + height_from_floor[0]
	status = True
	return status, sum
		
def scaled_people_floor_test(Y, height_from_floor):
	return people_floor_test(Y, height_from_floor)
	
		
def scaled_people_floor(features_number, features, height_from_floor):
	status, y = people_floor(features_number, features, height_from_floor)

	if not status:
		return status, y

	sum = 0
	m = 0

	for i in range(features_number):
		for j in range(i+1, features_number):
			if height_from_floor[i] - height_from_floor[j] < 2:
				continue
			t = (features[i][0][1] - features[j][0][1])/(height_from_floor[i] - height_from_floor[j])
			if t < 0:
				t = -t
			sum += t
			m +=1
	if m == 0:
		return status, y
	else:
		alpha = sum/m
		return status, alpha * y

def compute_distance(v):
	return (fku * h)/(v-v0)

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
	corners = cv2.goodFeaturesToTrack(cropped_image, mask = None, **feature_params)

	print ("\n")

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

	#return features, height_from_floor
	return features, (y2-y1)

def main():
	try:
  		# initialize leds
  		gpio.setmode(gpio.BCM)
  		gpio.setup(17, gpio.OUT)
  		gpio.setup(27, gpio.OUT)
  		gpio.output(27, True)

  		# initialize the HOG descriptor/person detector
  		camera = PiCamera()
  		camera.hflip = True
  		camera.vflip = True
  		camera.resolution = (320, 240)
  		camera.framerate = 8
  		rawCapture = PiRGBArray(camera, size=(320, 240)) 
  		time.sleep(0.25)
  		hog = cv2.HOGDescriptor()
  		hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

  		Threshold = 0
  		features_number = 0

  		tracked_features = None
  		detected = False

  		for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):

  			if not detected: # detection block
  				gpio.output(17, False)
  				Threshold = 0
  				unchangedPointsMap = dict()

  				current_frame = frame.array
  				#current_frame = imutils.resize(current_frame, width = 300)
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
  					rects[i][3] = h - 40

  				for (x, y, w, h) in rects:
  					cv2.rectangle(current_frame_copy, (x, y), (x + w, y + h), (0, 0, 255), 2)
  		 	
  		 		# Filter boxes
  				rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
  				pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
  		 
  				# draw the final bounding boxes
  				for (xA, yA, xB, yB) in pick:
  					cv2.rectangle(current_frame, (xA, yA), (xB, yB), (0, 255, 0), 2)
  			
  				print("{} original boxes, {} after suppression".format(len(rects), len(pick)))
  			
  				if len(rects) > 0:
  					features, height_from_floor = find_features(current_frame, rects[0], 0)
  					#print(features)
  					detected = True
  					gpio.output(17, True)
  		
  			if detected: # Tracking block
  				if Threshold == 0:
  					features_number = len(features)
  					Threshold = features_number * threshold_percent

  				#print ("Threshold" + str(Threshold))
  				if features_number < Threshold:
  					print ("Features less than threshold")
  					detected = False
  				else:
  					rawCapture.truncate(0)
  					next_frame = frame.array
  					#next_frame = imutils.resize(next_frame, width = 300)
  			
  					current_frame_copy = next_frame.copy()
  					next_frame = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)

  	 				#-----------Tracking using LK ---------------------------

  	 				try:
  	 					features = np.array(features)

  	 					(tracked_features, status, feature_errors) = cv2.calcOpticalFlowPyrLK(current_frame, next_frame, features, None, **lk_params)

  						arr_x = []
  						arr_y = []

  						for i in range(len(tracked_features)):
  							f = tracked_features[i]
  							x = f[0][0]
  							y = f[0][1]

  							arr_x.append(x)
  							arr_y.append(y)
  					
  						arr_x = sorted(arr_x)
  						arr_y = sorted(arr_y)

  						mid = len(arr_x)/2
  						X = arr_x[mid]
  						mid = len(arr_y)/2
  						Y = arr_y[mid]
              
  						#print(X)

  						new_feature_number = 0
  						temp_set_number = []
  						temp_distance = []
  						j = 0

  						print ("Height_from_floor" + str(height_from_floor))
  						print ("num" + str(features_number))
  						#print ("Status" + str(status))
  						#print ("Status[0]   " + str(status[0]))
  						#print ("Status[1]   " + str(status[1]))
  						#print ("Status[1][0]   " + str(status[1][0]))
						
  						for i in range(features_number):
  							if status[i][0] == 1:
  								new_feature_number += 1
  								#temp_distance.append(height_from_floor[i])
						
						temp_distance.append(height_from_floor-Y); 
						print (temp_distance)
  						height_from_floor = []

  						print ("Here")

  						for i in range(len(temp_distance)):
  						 	height_from_floor.append(temp_distance[i])

  						print ("Here2")

  						features_number = new_feature_number
  						features = []

  						for i in range(features_number):
  							features.append(tracked_features[i])

  						features = np.array(features)
  						tracked_features = []
  						current_frame = next_frame.copy()

  	 				except Exception, e:
  	 					raise e

  	 				#-------Compute Distance --------------------
  	 				#status, v = scaled_people_floor(features_number, features, height_from_floor)
					status, v = scaled_people_floor_test(Y, height_from_floor)

  	 				if status:
  	 					distance = compute_distance(v)
  	 					print (distance)
						cv2.putText(current_frame_copy, str(distance),(current_frame_copy.shape[1] - 200, current_frame_copy.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX,0.75, (0, 255, 0), 3)

  	 				#-------Showing Points ------------------------
  	 				for i in range(features_number):
  	 					cv2.circle(current_frame_copy,
  	 						   	tuple(features[i][0]),
  	 						   	3,
  	 						   	255,
  	 						   	-1)

  	 				cv2.circle(current_frame_copy,
  	 							(X,Y),
  	 							5,
  	 							(0,0,255),
  	 							-1)
  			
  			# show the output images
			
  			cv2.imshow("HOG", current_frame_copy)
  			key = cv2.waitKey(1) & 0xFF
  			rawCapture.truncate(0)
  		
  			if key == ord("w"):
  				break
  	except KeyboardInterrupt, SystemExit:
    		gpio.output(27, False)
		gpio.output(17, False)
		camera.release()
		cv2.destroyAllWindows()
		raise

if __name__ == '__main__':
	main()
