import argparse
import cv2
import math
import numpy as np
import os
import scipy
import subprocess
import sys

# Results variables
BASE_DIR		= "results"
RESULTS_DIR		= "video_results/"
OUTPUT_DIR		= "raw_output"
FILE_DIR 		= ""

FILE_NAME 		= "site_counts_"

RESULTS_EXT		= ".csv"

VIDEO_FILE 		= ""

FPS				= 30

def main(args):

	global RESULTS_DIR

	global FILE_NAME
	global VIDEO_FILE
	global FILE_DIR

	global FPS

	for arg in args:
		if "--three-valued" in arg:
			FILE_DIR += "three_valued"
			FILE_NAME += "three_valued"
		if "--mal-three-valued" in arg:
			FILE_DIR += "three_valued_robust_to_random"
			FILE_NAME += "three_valued_robust_to_random"
		if "--boolean-adopt" in arg:
			FILE_DIR += "boolean_adopt"
			FILE_NAME += "boolean_adopt"
		if "--mal-boolean-adopt" in arg:
			FILE_DIR += "boolean_adopt_robust_to_random"
			FILE_NAME += "boolean_adopt_robust_to_random"
		if "--boolean-uncertainty" in arg:
			FILE_DIR += "boolean_uncertainty"
			FILE_NAME += "boolean_uncertainty"
		if "--boolean-averaged" in arg:
			FILE_DIR += "boolean_averaged"
			FILE_NAME += "boolean_averaged"
		if "--majority-rule" in arg:
			FILE_DIR += "majority_rule"
			FILE_NAME += "majority_rule"

		if "--video" in arg:
			VIDEO_FILE = arg.split("=")[-1]
			if "moving" in VIDEO_FILE:
				RESULTS_DIR += "moving"
			elif "stationary" in VIDEO_FILE:
				RESULTS_DIR += "stationary"


	video_file = cv2.VideoCapture(VIDEO_FILE)

	interest_region = ()

	# List the LED BGR boundaries from min to max.
	LED_boundaries = [
		([170, 60, 80], [180, 255, 255]),	# Red
		([90, 60, 70], [115, 255, 255])	# Blue
		# ([60, 50, 0], [80, 255, 255])		# Green
	]

	site_num = 2
	site_counts = [[] for x in range(site_num)]

	skip_frames = 0
	count = 0

	while(True):
		proceed, frame = video_file.read()
		if not proceed:
			break
		#print(len(frame))
		#print(len(frame[0]))

		frame = np.array(frame)

		hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
		k = -1

		count += 1

		if count <= skip_frames:
			continue

		if FPS == 60 and count % 2 == 0:
			continue

		# loop over the colour boundaries and
		for i, (lower, upper) in enumerate(LED_boundaries):
			# create numpy arrays for the boundaries
			lower = np.array(lower)
			upper = np.array(upper)

			# find the colors withjin the specified boundaries and apply the mask
			mask = cv2.inRange(hsv, lower, upper)
			output = cv2.bitwise_and(frame, frame, mask = mask)

			# This works really well - morph open erodes, then dilates.
			# Great for removing noise
			# 4x4 kernel
			# kernel = np.ones((4,4),np.uint8)
			# mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

			# Dilates and then erodes, useful for closing small holes (in the centre of lights)
			# 6x6 kernel
			# kernel = np.ones((5,5),np.uint8)
			# mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
			# kernel = np.ones((5,5),np.uint8)
			# mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

			kernel = np.ones((3,3), np.uint8)
			erosion = cv2.erode(mask, kernel, iterations = 1)

			kernel = np.ones((8,8), np.uint8)
			dilate = cv2.dilate(erosion, kernel, iterations = 1)

			mask = dilate

			contours, _ = cv2.findContours(mask.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

			site_counts[i].append(len(contours))

	# 		cv2.imshow('frame', frame)
	# 		cv2.imshow('mask', mask)
	# 		cv2.imshow('output', output)
	# 		cv2.waitKey()

	# 		k = cv2.waitKey(5) & 0xFF

	# cv2.destroyAllWindows()

	with open(BASE_DIR + os.sep + RESULTS_DIR + os.sep + FILE_DIR + os.sep + OUTPUT_DIR + os.sep + FILE_NAME + RESULTS_EXT, "w") as outfile:

		for iteration in range(len(site_counts[0])):
			extra_comma = False
			for site in range(site_num):
				if not extra_comma:
					extra_comma = True
				else:
					outfile.write(",")
				outfile.write(str(site_counts[site][iteration]))
			outfile.write("\n")

if __name__ == '__main__':
	main(sys.argv)
