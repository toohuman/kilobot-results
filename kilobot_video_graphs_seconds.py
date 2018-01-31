import gc
import os
import random
import sys

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np

FILE_LOCATION 		= []
FILE_NAME 			= ""
DIRECTORY			= ""
RAW_OUTPUT			= "raw_output"

FILE_EXT 			= ".csv"
GRAPHS_EXT 			= ".pdf"

SITE_NUM  			= 0

fps = 30
frame_limit = 240 # Seconds
trim_threshold = 1
trim_to = 0	# Each category must have trim_to amount of bots dancing
				# before the plotting starts

figure_size = (18, 10)
font_size = 30
line_width = 3
line_colour_jump = 10

site_names = ["Choice A", "Choice B"]
linestyles = ['-', '--', '-.', ':']

# Percentiles

PERC_LOWER = 10
PERC_UPPER = 90

def main(args):

	FILE_NUMS = []
	legends = []

	for arg in args:
		if "--file" in arg:
			FILE_LOCATION = arg.split("=")[-1].split(os.sep)
			FILE_NAME = FILE_LOCATION[-1]
			GRAPH_OUTPUT = FILE_LOCATION[:-2]
			FILE_LOCATION = FILE_LOCATION[:-1]
		if "--range" in arg:
			file_range = list(map(int, arg.split("=")[-1].split(',')))
			FILE_NUMS = [x for x in range(file_range[0], file_range[-1] + 1)]

	site_stats = []
	frames = []

	for fnum in FILE_NUMS:
		try:
			site_stats.append(list())
			with open(os.sep.join(FILE_LOCATION) + os.sep + FILE_NAME + "_" + str(fnum) + FILE_EXT, "r") as infile:
				extend_list = True
				for line in infile:
					site_values = list(map(int, line.split(",")))
					if extend_list:
						for site in range(len(site_values)):
							site_stats[-1].append(list())
						extend_list = False

					for i, value in enumerate(site_values):
						site_stats[-1][i].append(value)

		except FileNotFoundError:
			pass

		if len(site_stats[-1][0]) > len(frames):
			frames = [i for i in range(len(site_stats[-1][0]))]

	# file_num = FILE_NAME.split("_")[-1]

	SITE_NUM = len(site_stats[0])


	# Comment out this code if not extending shorter runs to fit
	# the frame limit
	if len(frames) < fps * frame_limit:
		frames = [i for i in range(fps * frame_limit)]


	# Loop over the site_stats array for the largest amount of frames and
	# extend the array with the last value if it falls short..

	for i in range(len(site_stats)):
		for j in range(len(site_stats[i])):
			last_value = 0
			for k in range(len(frames)):
				try:
					last_value = site_stats[i][j][k]
				except:
					site_stats[i][j].append(last_value)

	frame_counts = [0] * len(frames)
	avg_site_stats = [[0 for x in range(len(frames))] for y in range(SITE_NUM)]
	std_dev = [[0 for x in range(len(frames))] for y in range(SITE_NUM)]
	perc_lower = [[0 for x in range(len(frames))] for y in range(SITE_NUM)]
	perc_upper = [[0 for x in range(len(frames))] for y in range(SITE_NUM)]
	temp_std_dev = [[[] for x in range(len(frames))] for y in range(SITE_NUM)]

	for i in range(len(FILE_NUMS)):
		for j in range(SITE_NUM):
			for k in range(len(frames)):
				if j == 0:
					frame_counts[k] += 1

				avg_site_stats[j][k] += site_stats[i][j][k]
				temp_std_dev[j][k].append(site_stats[i][j][k])


	for i in range(len(avg_site_stats)):
		for j in range(len(avg_site_stats[i])):
			if avg_site_stats[i][j] != 0:
				avg_site_stats[i][j] /= frame_counts[j]
				std_dev[i][j] = np.std(temp_std_dev[i][j])
				perc_lower[i][j] = avg_site_stats[i][j] - np.percentile(temp_std_dev[i][j], PERC_LOWER)
				perc_upper[i][j] = np.percentile(temp_std_dev[i][j], PERC_UPPER) - avg_site_stats[i][j]

	if trim_threshold > 0:
		for i in range(len(frames)):
			exceed_threshold = True
			for j in range(SITE_NUM):
				if avg_site_stats[j][i] < trim_threshold:
					exceed_threshold = False
					break
			if exceed_threshold == True:
				trim_to = i;
				break

		for i in range(SITE_NUM):
			avg_site_stats[i] = avg_site_stats[i][trim_to::]
			std_dev[i] = std_dev[i][trim_to::]
			perc_lower[i] = perc_lower[i][trim_to::]
			perc_upper[i] = perc_upper[i][trim_to::]
		frames = frames[:-trim_to:]

	cmap = cm.get_cmap('gray')
	c = [ cmap(float(x) / (20.0)) for x in range(0, 20) ]

	plt.figure(0, figsize = figure_size)
	font = { 'size' : font_size }
	plt.rc('font', **font)
	plt.grid(True)

	c_i = 0

	for site in range(SITE_NUM):
		print(avg_site_stats[site][::fps * 3][-1])
		# plt.plot([x/fps for x in frames[::fps]], avg_site_stats[site][::fps], color = c[c_i], linewidth = 2)
		# plt.errorbar([x/fps for x in frames[::fps * 3]], avg_site_stats[site][::fps * 3], std_dev[site][::fps * 3], 0, color = c[c_i], linewidth = line_width)
		plt.errorbar([x/fps for x in frames[::fps * 3]], avg_site_stats[site][::fps * 3],\
			yerr=np.vstack([perc_lower[site][::fps * 3], perc_upper[site][::fps * 3]]),\
			color = c[c_i], linewidth = line_width)
		c_i += line_colour_jump

	# plt.legend(["Site " + str(x) for x in range(SITE_NUM)], loc='best')
	c_i = 0
	for value in site_names:
		legends.append(mlines.Line2D([], [], color=c[c_i], linewidth = line_width, linestyle=linestyles[0], label=value))
		c_i += line_colour_jump

	plt.legend(handles=legends, loc='best')

	plt.xlabel('Seconds')
	plt.ylabel('No. of Kilobots signalling')
	if len(frames) < fps * frame_limit:
		plt.xlim([0,frames[-1]/fps])
	else:
		plt.xlim([0,frame_limit])
	plt.ylim([0,250])
	plt.savefig(os.sep.join(GRAPH_OUTPUT) + os.sep + "kilobot_trajectory" + GRAPHS_EXT, bbox_inches='tight')
	plt.clf()

if __name__ == "__main__":
	main(sys.argv)
