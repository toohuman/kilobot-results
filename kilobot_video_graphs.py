import gc
import os
import random
import sys

import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

FILE_LOCATION 		= []
FILE_NAME 			= ""
DIRECTORY			= ""
RAW_OUTPUT			= "raw_output"

FILE_EXT 			= ".csv"
GRAPHS_EXT 			= ".pdf"

SITE_NUM  			= 0

fps = 30
iterations = 1000 # Seconds
kilobot_updates = 4 # 8/32 per second
trim_threshold = 1
trim_to = 0	# Each category must have trim_to amount of bots dancing
				# before the plotting starts

figure_size = (18, 9)
font_size = 16
line_width = 1.5
line_colour_jump = 3

site_names = ["A", "B"]
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
	if len(frames) < fps * (iterations / kilobot_updates):
		frames = [i for i in range(round(fps * (iterations / kilobot_updates) + (iterations / kilobot_updates)))]


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

	# Convert resutls to percentages

	# print(len(site_stats))
	# print(len(site_stats[0]))
	# print(len(site_stats[0][0]))

	# for i in range(len(site_stats)):
	# 	for k in range(len(site_stats[i][0])):
	# 		total = site_stats[i][0][k] + site_stats[i][1][k]
	# 		if total == 0:
	# 			continue
	# 		site_stats[i][0][k] = (site_stats[i][0][k] / total) * 100
	# 		site_stats[i][1][k] = (site_stats[i][1][k] / total) * 100

	# Generate the averages of the results

	frame_counts = [0] * len(frames)
	avg_site_stats = [[0 for x in range(len(frames))] for y in range(SITE_NUM)]
	std_dev = [[0 for x in range(len(frames))] for y in range(SITE_NUM)]
	perc_lower = [[0 for x in range(len(frames))] for y in range(SITE_NUM)]
	perc_upper = [[0 for x in range(len(frames))] for y in range(SITE_NUM)]
	temp_std_dev = [[[] for x in range(len(frames))] for y in range(SITE_NUM)]

	# for i in range(len(FILE_NUMS)):
	# 	for j in range(SITE_NUM):
	# 		for k in range(len(frames)):
	# 			if j == 0:
	# 				frame_counts[k] += 1

	# 			avg_site_stats[j][k] += site_stats[i][j][k]
	# 			temp_std_dev[j][k].append(site_stats[i][j][k])

	for i in range(len(FILE_NUMS)):
		for k in range(len(frames)):
			total = 0
			for j in range(SITE_NUM):
				if j == 0:
					frame_counts[k] += 1
				total += site_stats[i][j][k]
			for j in range(SITE_NUM):
				if total != 0:
					avg_site_stats[j][k] += (site_stats[i][j][k] / total) * 100
					temp_std_dev[j][k].append((site_stats[i][j][k] / total) * 100)
				else:
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
		# frames = frames[:-trim_to:]

	# Now that the non-signalling period has been trimmed, we
	# extend the end of the values again to ensure that _iterations_
	# results values are recorded, and that the plots do not stop short

	for i in range(SITE_NUM):
		last_avg_value = 0
		last_std_value = 0
		for f in range(len(frames)):
			try:
				last_avg_value = avg_site_stats[i][f]
				last_std_value = std_dev[i][f]
				last_perc_lower = perc_lower[i][f]
				last_perc_upper = perc_upper[i][f]
			except:
				avg_site_stats[i].append(last_avg_value)
				std_dev[i].append(last_std_value)
				perc_lower[i].append(last_perc_lower)
				perc_upper[i].append(last_perc_upper)

	# for i in range(len(avg_site_stats[0])):
	# 	# print(str(avg_site_stats[0][i]) + " / " + str(std_dev[0][i]) + " = " + str(avg_site_stats[0][i] / std_dev[0][i]))
	# 	# std_dev[0][i] = avg_site_stats[0][i] / std_dev[0][i]
	# 	# std_dev[1][i] = avg_site_stats[1][i] / std_dev[1][i]
	# 	total = avg_site_stats[0][i] + avg_site_stats[1][i]
	# 	avg_site_stats[0][i] = (avg_site_stats[0][i] / total) * 100
	# 	avg_site_stats[1][i] = (avg_site_stats[1][i] / total) * 100
	# 	std_dev[0][i] = (std_dev[0][i] / total) * 100
	# 	std_dev[1][i] = (std_dev[1][i] / total) * 100
	# 	perc_lower[0][i] = (perc_lower[0][i] / total) * 100
	# 	perc_lower[1][i] = (perc_lower[1][i] / total) * 100
	# 	perc_upper[0][i] = (perc_upper[0][i] / total) * 100
	# 	perc_upper[1][i] = (perc_upper[1][i] / total) * 100

	# frames = [x for x in range(iterations + fps)]
	# for i in range(len(avg_site_stats)):
	# 	avg_site_stats[i] = avg_site_stats[i][:len(frames)]
	# 	std_dev[i] = std_dev[i][:len(frames)]

	cmap = cm.get_cmap('Set1')
	c = [ cmap(float(x) / (20.0)) for x in range(0, 20) ]
	line_colour_jump = 3

	plt.figure(0, figsize = figure_size)
	font = { 'size' : font_size }
	plt.rc('font', **font)
	plt.grid(True)

	tick_rate = int((fps/kilobot_updates) * fps)

	print(len(frames))
	print(len(avg_site_stats[0]))
	print(tick_rate)

	c_i = 0
	line_i = 0

	for site in reversed(range(SITE_NUM)):
		print(c_i)
		print(avg_site_stats[site][::tick_rate][-1])
		print([int(x/(fps/kilobot_updates)) for x in frames][::tick_rate])
		# plt.plot([x/fps for x in frames[::fps]], avg_site_stats[site][::fps], color = c[c_i], linewidth = 2)
		# plt.errorbar([int(x/(fps/kilobot_updates)) for x in frames][::tick_rate], avg_site_stats[site][::tick_rate], std_dev[site][::tick_rate], 0, color = c[c_i], linewidth = line_width, linestyle=linestyles[line_i])
		plt.errorbar([int(x/(fps/kilobot_updates)) for x in frames][::tick_rate], avg_site_stats[site][::tick_rate],\
			yerr=np.vstack([perc_lower[site][::tick_rate], perc_upper[site][::tick_rate]]),\
			color = c[c_i], linewidth = line_width, linestyle=linestyles[line_i])
		# c_i += line_colour_jump
		c_i += line_colour_jump

	# plt.legend(["Site " + str(x) for x in range(SITE_NUM)], loc='best')
	c_i = 0
	line_i = 0
	for value in site_names:
		print(c_i)
		legends.append(mlines.Line2D([], [], color=c[c_i], linewidth = line_width, linestyle=linestyles[line_i], label=value))
		# c_i += line_colour_jump
		c_i += line_colour_jump

	plt.legend(handles=legends, loc='best', prop={'size':font_size})

	plt.xlabel('Iterations')
	plt.ylabel('% of Kilobots signalling')
	plt.xlim([0,iterations])
	plt.ylim([0,105])
	plt.savefig(os.sep.join(GRAPH_OUTPUT) + os.sep + "kilobot_trajectory" + GRAPHS_EXT, bbox_inches='tight')
	plt.clf()

if __name__ == "__main__":
	main(sys.argv)
