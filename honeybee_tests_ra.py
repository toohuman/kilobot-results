import gc
import os
import random
import sys
import subprocess

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.cm as cm

# Simulator variables
ARGS			= ["--gui", "--seed="]

random.seed(512)
SEEDS			= []

SITE_PAYOFF 	= []
LANG_SIZE		= 0

# Radius variables
RADIUS_FILE		= "test_swarm_.world"
RADII			= ["{0:.2f}".format(x/100) for x in range(0, 21, 2)]

SINGLE_RADIUS	= "1.00"

# Movement variables

# Results variables
RESULTS_DIR		= "results/test_results"
SEEDS_FILE		= "test_seeds"
PAYOFF_FILE		= "site_payoffs"
RESULTS_FILE	= "simulator_results"
RESULTS_EXT		= ".csv"

# Graphs variables
# GRAPHS_DIR		= "results"
GRAPH_FILES		= [["mean_payoff", "unique_sites"], ["dancing_for_site"], ["unique_beliefs"], ["site_belief_values"], ["message_counts"]]
GRAPHS_EXT		= ".pdf"

figure_size = (18, 8) #(24, 10)
font_size = 30 # 18
line_width = 3
line_colour_jump = 3

def main(args):
	# Command-line arguments
	plot_traj	= False
	plot_steady	= False

	global RESULTS_DIR

	for arg in args:
		if "traj" in arg:
			plot_traj = True
		if "steady" in arg:
			plot_steady = True
		if "--three-valued" in arg:
			RESULTS_DIR += "/three_valued"
		if "--mal-three-valued" in arg:
			RESULTS_DIR += "/three_valued_mal"
		if "--static-mal-three-valued" in arg:
			RESULTS_DIR += "/three_valued_mal_static"
		if "--noisy-three-valued" in arg:
			RESULTS_DIR += "/three_valued_noisy"
		if "--majority-rule" in arg:
			RESULTS_DIR += "/majority_rule"
		if "--boolean-uncertainty" in arg:
			RESULTS_DIR += "/boolean_uncertainty"
		if "--boolean-averaged" in arg:
			RESULTS_DIR += "/boolean_averaged"
		if "--boolean-adopt" in arg:
			RESULTS_DIR += "/boolean_adopt_dancer"
		if "--mal-boolean-adopt" in arg:
			RESULTS_DIR += "/boolean_adopt_mal"
		if "--static-mal-boolean-adopt" in arg:
			RESULTS_DIR += "/boolean_adopt_mal_static"
		if "--noisy-boolean-adopt" in arg:
			RESULTS_DIR += "/boolean_adopt_noisy"
		if "--boolean-swap" in arg:
			RESULTS_DIR += "/boolean_50_50_swap"

		if "--sites" in arg:
			vars = arg.split("=")[-1].split(",")
			for value in vars:
				RESULTS_DIR += "/" + value

		if "--vars" in arg:
			vars = arg.split("=")[-1]
			RESULTS_DIR	+= "_" + vars

	radii_runs = list()

	linestyles = ['-', '--', '-.', ':']

	with open(RESULTS_DIR + os.sep + SEEDS_FILE + RESULTS_EXT, "r") as infile:
		for line in infile:
			file_seeds = line.split(',')
			random.seed(file_seeds[0])

			SEEDS 			= [x for x in file_seeds[1:]]
			TESTS 			= len(SEEDS)

	with open(RESULTS_DIR + os.sep + PAYOFF_FILE + RESULTS_EXT, "r") as infile:
		for line in infile:
			SITE_PAYOFF 	= list( map(int, line.split(',')) )
			LANG_SIZE		= len(SITE_PAYOFF)
			break

	steady_payoff_stats 		= list()
	steady_site_stats 			= list()
	steady_site_belief_stats 	= list()
	steady_unique_site_stats	= list()
	steady_unique_belief_stats 	= list()
	steady_message_stats		= list()

	for radius in RADII:
		radii_runs.append(radius)

		belief_results = list()
		unique_belief_results = list()
		nest_site_results = list()
		message_results = list()

		results_files = ["average_beliefs", "unique_beliefs", "nest_sites", "message_counts"]
		for file in results_files:
			with open(RESULTS_DIR + os.sep + file + "_radius_" + str(radius) + RESULTS_EXT, "r") as infile:
				iteration = 0
				for line in infile:
					if "average_b" in file:
						belief_results.append(list())
						for _ in range(len(line.split(","))):
							belief_results[-1].append(list())
					if "unique_b" in file:
						unique_belief_results.append(list())
					elif "nest" in file:
						nest_site_results.append(list())
						for _ in range(len(line.split(","))):
							nest_site_results[-1].append(list())
					elif "message" in file:
						message_results.append(list())

					for test, test_elements in enumerate(line.split(",")):
						if "average_b" in file:
							for element in test_elements.split(";"):
								belief_results[iteration][test].append(float(element))
						if "unique_b" in file:
							unique_belief_results[iteration].append(int(test_elements))
						elif "nest" in file:
							for element in test_elements.split(";"):
								nest_site_results[iteration][test].append(int(element))
						elif "message" in file:
							message_results[iteration].append(float(test_elements))

					iteration += 1

		iterations 	= len(nest_site_results)
		tests 		= len(nest_site_results[-1])

		payoff_stats = list()
		site_stats = list()
		site_belief_stats = list()
		unique_site_stats = list()
		unique_belief_stats = list()
		message_stats = list()

		for iter in range(iterations):			#for number of iterations
												#[avg, min, max, std. dev.]
			payoff_stats.append([0, 0, 0, 0])
			site_stats.append([[0, 0, 0, 0] for x in range(LANG_SIZE + 1)])
			site_belief_stats.append([[0, 0, 0, 0] for x in range(LANG_SIZE)])
			unique_site_stats.append([0, 0, 0, 0])
			unique_belief_stats.append([0, 0, 0, 0])
			message_stats.append([0, 0, 0, 0])

		# Start calculating the results and inserting them into the appropriate lists
		for iter in range(iterations):

			payoff_values = list()
			site_values = [list() for x in range(LANG_SIZE + 1)]
			site_belief_values = [list() for x in range(LANG_SIZE)]
			unique_site_values = list()
			unique_belief_values = list()
			message_values = list()

			for test in range(tests):

				payoff_sum = 0.0
				for l in range(LANG_SIZE + 1):
					site_values[l].append(nest_site_results[iter][test][l])
					if l == LANG_SIZE:
						continue

					site_belief_values[l].append(belief_results[iter][test][l])

					payoff_sum += belief_results[iter][test][l] * SITE_PAYOFF[l]
				payoff_values.append(payoff_sum)

				# Unique site results
				nonZeroCount = 0
				for dancers in nest_site_results[iter][test][0:-1]:
					if dancers > 0:
						nonZeroCount += 1
				unique_site_values.append(nonZeroCount)
				message_values.append(message_results[iter][test])

				unique_belief_values.append(unique_belief_results[iter][test])

			payoffs = np.array(payoff_values)
			payoff_stats[iter][0] = np.average(payoffs)
			payoff_stats[iter][1] = np.min(payoffs)
			payoff_stats[iter][2] = np.max(payoffs)
			payoff_stats[iter][3] = np.std(payoffs)

			msgs = np.array(message_values)
			message_stats[iter][0] = np.average(msgs)
			message_stats[iter][1] = np.min(msgs)
			message_stats[iter][2] = np.max(msgs)
			message_stats[iter][3] = np.std(msgs)

			unique_beliefs = np.array(unique_belief_values)
			unique_belief_stats[iter][0] = np.average(unique_beliefs)
			unique_belief_stats[iter][1] = np.min(unique_beliefs)
			unique_belief_stats[iter][2] = np.max(unique_beliefs)
			unique_belief_stats[iter][3] = np.std(unique_beliefs)

			unique_sites = np.array(unique_site_values)
			unique_site_stats[iter][0] = np.average(unique_sites)
			unique_site_stats[iter][1] = np.min(unique_sites)
			unique_site_stats[iter][2] = np.max(unique_sites)
			unique_site_stats[iter][3] = np.std(unique_sites)

			for prop in range(LANG_SIZE + 1):

				sites = np.array(site_values)
				site_stats[iter][prop][0] = np.average(sites[prop])
				site_stats[iter][prop][1] = np.min(sites[prop])
				site_stats[iter][prop][2] = np.max(sites[prop])
				site_stats[iter][prop][3] = np.std(sites[prop])

			for prop in range(LANG_SIZE):

				site_beliefs = np.array(site_belief_values)
				# for truth in range(0, 3):
				site_belief_stats[iter][prop][0] = np.average(site_beliefs[prop])
				site_belief_stats[iter][prop][1] = np.min(site_beliefs[prop])
				site_belief_stats[iter][prop][2] = np.max(site_beliefs[prop])
				site_belief_stats[iter][prop][3] = np.std(site_beliefs[prop])

		steady_payoff_stats.append(payoff_stats[-1])
		steady_site_stats.append(site_stats[-1])
		steady_unique_site_stats.append(unique_site_stats[-1])
		steady_message_stats.append(message_stats[-1])
		steady_unique_belief_stats.append(unique_belief_stats[-1])
		steady_site_belief_stats.append(site_belief_stats[-1])

		# GRAPHING TRAJECTORIES
		x_steps = [x for x in range(iterations)]
		cmap = cm.get_cmap('gray')
		c = [ cmap(float(x) / (10.0)) for x in range(0, 10) ]
		font = { 'size' : font_size }

		for files in GRAPH_FILES:
			plt.figure(0, figsize = figure_size)
			c_i = 0
			line_i = 0
			legends = []

			if "message" in files[0]:
				continue

			for file in files:

				results = list()
				std_dev = list()
				if file == "mean_payoff":
					results = [x[0] for x in payoff_stats]
					std_dev = [x[3] for x in payoff_stats]
				elif file == "unique_sites":
					results = [x[0] for x in unique_site_stats]
					std_dev = [x[3] for x in unique_site_stats]
				elif file == "unique_beliefs":
					results = [x[0] for x in unique_belief_stats]
					std_dev = [x[3] for x in unique_belief_stats]

				if file == "dancing_for_site":
					# c_i = 0
					for i in range(LANG_SIZE + 1):
						# z = iteration -> y = test -> x = nest_index
						results = [x[i][0] for x in site_stats]
						std_dev = [x[i][3] for x in site_stats]
						plt.errorbar(x_steps[::10] + [x_steps[-1]], results[::10] + [results[-1]],\
							std_dev[::10] + [std_dev[-1]], 0, color = c[c_i], linewidth = line_width, linestyle=linestyles[line_i])
						c_i += line_colour_jump
						line_i += 1
				elif file == "site_belief_values":
					c_i = 0
					for i in range(LANG_SIZE):
						# z = iteration -> y = test -> x = nest_index
						results = [x[i][0] for x in site_belief_stats]
						std_dev = [x[i][3] for x in site_belief_stats]
						plt.errorbar(x_steps[::10] + [x_steps[-1]], results[::10] + [results[-1]],\
							std_dev[::10] + [std_dev[-1]], 0, color = c[c_i], linewidth = line_width, linestyle=linestyles[line_i])
						c_i += line_colour_jump
						line_i += 1
				else:
					plt.errorbar(x_steps[::10] + [x_steps[-1]], results[::10] + [results[-1]],\
						std_dev[::10] + [std_dev[-1]], 0, color = c[c_i], linewidth = line_width, linestyle=linestyles[line_i])
					c_i += line_colour_jump
					line_i += 1

			line_i = 0

			FIG_TITLE = ""
			plt.rc('font', **font)
			plt.grid(True)
			x1,x2,y1,y2 = plt.axis()
			plt.axis((x1,x2,0,y2));

			if GRAPH_FILES.index(files) == 0:
				temp_legend = [" ".join(x.split("_")).capitalize() for x in files]
				c_i = 0
				for value in temp_legend:
					legends.append(mlines.Line2D([], [], color=c[c_i], linewidth = line_width, linestyle=linestyles[line_i], label=value))
					c_i += line_colour_jump
					line_i += 1
				# plt.annotate("Site Values: [ " + ", ".join([str(x) for x in SITE_PAYOFF]) + " ]",
					# (.8, .75), xycoords='axes fraction', backgroundcolor='w')
				# plt.annotate("Msg. Received: {0:.2f}".format(np.average([x[0] for x in message_stats])),
					# (.8, .70), xycoords='axes fraction', backgroundcolor='w')
				FIG_TITLE = "kilobot_stats"
			elif GRAPH_FILES.index(files) == 1:
				temp_legend = ["Site " + str(SITE_PAYOFF.index(x)) + ": " + str(x) for x in SITE_PAYOFF]\
					+ ["Not Dancing"]
				c_i = 0
				for value in temp_legend:
					legends.append(mlines.Line2D([], [], color=c[c_i], linewidth = line_width, linestyle=linestyles[line_i], label=value))
					c_i += line_colour_jump
					line_i += 1
				temp_y_label = " ".join(file.split("_")).capitalize()
				plt.ylabel("No. of Kilobots signalling")
				FIG_TITLE = file
			elif GRAPH_FILES.index(files) == 2:
				plt.ylabel(" ".join(file.split("_")).capitalize())
				FIG_TITLE = file
			elif GRAPH_FILES.index(files) == 3:
				temp_legend = ["Site " + str(x) for x in range(LANG_SIZE)]
				c_i = 0
				for value in temp_legend:
					legends.append(mlines.Line2D([], [], color=c[c_i], linewidth = line_width, linestyle=linestyles[line_i], label=value))
					c_i += line_colour_jump
					line_i += 1
				plt.ylabel("Average beliefs")
				FIG_TITLE = file
				#plt.axis((x1,x2,0,1));

			plt.xlabel('Iterations')
			if len(legends) > 0:
				plt.legend(handles=legends, loc='best')
			#plt.ylabel(" ".join(file.split("_")).capitalize())
			plt.savefig("results" + os.sep + "trajectories" + os.sep + FIG_TITLE + "_" + radii_runs[-1] + GRAPHS_EXT, bbox_inches='tight')
			plt.clf()

		gc.collect()

	# GRAPHING STEADY STATES FOR RADII
	radii = [float(x) for x in radii_runs]
	cmap = cm.get_cmap('gray')
	c = [ cmap(float(x) / (10.0)) for x in range(0, 10) ]
	font = { 'size' : font_size }

	for files in GRAPH_FILES:
		plt.figure(0, figsize = figure_size)
		c_i = 0
		line_i = 0
		add_legend = True
		legends = []

		for file in files:

			results = list()
			std_dev = list()
			if file == "mean_payoff":
				results = [x[0] for x in steady_payoff_stats]
				std_dev = [x[3] for x in steady_payoff_stats]
			elif file == "unique_sites":
				results = [x[0] for x in steady_unique_site_stats]
				std_dev = [x[3] for x in steady_unique_site_stats]
			elif file == "unique_beliefs":
				results = [x[0] for x in steady_unique_belief_stats]
				std_dev = [x[3] for x in steady_unique_belief_stats]
			elif file == "message_counts":
				results = [x[0] for x in steady_message_stats]
				std_dev = [x[3] for x in steady_message_stats]

			if file == "dancing_for_site":
				c_i = 0
				for i in range(LANG_SIZE + 1):
					# z = iteration -> y = test -> x = nest_index
					results = [x[i][0] for x in steady_site_stats]
					std_dev = [x[i][3] for x in steady_site_stats]
					plt.errorbar(radii, results, std_dev, 0, color = c[c_i], linewidth = line_width, linestyle=linestyles[line_i])
					c_i += line_colour_jump
					line_i += 1
			elif file == "site_belief_values":
				c_i = 0
				for i in range(LANG_SIZE):
					# z = iteration -> y = test -> x = nest_indexresults = [x[i][0] for x in steady_site_stats]
					results = [x[i][0] for x in steady_site_belief_stats]
					std_dev = [x[i][3] for x in steady_site_belief_stats]
					plt.errorbar(radii, results, std_dev, 0, color = c[c_i], linewidth = line_width, linestyle=linestyles[line_i])
					c_i += line_colour_jump
					line_i += 1
			else:
				plt.errorbar(radii, results, std_dev, 0, color = c[c_i], linewidth = line_width, linestyle=linestyles[line_i])
				c_i += line_colour_jump

			line_i += 1

		line_i = 0

		FIG_TITLE = ""
		plt.rc('font', **font)
		plt.grid(True)
		x1,x2,y1,y2 = plt.axis()
		plt.axis((x1,x2,0,y2));

		if GRAPH_FILES.index(files) == 0:
			temp_legend = [" ".join(x.split("_")).capitalize() for x in files]
			c_i = 0
			for value in temp_legend:
				legends.append(mlines.Line2D([], [], color=c[c_i], linewidth = line_width, linestyle=linestyles[line_i], label=value))
				c_i += line_colour_jump
				line_i += 1
			# plt.annotate("Site Values: [ " + ", ".join([str(x) for x in SITE_PAYOFF]) + " ]",
				# (.8, .75), xycoords='axes fraction', backgroundcolor='w')
			FIG_TITLE = "kilobot_stats"
		elif GRAPH_FILES.index(files) == 1:
			temp_legend = ["Site " + str(SITE_PAYOFF.index(x)) + ": " + str(x) for x in SITE_PAYOFF]\
				+ ["Not Dancing"]
			c_i = 0
			for value in temp_legend:
				legends.append(mlines.Line2D([], [], color=c[c_i], linewidth = line_width, linestyle=linestyles[line_i], label=value))
				c_i += line_colour_jump
				line_i += 1
			temp_y_label = " ".join(file.split("_")).capitalize()
			plt.ylabel("No. of Kilobots " + temp_y_label)
			FIG_TITLE = file
		elif GRAPH_FILES.index(files) == 2:
			plt.ylabel(" ".join(file.split("_")).capitalize())
			FIG_TITLE = file
			#plt.axis((x1,x2,0,1));
		elif GRAPH_FILES.index(files) == 3:
			temp_legend = ["Site: " + str(x) for x in range(LANG_SIZE)]
			c_i = 0
			for value in temp_legend:
				legends.append(mlines.Line2D([], [], color=c[c_i], linewidth = line_width, linestyle=linestyles[line_i], label=value))
				c_i += line_colour_jump
				line_i += 1
			plt.ylabel("Average beliefs")
			FIG_TITLE = file
			#plt.axis((x1,x2,0,1));
		elif GRAPH_FILES.index(files) == 4:
			plt.legend(["Message Counts"])
			plt.ylabel(" ".join(file.split("_")).capitalize())
			FIG_TITLE = file

		plt.xlabel('Radii')
		if len(legends) > 0:
			plt.legend(handles=legends, loc='best')
		plt.xlim([0,float(RADII[-1])])
		#plt.ylabel(" ".join(file.split("_")).capitalize())
		plt.savefig("results" + os.sep + "steadystates" + os.sep + FIG_TITLE + GRAPHS_EXT, bbox_inches='tight')
		plt.clf()

if __name__ == "__main__":
	main(sys.argv)
