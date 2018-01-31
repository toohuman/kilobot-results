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

# Simulator variables
ARGS			= ["--gui", "--seed="]

random.seed(512)
SEEDS			= []

SITE_PAYOFF 	= []
SITE_NAMES		= ["A", "B", "C", "D", "E"]
LANG_SIZE		= 0

# Radius variables
RADIUS_FILE		= "test_swarm_.world"
RADII			= ["{0:.2f}".format(x/100) for x in range(0, 21, 2)]

SINGLE_RADIUS	= "1.00"

# Percentiles

PERC_LOWER = 10
PERC_UPPER = 90

# Results variables
RESULTS_DIR		= "results/test_results"
RESULTS_DIRS	= []
SEEDS_FILE		= "test_seeds"
PAYOFF_FILE		= "site_payoffs"
RESULTS_EXT		= ".csv"

# Graphs variables
GRAPH_FILES		= [["mean_payoff", "unique_sites"], ["dancing_for_site"], ["unique_beliefs"], ["site_belief_values"], ["message_counts"]]
GRAPHS_EXT		= ".pdf"

figure_size = (18, 9) #(9, 5)
font_size = 16 # 18
line_width = 1.5
line_colour_jump = 3
colour_map = 'gray'

def main(args):
	# Command-line arguments
	plot_traj	= False
	plot_steady	= False

	add_model = False

	global RESULTS_DIR
	global RESULTS_DIRS

	CMP_FILES = []
	CMP_VALUES = [""]
	CMP_RATES = []

	results_files = ["average_beliefs", "unique_beliefs", "nest_sites", "message_counts"]

	iterations = 0
	tests = 0

	for arg in args:
		if "traj" in arg:
			plot_traj = True
		if "steady" in arg:
			plot_steady = True
		if "--three-valued" in arg:
			RESULTS_DIRS.append("three_valued")
			CMP_FILES.append("three-valued")
		if "--mal-three-valued" in arg:
			RESULTS_DIRS.append("three_valued_mal")
			CMP_FILES.append("three-valued malfunction")
		if "--static-mal-three-valued" in arg:
			RESULTS_DIRS.append("three_valued_mal_static")
			CMP_FILES.append("three-valued (static) malfunction")
		if "--noisy-three-valued" in arg:
			RESULTS_DIRS.append("three_valued_noisy")
			CMP_FILES.append("three-valued (noisy)")
		if "--site-drop-three-valued" in arg:
			RESULTS_DIRS.append("three_valued_site_drop")
			CMP_FILES.append("three-valued (site drop)")
		if "--fast-three-valued" in arg:
			RESULTS_DIRS.append("three_valued_fast")
			CMP_FILES.append("three-valued fast")
		if "--majority-rule" in arg:
			RESULTS_DIRS.append("majority_rule")
			CMP_FILES.append("majority rule")
		if "--boolean-uncertainty" in arg:
			RESULTS_DIRS.append("boolean_uncertainty")
			CMP_FILES.append("boolean (uncertainty)")
		if "--boolean-averaged" in arg:
			RESULTS_DIRS.append("boolean_averaged")
			CMP_FILES.append("boolean (averaged)")
		if "--boolean-adopt" in arg:
			RESULTS_DIRS.append("boolean_adopt")
			CMP_FILES.append("weighted voter")
		if "--mal-boolean-adopt" in arg:
			RESULTS_DIRS.append("boolean_adopt_mal")
			CMP_FILES.append("boolean malfunction")
		if "--static-mal-boolean-adopt" in arg:
			RESULTS_DIRS.append("boolean_adopt_mal_static")
			CMP_FILES.append("boolean (static) malfunction")
		if "--noisy-boolean-adopt" in arg:
			RESULTS_DIRS.append("boolean_adopt_noisy")
			CMP_FILES.append("boolean (noisy)")
		if "--site-drop-boolean-adopt" in arg:
			RESULTS_DIRS.append("boolean_adopt_site_drop")
			CMP_FILES.append("boolean (site drop)")
		if "--boolean-swap" in arg:
			RESULTS_DIRS.append("boolean_50_50_swap")
			CMP_FILES.append("boolean (50-50 swap)")

		if "--motion" in arg:
			vars = arg.split("=")[-1].split(",")
			for value in vars:
				RESULTS_DIR += "/" + value

		if "--sites" in arg:
			vars = arg.split("=")[-1].split(",")
			for value in vars:
				RESULTS_DIR += "/" + value

		if "--vars" in arg:
			vars = arg.split("=")[-1].split(",")
			for value in vars:
				CMP_VALUES.append("_" + value)

	linestyles = ['-', '--', '-.', ':']

	if len(RESULTS_DIRS) > 1:
		add_model = True

	collective_payoff_stats 		= list()
	collective_site_stats 			= list()
	collective_site_belief_stats 	= list()
	collective_unique_site_stats	= list()
	collective_unique_belief_stats 	= list()
	collective_message_stats		= list()

	for r, radius in enumerate(RADII):

		collective_payoff_stats.append(list())
		collective_site_stats.append(list())
		collective_site_belief_stats.append(list())
		collective_unique_site_stats.append(list())
		collective_unique_belief_stats.append(list())
		collective_message_stats.append(list())

		for cmp_i, cmp_dir in enumerate(RESULTS_DIRS):

			collective_payoff_stats[r].append(list())
			collective_site_stats[r].append(list())
			collective_site_belief_stats[r].append(list())
			collective_unique_site_stats[r].append(list())
			collective_unique_belief_stats[r].append(list())
			collective_message_stats[r].append(list())

	for r, radius in enumerate(RADII):

		CMP_RATES.append(list())

		for cmp_i, cmp_dir in enumerate(RESULTS_DIRS):

			cmp_vi = 0

			CMP_RATES[-1].append(list())

			for cmp_val in CMP_VALUES:

				try:
					with open(RESULTS_DIR + os.sep + cmp_dir + cmp_val + os.sep + SEEDS_FILE + RESULTS_EXT, "r") as infile:
						for line in infile:
							file_seeds = line.split(',')
							random.seed(file_seeds[0])

							SEEDS 			= [x for x in file_seeds[1:]]
							TESTS 			= len(SEEDS)


				except FileNotFoundError:
					continue

				CMP_RATES[r][cmp_i].append(cmp_val[1:])

				with open(RESULTS_DIR + os.sep + cmp_dir + cmp_val + os.sep + PAYOFF_FILE + RESULTS_EXT, "r") as infile:
					for line in infile:
						SITE_PAYOFF 	= list( map(int, line.split(',')) )[::-1]
						LANG_SIZE		= len(SITE_PAYOFF)
						break

				belief_results = list()
				unique_belief_results = list()
				nest_site_results = list()
				message_results = list()

				for file in results_files:
					with open(RESULTS_DIR + os.sep + cmp_dir + cmp_val + os.sep + file + "_radius_" + str(radius) + RESULTS_EXT, "r") as infile:
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
									for element in reversed(test_elements.split(";")):
										belief_results[iteration][test].append(float(element))
								if "unique_b" in file:
									unique_belief_results[iteration].append(int(test_elements))
								elif "nest" in file:
									elements = test_elements.split(";")
									elements.reverse()
									elements = elements[1:] + [elements[0]]
									for element in elements:
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
					payoff_stats.append([0, 0, 0, 0, 0, 0])
					site_stats.append([[0, 0, 0, 0, 0, 0] for x in range(LANG_SIZE + 1)])
					site_belief_stats.append([[0, 0, 0, 0, 0, 0] for x in range(LANG_SIZE)])
					unique_site_stats.append([0, 0, 0, 0, 0, 0])
					unique_belief_stats.append([0, 0, 0, 0, 0, 0])
					message_stats.append([0, 0, 0, 0, 0, 0])

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
					payoff_stats[iter][4] = payoff_stats[iter][0] - np.percentile(payoffs, PERC_LOWER)
					payoff_stats[iter][5] = np.percentile(payoffs, PERC_UPPER) - payoff_stats[iter][0]
					if payoff_stats[iter][4] < 0:
						payoff_stats[iter][4] = 0
					if payoff_stats[iter][5] < 0:
						payoff_stats[iter][5] = 0

					msgs = np.array(message_values)
					message_stats[iter][0] = np.average(msgs)
					message_stats[iter][1] = np.min(msgs)
					message_stats[iter][2] = np.max(msgs)
					message_stats[iter][3] = np.std(msgs)
					message_stats[iter][4] = message_stats[iter][0] - np.percentile(msgs, PERC_LOWER)
					message_stats[iter][5] = np.percentile(msgs, PERC_UPPER) - message_stats[iter][0]
					if message_stats[iter][4] < 0:
						message_stats[iter][4] = 0
					if message_stats[iter][5] < 0:
						message_stats[iter][5] = 0

					unique_beliefs = np.array(unique_belief_values)
					unique_belief_stats[iter][0] = np.average(unique_beliefs)
					unique_belief_stats[iter][1] = np.min(unique_beliefs)
					unique_belief_stats[iter][2] = np.max(unique_beliefs)
					unique_belief_stats[iter][3] = np.std(unique_beliefs)
					unique_belief_stats[iter][4] = unique_belief_stats[iter][0] - np.percentile(unique_beliefs, PERC_LOWER)
					unique_belief_stats[iter][5] = np.percentile(unique_beliefs, PERC_UPPER) - unique_belief_stats[iter][0]
					if unique_belief_stats[iter][4] < 0:
						unique_belief_stats[iter][4] = 0
					if unique_belief_stats[iter][5] < 0:
						unique_belief_stats[iter][5] = 0

					unique_sites = np.array(unique_site_values)
					unique_site_stats[iter][0] = np.average(unique_sites)
					unique_site_stats[iter][1] = np.min(unique_sites)
					unique_site_stats[iter][2] = np.max(unique_sites)
					unique_site_stats[iter][3] = np.std(unique_sites)
					unique_site_stats[iter][4] = unique_site_stats[iter][0] - np.percentile(unique_sites, PERC_LOWER)
					unique_site_stats[iter][5] = np.percentile(unique_sites, PERC_UPPER) - unique_site_stats[iter][0]
					if unique_site_stats[iter][4] < 0:
						unique_site_stats[iter][4] = 0
					if unique_site_stats[iter][5] < 0:
						unique_site_stats[iter][5] = 0

					total = 0.0
					for prop in range(LANG_SIZE + 1):

						sites = np.array(site_values)
						site_stats[iter][prop][0] = np.average(sites[prop])
						total += site_stats[iter][prop][0]
						site_stats[iter][prop][1] = np.min(sites[prop])
						site_stats[iter][prop][2] = np.max(sites[prop])
						site_stats[iter][prop][3] = np.std(sites[prop])
						site_stats[iter][prop][4] = np.percentile(sites[prop], PERC_LOWER)
						site_stats[iter][prop][5] = np.percentile(sites[prop], PERC_UPPER)

					# Convert site stats to percentages

					for prop in range(LANG_SIZE + 1):

						site_stats[iter][prop][0] = (site_stats[iter][prop][0] / total) * 100
						site_stats[iter][prop][1] = (site_stats[iter][prop][1] / total) * 100
						site_stats[iter][prop][2] = (site_stats[iter][prop][2] / total) * 100
						site_stats[iter][prop][3] = (site_stats[iter][prop][3] / total) * 100
						site_stats[iter][prop][4] = site_stats[iter][prop][0] - ((site_stats[iter][prop][4] / total) * 100)
						site_stats[iter][prop][5] = ((site_stats[iter][prop][5] / total) * 100) - site_stats[iter][prop][0]
						if site_stats[iter][prop][4] < 0:
							site_stats[iter][prop][4] = 0
						if site_stats[iter][prop][5] < 0:
							site_stats[iter][prop][5] = 0

					# ---------------------------------

					for prop in range(LANG_SIZE):

						site_beliefs = np.array(site_belief_values)
						# for truth in range(0, 3):
						site_belief_stats[iter][prop][0] = np.average(site_beliefs[prop])
						site_belief_stats[iter][prop][1] = np.min(site_beliefs[prop])
						site_belief_stats[iter][prop][2] = np.max(site_beliefs[prop])
						site_belief_stats[iter][prop][3] = np.std(site_beliefs[prop])
						site_belief_stats[iter][prop][4] = site_belief_stats[iter][prop][0] - np.percentile(site_beliefs[prop], PERC_LOWER)
						site_belief_stats[iter][prop][5] = np.percentile(site_beliefs[prop], PERC_UPPER) - site_belief_stats[iter][prop][0]
						if site_belief_stats[iter][prop][4] < 0:
							site_belief_stats[iter][prop][4] = 0
						if site_belief_stats[iter][prop][5] < 0:
							site_belief_stats[iter][prop][5] = 0

				collective_payoff_stats[r][cmp_i].append(payoff_stats)
				collective_site_stats[r][cmp_i].append(site_stats)
				collective_unique_site_stats[r][cmp_i].append(unique_site_stats)
				collective_message_stats[r][cmp_i].append(message_stats)
				collective_unique_belief_stats[r][cmp_i].append(unique_belief_stats)
				collective_site_belief_stats[r][cmp_i].append(site_belief_stats)

				cmp_vi += 1

	for files in GRAPH_FILES:

		for r, radius in enumerate(RADII):

			# GRAPHING TRAJECTORIES
			x_steps = [x for x in range(iterations)]
			cmap = cm.get_cmap(colour_map)
			legends = []
			c = [ cmap(float(x) / (20.0)) for x in range(0, 20) ]
			font = { 'size' : font_size }

			plt.figure(0, figsize = figure_size)

			add_legend = True

			skip_graphs = False

			FIG_TITLE = ""

			c_i = 0
			line_i = 0

			for cmp_i, cmp_dir in enumerate(RESULTS_DIRS):

				for cmp_vi in range(len(collective_payoff_stats[r][cmp_i])):

					if "message" in files[0]:
						skip_graphs = True
						continue

					c_i = 0
					# line_i = 0

					for file in files:

						results = list()
						std_dev = list()
						percs 	= list()
						if file == "mean_payoff":
							results = [x[0] for x in collective_payoff_stats[r][cmp_i][cmp_vi]]
							std_dev = [x[3] for x in collective_payoff_stats[r][cmp_i][cmp_vi]]
							percs.append([x[4] for x in collective_payoff_stats[r][cmp_i][cmp_vi]])
							percs.append([x[5] for x in collective_payoff_stats[r][cmp_i][cmp_vi]])
						elif file == "unique_sites":
							results = [x[0] for x in collective_unique_site_stats[r][cmp_i][cmp_vi]]
							std_dev = [x[3] for x in collective_unique_site_stats[r][cmp_i][cmp_vi]]
							percs.append([x[4] for x in collective_unique_site_stats[r][cmp_i][cmp_vi]])
							percs.append([x[5] for x in collective_unique_site_stats[r][cmp_i][cmp_vi]])
						elif file == "unique_beliefs":
							results = [x[0] for x in collective_unique_belief_stats[r][cmp_i][cmp_vi]]
							std_dev = [x[3] for x in collective_unique_belief_stats[r][cmp_i][cmp_vi]]
							percs.append([x[4] for x in collective_unique_belief_stats[r][cmp_i][cmp_vi]])
							percs.append([x[5] for x in collective_unique_belief_stats[r][cmp_i][cmp_vi]])

						if file == "dancing_for_site":
							c_i = 0
							for i in range(LANG_SIZE + 1):
								# z = iteration -> y = test -> x = nest_index
								results = [x[i][0] for x in collective_site_stats[r][cmp_i][cmp_vi]]
								std_dev = [x[i][3] for x in collective_site_stats[r][cmp_i][cmp_vi]]
								percs 	= list()
								percs.append([x[i][4] for x in collective_site_stats[r][cmp_i][cmp_vi]])
								percs.append([x[i][5] for x in collective_site_stats[r][cmp_i][cmp_vi]])
								# plt.errorbar(x_steps[::30] + [x_steps[-1]], results[::30] + [results[-1]],\
								# 	std_dev[::30] + [std_dev[-1]], 0, color = c[c_i], linewidth=line_width, linestyle=linestyles[line_i])
								plt.errorbar(x_steps[::30] + [x_steps[-1]], results[::30] + [results[-1]],\
									yerr=np.vstack([percs[0][::30] + [percs[0][-1]], percs[1][::30] + [percs[1][-1]]]),\
									color = c[c_i], linewidth=line_width, linestyle=linestyles[line_i])
								c_i += line_colour_jump
								# line_i += 1
						elif file == "site_belief_values":
							c_i = 0
							for i in range(1):
								# z = iteration -> y = test -> x = nest_index
								results = [x[i][0] for x in collective_site_belief_stats[r][cmp_i][cmp_vi]]
								std_dev = [x[i][3] for x in collective_site_belief_stats[r][cmp_i][cmp_vi]]
								percs 	= list()
								percs.append([x[i][4] for x in collective_site_belief_stats[r][cmp_i][cmp_vi]])
								percs.append([x[i][5] for x in collective_site_belief_stats[r][cmp_i][cmp_vi]])
								# plt.errorbar(x_steps[::30] + [x_steps[-1]], results[::30] + [results[-1]],\
								# 	std_dev[::30] + [std_dev[-1]], 0, color = c[c_i], linewidth=line_width, linestyle=linestyles[line_i])
								plt.errorbar(x_steps[::30] + [x_steps[-1]], results[::30] + [results[-1]],\
									yerr=np.vstack([percs[0][::30] + [percs[0][-1]], percs[1][::30] + [percs[1][-1]]]),\
									color = c[c_i], linewidth=line_width, linestyle=linestyles[line_i])
								c_i += line_colour_jump
								# line_i += 1
						else:
							# plt.errorbar(x_steps[::30] + [x_steps[-1]], results[::30] + [results[-1]],\
							# 	std_dev[::30] + [std_dev[-1]], 0, color = c[c_i], linewidth=line_width, linestyle=linestyles[line_i])
							plt.errorbar(x_steps[::30] + [x_steps[-1]], results[::30] + [results[-1]],\
								yerr=np.vstack([percs[0][::30] + [percs[0][-1]], percs[1][::30] + [percs[1][-1]]]),\
								color = c[c_i], linewidth=line_width, linestyle=linestyles[line_i])


						# line_i += 1
						c_i += line_colour_jump

					# line_i = 0
					c_i = 0

					FIG_TITLE = ""
					plt.rc('font', **font)
					plt.grid(True)
					x1,x2,y1,y2 = plt.axis()
					plt.axis((x1,x2,0,y2))

					if GRAPH_FILES.index(files) == 0:
						if add_legend == True:
							temp_legend = [" ".join(x.split("_")).capitalize() for x in files]
							c_i = 0
							for value in temp_legend:
								legends.append(mlines.Line2D([], [], color=c[c_i], linewidth=line_width, linestyle=linestyles[line_i], label=value))
								c_i += line_colour_jump
								# line_i += 1

						# plt.annotate("Site Values: [ " + ", ".join([str(x) for x in SITE_PAYOFF]) + " ]",
							# (.85, .10), xycoords='axes fraction', backgroundcolor='w')
						# plt.annotate("Msg. Received: {0:.2f}".format(np.average([x[0] for x in collective_message_stats[r][cmp_i][cmp_vi]])),
							# (.85, .05), xycoords='axes fraction', backgroundcolor='w')
						FIG_TITLE = "kilobot_stats"

					elif GRAPH_FILES.index(files) == 1:
						# temp_legend = ["Choice " + str(SITE_PAYOFF.index(x)) + ": " + str(x) for x in SITE_PAYOFF]\
						# 	+ ["Updating"]
						# for value in temp_legend:
						# 	legends.append(CMP_VALUES[cmp_vi][1:] + value)

						if add_legend == True:
							temp_legend = [SITE_NAMES[x] for x in range(LANG_SIZE)]\
								+ ["Updating"]
							c_i = 0
							for value in temp_legend:
								legends.append(mlines.Line2D([], [], color=c[c_i], linewidth=line_width, linestyle=linestyles[line_i], label=value))
								c_i += line_colour_jump
								# line_i += 1

						# temp_y_label = " ".join(file.split("_")).capitalize()
						plt.ylabel("% of Kilobots signalling")
						FIG_TITLE = files[0]

					elif GRAPH_FILES.index(files) == 2:
						plt.ylabel(" ".join(file.split("_")).capitalize())
						FIG_TITLE = files[0]
						plt.axis((x1,x2,0,3.5));

					elif GRAPH_FILES.index(files) == 3:
						# temp_legend = ["Site: " + str(x) for x in range(LANG_SIZE)]
						# for value in temp_legend:
						# 	legends.append(CMP_VALUES[cmp_vi][1:] + value)

						if add_legend == True:
							temp_legend = []
							c_i = 0
							for value in temp_legend:
								legends.append(mlines.Line2D([], [], color=c[c_i], linewidth=line_width, linestyle=linestyles[line_i], label=value))
								c_i += line_colour_jump
								# line_i += 1

						plt.ylabel("Average beliefs")
						FIG_TITLE = files[0]
						#plt.axis((x1,x2,0,1));

					# legends.append(mlines.Line2D([], [], color=c[c_i], linewidth=line_width, label=CMP_RATES[r][cmp_i][cmp_vi]))

					add_legend = False
					# c_i += line_colour_jump

				if add_model:
					legends.append(mlines.Line2D([], [], color='black', linewidth=line_width, linestyle=linestyles[line_i], label=CMP_FILES[cmp_i].capitalize()))
				# c_i += line_colour_jump
				line_i += 1

			if skip_graphs == True:
				continue

			plt.xlabel('Iterations')
			#plt.ylabel(" ".join(file.split("_")).capitalize())

			plt.legend(handles=legends, loc='best', prop={'size':font_size})

			plt.savefig("results" + os.sep + "trajectories" + os.sep + FIG_TITLE + "_" + str(radius) + GRAPHS_EXT, bbox_inches='tight')
			plt.clf()

	for files in GRAPH_FILES:

		# GRAPHING STEADY STATES FOR RADII
		radii = [int(float(x) * 100) for x in RADII]
		cmap = cm.get_cmap(colour_map)
		legends = []
		c = [ cmap(float(x) / (20.0)) for x in range(0, 20) ]
		font = { 'size' : font_size }

		plt.figure(0, figsize = figure_size)

		c_i = 0
		line_i = 0
		add_legend = True

		for cmp_i, cmp_dir in enumerate(RESULTS_DIRS):

			for cmp_vi in range(len(collective_payoff_stats[r][cmp_i])):

				c_i = 0
				# line_i = 0

				for file in files:

					results = list()
					std_dev = list()
					percs 	= list()
					if file == "mean_payoff":
						steady_payoff_stats = []
						for r, radius in enumerate(RADII):
							steady_payoff_stats.append(collective_payoff_stats[r][cmp_i][cmp_vi][-1])
						results = [x[0] for x in steady_payoff_stats]
						std_dev = [x[3] for x in steady_payoff_stats]
						percs.append([x[4] for x in steady_payoff_stats])
						percs.append([x[5] for x in steady_payoff_stats])
					elif file == "unique_sites":
						steady_unique_site_stats = []
						for r, radius in enumerate(RADII):
							steady_unique_site_stats.append(collective_unique_site_stats[r][cmp_i][cmp_vi][-1])
						results = [x[0] for x in steady_unique_site_stats]
						std_dev = [x[3] for x in steady_unique_site_stats]
						percs.append([x[4] for x in steady_unique_site_stats])
						percs.append([x[5] for x in steady_unique_site_stats])
					elif file == "unique_beliefs":
						steady_unique_belief_stats = []
						for r, radius in enumerate(RADII):
							steady_unique_belief_stats.append(collective_unique_belief_stats[r][cmp_i][cmp_vi][-1])
						results = [x[0] for x in steady_unique_belief_stats]
						std_dev = [x[3] for x in steady_unique_belief_stats]
						percs.append([x[4] for x in steady_unique_belief_stats])
						percs.append([x[5] for x in steady_unique_belief_stats])
					elif file == "message_counts":
						steady_message_stats = []
						for r, radius in enumerate(RADII):
							steady_message_stats.append(collective_message_stats[r][cmp_i][cmp_vi][-1])
						results = [x[0] for x in steady_message_stats]
						std_dev = [x[3] for x in steady_message_stats]
						percs.append([x[4] for x in steady_message_stats])
						percs.append([x[5] for x in steady_message_stats])

					if file == "dancing_for_site":
						c_i = 0
						steady_site_stats = []
						for r, radius in enumerate(RADII):
							steady_site_stats.append(collective_site_stats[r][cmp_i][cmp_vi][-1])
						print("Site stats:")
						for i in range(LANG_SIZE + 1):
							# z = iteration -> y = test -> x = nest_index
							results = [x[i][0] for x in steady_site_stats]
							print(results[5])
							std_dev = [x[i][3] for x in steady_site_stats]
							percs 	= list()
							percs.append([x[i][4] for x in steady_site_stats])
							percs.append([x[i][5] for x in steady_site_stats])
							#plt.errorbar(radii, results, std_dev, 0, color = c[c_i], linewidth=line_width, linestyle=linestyles[line_i])
							plt.errorbar(radii, results, yerr=np.vstack([percs[0], percs[1]]),\
								color = c[c_i], linewidth=line_width, linestyle=linestyles[line_i])
							c_i += line_colour_jump
							# line_i += 1
					elif file == "site_belief_values":
						c_i = 0
						steady_site_belief_stats = []
						for r, radius in enumerate(RADII):
							steady_site_belief_stats.append(collective_site_belief_stats[r][cmp_i][cmp_vi][-1])
						print("Belief state:")
						for i in range(1):
							# z = iteration -> y = test -> x = nest_indexresults = [x[i][0] for x in steady_site_stats]
							results = [x[i][0] for x in steady_site_belief_stats]
							print(results[5])
							std_dev = [x[i][3] for x in steady_site_belief_stats]
							print(std_dev[5])
							percs 	= list()
							percs.append([x[i][4] for x in steady_site_belief_stats])
							percs.append([x[i][5] for x in steady_site_belief_stats])
							#plt.errorbar(radii, results, std_dev, 0, color = c[c_i], linewidth=line_width, linestyle=linestyles[line_i])
							plt.errorbar(radii, results, yerr=np.vstack([percs[0], percs[1]]),\
								color = c[c_i], linewidth=line_width, linestyle=linestyles[line_i])
							c_i += line_colour_jump
							# line_i += 1
					else:
						#plt.errorbar(radii, results, std_dev, 0, color = c[c_i], linewidth=line_width, linestyle=linestyles[line_i])
						plt.errorbar(radii, results, yerr=np.vstack([percs[0], percs[1]]), color = c[c_i], linewidth=line_width, linestyle=linestyles[line_i])
						# c_i += line_colour_jump

					c_i += line_colour_jump
					# line_i += 1

				c_i = 0
				# line_i = 0

				FIG_TITLE = ""
				plt.rc('font', **font)
				plt.grid(True)
				x1,x2,y1,y2 = plt.axis()
				plt.axis((x1,x2,0,y2));

				if GRAPH_FILES.index(files) == 0:
					if add_legend == True:
						temp_legend = [" ".join(x.split("_")).capitalize() for x in files]
						c_i = 0
						for value in temp_legend:
							legends.append(mlines.Line2D([], [], color=c[c_i], linewidth=line_width, linestyle=linestyles[line_i], label=value))
							c_i += line_colour_jump
							# line_i += 1

					# plt.annotate("Site Values: [ " + ", ".join([str(x) for x in SITE_PAYOFF]) + " ]",
						# (.85, .05), xycoords='axes fraction', backgroundcolor='w')
					FIG_TITLE = "kilobot_stats"
				elif GRAPH_FILES.index(files) == 1:
					if add_legend == True:
						# temp_legend = ["Choice " + str(SITE_NAMES[SITE_PAYOFF.index(x)]) + ": " + str(x) for x in SITE_PAYOFF]\
						temp_legend = [SITE_NAMES[x] for x in range(LANG_SIZE)]\
						+ ["Updating"]
						c_i = 0
						for value in temp_legend:
							legends.append(mlines.Line2D([], [], color=c[c_i], linewidth=line_width, linestyle=linestyles[line_i], label=value))
							c_i += line_colour_jump
							# line_i += 1
					# temp_y_label = " ".join(file.split("_")).capitalize()
					plt.ylabel("% of Kilobots signalling")
					FIG_TITLE = file
				elif GRAPH_FILES.index(files) == 2:
					plt.ylabel(" ".join(file.split("_")).capitalize())
					FIG_TITLE = file
					plt.axis((x1,x2,0,3.5));
				elif GRAPH_FILES.index(files) == 3:
					if add_legend == True:
						temp_legend = []
						c_i = 0
						for value in temp_legend:
							legends.append(mlines.Line2D([], [], color=c[c_i], linewidth=line_width, linestyle=linestyles[line_i], label=value))
							c_i += line_colour_jump
							# line_i += 1
					plt.ylabel("Average beliefs")
					FIG_TITLE = file
					#plt.axis((x1,x2,0,1));
				elif GRAPH_FILES.index(files) == 4:
					# if add_legend == True:
					# 	temp_legend = ["Message Counts"]
					# 	# c_i = 0
					# 	for value in temp_legend:
					# 		legends.append(mlines.Line2D([], [], color='black', linewidth=line_width, linestyle=linestyles[line_i], label=value))
					# 		# c_i += line_colour_jump
					# 		line_i += 1
					# plt.ylabel(" ".join(file.split("_")).capitalize())

					plt.ylabel("Average no. of\nmessages received")
					FIG_TITLE = file

				# legends.append(mlines.Line2D([], [], color=c[c_i], linewidth=line_width, label=CMP_RATES[r][cmp_i][cmp_vi]))

				add_legend = False
				# c_i += line_colour_jump

			if add_model:
				legends.append(mlines.Line2D([], [], color='black', linewidth=line_width, linestyle=linestyles[line_i], label=CMP_FILES[cmp_i].capitalize()))
			# c_i += line_colour_jump
			line_i += 1

		plt.xlabel(r'Radius $r$')
		plt.xlim([0,float(radii[-1])])

		plt.legend(handles=legends, loc='best', ncol=2, prop={'size':font_size}) #  ncol=2,

		plt.savefig("results" + os.sep + "steadystates" + os.sep + FIG_TITLE + GRAPHS_EXT, bbox_inches='tight')
		plt.clf()


if __name__ == "__main__":
	main(sys.argv)
