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

radius = "0.10"

# Percentiles

PERC_LOWER = 10
PERC_UPPER = 90

# Results variables
RESULTS_DIR		= "results/test_results"
RESULTS_DIRS	= []
SEEDS_FILE		= "test_seeds"
PAYOFF_FILE		= "site_payoffs"
RESULTS_FILE	= "simulator_results"
RESULTS_EXT		= ".csv"

# Graphs variables
GRAPHS_DIR		= "results"
GRAPH_FILES		= [["belief_malfunction"]]
GRAPHS_EXT		= ".pdf"

figure_size = (18, 9) #(24, 10)
font_size = 16 # 18
line_width = 1.5
line_colour_jump = 3
colour_map = 'gray'

def main(args):
	# Command-line arguments
	plot_traj	= False
	plot_steady	= False

	global RESULTS_DIR
	global RESULTS_DIRS
	global radius

	CMP_FILES = []
	CMP_VALUES = [""]
	MAL_RATES = [0]

	results_files = ["average_beliefs"]

	iterations = 0
	tests = 0

	for arg in args:
		if "traj" in arg:
			plot_traj = True
		if "steady" in arg:
			plot_steady = True
		if "--three-valued" in arg:
			RESULTS_DIRS.append("three_valued")
			CMP_FILES.append("Three-valued")
		if "--mal-three-valued" in arg:
			RESULTS_DIRS.append("three_valued_mal")
			CMP_FILES.append("Three-valued")
		if "--static-mal-three-valued" in arg:
			RESULTS_DIRS.append("three_valued_mal_static")
			CMP_FILES.append("three-valued (static) malfunction")
		if "--noisy-three-valued" in arg:
			RESULTS_DIRS.append("three_valued_noisy")
			CMP_FILES.append("three-valued (noisy)")
		if "--site-drop-three-valued" in arg:
			RESULTS_DIRS.append("three_valued_site_drop")
			CMP_FILES.append("three-valued (site drop)")
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
			CMP_FILES.append("boolean")
		if "--mal-boolean-adopt" in arg:
			RESULTS_DIRS.append("boolean_adopt_mal")
			CMP_FILES.append("Weighted voter")
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
				MAL_RATES.append(int(value))

	print(MAL_RATES)

	linestyles = ['-', '--', '-.', ':']

	collective_site_belief_stats 	= list()

	for cmp_vi, cmp_val in enumerate(CMP_VALUES):

		collective_site_belief_stats.append(list())

		for cmp_i, cmp_dir in enumerate(RESULTS_DIRS):

			try:
				if MAL_RATES[cmp_vi] == 0:
					temp_dir = cmp_dir.split("_")[:-1]
					temp_dir = "_".join(temp_dir)
					cmp_dir = temp_dir
				print(RESULTS_DIR + os.sep + cmp_dir + cmp_val + os.sep + SEEDS_FILE + RESULTS_EXT)
				with open(RESULTS_DIR + os.sep + cmp_dir + cmp_val + os.sep + SEEDS_FILE + RESULTS_EXT, "r") as infile:
					for line in infile:
						file_seeds = line.split(',')
						random.seed(file_seeds[0])

						SEEDS 			= [x for x in file_seeds[1:]]
						TESTS 			= len(SEEDS)

			except FileNotFoundError:
				continue

			with open(RESULTS_DIR + os.sep + cmp_dir + cmp_val + os.sep + PAYOFF_FILE + RESULTS_EXT, "r") as infile:
				for line in infile:
					SITE_PAYOFF 	= list( map(int, line.split(',')) )
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

						for test, test_elements in enumerate(line.split(",")):
							if "average_b" in file:
								for element in reversed(test_elements.split(";")):
									belief_results[iteration][test].append(float(element))

						iteration += 1

			iterations 	= len(belief_results)
			tests 		= len(belief_results[-1])

			site_belief_stats = list()

			for iter in range(iterations):			#for number of iterations
													#[avg, min, max, std. dev.]
				site_belief_stats.append([[0, 0, 0, 0, 0, 0] for x in range(LANG_SIZE)])

			# Start calculating the results and inserting them into the appropriate lists
			for iter in range(iterations):

				site_belief_values = [list() for x in range(LANG_SIZE)]

				for test in range(tests):
					for l in range(LANG_SIZE):
						site_belief_values[l].append(belief_results[iter][test][l])

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

			collective_site_belief_stats[cmp_vi].append(site_belief_stats)

	for files in GRAPH_FILES:

		# GRAPHING STEADY STATES FOR RADII
		radii = [float(x) for x in RADII]
		cmap = cm.get_cmap(colour_map)
		legends = []
		c = [ cmap(float(x) / (20.0)) for x in range(0, 20) ]
		font = { 'size' : font_size }

		plt.figure(0, figsize = figure_size)

		FIG_TITLE = ""

		c_i = 0
		line_i = 0
		add_legend = True

		# for cmp_val in CMP_VALUES:

		for cmp_i, cmp_dir in enumerate(RESULTS_DIRS):

	# for cmp_i, cmp_dir in enumerate(RESULTS_DIRS):

	# 	for cmp_vi in range(len(collective_site_belief_stats[r][cmp_i])):

			c_i = 0
			# line_i = 0

			for file in files:

				results = list()
				std_dev = list()
				percs 	= list()

				c_i = 0
				steady_site_belief_stats = []
				for r, rate in enumerate(MAL_RATES):
					steady_site_belief_stats.append(collective_site_belief_stats[r][cmp_i][-1])
				print("Belief state:")
				for i in range(1):
					# z = iteration -> y = test -> x = nest_indexresults = [x[i][0] for x in steady_site_stats]
					results = [x[i][0] for x in steady_site_belief_stats]
					print(results)
					std_dev = [x[i][3] for x in steady_site_belief_stats]
					percs 	= list()
					percs.append([x[i][4] for x in steady_site_belief_stats])
					percs.append([x[i][5] for x in steady_site_belief_stats])
					#plt.errorbar(radii, results, std_dev, 0, color = c[c_i], linewidth=line_width, linestyle=linestyles[line_i])
					plt.errorbar(MAL_RATES, results, yerr=np.vstack([percs[0], percs[1]]),\
						color = c[c_i], linewidth=line_width, linestyle=linestyles[line_i])
					c_i += line_colour_jump
					# line_i += 1

				c_i += line_colour_jump
				# line_i += 1

				c_i = 0
				# line_i = 0

				FIG_TITLE = file
				plt.rc('font', **font)
				plt.grid(True)
				x1,x2,y1,y2 = plt.axis()
				plt.axis((x1,x2,0,y2));


				if add_legend == True:
					temp_legend = []
					c_i = 0
					for value in temp_legend:
						legends.append(mlines.Line2D([], [], color=c[c_i], linewidth = line_width, linestyle=linestyles[line_i], label=value))
						c_i += line_colour_jump
						# line_i += 1
				plt.ylabel("Average belief in best choice")
				#plt.axis((x1,x2,0,1));


				legends.append(mlines.Line2D([], [], color='black', linewidth = line_width, linestyle=linestyles[line_i], label=CMP_FILES[cmp_i]))
				# c_i += line_colour_jump
				line_i += 1

				add_legend = False

		plt.xlabel(r'Malfunction rate $\lambda$')
		plt.xlim([0,MAL_RATES[-1]])

		plt.legend(handles=legends, loc='best', prop={'size':font_size})

		plt.savefig("results" + os.sep + "steadystates" + os.sep + FIG_TITLE + GRAPHS_EXT, bbox_inches='tight')
		plt.clf()


if __name__ == "__main__":
	main(sys.argv)
