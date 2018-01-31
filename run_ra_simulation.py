import gc
import os
import random
import sys
import subprocess

import numpy as np

# Simulator variables
DIRECTORY		= "../kilobox_build/Testbed/"
FILE_NAME		= "Testbed"
WORLD_DIR		= "../kilobox/worlds/"
WORLD_DIR_		= "../kilobox/worlds/generated/"
WORLD_FILE		= "test_forage.world"
ARGS			= ["--gui", "--seed="]

TESTS			= 50

# Kilobot vars
KILOBOTS 		= 400
KILOBOT_BASE 	= ["r0( pose [", "", "", "0", "", "])"]
KILO_SPACING	= 6

BASE_SEED 		= 8518
random.seed(BASE_SEED)
SEEDS			= sorted([random.randint(0, 10000) for x in range(TESTS)])
SEEDS			= [str(x) for x in SEEDS]

FILE_NUM		= [x for x in range(len(SEEDS))]
SITE_PAYOFF 	= [7, 9]# [5, 12, 19, 26, 33]
STR_SITE_PAYOFF = [str(x) for x in SITE_PAYOFF]
LANG_SIZE		= len(SITE_PAYOFF)

# Radius variables
RADIUS_FILE		= "test_swarm.world"
RADII			= ["{0:.2f}".format(x/100) for x in range(0, 21, 2)]
# RADII			+= ["{0:.2f}".format(x/100) for x in range(30, 71, 20)]
print(RADII)
print(SITE_PAYOFF)

# Movement variables

# Results variables
BASE_DIR		= "results"
RESULTS_DIR		= "test_results"

SEEDS_FILE		= "test_seeds"
PAYOFF_FILE		= "site_payoffs"
RESULTS_FILE	= "simulator_results"
RESULTS_EXT		= ".csv"


def main(args):
	# Command-line arguments
	global FILE_NAME
	global RESULTS_DIR

	dry_run		= False
	boolean		= False
	three_valued = False
	majority_rule = False
	bool_append = ""
	bool_dir 	= ""
	for arg in sys.argv:
		if "--p-value" in arg:
			p_value = int(float(arg.split('=')[1]) * 100)
			bool_append = "_bool_" + str(p_value)
			bool_dir = "boolean_" + str(p_value) + os.sep
		if "--three-valued" in arg:
			FILE_NAME += "_three_valued"
			RESULTS_DIR	+= "/three_valued"
			three_valued = True
		if "--mal-three-valued" in arg:
			FILE_NAME += "_three_valued_mal"
			RESULTS_DIR	+= "/three_valued_mal"
			three_valued = True
		if "--static-mal-three-valued" in arg:
			FILE_NAME += "_three_valued_mal_static"
			RESULTS_DIR	+= "/three_valued_mal_static"
			three_valued = True
		if "--noisy-three-valued" in arg:
			FILE_NAME += "_three_valued_noisy"
			RESULTS_DIR	+= "/three_valued_noisy"
			three_valued = True
		if "--site-drop-three-valued" in arg:
			FILE_NAME += "_three_valued_site_drop"
			RESULTS_DIR	+= "/three_valued_site_drop"
			three_valued = True
		if "--fast-three-valued" in arg:
			FILE_NAME += "_three_valued_fast"
			RESULTS_DIR	+= "/three_valued_fast"
			three_valued = True
		if "--majority-rule" in arg:
			FILE_NAME += "_majority_rule"
			RESULTS_DIR	+= "/majority_rule"
			majority_rule = True
		if "--boolean-uncertainty" in arg:
			FILE_NAME += "_boolean_uncertainty"
			RESULTS_DIR	+= "/boolean_uncertainty"
		if "--boolean-averaged" in arg:
			FILE_NAME += "_boolean_averaged"
			RESULTS_DIR	+= "/boolean_averaged"
		if "--boolean-adopt" in arg:
			FILE_NAME += "_boolean_adopt"
			RESULTS_DIR	+= "/boolean_adopt"
		if "--mal-boolean-adopt" in arg:
			FILE_NAME += "_boolean_adopt_mal"
			RESULTS_DIR	+= "/boolean_adopt_mal"
		if "--static-mal-boolean-adopt" in arg:
			FILE_NAME += "_boolean_adopt_mal_static"
			RESULTS_DIR	+= "/boolean_adopt_mal_static"
		if "--noisy-boolean-adopt" in arg:
			FILE_NAME += "_boolean_adopt_noisy"
			RESULTS_DIR	+= "/boolean_adopt_noisy"
		if "--site-drop-boolean-adopt" in arg:
			FILE_NAME += "_boolean_adopt_site_drop"
			RESULTS_DIR	+= "/boolean_adopt_site_drop"
		if "--boolean-swap" in arg:
			FILE_NAME += "_boolean_50_50_swap"
			RESULTS_DIR	+= "/boolean_50_50_swap"
		if "--boolean-pref" in arg:
			FILE_NAME += "_boolean_pref"
			RESULTS_DIR	+= "/boolean_pref"
		if "--boolean-borda-pref" in arg:
			FILE_NAME += "_boolean_pref_borda"
			RESULTS_DIR	+= "/boolean_pref_borda"

		if "--motion" in arg:
			vars = arg.split("=")[-1]
			RESULTS_DIR	+= "/" + vars

		if "--vars" in arg:
			vars = arg.split("=")[-1]
			FILE_NAME += "_" + vars
			RESULTS_DIR	+= "_" + vars


	print(sys.argv)

	temp_output_files = list()
	temp_world_files = list()


	with open(BASE_DIR + os.sep + RESULTS_DIR + os.sep + bool_dir + SEEDS_FILE + RESULTS_EXT, "w") as outfile:
		outfile.write(','.join([str(BASE_SEED)] + SEEDS))

	with open(BASE_DIR + os.sep + RESULTS_DIR + os.sep + bool_dir + PAYOFF_FILE + RESULTS_EXT, "w") as outfile:
		outfile.write(','.join(STR_SITE_PAYOFF))

	for radius in RADII:

		belief_results				= list()
		unique_belief_results		= list()
		# nest_site_results 			= list()
		nest_site_counts_results	= list()
		message_results 			= list()

		temp_rad_file = RADIUS_FILE.split(".")
		temp_rad_file.insert(-1, ".")

		for test in range(len(FILE_NUM)):

			# Set the seed to match this test seed
			random.seed(int(SEEDS[test]))

			# Generate the kilobots in random positions based on the seed.
			kilobots = []
			# 0.6 for 400 Kilobots in 1.2m^2 arena.
			# 0.85 for 800 Kilobots in 1.8m^2 arena.
			default_pos = 0.6
			xstart = -default_pos + 0.03
			ystart = -default_pos + 0.03
			xpos = xstart * 100
			ypos = ystart * 100
			xstop = default_pos
			ystop = default_pos
			for index in range(KILOBOTS):
				kilobot = KILOBOT_BASE
				# kilobot[1] = "{0:.6f}".format(random.uniform(-1.4, 1.4))
				# kilobot[2] = "{0:.6f}".format(random.uniform(-0.9, 0.9))
				# kilobot[4] = "{0:.6f}".format(random.uniform(-180.0, 180.0))
				kilobot[1] = "{0:.6f}".format(xpos/100)
				kilobot[2] = "{0:.6f}".format(ypos/100)
				kilobot[4] = "{0:.6f}".format(-90.0)
				kilobots.append(" ".join(kilobot))

				xpos += KILO_SPACING
				if xpos/100 >= xstop:
					xpos = xstart * 100
					ypos += KILO_SPACING

			temp_rad_file_str = "".join(temp_rad_file)
			radfile_beginning = ""

			with open(WORLD_DIR + temp_rad_file_str, "r") as radfile:
				for line in radfile:
					radfile_beginning += line

			temp_rad_file_str = list(temp_rad_file)
			temp_rad_file_str.insert(-2, "_" + str(radius) + "_" + SEEDS[test])
			temp_rad_file_str = "".join(temp_rad_file_str)

			radfile_beginning = radfile_beginning.split(" ")
			value_index = radfile_beginning.index("$VALUE$")
			radfile_beginning[value_index] = radius

			with open(WORLD_DIR_ + temp_rad_file_str, "w") as radfile:
				radfile.write(" ".join(radfile_beginning))
				for bot in kilobots:
					radfile.write(bot + '\n')

			temp_world_files.append(WORLD_DIR_ + temp_rad_file_str)

			mode = "w"# if SEED == SEEDS[0] else "a"
			#temp_output_files.append(BASE_DIR + os.sep + RESULTS_DIR + os.sep + bool_dir + RESULTS_FILE + "_RADII_" + str(radius) + "_" + SEEDS[test] + RESULTS_EXT)

			print("Running file: ", end="")
			print(DIRECTORY + FILE_NAME + bool_append + " " + " ".join(ARGS) + SEEDS[test] + " " + WORLD_DIR_ + temp_rad_file_str)
			cmd = [DIRECTORY + FILE_NAME + bool_append, ARGS[0], ARGS[1] + SEEDS[test], WORLD_DIR_ + temp_rad_file_str]

			#with open(temp_output_files[-1], mode) as outfile:
			proc = subprocess.Popen(cmd, bufsize = 1, stdout=subprocess.PIPE)

			belief_tests = list()
			nest_site_tests = list()
			message_tests = list()

			for line in iter(proc.stdout.readline, ''):

				if not line:
					break
				line = str(line.rstrip(), 'utf-8')

				if line[0] == "+":
					line_elements = [x for x in line.split(":")]
					for elem in range(len(line_elements)):
						if elem == 0:
							continue
						elif elem == 4:
							if three_valued == True:
								line_elements[elem] = [float(x) for x in line_elements[elem].split(";")]
								for index, item in enumerate(line_elements[elem]):
									if item == 2:
										line_elements[elem][index] = 1
									elif item == 1:
										line_elements[elem][index] = 0.5
							else:
								line_elements[elem] = [float(x) for x in line_elements[elem].split(";")]
						else:
							line_elements[elem] = int(line_elements[elem])

					if line_elements[1] == len(belief_tests):
						belief_tests.append(list())
						nest_site_tests.append([0 for x in range(len(SITE_PAYOFF) + 1)])
						message_tests.append(list())


					if line_elements[1] >= len(belief_results):
						belief_results.append(list())
						unique_belief_results.append(list())
						# nest_site_results.append(list())
						nest_site_counts_results.append(list())
						message_results.append(list())

					if test == len(nest_site_counts_results[line_elements[1]]):
						# belief_tests[line_elements[1]].append(list())
						# nest_site_results[line_elements[1]].append(list())
						nest_site_counts_results[line_elements[1]].append([0 for x in range(len(SITE_PAYOFF) + 1)])
						# message_tests[line_elements[1]].append(list())

					belief_tests[line_elements[1]].append(line_elements[4])
					if line_elements[2] == 0:
						# nest_site_results[line_elements[1]][test].append(-1)
						nest_site_counts_results[line_elements[1]][test][-1] += 1
					else:
						# nest_site_results[line_elements[1]][test].append(line_elements[3])
						nest_site_counts_results[line_elements[1]][test][line_elements[3]] += 1
					message_tests[line_elements[1]].append(line_elements[5])

			for index in range(len(belief_tests)):

				belief_results[index].append([np.average([belief[l] for belief in belief_tests[index]]) for l in range(LANG_SIZE)])

				unique_beliefs = list()
				for belief in belief_tests[index]:
					if belief not in unique_beliefs:
						unique_beliefs.append(belief)
				unique_belief_results[index].append(len(unique_beliefs))

				message_results[index].append(np.average(message_tests[index]))

			gc.collect()

		with open(BASE_DIR + os.sep + RESULTS_DIR + os.sep + bool_dir + "average_beliefs" + "_radius_" + str(radius) + RESULTS_EXT, 'w') as outfile:

			for beliefs in belief_results:
				extra_comma = False
				for belief in beliefs:
					if not extra_comma:
						extra_comma = True
					else:
						outfile.write(',')
					semi_colon = False
					for l in range(LANG_SIZE):
						if not semi_colon:
							semi_colon = True
						else:
							outfile.write(';')
						outfile.write(str(belief[l]))
				outfile.write('\n')

		with open(BASE_DIR + os.sep + RESULTS_DIR + os.sep + bool_dir + "unique_beliefs" + "_radius_" + str(radius) + RESULTS_EXT, 'w') as outfile:

			for beliefs in unique_belief_results:
				extra_comma = False
				for belief in beliefs:
					if not extra_comma:
						extra_comma = True
					else:
						outfile.write(',')
					outfile.write(str(belief))
				outfile.write('\n')

		with open(BASE_DIR + os.sep + RESULTS_DIR + os.sep + bool_dir + "nest_sites" + "_radius_" + str(radius) + RESULTS_EXT, 'w') as outfile:

			for sites in nest_site_counts_results:
				extra_comma = False
				for site in sites:
					if not extra_comma:
						extra_comma = True
					else:
						outfile.write(',')
					semi_colon = False
					for l in range(LANG_SIZE + 1):
						if not semi_colon:
							semi_colon = True
						else:
							outfile.write(';')
						outfile.write(str(site[l]))
				outfile.write('\n')

				# 	if not extra_comma:
				# 		extra_comma = True
				# 	else:
				# 		outfile.write(',')
				# 	outfile.write(str([s for s in site if s != -1]))
				# outfile.write('\n')

		# with open(BASE_DIR + os.sep + RESULTS_DIR + os.sep + bool_dir + "average_payoff" + "_radius_" + str(radius) + RESULTS_EXT, 'w') as outfile:

		# 	for i in range(len(belief_results)):
		# 		extra_comma = False
		# 		for j in range(len(belief_results[i])):
		# 			if not extra_comma:
		# 				extra_comma = True
		# 			else:
		# 				outfile.write(',')
		# 			payoff = list()
		# 			for k in range(len(belief_results[i][j])):
		# 				if nest_site_results[i][j][k] == -1:
		# 					continue
		# 				temp_payoff = 0.0
		# 				if three_valued == True:
		# 					for l in range(LANG_SIZE):
		# 						temp_payoff += belief_results[i][j][k][l] * SITE_PAYOFF[l]
		# 				else:
		# 					for l in range(LANG_SIZE - 1):
		# 						temp_payoff += belief_results[i][j][k][l] * SITE_PAYOFF[l]
		# 					temp_payoff += (1.0 - sum(belief_results[i][j][k])) * SITE_PAYOFF[-1]
		# 				payoff.append(temp_payoff)
		# 			outfile.write(str(np.average(payoff)))
		# 		outfile.write('\n')

		with open(BASE_DIR + os.sep + RESULTS_DIR + os.sep + bool_dir + "message_counts" + "_radius_" + str(radius) + RESULTS_EXT, 'w') as outfile:

			for messages in message_results:
				extra_comma = False
				for message in messages:
					if not extra_comma:
						extra_comma = True
					else:
						outfile.write(',')
					outfile.write(str(message))
				outfile.write('\n')

		#remove_files(temp_output_files)
		remove_files(temp_world_files)
		#temp_output_files = list()
		temp_world_files = list()

		gc.collect()

def remove_files(files_to_delete):
	for file in files_to_delete:
		try:
			os.remove(file)
		except FileNotFoundError:
			continue

if __name__ == "__main__":
	main(sys.argv)
