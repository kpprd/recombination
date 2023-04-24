from recombination import *

#Analyze the Copenhaver Arabidopsis chromosome 3 dataset using the gamma + mu model
investigation = Investigation("ABCDEFGH", model="free", interference_parameters=1, interference_bounds=(0,20), extra_pathway= True, alpha = 0, beta = 0, include_breakpoint_loci = False, include_patterns_in_report= False, output_file="results/copenhaver3_results.txt", tetrad = True, error2 = 1e-6, min_x= 2)	
investigation.read_input("copenhaver3.txt")
investigation.run_nelder_mead(repeat = 30, verbose = True, report_frequency = 500)

