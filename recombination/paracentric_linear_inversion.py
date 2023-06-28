from recombination import *

investigation = Investigation("A[BCDE]@", model="free", extra_pathway= False, interference_parameters = 1, interference_bounds = (0,20), alpha = 0, beta = 0, include_breakpoint_loci = False, include_patterns_in_report= True, output_file="sturtevant_sc8_unbalanced_results.txt", interference_direction = "original", linear_meiosis= True, error2 = 1e-6)
investigation.read_input("datasets/sturtevant_sc8_unbalanced.txt")
investigation.run_nelder_mead(repeat = 30, verbose = True, report_frequency = 100)