from recombination import *

investigation = Investigation("A[BC@]DEFGH", extra_pathway=False, interference_parameters=1, interference_bounds=(0,20), alpha = 0, beta = 0, include_breakpoint_loci = False, include_patterns_in_report = False, output_file="results/roberts165_results.txt", interference_direction="original", error1 = 1e-6)
investigation.read_input("datasets/roberts165.txt")
investigation.run_nelder_mead(repeat = 30, verbose = True, report_frequency=500)