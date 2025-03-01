from recombination import *

# Run Nelder-Nead and bootstrap with the gamma model on the homokaryotype dataset from Roberts 197
investigation = Investigation("ABC@DEFGH", interference_parameters = 1, interference_bounds = (0,12), beta = 0, extra_pathway= False, output_file= "roberts_control_results.txt", error1 = 1e-6, include_patterns_in_report=True)
investigation.read_input("datasets/roberts_control.txt")
investigation.run_nelder_mead(repeat = 100, verbose = True, report_frequency = 500)

boot = ParametricBootstrap(investigation.loci, investigation.all_patterns_order, investigation.all_best_patterns, investigation.n, "Control", investigation.parameter_estimates, investigation.evaluation, investigation.all_data, investigation.unbalanced_proportion, interference_bounds=investigation.interference_bounds, seed = investigation.seed, breakpoint_nonstationarity=investigation.breakpoint_nonstationarity, stationary_test = investigation.stationary_test, repeat=100, replications=100, different_breakpoint_interference = investigation.difference_breakpoint_interference)
boot.run()

# Run Nelder-Nead and bootstrap with the gamma model and breakpoint interference (H1 model) on the inversion 165 dataset from Roberts 197
investigation = Investigation("A[BC@]DEFGH", extra_pathway=False, interference_parameters=1, interference_bounds=(0,12), alpha = 0, beta = 0, include_breakpoint_loci = False, include_patterns_in_report = True, output_file="roberts165_results.txt", breakpoint_nonstationarity = True, different_breakpoint_interference=False, stationary_test = False, error1 = 1e-6, print_probabilities=True)
investigation.read_input("datasets/roberts165.txt")
investigation.run_nelder_mead(repeat = 100, verbose = True, report_frequency=500)

boot = ParametricBootstrap(investigation.loci, investigation.all_patterns_order, investigation.all_best_patterns, investigation.n, "165_h1", investigation.parameter_estimates, investigation.evaluation, investigation.all_data, investigation.unbalanced_proportion, breakpoint_nonstationarity=investigation.breakpoint_nonstationarity, stationary_test=investigation.stationary_test, interference_bounds=investigation.interference_bounds, seed = investigation.seed, repeat=100, replications=100, different_breakpoint_interference=investigation.difference_breakpoint_interference)
boot.run()


# Run Nelder-Mead with the gamma + mu model on the Copenhaver Arabidopsis chromosome 3
investigation = Investigation("ABCDEFGH", model="free", interference_parameters=1, interference_bounds=(0,20), extra_pathway= True, alpha = 0, beta = 0, include_breakpoint_loci = False, include_patterns_in_report= False, output_file="copenhaver3_results.txt", tetrad = True, error2 = 1e-6, min_x= 2)	
investigation.read_input("datasets/copenhaver3.txt")
investigation.run_nelder_mead(repeat = 100, verbose = True, report_frequency = 500)


# Run Nelder-Mead with the mix3 model on the Morgan Drosophila dataset
investigation = Investigation("ABCDEFGHI", model="free", interference_parameters = 3, interference_bounds = (0,20), include_patterns_in_report = False, extra_pathway= False, beta = 0, output_file = "morgan_results.txt", error2 = 1e-6, error1 = 1e-6)
investigation.read_input("datasets/morgan.txt")
investigation.run_nelder_mead(repeat = 100, verbose = True, report_frequency = 200)

