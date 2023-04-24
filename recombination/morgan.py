from recombination import *

# Analyze the Morgan Drosophila datasets with the gamma model
investigation = Investigation("ABCDEFGHI", model="free", interference_parameters = 1, interference_bounds = (0,20), include_patterns_in_report = False, extra_pathway= False, beta = 0, output_file = "results/morgan_results.txt", error2 = 1e-6, error1 = 1e-6)
investigation.read_input("datasets/morgan.txt")
investigation.run_nelder_mead(repeat = 30, verbose = True, report_frequency = 200)

# Analyze the Morgan Drosophila datasets with the mix3 model
investigation = Investigation("ABCDEFGHI", model="free", interference_parameters = 3, interference_bounds = (0,20), include_patterns_in_report = False, extra_pathway= False, beta = 0, output_file = "results/morgan_results.txt", error2 = 1e-6, error1 = 1e-6)
investigation.read_input("datasets/morgan.txt")
investigation.run_nelder_mead(repeat = 30, verbose = True, report_frequency = 200)
