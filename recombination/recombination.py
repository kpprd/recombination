#!/usr/local/bin/python3
from ast import Lambda, Param
from math import inf
from ftplib import all_errors
#from multiprocessing import parent_process
import re
from tabnanny import verbose
from tkinter import CURRENT
from tracemalloc import start
import numpy
import itertools
import scipy.misc
import scipy.special
import copy
from math import cos, gamma, lgamma, sin, exp, pi, factorial, log, log10, inf
import sys
import time
import random
import timeit
from scipy.optimize import minimize
from datetime import datetime
import numba
from numba import jit, njit
import matplotlib.pyplot as pt
from scipy.stats import gamma
from scipy.stats import chi2


def AIC(lnL, variables):
    return 2*variables -2*lnL

def likelihood_ratio(lnL1, lnL2):
	return -2*(lnL1-lnL2)

def calculate_p(test_statistic, df):
    return 1-chi2.cdf(test_statistic, df)

def test(lnL1, lnL2, delta_parameters):
	test_stat = likelihood_ratio(lnL1, lnL2)
	p = calculate_p(test_stat, delta_parameters)
	return p


@jit("float64(float64, int64)", nopython = True)
def poisson(l, x):
		p = exp(-l)
		for i in range(1, x+1):
			p = (p*l)/(i)
		return p


def recombination_rate(m, pi_vector,lambda_value, mu_value):
	f = 0
	for q in range(m+1):
		for c in range(q+1):
			f+= pi_vector[q]*(poisson(lambda_value, c))
	return 0.5*(1-exp(-mu_value)*f)


def generate_pi_vector(gamma_values):
		'''
		Generates the stationary distribution and stores it as self.pi_vector
		'''
		m = len(gamma_values) -1
		pi_vector = numpy.zeros(m+1, numpy.float64)
		denominator = 0.0
		for q in range(m+1):
			denominator += (q+1)*gamma_values[q]
		for i in range(m+1):
			nominator = 0.0
			for q in range(i, m+1):
				nominator += gamma_values[q]
			pi_vector[i] = nominator/denominator
		return pi_vector



@jit("f8[:,:](u8, u8, f8, f8, f8[:], f8[:,:])",nopython=True)
def numba_generate_D(m, x, lambda_value, mu_value, gamma_values, g_values):
	'''
	Generates and returns a single D(x) matrix as defined in the paper.
		
	Arguments:
	x: (int) the number of chiasma events
	lambda_value/mu_value/g_value: see class Chromosome
	'''
	matrix = numpy.zeros((m+1,m+1), numpy.float64)
	maxint = 9223372036854775807
	if lambda_value+mu_value != 0:
		p1 = mu_value/(lambda_value+mu_value)
		p2 = lambda_value/(lambda_value+mu_value)
	else:
		p1 = 0.0
		p2 = 0.0
	if x == 0:
		for i in range(m+1):
			for j in range(m+1):
				if i>=j:
					f = 1
					for v in range(1,i-j+1):
						f = f*v
					matrix[i][j] = ((exp(-lambda_value)*lambda_value**(i-j))/f)*exp(-mu_value) #poisson(lambda_value, i-j)*exp(-mu_value)
	else:
		for i in range(m+1):
			for j in range(m+1):
				s = 0.0
				for l in range(x):
					for n in range(x-l-1, (x-l-1)*(m+1)+1):
						for q in range(j, m+1):
							h = i+1+l+n+q-j
							to_add1 = g_values[n][x-l-1]*gamma_values[q]*exp(-(lambda_value+mu_value))*(p1**l)*(p2**(h-l))
							for v in range(h):
								to_add1 = to_add1 * (lambda_value + mu_value)
							for v2 in range(1, l+1):
								to_add1 = to_add1/v2
							for v3 in range(1,h-l+1):
								to_add1 = to_add1/v3
							s += to_add1
				if i>=j:
					to_add2 = (exp(-mu_value))*(mu_value**x)*(exp(-lambda_value)*lambda_value ** (i-j) )
					for v4 in range(1,x+1):
						to_add2 = to_add2/v4
					for v5 in range(1, i-j+1):
						to_add2 = to_add2/v5
					s += to_add2
				matrix[i][j] = s
		
	return matrix


class CoincidenceFigure:
	def __init__(self, title = ""):
		self.coincidence_plots = []
		self.title = title

class CoincidencePlot:
	def __init__(self, gamma_values, p, plot_marker = None):
		self.gamma_values = gamma_values
		self.p = p
		self.h = self.calculate_h(p, gamma_values)
		self.plot_marker = plot_marker

	def get_sum(self, gamma_values):
		gamma_sum = 0
		for q in range(len(gamma_values)):
			gamma_sum += (q+1)*gamma_values[q]
		return gamma_sum

	def calculate_h(self, p, gamma_values):
		if p == 1:
			return inf
		else:
			gamma_sum = self.get_sum(gamma_values)
			return (p/gamma_sum)/(1-p)

class Investigation:
	"""
	A class for performing maximum likelihood parameter estimations on recombination and tetrad data.
	Key attributes:
	model (string): Which parametrization to use. You can choose between the following:
		'free': runs the analysis with the gamma, mix3, mix7, mix15 or mix31 models, depending on the number given number of interference_parameters
		'negative': the gamma + gamma_0 model
		'map': estimates only the genetic lengths of the interval
	interference_parameters (int): The number of interference parameters (1, 3, 7, 15 or 31)
	extra_pathway (bool): Whether or not to include an additional non-interfering pathway (+ mu)
	tetrad (bool): True if the input data is for a tetrad
	linear_meiosis (bool): True  if the input data is for a heterokaryotype with linear meiosis

	"""
	def __init__(self, loci, model = "free", gamma_values = None, lambda_values = None, h = None, p = None, extra_pathway = False, start_at_gamma = 0, include_breakpoint_loci = False, output_file = None, alpha = 0, beta = 1, linear_meiosis = False, include_centromere = False, include_patterns_in_report = True, d_values = None, interference_direction = "original", tetrad = False, distances = False, error1 = 10e-12, error2 = 10e-11, error3 = 1e-12, min_x = 2, max_x = 20, benchmark = 0, plot_span = 1, plot_resolution = 1000, plot_name = None, plot_marker = None, figure_name = None, figure_header = None, coincidence4_interval_size = 0.001, interference_bounds = 10, interference_parameters = 1, closed_form = False, seed = None):
		self.chromosome = None
		self.error1 = error1
		self.error2 = error2
		self.error3 = error3
		self.report_frequency = 100
		self.repetition = 0
		self.interference_parameters_n = interference_parameters
		if self.interference_parameters_n not in [0,1,3,7,15,31]:
			print("Error! Input 'interference_parameters' must be either 0, 1, 3, 7, 15 or 31")
			sys.exit(1)
		if seed == None and (model == "free" or model == "negative"):
			self.seed = interference_bounds
		else:
			self.seed = seed
		self.gamma_parameters_n = int((self.interference_parameters_n+1)/2)
		self.weight_parameters_n = int(self.gamma_parameters_n -1)
		self.interference_bounds = interference_bounds
		self.closed_form = closed_form
		self.max_truncate = None
		self.linear_meiosis = True
		self.genetic_algorithm = None
		self.coincidence4_interval_size = coincidence4_interval_size
		self.nelder_mead = None
		self.max_x = max_x
		self.min_x = min_x
		self.data = []
		self.data_dict = {}
		self.loci = loci
		self.benchmark = benchmark
		self.interference_direction = interference_direction
		self.intervals_n = len(loci)-1
		self.reduced_intervals_n = self.intervals_n
		self.include_patterns_in_report = include_patterns_in_report
		self.inversion = False
		self.unbalanced_proportion = 0
		self.tetrad = tetrad
		self.p = p
		if ("[" in self.loci and not "]" in self.loci) or ("]" in self.loci and not "[" in self.loci):
			print("Error! You have included only one breakpoint. You must include either two or none.")
			sys.exit(1)
		if "[" in self.loci and (not include_breakpoint_loci):
			self.reduced_intervals_n = self.reduced_intervals_n - 2
		if "@" in self.loci and (not include_centromere):
			self.reduced_intervals_n = self.reduced_intervals_n - 1
		if "[" in self.loci:
			self.inversion = True
		if model == "coincidence" or model == "sterility":
			self.coincidence_plots = [CoincidencePlot(gamma_values = gamma_values, p = p, plot_marker= plot_marker)]
			self.coincicdence_figures = CoincidenceFigure(title = figure_name)
			self.coincicdence_figures.coincidence_plots = self.coincidence_plots
		self.plot_resolution = plot_resolution
		self.plot_span = plot_span
		self.counter = 0
		self.solution = None
		self.map = []
		self.pattern_order = []
		self.all_patterns_order = []
		self.model = model
		self.input_gamma_values = gamma_values
		self.input_lambda_values = lambda_values
		self.input_h = h
		self.extra_pathway = extra_pathway
		self.start_at_gamma = start_at_gamma
		self.verbose = True
		self.include_breakpoint_loci = include_breakpoint_loci
		self.alpha = alpha
		self.output_file = output_file
		self.input_file = None
		self.n = None
		self.best_patterns = None
		self.all_best_patterns = None
		self.beta = beta
		self.linear_meiosis = linear_meiosis
		self.include_centromere = include_centromere
		self.model = model
		self.figure_header = figure_header
		if model != "d" and "[" in loci:
			self.d_values = "["
		if(model == "map"):
			self.parameters = self.intervals_n
		elif(model == "negative"):
			self.parameters = self.intervals_n + 2
		elif(model == "free"):
			self.parameters = self.intervals_n + self.interference_parameters_n
		elif(model != 'coincidence' and model != 'plot_distances' and model != "free" and model != 'sterility'):
			print("Error! There's no model called " + model + ". Please choose between 'map', 'free', 'negative', and 'coincidence'")
			sys.exit(1)
		if(extra_pathway):
			self.parameters += 1
		else:
			if p is None:
				self.p = 0
			else:
				self.p = p
			if h is None:
				self.h = 0
			else:
				self.h = h


	def add_coincidence_plot(self, gamma_values, p, plot_marker):
		self.coincidence_plots.append(CoincidencePlot(gamma_values, p, plot_marker))



	def plot_coincidence3(self):
		Xs = numpy.linspace(1e-5, self.plot_span, self.plot_resolution)
		Cs = [numpy.zeros(self.plot_resolution) for i in range(len(self.coincidence_plots))]
		for i in range(self.plot_resolution):
			x = Xs[i]
			for pl in range(len(self.coincidence_plots)):
				plot = self.coincidence_plots[pl]
				gamma_values = plot.gamma_values
				p = plot.p
				h = plot.h
				lambda_value = self.calculate_lambda(x, h, gamma_values)
				pi_vector = generate_pi_vector(gamma_values)
				m = len(gamma_values) -1
				R0 = recombination_rate(m, pi_vector, lambda_value, h*lambda_value)
				R1 = R0
				R12 = recombination_rate(m, pi_vector, 2*lambda_value, 2*h*lambda_value)
				c = (0.5*(R0 + R1 - R12))/(R0*R1)
				Cs[pl][i] = (0.5*(R0 + R1 - R12))/(R0*R1)
		
		pt.axis([0, self.plot_span, 0.0, 2])
		linestyles = ['-', '--', '-.', ':', ",", "o"]
		for pl in range(len(self.coincidence_plots)):
			ls = linestyles[pl%6]
			pt.plot(Xs, Cs[pl], label = self.coincidence_plots[pl].plot_marker, ls = ls)
		pt.xlabel('X')
		pt.ylabel('Coincidence')
		pt.legend(loc = 'lower right')
		pt.show()

	def get_coincidence_arrays(self):
		Xs = numpy.linspace(1e-10, self.plot_span, self.plot_resolution)
		Cs = [numpy.zeros(self.plot_resolution) for i in range(len(self.coincidence_plots))]
		tf = [[False, True] for i in range(3)]
		for pattern in itertools.product(*tf):
			self.pattern_order.append(pattern)
			self.data_dict[pattern] = 0
		for pl in range(len(self.coincidence_plots)):
			plot = self.coincidence_plots[pl]
			gamma_values = plot.gamma_values
			p = plot.p
			h = plot.h
			pi_vector = generate_pi_vector(gamma_values)
			m = len(gamma_values) -1
			lambda02 = self.calculate_lambda(self.coincidence4_interval_size, h, gamma_values)
			R0 = recombination_rate(m, pi_vector, lambda02, h*lambda02)
			R2 = R0
			for i in range(self.plot_resolution):
				x = Xs[i]
				lambda1 = self.calculate_lambda(x, h, gamma_values)
				lambda_values = [lambda02, lambda1, lambda02]
				mu_values = [h*lambda02, h*lambda1, h*lambda02]
				chromosome = Chromosome("ABCD", lambda_values, [1], gamma_values, self, h)
				chromosome.calculate_recombination_pattern_probabilities()
				chromosome.get_recombination_pattern_probabilities()
				R02 = chromosome.pattern_probabilities_array[self.pattern_order.index((True, False, True,))] + chromosome.pattern_probabilities_array[self.pattern_order.index((True, True, True,))]
				Cs[pl][i] = R02/(R0*R2)
		return Cs


	def get_standard_sterility_arrays(self):
		Xs = numpy.linspace(1e-10, self.plot_span, self.plot_resolution)
		Ss = [numpy.zeros(self.plot_resolution) for i in range(len(self.coincidence_plots))]
		for pl in range(len(self.coincidence_plots)):
			plot = self.coincidence_plots[pl]
			gamma_values = plot.gamma_values
			p = plot.p
			h = plot.h
			pi_vector = generate_pi_vector(gamma_values)
			m = len(gamma_values) -1
			for i in range(self.plot_resolution):
				x = Xs[i]
				lambda1 = self.calculate_lambda(x, h, gamma_values)
				mu1 = h*lambda1
				r = recombination_rate(m, pi_vector, lambda1, mu1)
				Ss[pl][i] = r
		return Ss



	def plot_coincidence4(self):
		Xs = numpy.linspace(1e-10, self.plot_span, self.plot_resolution)
		Cs = self.get_coincidence_arrays()
		pt.axis([0, self.plot_span, 0.0, 2])
		linestyles = ['-', '--', ':', '-.']
		for pl in range(len(self.coincidence_plots)):
			ls = linestyles[pl%4]
			pt.plot(Xs, Cs[pl], label = self.coincidence_plots[pl].plot_marker, ls = ls, color = "k")
		pt.xlabel('Morgan')
		pt.ylabel('Coincidence')
		pt.legend(loc = 'lower right')
		pt.show()


	
	def plot_standard_sterility(self):
		Xs = numpy.linspace(1e-10, self.plot_span, self.plot_resolution)
		Ss = self.get_standard_sterility_arrays()
		pt.axis([0, self.plot_span, 0.0, 0.6])
		linestyles = ['-', '--', ':', '-.']
		for pl in range(len(self.coincidence_plots)):
			ls = linestyles[pl%4]
			pt.plot(Xs, Ss[pl], label = self.coincidence_plots[pl].plot_marker, ls = ls, color = "k")
		pt.xlabel('Morgan')
		pt.ylabel('Sterility')
		pt.legend(loc = 'lower right')
		pt.show()

	def plot_crossover_distances(self):
		self.calculate_weights()
		Xs = numpy.linspace(0, self.plot_span, self.plot_resolution)
		summed_distribution = self.Ws[0]*gamma.pdf(Xs, 1, scale = 1/2)
		for i in range(1, len(self.Ws)):
			w = self.Ws[i]
			summed_distribution += w*gamma.pdf(Xs, i+1, scale = 1/(2*(i+1)))
		step_size = self.plot_span/self.plot_resolution
		pt.plot(Xs, summed_distribution)
		p = self.p
		pt.legend()
		pt.xlabel('Genetic distance in Morgans')
		pt.ylabel('Density')
		pt.show()

	def calculate_weights(self):
		gamma_values = self.input_gamma_values
		h = self.calculate_h(self.p, gamma_values)
		p1 = h/(h+1)
		self.Ws = [p1 + gamma_values[0]*(1-p1)]
		n = 2
		while n <= len(gamma_values):
			gamma_sum = 0
			for q in range(n-1):
				gamma_sum += gamma_values[q]
			self.Ws.append( ((1-p1)**(n-1)) * ((1-gamma_sum)*(p1) + gamma_values[n-1]*(1-p1)))
			n += 1


	def calculate_lambda(self, x, h, gamma_values):
		gamma_sum = self.get_sum(gamma_values)
		return x/(0.5*(h + (1/gamma_sum)))


	def analyze_solution(self, solution):
		evaluation = - self.minus_log_likelihood(solution, final = True)
		self.calculate_map(solution, evaluation)

	def calculate_p(self, h, gamma_values):
		if h == 0:
			return 0
		else:
			gamma_sum = self.get_sum(gamma_values)
			#for q in range(len(gamma_values)):
			#	gamma_sum += (q+1)*gamma_values[q]
			return h/(h+(1/gamma_sum))

	def calculate_h(self, p, gamma_values):
		if p is None:
			return 0
		elif p == 1:
			return inf
		else:
			gamma_sum = self.get_sum(gamma_values)
			return (p/gamma_sum)/(1-p)


	def get_sum(self, gamma_values):
		gamma_sum = 0
		for q in range(len(gamma_values)):
			gamma_sum += (q+1)*gamma_values[q]
		return gamma_sum

	def finalize_distances(self, solution, evaluation):
		gamma_values = self.get_gamma_values_distances(solution)
		output_string = "Gamma values:" + str(gamma_values)
		if self.extra_pathway:
			output_string += "\np: " + str(solution[-1])
		output_string += "\nEvaluation: " + str(evaluation)
		print(output_string)


	def calculate_map(self, parameters, evaluation, best = False):
		self.map = []
		alpha = self.alpha
		beta = self.beta
		include_patterns = self.include_patterns_in_report
		if self.model == "free":
			if self.interference_parameters_n == 0:
				if self.input_gamma_values == None:
					gamma_values = [1.0]
				else:
					gamma_values = self.input_gamma_values
			else:
				gamma_parameters = parameters[self.intervals_n:self.intervals_n+self.gamma_parameters_n]
				weight_parameters = parameters[self.intervals_n + self.gamma_parameters_n:self.intervals_n + self.gamma_parameters_n + self.weight_parameters_n]
				m = int( max(gamma_parameters) // 1) + 2
				gamma_values = [0.0 for i in range(m)]

				if self.interference_parameters_n == 1:
					q = parameters[self.intervals_n]
					k = int(q // 1)
					gamma_values =  [0 for i in range(k)]
					gamma_values.append(1-(q%1))
					gamma_values.append(q%1)
				elif self.interference_parameters_n == 3:
					a = weight_parameters[0]
					weights = [a, 1-a]
					for i in range(len(gamma_parameters)):
						q = gamma_parameters[i]
						k = int(q // 1)
						q0 = 1-(q%1)
						q1 = q%1
						gamma_values[k] += weights[i]*q0
						gamma_values[k+1] += weights[i]*q1
				elif self.interference_parameters_n == 7:
					a = weight_parameters[0]
					b = weight_parameters[1]
					c = weight_parameters[2]
					weights = [a*b, a*(1-b), (1-a)*c, (1-a)*(1-c)]
					for i in range(len(gamma_parameters)):
						q = gamma_parameters[i]
						k = int(q // 1)
						q0 = 1-(q%1)
						q1 = q%1
						gamma_values[k] += weights[i]*q0
						gamma_values[k+1] += weights[i]*q1
				elif self.interference_parameters_n == 15:
					a = weight_parameters[0]
					b = weight_parameters[1]
					c = weight_parameters[2]
					d = weight_parameters[3]
					e = weight_parameters[4]
					f = weight_parameters[5]
					g = weight_parameters[6]
					weights = [a*b*c, a*b*(1-c), a*(1-b)*d, a*(1-b)*(1-d), (1-a)*e*f, (1-a)*e*(1-f), (1-a)*(1-e)*g, (1-a)*(1-e)*(1-g)]
					for i in range(len(gamma_parameters)):
						q = gamma_parameters[i]
						k = int(q // 1)
						q0 = 1-(q%1)
						q1 = q%1
						gamma_values[k] += weights[i]*q0
						gamma_values[k+1] += weights[i]*q1
				elif self.interference_parameters_n == 31:
					a = weight_parameters[0]
					b = weight_parameters[1]
					c = weight_parameters[2]
					d = weight_parameters[3]
					e = weight_parameters[4]
					f = weight_parameters[5]
					g = weight_parameters[6]
					h = weight_parameters[7]
					i = weight_parameters[8]
					j = weight_parameters[9]
					k = weight_parameters[10]
					l = weight_parameters[11]
					m = weight_parameters[12]
					n = weight_parameters[13]
					o = weight_parameters[14]
					weights = [a*b*c*d, a*b*c*(1-d), a*b*(1-c)*e, a*b*(1-c)*(1-e), a*(1-b)*f*g,  a*(1-b)*f*(1-g), a*(1-b)*(1-f)*h, a*(1-b)*(1-f)*(1-h), (1-a)*i*j*k, (1-a)*i*j*(1-k), (1-a)*i*(1-j)*l, (1-a)*i*(1-j)*(1-l), (1-a)*(1-i)*m*n, (1-a)*(1-i)*m*(1-n), (1-a)*(1-i)*(1-m)*o, (1-a)*(1-i)*(1-m)*(1-o)]
					for i in range(len(gamma_parameters)):
						q = gamma_parameters[i]
						k = int(q // 1)
						q0 = 1-(q%1)
						q1 = q%1
						gamma_values[k] += weights[i]*q0
						gamma_values[k+1] += weights[i]*q1
		if self.model == "negative":
			if self.extra_pathway:
				gamma_parameter = parameters[-3]
				weight_parameter = parameters[-2]
			else:
				gamma_parameter = parameters[-2]
				weight_parameter = parameters[-1]
			m = int(gamma_parameter // 1) + 2
			gamma_values = [0.0 for i in range(m)]
			gamma_values[0] = weight_parameter
			q = gamma_parameter
			k = int(q // 1)
			q0 = 1-(q%1)
			q1 = q%1
			gamma_values[k] += (1-weight_parameter)*q0
			gamma_values[k+1] += (1-weight_parameter)*q1
		if self.model == "map":
			gamma_values = self.input_gamma_values
		if not self.extra_pathway:
			p = self.p
		else:
			p = parameters[-1]
		h = self.calculate_h(p, gamma_values)
		Xs = parameters[0:self.intervals_n]
		lambda_values = []
		for x in Xs:
			lambda_values.append(self.calculate_lambda(x, h, gamma_values))
	
		denominator = 0
		for i in range(len(gamma_values)):
			denominator += (i+1)*gamma_values[i]
		for i in range(self.intervals_n):
			map_length = parameters[i]
			self.map.append(map_length)
		if not best:
			output_string = "\n**********"
		if best:
			output_string = "\n*****BEST RESULT*****"
		output_string += "\nFile: " + self.input_file + "\nTime: " + str(datetime.now()) + "\nLoci: " + self.loci + "\nModel: " + self.model + "\nExtra pathway: " + str(self.extra_pathway) + "\nh: " + str(h) + "\nu: " + str(p)
		output_string += "\nn: " + str(self.n)
		output_string += "\nRepetition " + str(self.repetition)
		if self.inversion and self.alpha:
			output_string = output_string + "\nDirection: " + self.interference_direction
		output_string += "\nClosed form: " + str(self.closed_form)
		
		output_string = output_string + "\nAlpha: " + str(alpha)
		output_string = output_string + "\nBeta: " + str(beta)
		if self.interference_parameters_n > 0:
			output_string += "\nInterference bounds: " + str(self.interference_bounds)
		if self.model == "free":
			output_string += "\nInterference parameters: " + str(self.interference_parameters_n)
		output_string = output_string + "\nGamma values: " + str(gamma_values)
		if self.model == "free" and self.interference_parameters_n == 1:
			output_string += "\nq: " + str(q)
		else:
			output_string = output_string + "\nm: " + str(self.m)
		output_string = output_string + "\nLambda values: " + str(lambda_values)+ "\nMap: " + str(100*(numpy.array(self.map)))
		output_string = output_string + "\nSolution: " + str(list(parameters))
		if not self.tetrad:
			if not "unbalanced" in self.data_dict:
				output_string = output_string + "\nEstimated sterility: " + str(self.unbalanced_proportion)
		output_string = output_string + "\nLog10 error1: " + str(log10(self.error1))
		if self.tetrad or self.linear_meiosis:
			output_string = output_string + "\nLog10 error2: " + str(log10(self.error2))
			output_string = output_string + "\nMin x: " + str(self.min_x)
			output_string = output_string + "\nMax x: " + str(self.max_x)
		output_string = output_string + "\nEvaluation: " + str(evaluation)
		output_string = output_string + "\n"
		if evaluation > self.benchmark:
			output_string = output_string + "\nBENCHMARK REACHED!"
		if include_patterns or evaluation > self.benchmark or best:
			for i in range(len(self.pattern_order)):
				pattern = self.pattern_order[i]
				if pattern in self.data_dict:
					observed = self.data_dict[pattern]
				else:
					observed = 0
				output_string = output_string + "\n----"
				output_string = output_string + "\nPattern: " + str(pattern)
				output_string = output_string + "\nExpected: " + str(self.n*self.best_patterns[i])
				output_string = output_string + "\nObserved: " + str(observed)
			if 'unbalanced' in self.data_dict:
				output_string = output_string + "\n----"
				output_string = output_string + "\nUnbalanced"
				output_string = output_string + "\nExpected: " + str(self.n*self.unbalanced_proportion)
				output_string = output_string + "\nObserved: " + str(self.data_dict['unbalanced'])

		output_string = output_string + "\n"


		print(output_string)
		if(self.output_file is not None):
			f = open(self.output_file, 'a')
			f.write(output_string)
			f.close()



	def read_input(self, path):
		self.input_file = path
		input_file = open(path)
		if self.tetrad:
			for line in input_file:
				line = line.strip()
				elements = line.split("\t")
				self.data_dict[tuple(map(int, elements[0]))] = int(elements[1])
		else:
			for line in input_file:
				line = line.strip()
				elements = line.split("\t")
				if(elements[0] != "unbalanced"):
					self.data_dict[tuple(map(bool,tuple(map(int, elements[0]))))] = int(elements[1])
				else:
					self.data_dict["unbalanced"] =int(elements[1])
		x = self.data_dict
		input_file.close()
		if self.tetrad:
			intervals_n = self.reduced_intervals_n
			zot = [[0,1,2] for i in range(intervals_n)]
			for pattern in itertools.product(*zot):
				if pattern in self.data_dict:
					self.data.append(self.data_dict[pattern])
					self.pattern_order.append(pattern)
				self.all_patterns_order.append(pattern)
		else:
			intervals_n = self.reduced_intervals_n
			tf = [[False, True] for i in range(intervals_n)]
			for pattern in itertools.product(*tf):
				if pattern in self.data_dict:
					self.data.append(self.data_dict[pattern])
					self.pattern_order.append(pattern)
				self.all_patterns_order.append(pattern)
			if "unbalanced" in self.data_dict and self.data_dict["unbalanced"] != 0:
				self.data.append(self.data_dict["unbalanced"])
				self.pattern_order.append("unbalanced")
				self.all_patterns_order.append("unbalanced")
		self.data = numpy.array(self.data)
		self.n = sum(self.data)


	def tetrad_log_likelihood_test(self, tetrad):
		y = len(tetrad)
		p = self.p
		s = 0
		for i in range(1, y-1):
			s += log(self.f(tetrad[i], p, self.gamma_values))
		return s

	def tetrad_log_likelihood(self, tetrad):
		y = len(tetrad)
		p = self.p
		if y == 1:
			return log(self.g(tetrad[0], p, self.pi_vector))
		elif y == 2:
			return log(self.f(tetrad[0], p, self.pi_vector)) + log(self.g(tetrad[1], p, self.gamma_values))
		else:
			s = 0
			for i in range(1, y-1):
				s += log(self.f(tetrad[i], p, self.gamma_values))
			return  log(self.f(tetrad[0], p, self.pi_vector)) + s + log(self.g(tetrad[y-1], p, self.gamma_values))

	def f(self, x, p, theta_values):
		m = len(theta_values)-1
		h = self.h
		s = 0
		for n in range(1, m+2):
			W = self.w(n, h, theta_values)
			s += W*gamma.pdf(x, n, scale = 1/(2*n))
		return s

	def g(self, x, p, theta_values):
		h = self.h
		m =len(theta_values) -1
		lambda_value = self.calculate_lambda(x, h, self.gamma_values)
		s = 0
		for q in range(m+1):
			for c in range(q+1):
				s += theta_values[q]*poisson(lambda_value, c)
		return exp(-h*lambda_value)*s

	def w(self, n, h,theta_values):
		return ((1/(h+1))**(n-1))*( self.Z(n, theta_values)*(h/(h+1)) + theta_values[n-1]*(1/(h+1)) )

	def Z(self, n, theta_values):
		if n == 1:
			return 1
		else:
			s = 0
			for q in range(n-1):
				s += theta_values[q]
			return 1 - s



	def minus_log_likelihood(self, parameters, final = False):	
		d_values = [1.0]
		if self.model == "free":
			if self.interference_parameters_n == 0:
				if self.input_gamma_values == None:
					gamma_values = [1.0]
				else:
					gamma_values = self.input_gamma_values
			else:
				gamma_parameters = parameters[self.intervals_n:self.intervals_n+self.gamma_parameters_n]
				weight_parameters = parameters[self.intervals_n + self.gamma_parameters_n:self.intervals_n + self.gamma_parameters_n + self.weight_parameters_n]
				m = int( max(gamma_parameters) // 1) + 2
				gamma_values = [0.0 for i in range(m)]

				if self.interference_parameters_n == 1:
					q = parameters[self.intervals_n]
					k = int(q // 1)
					gamma_values =  [0 for i in range(k)]
					gamma_values.append(1-(q%1))
					gamma_values.append(q%1)
				elif self.interference_parameters_n == 3:
					a = weight_parameters[0]
					weights = [a, 1-a]
					for i in range(len(gamma_parameters)):
						q = gamma_parameters[i]
						k = int(q // 1)
						q0 = 1-(q%1)
						q1 = q%1
						gamma_values[k] += weights[i]*q0
						gamma_values[k+1] += weights[i]*q1
				elif self.interference_parameters_n == 7:
					a = weight_parameters[0]
					b = weight_parameters[1]
					c = weight_parameters[2]
					weights = [a*b, a*(1-b), (1-a)*c, (1-a)*(1-c)]
					for i in range(len(gamma_parameters)):
						q = gamma_parameters[i]
						k = int(q // 1)
						q0 = 1-(q%1)
						q1 = q%1
						gamma_values[k] += weights[i]*q0
						gamma_values[k+1] += weights[i]*q1
				elif self.interference_parameters_n == 15:
					a = weight_parameters[0]
					b = weight_parameters[1]
					c = weight_parameters[2]
					d = weight_parameters[3]
					e = weight_parameters[4]
					f = weight_parameters[5]
					g = weight_parameters[6]
					weights = [a*b*c, a*b*(1-c), a*(1-b)*d, a*(1-b)*(1-d), (1-a)*e*f, (1-a)*e*(1-f), (1-a)*(1-e)*g, (1-a)*(1-e)*(1-g)]
					for i in range(len(gamma_parameters)):
						q = gamma_parameters[i]
						k = int(q // 1)
						q0 = 1-(q%1)
						q1 = q%1
						gamma_values[k] += weights[i]*q0
						gamma_values[k+1] += weights[i]*q1
				elif self.interference_parameters_n == 31:
					a = weight_parameters[0]
					b = weight_parameters[1]
					c = weight_parameters[2]
					d = weight_parameters[3]
					e = weight_parameters[4]
					f = weight_parameters[5]
					g = weight_parameters[6]
					h = weight_parameters[7]
					i = weight_parameters[8]
					j = weight_parameters[9]
					k = weight_parameters[10]
					l = weight_parameters[11]
					m = weight_parameters[12]
					n = weight_parameters[13]
					o = weight_parameters[14]
					weights = [a*b*c*d, a*b*c*(1-d), a*b*(1-c)*e, a*b*(1-c)*(1-e), a*(1-b)*f*g,  a*(1-b)*f*(1-g), a*(1-b)*(1-f)*h, a*(1-b)*(1-f)*(1-h), (1-a)*i*j*k, (1-a)*i*j*(1-k), (1-a)*i*(1-j)*l, (1-a)*i*(1-j)*(1-l), (1-a)*(1-i)*m*n, (1-a)*(1-i)*m*(1-n), (1-a)*(1-i)*(1-m)*o, (1-a)*(1-i)*(1-m)*(1-o)]
					for i in range(len(gamma_parameters)):
						q = gamma_parameters[i]
						k = int(q // 1)
						q0 = 1-(q%1)
						q1 = q%1
						gamma_values[k] += weights[i]*q0
						gamma_values[k+1] += weights[i]*q1
		if self.model == "map":
			if self.input_gamma_values == None:
				gamma_values = [1.0]
			else:
				gamma_values = self.input_gamma_values
		if self.model == "negative":
			if self.extra_pathway:
				gamma_parameter = parameters[-3]
				weight_parameter = parameters[-2]
			else:
				gamma_parameter = parameters[-2]
				weight_parameter = parameters[-1]
			m = int(gamma_parameter // 1) + 2
			gamma_values = [0.0 for i in range(m)]
			gamma_values[0] = weight_parameter
			q = gamma_parameter
			k = int(q // 1)
			q0 = 1-(q%1)
			q1 = q%1
			gamma_values[k] += (1-weight_parameter)*q0
			gamma_values[k+1] += (1-weight_parameter)*q1

		if not self.extra_pathway:
			p = self.p
		else:
			p = parameters[-1]
		h = self.calculate_h(p, gamma_values)
		Xs = parameters[0:self.intervals_n]
		lambda_values = []
		for x in Xs:
			lambda_values.append(self.calculate_lambda(x, h, gamma_values))

		chromosome = Chromosome(self.loci, lambda_values, d_values, gamma_values, self, h = h, alpha = self.alpha, beta = self.beta, linear_meiosis = self.linear_meiosis, tetrad = self.tetrad) 
		self.m = chromosome.m
		if self.tetrad:
			chromosome.calculate_tetrad_pattern_probabilities()
		else:
			chromosome.calculate_recombination_pattern_probabilities()
		chromosome.calculate_likelihood(final)
		self.counter += 1
		if self.verbose and self.counter % self.report_frequency == 0:
			print(str(datetime.now()))
			print(parameters)
			print(chromosome.likelihood)
		return -chromosome.likelihood

	
	def add_calculator(self, calculator):
		self.calculator = calculator


	def run_nelder_mead(self, repeat = 1, verbose = True, report_frequency = 100):
		self.verbose = verbose
		self.repetition = 0
		self.current_best = -inf
		self.is_current_best = True
		for i in range(repeat):
			self.repetition += 1
			bounds = tuple((0, 3) for i in range(self.intervals_n))
			parameters = [random.uniform(0,0.2) for i in range(self.intervals_n)]
			if self.model == "free":
				bounds += tuple([self.interference_bounds for i in range(self.gamma_parameters_n)])
				bounds += tuple([(0,1) for i in range(self.weight_parameters_n)])
				for i in range(self.gamma_parameters_n):
					parameters.append(random.uniform(self.seed[0], self.seed[1]))
				for i in range(self.weight_parameters_n):
					parameters.append(random.uniform(0,1))
			elif self.model == "negative":
				bounds = bounds + tuple([self.interference_bounds])
				bounds = bounds + ((0.0, 1.0),)
				parameters.append(random.uniform(self.seed[0], self.seed[1]));
				parameters.append(random.uniform(0.5, 1.0))

			if self.extra_pathway:
				bounds = bounds + ((0,0.9999),)
				parameters.append(random.uniform(0,0.9999))
			self.report_frequency = report_frequency
			result = minimize(self.minus_log_likelihood, parameters, bounds = bounds, method = "Nelder-Mead")
			solution = result['x']
			self.solution = solution
			evaluation = - self.minus_log_likelihood(solution, final = True)
			if evaluation > self.current_best:
				self.current_best = evaluation
				self.current_best_solution = self.solution
				self.is_current_best = True
			else:
				self.is_current_best = False
			self.calculate_map(solution, evaluation)
		best_solution = self.current_best_solution
		best_evaluation = self.current_best
		self.minus_log_likelihood(best_solution, final = True)
		self.calculate_map(best_solution, best_evaluation, best = True)
	
	



class Chromosome:
	'''
	A class for storing and manipulating information relating to a single chromosome
	
	Key attributes:
	self.loci: List of instances of class Locus
	self.loci_keys: A list of locus keys (single charachters)
	self.allele_ns_dictionary: A dictionary of allele numbers for each locus. e.g. {'A': 2, 'B':3} indicate that Locus A has two alleles and locus B has three alleles.
		If a locus is not listed, then the allele number is set to 2 by default.
	self.lambda_values/mu_values/d_values: (list) The lambda/mu/d values as defined in the thesis. The values are sorted so that index i of the list
		correspond to interval i.
	self.haplotypes/diplotypes/karyotypes: A list of Chromosome_haplotype/Chromosome_diplotype/Karyotype instances
	self.inversion: (bool) True if the chromosome has an inversion polymorphism
	self.Q: The Q matrix as defined in the thesis
	self.transition_matrices: The P matrices as defined in the thesis (theorem 7)
	self.intervals_n: (int) The number of intervals on the chromosome
	self.left_intervals_n: (int) The number of intervals to the left of the left inversion breakpoint, if self.inversion is True (equal to self.intervals_n if self.inversion is False)
	self.inversion_intervals_n: (int) The number of intervals in the inverted region
	self.proximal_intervals_n: (int) The number of intervals in the proximal region
	self.right_intervals_n: (int) The number of intervals in the region to the right of the proximal region if self.inversion is True and self.proximal_intervals_n>0
		or to the right of the inverted region if self.inversion is True and self.proximal_intervals_n == 0. Zero if self.inversion is False.
	self.original_homokaryotype/reversed_homokaryotype: If the chromosome has an inversion, then the order of the loci inside the inverted region will be reversed
		in the derived homokaryotype. self.original_homokaryotype points to the Karyotype instance with the original order, and self.reversed_homokaryotype to the
		Karyotype instance with the reversed order.
	self.error1/error2: (float) The accepted error when estimating infinite series expressions for homokaryotypes/heterokaryotypes.
		The results for lower values are more accurate, but take longer to compute.
	self.original_heterokaryotype/reversed_heterokaryotype: 'original' and 'reversed' here refers to the direction of the interference signal through the
		inversion loop (see figure 2.3 in the thesis)

	
	'''
	def __init__(self, loci_keys, lambda_values, d_values, gamma_values, investigation, h = 0, alpha = 0, beta = 1, linear_meiosis = False, tetrad = False, allele_ns_dictionary = {}):
		self.calculator = Calculator()
		self.investigation = investigation
		gamma_values = [float(x) for x in gamma_values]
		self.loci_keys = loci_keys
		self.loci_keys_reversed = None
		self.loci_n = len(self.loci_keys)
		self.allele_ns_dictionary = allele_ns_dictionary
		self.gamma_values = list(gamma_values)
		self.set_gamma_values(gamma_values);
		self.alpha = alpha
		self.g_values = None
		self.pi_vector = None
		self.pattern_probabilities_array = None
		self.all_pattern_probabilities_array = [0.0 for i in range(len(investigation.all_patterns_order))]
		self.b_values = [1.0]
		self.final = False
		self.linear_meiosis = linear_meiosis
		self.tetrad = tetrad
		if len(lambda_values) == 1 and self.loci_n != 2:
			self.lambda_values = [lambda_values[0] for i in range(self.loci_n-1)] # If only one lambda value is given, assume the user want to use this value for all intervals
		elif len(lambda_values) != self.loci_n - 1:
			print('Error! The number of lambda values does not correspond to the number of intervals!')
			sys.exit(1)
		
		else:
			self.lambda_values = list(lambda_values)
		self.mu_values = (h*numpy.array(lambda_values)).tolist()
		

		
		self.d_values = list(d_values)
		if sum(self.d_values) == 0: # no recombination in heterokaryotypes
			self.heteromorphic = True
		else:
			self.heteromorphic = False
		self.loci = []
		self.haplotypes = []
		self.diplotypes = []
		self.diplotype_keys = []
		self.paracentric = False
		self.karyotypes = []
		self.heterokaryotypes = []
		self.homokaryotypes = []
		self.statespace = None
		self.transition_matrices = None
		self.Q = None
		self.centromere_index = None
		self.beta = beta
		self.first_centromere_interval = None
		self.left_of_centromere_intervals_n = len(loci_keys)-1
		self.right_of_centromere_intervals_n = 0
		self.has_centromere = False
		self.intervals_n = len(self.loci_keys) -1
		self.loci_to_remove = []
		self.loci_to_remove_reversed = []
		self.independent_regions = []
		self.breakpoint_intervals = [-2]
		self.breakpoint_intervals_reversed = [-2]
		self.paracentric_linear = False

		if '@' in self.loci_keys:
			self.centromere_index = self.loci_keys.index('@')
			self.has_centromere = True
		self.inversion = False
		if '[' in self.loci_keys and ']' in self.loci_keys:
			self.inversion = True
			self.left_breakpoint_index = self.loci_keys.index('[')
			self.right_breakpoint_index = self.loci_keys.index(']')
			self.loci_keys_reversed = self.loci_keys[0:self.left_breakpoint_index+1]+self.loci_keys[self.left_breakpoint_index+1:self.right_breakpoint_index][::-1]+self.loci_keys[self.right_breakpoint_index:]
			if self.linear_meiosis:
				if '@' in self.loci_keys and ((self.loci_keys.index('@') < self.loci_keys.index('[')) or (self.loci_keys.index(']') < self.loci_keys.index('@'))):
					self.has_centromere = True
					self.paracentric = True
					self.paracentric_linear = True
					self.has_centromere = True
					self.centromere_index = self.loci_keys.index('@')
					self.linear_meiosis = True
					self.male_recombinant = False
					if self.centromere_index<self.left_breakpoint_index: # make sure that the proximal region is to the right of the inverted region
						self.loci_keys = self.loci_keys[::-1]
						self.lambda_values = self.lambda_values[::-1]
						self.mu_values = self.mu_values[::-1]
						self.left_breakpoint_index = self.loci_keys.index('[')
						self.right_breakpoint_index = self.loci_keys.index(']')
						self.centromere_index = self.loci_keys.index('@')
			else:
				self.paracentric_linear = False
				if '@' in self.loci_keys:
					self.has_centromere = True
					self.centromere_index = self.loci_keys.index('@')
					self.centromere_index_reversed = self.loci_keys_reversed.index("@")

		if '@' in self.loci_keys and '[' in self.loci_keys and ']' in self.loci_keys:
			self.independent_regions = []
			if self.alpha == 1 and self.beta == 1:
				self.independent_regions.append(list(range(self.intervals_n)))
			elif self.alpha == 1 and self.beta == 0:
				self.breakpoint_intervals.append(self.centromere_index -1)
				self.independent_regions.append(list(range(self.centromere_index)))
				self.independent_regions.append(list(range(self.centromere_index, self.intervals_n)))
			elif self.alpha == 0 and self.beta == 1:
				self.breakpoint_intervals.append(self.left_breakpoint_index -1)
				self.breakpoint_intervals.append(self.right_breakpoint_index -1)
				self.breakpoint_intervals_reversed.append(self.left_breakpoint_index -1)
				self.breakpoint_intervals_reversed.append(self.right_breakpoint_index -1)
				self.independent_regions.append(list(range(self.left_breakpoint_index)))
				self.independent_regions.append(list(range(self.left_breakpoint_index, self.right_breakpoint_index)))
				self.independent_regions.append(list(range(self.right_breakpoint_index, self.intervals_n)))
			elif self.alpha == 0 and self.beta == 0:
				indices = [self.centromere_index, self.left_breakpoint_index, self.right_breakpoint_index]
				indices.sort()
				indices.append(self.intervals_n)
				self.independent_regions.append(list(range(indices[0])))
				self.independent_regions.append(list(range(indices[0], indices[1])))
				self.independent_regions.append(list(range(indices[1], indices[2])))
				self.independent_regions.append(list(range(indices[2], indices[3])))
				self.breakpoint_intervals.append(self.centromere_index -1)
				self.breakpoint_intervals.append(self.left_breakpoint_index -1)
				self.breakpoint_intervals.append(self.right_breakpoint_index -1)
		elif '@' in self.loci_keys:
			if self.beta == 0:
				self.breakpoint_intervals.append(self.centromere_index -1)
				self.independent_regions.append(list(range(self.centromere_index)))
				self.independent_regions.append(list(range(self.centromere_index, self.intervals_n)))
			else:
				self.independent_regions.append(list(range(4)))
		elif '[' in self.loci_keys and ']' in self.loci_keys:
			if self.alpha == 0:
				self.independent_regions.append(list(range(self.left_breakpoint_index)))
				self.independent_regions.append(list(range(self.left_breakpoint_index, self.right_breakpoint_index)))
				self.independent_regions.append(list(range(self.right_breakpoint_index, self.intervals_n)))
				self.breakpoint_intervals.append(self.left_breakpoint_index -1)
				self.breakpoint_intervals.append(self.right_breakpoint_index -1)
			else:
				self.independent_regions.append(list(range(self.intervals_n)))

		else:
			self.independent_regions.append(list(range(self.intervals_n)))


		if '@' in self.loci_keys:
			self.has_centromere = True
			self.intervals_n = len(self.loci) -1
			self.centromere_index = self.loci_keys.index('@')
			self.first_centromere_interval = self.centromere_index -1
			self.left_of_centromere_intervals_n = self.centromere_index
			self.right_of_centromere_intervals_n = self.intervals_n - self.left_of_centromere_intervals_n
		if not self.inversion:
			self.left_intervals_n = self.loci_n-1
			self.intervals_n = self.left_intervals_n
			self.inversion_intervals_n = 0
			self.proximal_intervals_n = 0
			self.right_intervals_n = 0
		else:
			if not self.paracentric:
				self.left_intervals_n = self.left_breakpoint_index
				self.inversion_intervals_n = self.right_breakpoint_index - self.left_breakpoint_index
				self.proximal_intervals_n = 0
				self.right_intervals_n = self.loci_n - 1 - self.right_breakpoint_index
			else:
				self.left_intervals_n = self.left_breakpoint_index
				self.inversion_intervals_n = self.right_breakpoint_index - self.left_breakpoint_index
				self.proximal_intervals_n = self.centromere_index - self.right_breakpoint_index
				self.right_intervals_n = self.loci_n - 1 - self.centromere_index
		self.intervals_n = self.left_intervals_n + self.inversion_intervals_n + self.proximal_intervals_n + self.right_intervals_n
		if '[' in self.loci_keys and ']' in self.loci_keys:
			if not self.investigation.include_breakpoint_loci:
				self.loci_to_remove.append(self.left_breakpoint_index)
				self.loci_to_remove.append(self.right_breakpoint_index)
				self.loci_to_remove_reversed.append(self.left_breakpoint_index)
				self.loci_to_remove_reversed.append(self.right_breakpoint_index)
		if '@' in self.loci_keys:
			if not self.investigation.include_centromere:
				self.loci_to_remove.append(self.centromere_index)

		if (not self.inversion) or self.tetrad:
			self.original_homokaryotype = Karyotype(self, original = True, homokaryotype = True, tetrad = self.tetrad)
			self.homokaryotypes.append(self.original_homokaryotype)
			self.karyotypes.append(self.original_homokaryotype)
		if self.inversion:
			if self.alpha:
				if self.investigation.interference_direction == "original":
					self.original_heterokaryotype = Karyotype(self, original = True, homokaryotype = False)
					self.karyotypes.append(self.original_heterokaryotype)
					self.heterokaryotypes.append(self.original_heterokaryotype)
				elif self.investigation.interference_direction == "reversed":
					self.reversed_heterokaryotype = Karyotype(self, original = False, homokaryotype = False)
					self.heterokaryotypes.append(self.reversed_heterokaryotype)
					self.karyotypes.append(self.reversed_heterokaryotype)
				elif self.investigation.interference_direction == "mixed":
					self.original_heterokaryotype = Karyotype(self, original = True, homokaryotype = False)
					self.karyotypes.append(self.original_heterokaryotype)
					self.heterokaryotypes.append(self.original_heterokaryotype)
					self.reversed_heterokaryotype = Karyotype(self, original = False, homokaryotype = False)
					self.heterokaryotypes.append(self.reversed_heterokaryotype)
					self.karyotypes.append(self.reversed_heterokaryotype)
				else:
					print("Error! Invalid input for interference_direction!")
					sys.exit(1)
			else:
				self.original_heterokaryotype = Karyotype(self, original = True, homokaryotype = False)
				self.karyotypes.append(self.original_heterokaryotype)
				self.heterokaryotypes.append(self.original_heterokaryotype)
		self.independent_region_lengths = []
		for r in self.independent_regions:
			self.independent_region_lengths.append(len(r))
		self.pericentric = '@' in self.loci_keys and '[' in self.loci_keys and ']' in self.loci_keys and self.centromere_index > self.left_breakpoint_index and self.centromere_index < self.right_breakpoint_index
		if self.pericentric:
			self.lambda_sum_left_of_centromere_in_inversion = sum(numpy.array(self.lambda_values)[self.left_breakpoint_index:self.centromere_index])
			self.lambda_sum_right_of_centromere_in_inversion = sum(numpy.array(self.lambda_values)[self.centromere_index: self.right_breakpoint_index])
			self.mu_sum_left_of_centromere_in_inversion = sum(numpy.array(self.mu_values)[self.left_breakpoint_index:self.centromere_index])
			self.mu_sum_right_of_centromere_in_inversion = sum(numpy.array(self.mu_values)[self.centromere_index: self.right_breakpoint_index])

		if self.inversion:
			self.lambda_sum_inversion = sum(numpy.array(self.lambda_values)[self.left_breakpoint_index:self.right_breakpoint_index])
			self.mu_sum_inversion = sum(numpy.array(self.mu_values)[self.left_breakpoint_index:self.right_breakpoint_index])
			
	def calculate_likelihood(self, final = False):
		x = self.investigation.data
		if self.investigation.tetrad:
			self.get_tetrad_pattern_probabilties(final = final)
		else:
			self.get_recombination_pattern_probabilities(final = final)
		self.investigation.best_patterns = self.pattern_probabilities_array
		if final:
			self.investigation.all_best_patterns = self.all_pattern_probabilities_array
		if 0 in self.pattern_probabilities_array:
			self.likelihood = -inf
		else:
			y = numpy.log(self.pattern_probabilities_array)
			self.likelihood = sum(x*y)

	def get_tetrad_pattern_probabilties(self, final = False):
		self.pattern_probabilities_array = self.karyotypes[0].pattern_probabilities_array
		if final:
			self.all_pattern_probabilities_array = self.karyotypes[0].all_pattern_probabilities_array

	def get_recombination_pattern_probabilities(self, final = False):
		if len(self.karyotypes) == 1:
			self.pattern_probabilities_array = self.karyotypes[0].pattern_probabilities_array
			if final:
				self.investigation.unbalanced_proportion = self.karyotypes[0].unbalanced
		elif len(self.karyotypes) == 2:
			self.pattern_probabilities_array = (self.karyotypes[0].pattern_probabilities_array + self.karyotypes[1].pattern_probabilities_array)/2.0
			self.investigation.unbalanced_proportion = (self.karyotypes[0].unbalanced + self.karyotypes[1].unbalanced)/2
			if final:
				self.investigation.unbalanced_proportion = (self.karyotypes[0].unbalanced+self.karyotypes[1].unbalanced)/2

	def calculate_tetrad_pattern_probabilities(self):
		for k in self.karyotypes:
			k.calculate_tetrad_pattern_probabilities()

	def calculate_recombination_pattern_probabilities(self):
		for k in self.karyotypes:
			k.calculate_recombination_pattern_probabilities()
	
	def calculate_gamete_frequencies(self):
		self.calculate_recombination_pattern_probabilities()
		for d in self.diplotypes:
			d.calculate_gamete_frequencies()
	
	def initialize_loci(self):
		'''
		Set the default allele numbers for the special loci '[' and ']' (left and right breakpoints) and '@' (centromere),
		and initialize and store pointers to all loci.
		'''
		for i in range(len(self.loci_keys)):
			locus_key = self.loci_keys[i]
			if locus_key == '[' or locus_key == ']':
				breakpoint = True
				if locus_key in self.allele_ns_dictionary:
					if self.allele_ns_dictionary[locus_key] != 2:
						print('Error! The number of alleles for a breakpoint cannot be different from 2!')
						sys.exit(1)
			else:
				breakpoint = False
			if locus_key in self.allele_ns_dictionary:
				allele_n = self.allele_ns_dictionary[locus_key]
			else: # the default number of alleles is 2 for all loci except the centromere (@)
				if locus_key == '@':
					allele_n = 1
				else:
					allele_n = 2
			locus = Locus(chromosome = self, key = locus_key, allele_n = allele_n, macrolinked = False, microlinked = False, chromosome_index = i, breakpoint = breakpoint)
			locus.initialize_alleles()
			self.loci.append(locus)
	
	def initialize_haplotypes(self, haplotypes):
		allele_keys = []
		for locus in self.loci:
			new_allele_keys = []
			for allele in locus.alleles:
				new_allele_keys.append(allele.allele_key)
			allele_keys.append(new_allele_keys)
		#iterator = itertools.product(*allele_keys)
		for haplotype_allele_keys in haplotypes:
			passed = True
			if self.inversion:
				if haplotype_allele_keys[self.left_breakpoint_index] != haplotype_allele_keys[self.right_breakpoint_index]: # the two inversion breakpoints must belong to the same arrangement (ancestral/derived), otherwise the haplotype is unbalanced.
					passed = False
			if passed:
				haplotype = Chromosome_haplotype(allele_keys=haplotype_allele_keys, chromosome=self)
				if passed:
					self.haplotypes.append(haplotype)


	def set_gamma_values(self, gamma_values):
		'''
		Removes trailing zeros from gamma_values.
		'''
		while gamma_values[-1] == 0:
			gamma_values = gamma_values[:-1]
		self.gamma_values = gamma_values
		
		self.m = len(self.gamma_values) -1

	def generate_g_values(self, maximum):
		'''
		Initializes the g values matrix
		'''
		g_values = numpy.zeros((maximum,maximum), numpy.float64)
		g_values[0][0] = 1.0
		for n in range(1,maximum):
			for s in range(1,maximum):
				if n<s:
					break
				t = 0
				for k in range(s-1, n):
					if n-1-k<len(self.gamma_values):
						t+=g_values[k][s-1]*self.gamma_values[n-1-k]
				g_values[n][s] = t
				
		self.g_values = g_values
		
	
	def extend_g_values(self, extend_by):
		'''
		Extends the g values matrix if needed
		'''
		old_size = self.g_values.shape[0]
		new_size = old_size+extend_by
		new_g_values = numpy.zeros((new_size, new_size))
		new_g_values[0:old_size, 0:old_size] = self.g_values
		for n in range(old_size,new_size):
			for s in range(1,new_size):
				if n<s:
					break
				t = 0
				for k in range(s-1, n):
					if n-1-k<len(self.gamma_values):
						t+=new_g_values[k][s-1]*self.gamma_values[n-1-k]
				new_g_values[n][s] = t
		self.g_values = new_g_values

	def generate_pi_vector(self):
		'''
		Generates the stationary distribution and stores it as self.pi_vector
		'''
		pi_vector = numpy.zeros(self.m+1, numpy.float64)
		denominator = 0.0
		gamma_values = self.gamma_values
		for q in range(self.m+1):
			denominator += (q+1)*gamma_values[q]
		for i in range(self.m+1):
			nominator = 0.0
			for q in range(i, self.m+1):
				nominator += gamma_values[q]
			pi_vector[i] = nominator/denominator
		
		self.pi_vector = pi_vector

	def extend_b_values(self, extend_by = 10):
		'''
		Initialize or extend the array of b values.
		'''
		gamma_values = self.gamma_values
		b_values = self.b_values
		m = self.m
		current_length = len(b_values)
		for n in range(current_length, current_length + extend_by):
			s = 0
			if n-1>=m:
				for q in range(n-1-m, n):
					s += b_values[q]*gamma_values[n-1-q]
			else:
				for q in range(n):
					s += b_values[q]*gamma_values[n-1-q]
			b_values.append(s)
		self.b_values = b_values

class Karyotype:
	'''
	A class for storing and manipulating information relating to a single karyotype. Karyotype here refers to information about the arrangement on a single chromosome.
	The boolean variables self.original and self.homokaryotype define four different karyotypes:
	Original/ancestral homokaryotype: (self.original = True, self.homokaryotype = True) an inversion homokaryotype with loci arranged in the order given in the input file
	Reversed/derived homokaryotype: (self.original = False, self.homokaryotype = True) an inversion homokaryotype with the order of the loci inside the inverted region reversed
		with respect to the order in the input file
	Original heterokaryotype: (self.original = True, self.homokaryotype = False) an inversion heterokaryotype where the interference signal move in the direction given by
		the order of the loci in the input file
	Reversed heterokaryotype: (self.original = False, self.homokaryotype = False) an inversion heterokaryotype where the interference signal move in the direction given when
		the order of the loci inside the inverted region is reversed with respect to the order in the input file. (see figure 2.3 in the thesis).
	
	For heterokaryotypes with interference across the breakpoint boundaries (self.population.alpha = True), the recombination pattern for both the original and the reversed
		heterokaryotype is calculated, and the average of the two is used to calculate the gamete frequencies of the chromosome diplotype.
	
	Other key attributes:
	self.chromosome: Pointer to the Chromosome instance
	self.heteromorphic: (bool) True if there is no recombination on the chromosome
	self.recombination_patterns: A dictionary with the frequencies of the different recombination patterns.
	self.lambda_values/mu_values/d_values/loci/intervals_n etc: see class Chromosome
	
	
	
	'''
	def __init__(self, chromosome, original, homokaryotype, heteromorphic = False, tetrad = False):
		self.chromosome = chromosome
		self.original = original
		self.homokaryotype = homokaryotype
		self.heteromorphic = heteromorphic
		self.patterns_dict = {}
		self.Ds_sterility = None
		self.Th = None
		self.Tp = None
		self.sterility_initialized = False
		if chromosome.inversion:
			self.first_inversion_interval = self.chromosome.left_breakpoint_index
			self.last_inversion_interval = self.chromosome.right_breakpoint_index -1
		else:
			self.first_inversion_interval = 0
			self.last_inversion_interval = 0
		self.pattern_probabilities_array = []
		self.tetrad = tetrad
		self.unbalanced = 0.0
		self.all_pattern_probabilities_array = [0.0 for i in range(len(self.chromosome.investigation.all_patterns_order))]
		if self.heteromorphic:
			self.key = 'heteromorphic'
		else:
			if self.original:
				if self.homokaryotype:
					self.key = 'original_homokaryotype'
				else:
					self.key = 'original_heterokaryotype'
			else:
				if self.homokaryotype:
					self.key = 'reversed_homokaryotype'
				else:
					self.key = 'reversed_heterokaryotype'
		
		if self.heteromorphic:
			self.lambda_values = [0.0 for i in range(len(self.chromosome.lambda_values))]
			self.mu_values = [0.0 for i in range(len(self.chromosome.mu_values))]
			self.d_values = [0.0 for i in range(len(self.chromosome.d_values))]
			self.loci = self.chromosome.loci
		elif self.original:
			self.lambda_values = self.chromosome.lambda_values
			self.mu_values = self.chromosome.mu_values
			self.d_values = self.chromosome.d_values
			self.loci = self.chromosome.loci
		else:
			l = self.chromosome.left_breakpoint_index
			r = self.chromosome.right_breakpoint_index
			lv = self.chromosome.lambda_values
			mv = self.chromosome.mu_values
			dv = self.chromosome.d_values
			self.lambda_values = lv[0:l] + lv[l:r][::-1] + lv[r:]
			self.mu_values = mv[0:l] + mv[l:r][::-1] + mv[r:]
			self.d_values = dv[0:l] + dv[l:r][::-1] + dv[r:]
			
		if self.homokaryotype or self.heteromorphic:
			self.intervals_n = self.chromosome.intervals_n
			self.left_intervals_n = self.intervals_n
			self.inversion_intervals_n = 0
			self.proximal_intervals_n = 0
			self.right_intervals_n = 0
		
		else:
			self.intervals_n = self.chromosome.intervals_n
			self.left_intervals_n = self.chromosome.left_intervals_n
			self.inversion_intervals_n = self.chromosome.inversion_intervals_n
			self.proximal_intervals_n = self.chromosome.proximal_intervals_n
			self.right_intervals_n = self.chromosome.right_intervals_n
			self.lambda_values = list(numpy.array(self.lambda_values)*numpy.array(self.d_values))
			self.mu_values = list(numpy.array(self.mu_values)*numpy.array(self.d_values))
	
	def find_indices(self, pattern, statespace):
		'''
		Returns the indices of states in statespace that match the pattern. The pattern is ordinarily given as a list of boolean variables indicating recombination
		or non-recombination in each interval (first n elements, where n is the number of intervals) and an integer indicating the tetrad configuration (last element).
		Wild-card elements in the pattern are indicated with an 'x' (str).
		See the thesis, chapter 2, for details on the statespace and on tetrad configurations. 
		
		Pattern can also be a string indicating five special cases:
		pattern = 'balanced'/'unbalanced': returns the indices of states with balanced/unbalanced chromatids in the statespace
		pattern = 'balanced_a1'/'unbalanced_a1': returns the indices of states with balanced/unbalanced chromatids and anaphase I tetrad configuration in the statespace
		pattern = 'a1': returns the indices of states with anaphase I tetrad configuration in the statespace
		'''
		left_intervals_n = self.left_intervals_n
		inversion_intervals_n = self.inversion_intervals_n
		match = []
		for i in range(len(statespace)):
			state = statespace[i]
			found = True
			start = left_intervals_n
			end = left_intervals_n + inversion_intervals_n
			if pattern == 'unbalanced':
				if sum(state[start:end])%2 !=0:
					match.append(i)
		
			elif pattern == 'balanced':
				if sum(state[start:end])%2 == 0:
					match.append(i)
		
			elif pattern == 'unbalanced_a1':
				if sum(state[start:end])%2 !=0 and state[-1] == 1:
					match.append(i)
			elif pattern == 'balanced_a1':
				if sum(state[start:end])%2 == 0 and state[-1] == 1:
					match.append(i)
			elif pattern == 'a1':
				if state[-1] == 1:
					match.append(i)
			else:
				for j in range(len(state)):
					if pattern[j] != 'x' and state[j] != pattern[j]:
						found = False
						break
				if found:
					match.append(i)
		return match


	def calculate_sterility_1(self):
		if not self.sterility_initialized:
			self.initialize_sterility_1()
		args = [range(self.sterility_truncate[i]) for i in range(2)]
		chiasmata_iter = numpy.array(list(itertools.product(*args)))
		Ds = self.Ds_sterility
		Ps = self.Ps_sterility
		if self.chromosome.pi_vector is None:
			self.chromosome.generate_pi_vector()
		pi_vector = self.chromosome.pi_vector
		v0 = self.v0_sterility
		Q = self.chromosome.Q
		v = numpy.zeros(4, numpy.float64)
		if self.chromosome.alpha:
			independent_regions = [[0,1],[]]
		else:
			independent_regions = [[0], [1]]
		dot = numpy.dot
		for chiasmata_combination in chiasmata_iter: # equivalent to a nested for-loop over all number of chiasma events in each interval.
			p = 0.0
			mid_vector_p = pi_vector
			mid_vector_s = v0
			for r in range(len(independent_regions)):
				for i in independent_regions[r]:
					x = chiasmata_combination[i]
					mid_vector_p = dot(mid_vector_p, Ds[i][x])
					mid_vector_s = dot(mid_vector_s, Ps[i][x])
				if r < len(independent_regions)-1:
					mid_vector_p = dot(mid_vector_p, Q)
			v += numpy.sum(mid_vector_p)*mid_vector_s
		
		self.sterility = numpy.dot(v, self.w_sterility)


	def calculate_sterility2(self):
		if self.chromosome.pericentric and self.chromosome.beta == 0:
			rec1 = recombination_rate(self.chromosome.m, self.pi_vector, self.chromosome.lambda_sum_left_of_centromere_in_inversion, self.chromosome.mu_sum_left_of_centromere_in_inversion) * (1- recombination_rate(self.chromosome.m, self.pi_vector, self.chromosome.lambda_sum_right_of_centromere_in_inversion, self.chromosome.mu_sum_right_of_centromere_in_inversion))
			rec2 = recombination_rate(self.chromosome.m, self.pi_vector, self.chromosome.lambda_sum_right_of_centromere_in_inversion, self.chromosome.mu_sum_right_of_centromere_in_inversion) * (1- recombination_rate(self.chromosome.m, self.pi_vector, self.chromosome.lambda_sum_left_of_centromere_in_inversion, self.chromosome.mu_sum_left_of_centromere_in_inversion))
			return rec1 + rec2
		else:
			return recombination_rate(self.chromosome.m, self.pi_vector, self.chromosome.lambda_sum_inversion, self.chromosome.mu_sum_inversion)


	def initialize_sterility_1(self):
		left = self.chromosome.left_breakpoint_index
		right = self.chromosome.right_breakpoint_index
		cent = self.chromosome.centromere_index
		self.v0_sterility = numpy.array([1.0, 0.0, 0.0, 0.0])
		self.w_sterility = numpy.array([0.0, 0.0, 0.5, 1])
		self.Th = numpy.array([[0.0, 1.0, 0.0, 0.0], [0.25, 0.5, 0.0, 0.25], [0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
		self.Tp = numpy.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.5, 0.5, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
		self.lambda_inv = sum(numpy.array(self.lambda_values)[left:right])
		self.lambda_prox = sum(numpy.array(self.lambda_values)[right:cent])
		self.mu_inv = sum(numpy.array(self.mu_values)[left:right])
		self.mu_prox = sum(numpy.array(self.mu_values)[right:cent])
		self.generate_sterility_matrices()
		self.sterility_initialized = True


	

	def calculate_recombination_pattern_probabilities(self):
		'''
		Chooses the appropriate algorithm for calculating recombination pattern probabilities
		'''
		if self.chromosome.paracentric_linear and not self.homokaryotype:
			self.calculate_recombination_pattern_probabilities_1()
		else:
			self.calculate_recombination_pattern_probabilities_2()
	
	def calculate_recombination_pattern_probabilities_1(self):
		calculator = self.chromosome.calculator
		comb = scipy.special.comb
		m = self.chromosome.m
		lambda_values = self.lambda_values
		mu_values = self.mu_values
		gamma_values = self.chromosome.gamma_values
		distal_intervals_n = self.left_intervals_n
		inversion_intervals_n = self.inversion_intervals_n
		proximal_intervals_n = self.proximal_intervals_n
		right_intervals = self.right_intervals_n
		intervals_n = self.intervals_n
		alpha = self.chromosome.alpha
		pattern_order = self.chromosome.investigation.pattern_order
		all_patterns_order = self.chromosome.investigation.all_patterns_order
		pattern_probabilities_array = [0.0 for i in range(len(pattern_order))]
		all_pattern_probabilities_array = [0.0 for i in range(len(all_patterns_order))]
		if self.chromosome.pi_vector is None:
			self.chromosome.generate_pi_vector()
		pi_vector = self.chromosome.pi_vector
		self.pi_vector = pi_vector
		if self.chromosome.statespace is None:
			self.generate_statespace()
		self.statespace = self.chromosome.statespace
		if self.chromosome.transition_matrices is None:
			self.generate_transition_matrices() 
		
		if self.chromosome.g_values is None:
			self.chromosome.generate_g_values(100)
		
		self.g_values = self.chromosome.g_values
		
		self.generate_matrices()
		if self.chromosome.pi_vector is None:
			self.chromosome.generate_pi_vector()
		pi_vector = self.chromosome.pi_vector
		self.pi_vector = pi_vector # stationary phase distribution
		
		start = [False for i in range(self.intervals_n)]
		start.append(0)
		v0 = numpy.zeros(len(self.statespace))
		start_index = self.find_indices(start, self.statespace)[0]
		v0[start_index] = 1.0
		dot = numpy.dot
		v = numpy.zeros(len(self.statespace), numpy.float64)
		if self.chromosome.Q is None:
			self.generate_Q()
		Q = self.chromosome.Q # Q matrix
		truncate = self.new_truncate_D # truncate (list) gives the upper limit on the number of chiasma events to be considered in each interval. See method generate_matrices
		args = [range(truncate[i]) for i in range(self.intervals_n)]
		chiasmata_iter = numpy.array(list(itertools.product(*args)), numpy.int32)

		max_d = max(truncate)
		alpha = self.chromosome.alpha
		left_intervals_n = self.left_intervals_n
		inversion_intervals_n = self.inversion_intervals_n
		intervals_n = self.intervals_n
		
		Ds = self.Ds # Ds[i][x] is the D matrix for interval i, x chiasma events
		Ps = self.Ps # Ps[i][x] is the P matrix for interval i to the xth power.
		
		v = numpy.zeros(len(self.statespace), numpy.float64)
		for chiasmata_combination in chiasmata_iter: # equivalent to a nested for-loop over all number of chiasma events in each interval.
			p = 0.0
			mid_vector_p = pi_vector
			mid_vector_s = v0
			for r in range(len(self.chromosome.independent_regions)):
				for i in self.chromosome.independent_regions[r]:
					x = chiasmata_combination[i]
					mid_vector_p = dot(mid_vector_p, Ds[i][x])
					mid_vector_s = dot(mid_vector_s, Ps[i][x])
				if r < len(self.chromosome.independent_regions)-1:
					mid_vector_p = dot(mid_vector_p, Q)
			v += numpy.sum(mid_vector_p)*mid_vector_s
		self.v = v
		is_balanced = self.is_balanced
		find_indices = self.find_indices
		
		tf = []
		for i in range(intervals_n):
			tf.append([True, False])
		patterns = {}
		for pattern in itertools.product(*tf):
			if is_balanced(pattern): # balanced
				reduced_pattern = copy.copy(pattern)
				if not self.original:
					reduced_pattern = self.chromosome.calculator.reverse_intervals(list(reduced_pattern), self.first_inversion_interval, self.last_inversion_interval)
				if len(self.chromosome.loci_to_remove) > 0:
					reduced_pattern = self.chromosome.calculator.remove_loci(list(reduced_pattern), self.chromosome.loci_to_remove)
				if reduced_pattern in self.chromosome.investigation.pattern_order:
					w = numpy.zeros(len(self.statespace), numpy.float64)
					state = list(pattern)
					state = state + ['x']
					pattern_indices = find_indices(state, self.statespace)
					w[pattern_indices] = 1.0
					state_a1 = copy.copy(state)
					state_a1[-1] = 1
					pattern_a1_indices = find_indices(state_a1, self.statespace)
					w[pattern_a1_indices] = 2.0 # balanced chromatids in an anaphase I tetrad configuration are counted twice.
					p = numpy.dot(v, w)
					patterns[pattern] = p
					pattern_probabilities_array[pattern_order.index(reduced_pattern)] += p
					#all_pattern_probabilities_array[all_patterns_order.index(pattern)] += p

		w = numpy.zeros(len(self.statespace), numpy.float64)
		unbalanced_indices = find_indices('unbalanced', self.statespace)
		w[unbalanced_indices] = 1.0
		unbalanced_a1_indices = find_indices('unbalanced_a1', self.statespace)
		w[unbalanced_a1_indices] = 0.0 # unbalanced chromatids in an anaphase I tetrad are retained in the polar bodies.
		unbalanced_p = numpy.dot(v, w)

		self.unbalanced = unbalanced_p
		patterns['unbalanced'] = unbalanced_p
		if "unbalanced" in pattern_order:
			pattern_probabilities_array[pattern_order.index("unbalanced")] += unbalanced_p
			all_pattern_probabilities_array[all_patterns_order.index("unbalanced")] += p

		

		self.pattern_probabilities_array = numpy.array(pattern_probabilities_array)
		s = sum(self.pattern_probabilities_array)
		if not "unbalanced" in self.chromosome.investigation.data_dict:
				self.pattern_probabilities_array = (self.pattern_probabilities_array*(s+self.unbalanced))/s
		self.all_pattern_probabilities_array = numpy.array(all_pattern_probabilities_array)
		self.patterns_dict = patterns
		
	def calculate_recombination_pattern_probabilities_2(self):
		'''
		Implements theorems 1/2/5, depending on circumstances.
		'''
		self.generate_GHs()
		self.generate_RNs()
		if self.chromosome.pi_vector is None:
			self.chromosome.generate_pi_vector()
		self.pi_vector = self.chromosome.pi_vector # stationary phase distribution
		if ((not self.homokaryotype) or self.chromosome.beta == 0) and self.chromosome.Q is None:
			self.generate_Q()
		self.Q = self.chromosome.Q # Q matrix
		pi_vector = self.pi_vector
		recombination_patterns = {}
		left_intervals_n = self.left_intervals_n
		inversion_intervals_n = self.inversion_intervals_n
		left_of_centromere_intervals_n = self.chromosome.left_of_centromere_intervals_n
		right_of_centromere_intervals_n = self.chromosome.right_of_centromere_intervals_n
		intervals_n = self.intervals_n
		dot = numpy.dot
		inversion_pattern_probabilities_array = [0.0 for i in range(len(self.chromosome.investigation.pattern_order))]
		# Matrices R and N are the same as matrix M in thesis, for the two conditions.
		Rs = self.Rs # Rs[i] give matrix R for interval i, where matrix R is equal to 1/2 times matrix G
		Ns = self.Ns # Ns[i] give matrix N for interval i, where matrix N is equal to 1/2 times matrix G plus matrix H
		Q = self.Q
		alpha = self.chromosome.alpha
		beta = self.chromosome.beta
		patterns = {'unbalanced': 0.0}
		pattern_order = self.chromosome.investigation.pattern_order
		all_patterns_order = self.chromosome.investigation.all_patterns_order
		unbalanced = 0.0
		independent_regions = self.chromosome.independent_regions
		if self.original:
			independent_intervals = self.chromosome.breakpoint_intervals
			loci_to_remove = self.chromosome.loci_to_remove
		else:
			independent_intervals = self.chromosome.breakpoint_intervals_reversed
			loci_to_remove = self.chromosome.loci_to_remove

		if self.chromosome.inversion or len(self.chromosome.loci_to_remove)>0:
			tf = [[False, True] for i in range(intervals_n)]
			iterator = itertools.product(*tf)
		else:
			iterator = pattern_order
		for pattern in iterator: # equivalent to a for-loop over all possible patterns.
			mid_vector = pi_vector
			skip = False
			unbalanced_pattern = False
			if self.chromosome.inversion or len(loci_to_remove)>0:
				if sum(pattern[left_intervals_n: left_intervals_n+inversion_intervals_n])%2 != 0: # unbalanced pattern
					unbalanced_pattern = True
					skip = True
				else:
					reduced_pattern = copy.copy(pattern)
					if (not self.homokaryotype):
						if not self.original:
							reduced_pattern = self.chromosome.calculator.reverse_intervals(list(pattern), self.first_inversion_interval, self.last_inversion_interval)
					if len(self.chromosome.loci_to_remove) > 0:
						reduced_pattern = self.chromosome.calculator.remove_loci(list(reduced_pattern), loci_to_remove)
					if not reduced_pattern in pattern_order:
						skip = True
			if not skip:
				for i in range(intervals_n):
					if pattern[i]:
						mid_vector = dot(mid_vector, Rs[i])
					else:
						mid_vector = dot(mid_vector, Ns[i])
					if i in independent_intervals:
						mid_vector = dot(mid_vector, Q)

				prob = sum(mid_vector)
				if prob < 0:
					prob = 0
				if self.chromosome.inversion or len(loci_to_remove)>0:
					pattern = reduced_pattern
				if pattern in patterns:
					patterns[pattern] += prob
				else:
					patterns[pattern] = prob
				if pattern in self.chromosome.investigation.data_dict:
					if self.homokaryotype and (not self.chromosome.has_centromere):
						self.pattern_probabilities_array.append(prob)
						inversion_pattern_probabilities_array[pattern_order.index(pattern)] += prob
					else:
						inversion_pattern_probabilities_array[pattern_order.index(pattern)] += prob
		if self.chromosome.inversion:
			unbalanced = self.calculate_sterility2()
		else:
			unbalanced = 0
		if 'unbalanced' in self.chromosome.investigation.data_dict and self.chromosome.investigation.data_dict["unbalanced"] != 0:
			
			inversion_pattern_probabilities_array[pattern_order.index('unbalanced')] = unbalanced

		self.recombination_patterns = patterns
		if self.homokaryotype and (not self.chromosome.has_centromere):
			self.pattern_probabilities_array = numpy.array(inversion_pattern_probabilities_array)
		else:
			self.pattern_probabilities_array = numpy.array(inversion_pattern_probabilities_array)
		self.all_pattern_probabilities_array = numpy.array(self.all_pattern_probabilities_array)
		self.patterns_dict = patterns
		self.unbalanced = unbalanced
		
		if self.chromosome.inversion:
			s = sum(self.pattern_probabilities_array)
			if not "unbalanced" in self.chromosome.investigation.data_dict:
				self.pattern_probabilities_array = (self.pattern_probabilities_array*(s+unbalanced))/s
			
	
	def calculate_tetrad_pattern_probabilities(self):
		self.generate_tetrad_matrices()
		Fs = self.Fs
		Ts = self.Ts
		Ns = self.Ns
		dot = numpy.dot
		patterns = {}
		pattern_order = self.chromosome.investigation.pattern_order
		all_patterns_order = self.chromosome.investigation.all_patterns_order
		pattern_probabilities = [0.0 for i in range(len(pattern_order))]
		all_pattern_probabilities = [0.0 for i in range(len(all_patterns_order))]
		independent_intervals = self.chromosome.breakpoint_intervals
		if self.chromosome.pi_vector is None:
			self.chromosome.generate_pi_vector()
		pi_vector = self.chromosome.pi_vector # stationary phase distribution
		remove_centromere = False
		calculator = self.chromosome.calculator
		if self.chromosome.has_centromere:
			zot = [[0,1,2] for i in range(self.intervals_n)]
			iterator = list(itertools.product(*zot))
			remove_centromere = True
			centromere_locus = self.chromosome.centromere_index
			self.generate_Q()
			Q = self.chromosome.Q
		else:
			iterator = pattern_order
		for p in range(len(iterator)):
			pattern = iterator[p]
			skip = False
			if remove_centromere:
				reduced_pattern = copy.copy(pattern)
				new_patterns = calculator.remove_tetrad_centromere(pattern, centromere_locus)
				if not any(x in new_patterns for x in pattern_order):
					skip = True
			if not skip:
				mid_vector = pi_vector
				for i in range(len(pattern)):
					if pattern[i] == 0:
						mid_vector = dot(mid_vector, Fs[i])
					elif pattern[i] == 1:
						mid_vector = dot(mid_vector, Ts[i])
					elif pattern[i] == 2:
						mid_vector = dot(mid_vector, Ns[i])
					if i in independent_intervals:
						mid_vector = dot(mid_vector, Q)
				prob = sum(mid_vector)
				if remove_centromere:
					for pt in new_patterns:
						if pt in pattern_order:
							pattern_probabilities[pattern_order.index(pt)] += prob*new_patterns[pt]
				else:
					pattern_probabilities[p] = prob
		
		self.pattern_probabilities_array = numpy.array(pattern_probabilities)
		


	def generate_tetrad_matrices(self):
		self.generate_matrices(include_P = False)
		truncate_D = self.new_truncate_D
		Ds = self.Ds
		Fs = []
		Ts = []
		Ns = []
		for i in range(self.intervals_n):
			F = Ds[i][0]
			T = Ds[i][1]
			N = 0
			for s in range(2, truncate_D[i]):
				F += (1.0/3)*(0.5 + (-0.5)**s)*Ds[i][s]
				T += (2.0/3)*(1-(-0.5)**s)*Ds[i][s]
				N += (1.0/3)*(0.5+(-0.5)**s)*Ds[i][s]
			Fs.append(F)
			Ts.append(T)
			Ns.append(N)
		self.Fs = Fs
		self.Ts = Ts
		self.Ns = Ns

	def generate_sterility_matrices(self):
		Ds = [[],[]]
		Ps = [[],[]]
		transition_matrices = [self.Th, self.Tp]
		lambda_values = [self.lambda_inv, self.lambda_prox]
		mu_values = [self.mu_inv, self.mu_prox]
		if self.chromosome.g_values is None:
			self.chromosome.generate_g_values(00)
		self.g_values = self.chromosome.g_values
		g_values = self.chromosome.g_values
		if self.chromosome.pi_vector is None:
			self.chromosome.generate_pi_vector()
		pi_vector = self.chromosome.pi_vector
		error = self.chromosome.investigation.error2
		matrix_power = self.chromosome.calculator.matrix_power
		sterility_truncate = []
		for i in range(2):
			s = 0.0
			x = 0
			while True:
				while g_values.shape[0] < x*(self.chromosome.m+1): # extend g values matrix if needed.
					self.chromosome.extend_g_values(10)
					g_values = self.chromosome.g_values
				D = numba_generate_D(self.chromosome.m, x, lambda_values[i], mu_values[i], numpy.array(self.chromosome.gamma_values, numpy.float64), g_values)
				Ds[i].append(D)
				P1 = matrix_power(transition_matrices[i], x)
				Ps[i].append(P)
				s += sum(pi_vector.dot(D))
				if (s > 1-error) and x > 0:
					sterility_truncate.append(x+1)
					break
				x += 1
		self.sterility_truncate = sterility_truncate
		self.Ds_sterility = Ds
		self.Ps_sterility = Ps

	def generate_matrices(self, include_P = True):
		'''
		Generates and stores matrices D(x) and P^x for each interval.
		'''
		new_truncate = []
		max_x = self.chromosome.investigation.max_x
		min_x = self.chromosome.investigation.min_x
		lambda_values = self.lambda_values
		mu_values = self.mu_values
		if self.chromosome.g_values is None:
			self.chromosome.generate_g_values(100)
		self.g_values = self.chromosome.g_values
		g_values = self.chromosome.g_values
		if self.chromosome.pi_vector is None:
			self.chromosome.generate_pi_vector()
		pi_vector = self.chromosome.pi_vector
		matrix_power = self.chromosome.calculator.matrix_power
		mega_D = []
		mega_P = []
		error = self.chromosome.investigation.error2 # error is defined so that the recombination patterns sum to approximately 1-error.
		Ds = [[] for i in range(self.intervals_n)]
		if include_P:
			Ps = [[] for i in range(self.intervals_n)]
		for i in range(self.intervals_n):
			s = 0.0
			x = 0
			while True:
				while g_values.shape[0] < x*(self.chromosome.m+1): # extend g values matrix if needed.
					self.chromosome.extend_g_values(10)
					g_values = self.chromosome.g_values
				D = numba_generate_D(self.chromosome.m, x, lambda_values[i], mu_values[i], numpy.array(self.chromosome.gamma_values, numpy.float64), g_values)
				Ds[i].append(D)
				if include_P:
					if x == 0:
						P =  numpy.identity(self.chromosome.transition_matrices[i].shape[0])
					else:
						P = numpy.dot(Ps[i][x-1], self.chromosome.transition_matrices[i])
					Ps[i].append(P)
				s += sum(pi_vector.dot(D))
								
				if s > 1-error and x > 0 and x >= min_x:
					new_truncate.append(x+1)
					break
				x += 1
		self.new_truncate_D = new_truncate
		self.Ds = Ds
		self.mega_D = numpy.array(mega_D)
		if include_P:
			self.Ps = Ps
			self.mega_P = numpy.array(mega_P)
	
	
	def generate_D(self, x, lambda_value, mu_value, g_values):
		'''
		Generates and returns a single D(x) matrix as defined in the thesis.
		
		Arguments:
		x: (int) the number of chiasma events
		lambda_value/mu_value/g_value: see class Chromosome
		'''
		m = self.chromosome.m
		matrix = numpy.zeros((m+1,m+1), numpy.float64)
		if lambda_value+mu_value != 0:
			p1 = mu_value/(lambda_value+mu_value)
			p2 = lambda_value/(lambda_value+mu_value)
		else:
			p1 = 0.0
			p2 = 0.0
		comb = scipy.special.comb
		gamma_values = self.chromosome.gamma_values
		if x == 0:
			for i in range(m+1):
				for j in range(m+1):
					if i>=j:
						matrix[i][j] = poisson(lambda_value, i-j)*exp(-mu_value)
		else:
			for i in range(m+1):
				for j in range(m+1):
					s = 0.0
					for l in range(x):
						for n in range(x-l-1, (x-l-1)*(m+1)+1):
							for q in range(j, m+1):
								h = i+1+l+n+q-j
								s += g_values[n][x-l-1]*gamma_values[q]*( (exp(-(lambda_value+mu_value))*(lambda_value+mu_value)**h)/factorial(h) ) * comb(h,l)*(p1**l)*(p2**(h-l))
					if i>=j:
						s += ( (exp(-mu_value)*mu_value**x)/factorial(x) ) * ( (exp(-lambda_value)*lambda_value ** (i-j) )/factorial(i-j) )
					matrix[i][j] = s
		
		return matrix
		
	def is_balanced(self, state):
		'''
		Returns True if state is balanced.
		'''
		return sum(state[self.left_intervals_n:self.left_intervals_n + self.inversion_intervals_n])%2 == 0
		
		
	def generate_statespace(self):
		'''
		Generates the statespace as defined in the thesis.
		'''
		distal_intervals_n = self.left_intervals_n
		inversion_intervals_n = self.inversion_intervals_n
		proximal_intervals_n = self.proximal_intervals_n
		opposite_intervals_n = self.right_intervals_n
		intervals_n = self.intervals_n
		is_balanced = self.is_balanced
		
		rec_mid = []
		statespace = []
		if inversion_intervals_n == 0:
			for i in range(distal_intervals_n):
				rec_mid.append([False,True])
			for r in itertools.product(*rec_mid):
				state = list(r)
				state_copy = copy.copy(state)
				state_copy.append(0)
				statespace.append(state_copy)
		else:	
			for i in range(distal_intervals_n + inversion_intervals_n + proximal_intervals_n + opposite_intervals_n):
				rec_mid.append([False, True])
			for r in itertools.product(*rec_mid): # loops over all possible recombination patterns, r
				state = list(r)
				for j in range(4): # j indicate the tetrad configuration as follows. 0: no bridge; 1: single a1 bridge; 2: single a2 bridge; 3: double bridge
					if j == 0: # no bridge
						if is_balanced(state): # only balanced chromatids can be in configuration no bridge
							state_copy = copy.copy(state)
							state_copy.append(j)
							statespace.append(state_copy)
					elif j == 1 or j == 2: # single a1 or a2 bridge
						state_copy = copy.copy(state)
						if not is_balanced(state):
							for k in range(distal_intervals_n + inversion_intervals_n, intervals_n): # ignore unbalanced chromatids with recombination in intervals to the right of the inversion
								state_copy[k] = False
						state_copy.append(j)
						if state_copy not in statespace:
							statespace.append(state_copy)

					else: # double bridge
						state_copy = copy.copy(state)
						if not is_balanced(state): # only unbalanced chromatids can be in configuration double bridge
							for k in range(distal_intervals_n + inversion_intervals_n, intervals_n):
								state_copy[k] = False
							state_copy.append(j)
							if state_copy not in statespace:
								statespace.append(state_copy)

	
		self.chromosome.statespace = statespace
	
	def generate_transition_matrices(self):
		'''
		Generates matrix P for each interval, as defined and discussed in the thesis.
		'''
		distal_intervals_n = self.left_intervals_n
		inversion_intervals_n = self.inversion_intervals_n
		proximal_intervals_n = self.proximal_intervals_n
		opposite_intervals_n = self.right_intervals_n
		intervals_n = self.intervals_n
		is_balanced = self.is_balanced
		statespace = self.chromosome.statespace
		matrices = []
		for i in range(distal_intervals_n): # Distal matrices
			matrix = numpy.zeros((len(statespace), len(statespace)))
			for state_from in statespace:
				if sum(state_from[i+1:intervals_n]) == 0 and state_from[-1] == 0: # ignore states that show recombination in more proximal intervals
					from_index = statespace.index(state_from)
				
					state_to1 = copy.copy(state_from)
					state_to2 = copy.copy(state_from)	
					
					state_to1[i] = True
					state_to2[i] = False
					
					to_index1 = statespace.index(state_to1)
					to_index2 = statespace.index(state_to2)
					
					matrix[from_index][to_index1] = 0.5
					matrix[from_index][to_index2] = 0.5
		
			matrices.append(matrix)
			
		for i in range(distal_intervals_n ,distal_intervals_n + inversion_intervals_n): # Inversion matrices
			matrix = numpy.zeros((len(statespace),len(statespace)))
			for state_from in statespace:
				if sum(state_from[i+1:intervals_n]) == 0: # ignore states that show recombination in more proximal intervals
					if state_from[-1] == 0: # no bridge
						state_to1 = copy.copy(state_from)
						state_to2 = copy.copy(state_from)
						state_to1[i] = True
						state_to2[i] = False
						state_to1[-1] = 1
						state_to2[-1] = 1
						from_index = statespace.index(state_from)
						to_index1 = statespace.index(state_to1)
						to_index2 = statespace.index(state_to2)
						matrix[from_index][to_index1] = 0.5
						matrix[from_index][to_index2] = 0.5
					elif state_from[-1] == 1: # single a1 bridge
						state_to1 = copy.copy(state_from)
						state_to2 = copy.copy(state_from)
						state_to3 = copy.copy(state_from)
						state_to4 = copy.copy(state_from)
			
						if is_balanced(state_from): #balanced
							if state_from[i] == False:
								state_to1[i] = False
								state_to1[-1] = 0
				
								state_to2[i] = True
								state_to2[-1] = 1
				
								state_to3[i] = False
								state_to3[-1] = 1
				
								state_to4[i] = True
								state_to4[-1] = 3
							else:
								state_to1[i] = True
								state_to1[-1] = 0
				
								state_to2[i] = True
								state_to2[-1] = 1
				
								state_to3[i] = False
								state_to3[-1] = 1
				
								state_to4[i] = False
								state_to4[-1] = 3
					
						else: #unbalanced
							if state_from[i] == False:
								state_to1[i] = True
								state_to1[-1] = 0
					
								state_to2[i] = True
								state_to2[-1] = 1
					
								state_to3[i] = False
								state_to3[-1] = 1
					
								state_to4[i] = False
								state_to4[-1] = 3
					
							else:
								state_to1[i] = False
								state_to1[-1] = 0
					
								state_to2[i] = False
								state_to2[-1] = 1
					
								state_to3[i] = True
								state_to3[-1] = 1
					
								state_to4[i] = True
								state_to4[-1] = 3
			
						from_index = statespace.index(state_from)
						to_index1 = statespace.index(state_to1)
						to_index2 = statespace.index(state_to2)
						to_index3 = statespace.index(state_to3)
						to_index4 = statespace.index(state_to4)
			
						matrix[from_index][to_index1] = 0.25
						matrix[from_index][to_index2] = 0.25
						matrix[from_index][to_index3] = 0.25
						matrix[from_index][to_index4] = 0.25
			
			
					elif state_from[-1] == 3: # double bridge
						state_to1 = copy.copy(state_from)
						state_to2 = copy.copy(state_from)
						state_to1[i] = True #5
						state_to1[-1] = 1
						state_to2[i] = False #5
						state_to2[-1] = 1
			
						from_index = statespace.index(state_from)
			
						to_index1 = statespace.index(state_to1)
						to_index2 = statespace.index(state_to2)
			
						matrix[from_index][to_index1] = 0.5
						matrix[from_index][to_index2] = 0.5
					
			matrices.append(matrix)
			
		for i in range(distal_intervals_n + inversion_intervals_n, distal_intervals_n + inversion_intervals_n + proximal_intervals_n): # proximal matrices
			matrix = numpy.zeros((len(statespace),len(statespace)))
			for state_from in statespace:
				from_index = statespace.index(state_from)
				if sum(state_from[i+1:intervals_n]) == 0: # ignore states that show recombination in more proximal intervals
					if state_from[-1] == 0: # no bridge
						state_to1 = copy.copy(state_from)
						state_to2 = copy.copy(state_from)
						state_to1[i] = True #1
						state_to2[i] = False #1
				
						to_index1 = statespace.index(state_to1)
						to_index2 = statespace.index(state_to2)
				
						matrix[from_index][to_index1] = 0.5
						matrix[from_index][to_index2] = 0.5
			
					elif state_from[-1] == 1: #single a1 bridge
						if is_balanced(state_from): # balanced
							state_to1 = copy.copy(state_from)
							state_to2 = copy.copy(state_from)
							state_to3 = copy.copy(state_from)
							state_to4 = copy.copy(state_from)
							if state_from[i] == False:
						
								state_to1[i] = True
								state_to1[-1] = 2
						
								state_to2[i] = True
								state_to2[-1] = 1
						
								state_to3[i] = False
								state_to3[-1] = 1
						
								state_to4[i] = False
								state_to4[-1] = 2
						
							else:
						
								state_to1[i] = False
								state_to1[-1] = 2
						
								state_to2[i] = False
								state_to2[-1] = 1
						
								state_to3[i] = True
								state_to3[-1] = 1
						
								state_to4[i] = True
								state_to4[-1] = 2
					
							to_index1 = statespace.index(state_to1)
							to_index2 = statespace.index(state_to2)
							to_index3 = statespace.index(state_to3)
							to_index4 = statespace.index(state_to4)
				
							matrix[from_index][to_index1] = 0.25
							matrix[from_index][to_index2] = 0.25
							matrix[from_index][to_index3] = 0.25
							matrix[from_index][to_index4] = 0.25
						
						else: #unbalanced
						
							state_to1 = copy.copy(state_from)
							state_to2 = copy.copy(state_from)
					
							# ignore recombination in the proximal region, as states will be unbalanced anyway
							state_to1[-1] = 1 #2
							state_to2[-1] = 2 #3

					
							to_index1 = statespace.index(state_to1)
							to_index2 = statespace.index(state_to2)
					
							matrix[from_index][to_index1] = 0.5 
							matrix[from_index][to_index2] = 0.5
						
				
					elif state_from[-1] == 2: # single a2 bridge
						if is_balanced(state_from):
							state_to1 = copy.copy(state_from)
							state_to2 = copy.copy(state_from)
							if state_from[i] == False:
								state_to1[i] = True
								state_to1[-1] = 1
						
								state_to2[i] = False
								state_to2[-1] = 1
					
							else:
								state_to1[i] = False
								state_to1[-1] = 1
						
								state_to2[i] = True
								state_to2[-1] = 1
					
							to_index1 = statespace.index(state_to1)
							to_index2 = statespace.index(state_to2)
				
							matrix[from_index][to_index1] = 0.5
							matrix[from_index][to_index2] = 0.5
					
					
					
						else: #unbalanced
							state_to = copy.copy(state_from)
							state_to[-1] = 1
					
							to_index = statespace.index(state_to)
							matrix[from_index][to_index] = 1.0 # unbalanced gamete no matter what happens in the proximal region.
					
							
					else: # double bridge
						state_to = copy.copy(state_from)
						to_index = statespace.index(state_to)
						matrix[from_index][to_index] = 1.0 # unbalanced gamete no matter what happens in the proximal region.
			matrices.append(matrix)
			
		for i in range(distal_intervals_n + inversion_intervals_n + proximal_intervals_n, intervals_n): # opposite arm matrices
			matrix = numpy.zeros((len(statespace),len(statespace)))
			for state_from in statespace:
				if sum(state_from[i+1:intervals_n]) == 0: # ignore states that show recombination in intervals to the right of current interval
				
					if is_balanced(state_from):
						state_to1 = copy.copy(state_from)
						state_to2 = copy.copy(state_from)
					
						state_to1[i] = False
						state_to2[i] = True
					
						from_index = statespace.index(state_from)
						to_index1 = statespace.index(state_to1)
						to_index2 = statespace.index(state_to2)
					
						matrix[from_index][to_index1] = 0.5
						matrix[from_index][to_index2] = 0.5
						
						
						
					else: #unbalanced
						state_to = copy.copy(state_from)
						from_index = statespace.index(state_from)
						to_index = statespace.index(state_to)
						matrix[from_index][to_index] = 1.0 # unbalanced gamete no matter what happens in the opposite arm.
						
		
			matrices.append(matrix)
		
		self.chromosome.transition_matrices = matrices
		
			

	
	def generate_Q(self):
		'''
		Generates and stores the Q matrix, as defined in the thesis.
		'''
		Q = []
		pi_vector = list(self.chromosome.pi_vector)
		for i in range(len(pi_vector)):
			Q.append(pi_vector)
		Q = numpy.array(Q)
		self.Q = Q
		self.chromosome.Q = Q
	
	def generate_RNs(self):
		'''
		Generates and stores matrices R and N for each interval. These are the M matrices as defined in the thesis, for the two conditions (recombination and non-recombination)
		'''
		Rs = []
		Ns = []
		for i in range(len(self.GHs)):
			Rs.append(0.5*self.GHs[i][0])
			Ns.append(0.5*self.GHs[i][0]+self.GHs[i][1])
		self.Ns = Ns
		self.Rs = Rs
		
	def generate_GHs(self):
		'''
		Generates and store matrices G and H for each interval, as defined in the thesis.
		'''
		GHs = []
		error = self.chromosome.investigation.error1 # error is defined so that the recombination patterns sum to approximate 1-error when the infinite series form of matrix G is used.
		intervals_n = self.chromosome.intervals_n
		closed_form = self.chromosome.investigation.closed_form # True if the closed form of matrix G (theorem 2 in the thesis) is to be used.
		for k in range(intervals_n):
			lambda_value = self.lambda_values[k]
			mu_value = float(self.mu_values[k])
			gamma_values = self.chromosome.gamma_values
			m = self.chromosome.m
			if len(self.chromosome.b_values) == 1 and not closed_form:
					self.chromosome.extend_b_values(100)
			b_values = self.chromosome.b_values
			G = numpy.zeros((m+1,m+1))
			H = numpy.zeros((m+1,m+1))
			f = self.chromosome.calculator.f # function f in theorem 2 in the thesis.
			if gamma_values[m] == 1.0 and closed_form: # use theorem 2
				for i in range(m+1):
					for j in range(m+1):
						if i>=j:
							pois_term = poisson(lambda_value, i-j)
							psi = exp(-lambda_value)*f(i-j, m+1, lambda_value)-pois_term
						else:
							pois_term = 0.0
							psi = exp(-lambda_value)*f(m+1+i-j, m+1, lambda_value)
						G[i][j] = exp(-mu_value)*psi+(1-exp(-mu_value))*(psi+pois_term)
						H[i][j] = pois_term*exp(-mu_value)
			else:
				if len(self.chromosome.b_values) == 1:
					self.chromosome.extend_b_values(100)
				b_values = self.chromosome.b_values
				for i in range(m+1):
					for j in range(m+1):
						psi = 0.0
						n = 0
						check = 0.0
						while check < 1-error:
							while len(b_values) <= n:
								self.chromosome.extend_b_values()
								b_values = self.chromosome.b_values

							for q in range(j, m+1):
								try:
									pois_factor = poisson(lambda_value, i+1+n+q-j)
								except OverflowError:
									print('Error! error1 is too small! Please choose a higher value for parameter error1 under heading # population')
									sys.exit(1)
								add = b_values[n]*gamma_values[q]*pois_factor
								psi += add
							check += poisson(lambda_value, n)
							n += 1
															
						if i>=j:
							pois_term = poisson(lambda_value, i-j)
						else:
							pois_term = 0.0
						G[i][j] = exp(-mu_value)*psi+(1-exp(-mu_value))*(psi+pois_term)
						H[i][j] = pois_term*exp(-mu_value)
			GHs.append([G,H])
	
		self.GHs = GHs
	
	

			
class Chromosome_diplotype:
	'''
	A class for storing and manipulating information about a single chromosome diplotype (two chromosome haplotypes)
	
	Key attributes:
	self.haplotypes: A list of the two Chromosome_haplotype instances, sorted according to the Chromosome_haplotype function __lt__.
	self.karyotypes: A List of instances of class Karyotype.
	self.original_homokaryotype: (bool) True if the diplotype is homozygous for the ancestral arrangement.
	self.reversed_homokaryotype: (bool) True if the diplotype is homozygous for the derived arrangement.
	self.heterokaryotype: (bool) True if the diplotype is an inversion heterokaryotype
	self.chromosome: Pointer to the chromosome
	self.male: (bool) None if self.chromosome is not the sex chromosome. True if self.chromosome is the sex chromosome and the diplotype is male. False if
		self.chromosome is the sex chromosome and the diplotype is female.
	self.gametes: A dictionary of gamete frequencies. A gamete is here represented as a string of allele keys (integers), where the character at index i gives the allele key for locus with index i
		(as given by the ordering of the loci in the input file). E.g. if for the given chromosome, the input file reads 'Loci = ABC', then the gamete '001' indicate alleles A0, B0, and C1.
		Note that the ordering of the loci in the gametes is always the same as in the input file, even for derived homokaryotypes.
	
	
	
	'''
	def __init__(self, haplotypes, chromosome, karyotypes, original_homokaryotype = False, reversed_homokaryotype = False, heterokaryotype = False, male = None):
		self.haplotypes = haplotypes
		self.chromosome = chromosome
		self.karyotypes = karyotypes
		self.original_homokaryotype = original_homokaryotype
		self.reversed_homokaryotype = reversed_homokaryotype
		self.heterokaryotype = heterokaryotype
		self.male = male
		self.gametes = []
		self.male_indices = []
		self.female_indices = []
		self.nonrecombinant_gametes = None
	
	def add_index(self, index, male):
		if male:
			self.male_indices.append(index)
		else:
			self.female_indices.append(index)
		for haplotype in self.haplotypes:
			haplotype.add_index(index, male)
		
	def generate_nonrecombinant_gametes(self):
		'''
		When there is no recombination, simply copy the chromosome haplotypes to make gametes.
		'''
		self.nonrecombinant_gametes = {}
		if self.haplotypes[0].allele_keys == self.haplotypes[1].allele_keys:
			self.nonrecombinant_gametes[str(self.haplotypes[0].allele_keys).replace('(', '').replace(')','').replace(',','').replace(' ','')] = 1.0
		else:
			for haplotype in self.haplotypes:
				self.nonrecombinant_gametes[str(haplotype.allele_keys).replace('(', '').replace(')','').replace(',','').replace(' ','')] = 0.5
		
	
	def set_initial_frequencies(self, initial_frequencies):
		self.initial_frequencies = initial_frequencies
	
	def make_key(self):
		key = ''
		for i in range(len(self.chromosome.loci)):
			l = self.chromosome.loci[i]
			key += l.key
			key += str(self.haplotypes[0].allele_keys[i])
			key += str(self.haplotypes[1].allele_keys[i])
		self.key = key
		return key
	
	def print_initial_frequencies(self):
		print(self.key,':', self.initial_frequencies)
	
	
	def calculate_gamete_frequencies(self):
		'''
		Translates the recombination patterns (calculated in class Karyotype) into gametes.
		'''
		allele_keys_0 = str(self.haplotypes[0].allele_keys).replace(' ','').replace(',','').replace('(','').replace(')','').replace('\'','') # convert from tuple to string
		allele_keys_1 = str(self.haplotypes[1].allele_keys).replace(' ','').replace(',','').replace('(','').replace(')','').replace('\'','')
		reverse = self.chromosome.calculator.reverse
		self.gametes = []
		for karyotype in self.karyotypes: 
			'''
			Loop over karyotypes. For homokaryotypes, self.karyotypes only have a single element. For heterokaryotypes, it has two, 
			corresponding to the two possible directions of the interference signal across inversion breakpoint boundaries (see figure 2.3 in the thesis). 
			If there is no interference across breakpoint boundaries (alpha = 0), these karyotypes give identical results.
			
			'''
			if not (karyotype.original or karyotype.heteromorphic): # reverse loci order in the inverted region if karyotype has the derived arrangement (reversed back again later)
				l = self.chromosome.left_breakpoint_index
				r = self.chromosome.right_breakpoint_index
				allele_keys_0 = reverse(allele_keys_0, l, r)
				allele_keys_1 = reverse(allele_keys_1, l, r)
			gametes = {}
			for pattern, frequency in karyotype.recombination_patterns.items():
				if pattern == 'unbalanced':
					if frequency != 0:
						gametes['unbalanced'] = frequency
				else:
					gamete_0 = allele_keys_0[0]
					gamete_1 = allele_keys_1[0]
					new_gamete_0 = ''
					new_gamete_1 = ''
					for i in range(len(pattern)):
						# build up gametes by appending from the appropriate homologue
						if pattern[i]: # pattern[i] is True if the pattern shows recombination in interval i, and False if it does not.
							new_gamete_0 = gamete_1 + allele_keys_0[i+1]
							new_gamete_1 = gamete_0 + allele_keys_1[i+1]
						else:
							new_gamete_0 = gamete_0 + allele_keys_0[i+1]
							new_gamete_1 = gamete_1 + allele_keys_1[i+1]
					
						gamete_0 = new_gamete_0
						gamete_1 = new_gamete_1
					if not (karyotype.original or karyotype.heteromorphic): # reverse back
						gamete_0 = reverse(gamete_0, l, r)
						gamete_1 = reverse(gamete_1, l, r)
					frequency = frequency/2.0 # The two gametes are equally likely
					if frequency > 0:
						if gamete_0 in gametes:
							gametes[gamete_0] += frequency
						else:
							gametes[gamete_0] = frequency
						if gamete_1 in gametes:
							gametes[gamete_1] += frequency
						else:
							gametes[gamete_1] = frequency
						
			
			self.gametes.append(gametes)
		if len(self.gametes) == 1:
			self.gametes = self.gametes[0]
		else:
			g = {}
			for key in self.gametes[0]:
				g[key] = (self.gametes[0][key]+self.gametes[1][key])/2.0
			self.gametes = g


				
			
class Chromosome_haplotype:
	'''
	A class for storing and manipulation information relating to a single chromosome haplotype.
	
	Key attributes:
	self.allele_keys: The tuple of allele keys (integers; sorted by locus index).
	self.chromosome: A pointer to the Chromosome instance.
	self.locus_allele_keys: A list of locus_allele_keys, where a single such key consist of a locus key (single character) and an allele key (single digit)
	self.key: (str) A string of concatenated locus_allele_keys for each locus in the order they appear on the chromsome. E.g. the key A0B0C1 indicate
		alleles A0, B0 and C1.

	
	'''
	def __init__(self, allele_keys, chromosome):
		self.allele_keys = allele_keys
		self.chromosome = chromosome
		self.male_indices = []
		self.female_indices = []
		self.genotype_indices = []
		self.make_keys()
	
	def make_keys(self):
		self.locus_allele_keys = []
		self.key = ''
		loci_keys = self.chromosome.loci_keys
		for i in range(len(loci_keys)):
			locus_allele_key = loci_keys[i]+str(self.allele_keys[i])
			self.key += locus_allele_key
			self.locus_allele_keys.append(locus_allele_key)
	
	def add_index(self, index, male):
		if male:
			self.male_indices.append(index)
		else:
			self.female_indices.append(index)
		for allele in self.alleles:
			allele.add_index(index, male)
	
	def find_alleles(self):
		alleles = []
		for i in range(len(self.allele_keys)):
			locus = self.chromosome.loci[i]
			allele_key = self.allele_keys[i]
			if allele_key == '-':
				allele = locus.alleles[-1]
			else:
				allele = locus.alleles[allele_key]
			alleles.append(allele)
		self.alleles = alleles
		
	def find_initial_frequencies(self):
		initial_frequencies = [1.0 for i in range(self.chromosome.investigation.demes_n)]
		for d in range(len(initial_frequencies)):
			for allele in self.alleles:
				initial_frequencies[d] *= allele.initial_frequencies[d] # Hardy-Weinberg
		self.initial_frequencies = initial_frequencies
	
	def __lt__(self, other):
		'''
		A function used for sorting Allele instances. Returns True if self is smaller than other. For two autosomal or two macrolinked haplotypes, the smaller instance
		is the one with the smaller tuple of allele keys when these are read as binary numbers. For one macrolinked and one microlinked haplotype, the latter is the smaller.
		'''

		return self.allele_keys<other.allele_keys
		

class Locus:
	'''
	A class for storing and manipulating information about a single locus.
	
	Key attributes:
	self.chromsome: The chromosome instance
	self.key: (str) A unique single character identifying the locus
	self.alleles_n: (int) The number of alleles of the locus
	self.alleles: The list of Allele instances.
	self.allele_keys: The list of allele keys (integers).
	self.macrolinked: (bool) True if the locus is linked to the macro sex chromosome (X/Z).
	self.microlinked: (bool) True if the locus is linked to the micro sex chromosome (Y/W).
	self.sex: (bool) True if the locus is the sex determination locus.
	self.breakpoint: (bool) True if the locus is an inversion breakpoint
	self.chromosome_index: (int) The index of the locus among the loci on the same chromosome.
	
	'''
	
	
	def __init__(self, chromosome, key, allele_n, macrolinked, microlinked, population, chromosome_index, breakpoint):
		self.chromosome = chromosome
		self.key = key
		self.allele_n = allele_n
		self.alleles = []
		self.macrolinked = macrolinked
		self.microlinked = microlinked
		self.mitochondria = False # This function is not currently in use. In future versions of the program, I will allow for mitochondrial inheritance
		if self.key == '$':
			self.sex = True
			self.allele_n = 2
		else:
			self.sex = False
		self.population = population
		self.chromosome_index = chromosome_index 
		self.breakpoint = breakpoint
		self.allele_keys = []
	
	def initialize_alleles(self):
		'''
		Create and store the appropriate number of Allele instances
		'''
		for i in range(self.allele_n):
			allele = Allele(self, i, self.population) # new Allele instance
			self.alleles.append(allele)
			self.population.alleles.append(allele)
			self.allele_keys.append(i)
			self.population.alleles_dictionary[allele.key] = allele
		if self.microlinked or self.macrolinked:
			allele = Allele(self, '-', self.population) # If the allele is sex-linked, make a dummy allele that serves as a placeholder for the non-existing homologue.
			self.alleles.append(allele)
			self.population.alleles.append(allele)
			self.allele_keys.append(allele.allele_key)
			self.population.alleles_dictionary[allele.key] = allele
			
		
class Allele:
	'''
	A class for storing and manipulating information relating to a single allele.
	
	Key attributes:
	self.allele_key: (int) An integer identifying the allele among the other allels of the same locus.
	self.locus_key: (str) The single character identifying the locus of the allele.
	self.key: (str) The concatenation of the locus_key and the allele_key.
	self.locus: The Locus instance.
	self.initial_frequencies: The initial frequencies of the allele in each deme.
	self.male_indices/female_indices: The indices of male/female instances that has the allele. Homozygote indices are listed twice.
	'''
	def __init__(self, locus, allele_key):
		self.allele_key = allele_key
		self.locus_key = locus.key
		self.key = self.locus_key+str(self.allele_key)
		self.locus = locus
		self.initial_frequencies = []
		self.male_indices = []
		self.female_indices = []
	
	def add_index(self, index, male):
		if male:
			self.male_indices.append(index)
		else:
			self.female_indices.append(index)
	
	def set_initial_frequency(self, frequency):
		self.initial_frequencies.append(frequency)
	
	def find_frequencies(self, male_frequencies, female_frequencies, print_to_screen = True):
		'''
		Find and return the frequency of the allele in each deme.
		'''
		if self.locus.microlinked:
			factor = 1.0/0.25
		elif self.locus.macrolinked:
			factor = 1.0/0.75
		else:
			factor = 1.0
		f = factor*0.25*(numpy.sum(male_frequencies[:,self.male_indices], axis = 1) + numpy.sum(female_frequencies[:,self.female_indices], axis = 1))
		if print_to_screen:
			print(self.key)
			print(f)
		return f

class Calculator:
	'''
	A class with miscellaneous methods used for calculation
	'''
	def __init__(self):
		self.memory = {'f':{}}
	
	def reverse(self, keys, left_breakpoint, right_breakpoint):
		'''
		Returns loci keys with the order inside the inverted region reversd
		'''
		l = left_breakpoint
		r = right_breakpoint
		return keys[0:l+1]+keys[l+1:r][::-1]+keys[r:]

	def remove_tetrad_centromere(self, pattern, locus):
		t1 = pattern[locus-1]
		t2 = pattern[locus]
		t_new = [0.0, 0.0, 0.0]
		if t1 == 0:
			t_new[t2] = 1.0
		elif t2 == 0:
			t_new[t1] = 1.0
		elif (t1 == 1 and t2 == 2 ) or (t1 == 2 and t2 == 1):
			t_new[1] = 1
		elif (t1 == 2 and t2 == 2):
			t_new[0] = 1
		elif t1 == 1 and t2 == 1:
			t_new = [0.25, 0.5, 0.25]
		else:
			print("Error! Unknown pattern!")
			sys.exit(1)
		p1 = list(pattern[0: locus-1])
		p2 = list(pattern[locus+1:])
		new_patterns = {}
		for i in range(len(t_new)):
			if t_new[i] != 0:
				new_pattern = p1 + [i]  + p2
				new_patterns[tuple(new_pattern)] = t_new[i]
		return new_patterns

	def remove_loci(self, pattern, loci):
		loci.sort()
		loci.reverse()
		while(len(loci)>0):
			current_locus = loci[0]
			loci = loci[1:]
			if current_locus == len(pattern):
				output_pattern = pattern[0:-1]
			elif current_locus == 0:
				output_pattern = pattern[1:]
			else:
				s = pattern[current_locus-1] + pattern[current_locus]
				if s!= 1:
					output_pattern = pattern[0:current_locus-1] + [False] + pattern[current_locus+1:]
				else:
					output_pattern = pattern[0:current_locus-1] + [True] + pattern[current_locus+1:]
			pattern = copy.copy(output_pattern)
		return tuple(output_pattern)

	def remove_breakpoint_intervals(self, pattern, first_inversion_interval, last_inversion_interval):
		output_pattern = copy.copy(pattern)
		if(last_inversion_interval == len(pattern)-1):
			output_pattern = output_pattern[:-1]
		else:
			s = pattern[last_inversion_interval]+pattern[last_inversion_interval+1]
			if s != 1:
				output_pattern = pattern[0:last_inversion_interval] + [False] + pattern[last_inversion_interval+2:]
			else:
				output_pattern = pattern[0:last_inversion_interval] + [True] + pattern[last_inversion_interval+2:]
		if first_inversion_interval == 0:
			output_pattern = output_pattern[1:]
		else:
			s = pattern[first_inversion_interval-1] + pattern[first_inversion_interval]
			if s!= 1:
				output_pattern = output_pattern[0:first_inversion_interval-1] + [False] + output_pattern[first_inversion_interval+1:]
			else:
				output_pattern = output_pattern[0:first_inversion_interval-1] + [True] + output_pattern[first_inversion_interval+1:]
		return tuple(output_pattern)

	def reverse_intervals(self, pattern, first_inversion_interval, last_inversion_interval):
		inv = pattern[first_inversion_interval:last_inversion_interval+1]
		inv.reverse()
		output_pattern = pattern[0:first_inversion_interval] + inv + pattern[last_inversion_interval + 1:]
		return tuple(output_pattern)

	
	def f(self, r, q, l):
		'''
		The function f as defined in theorem 2 in thesis
		'''
		if (r,q,l) in self.memory['f']:
			return self.memory['f'][(r,q,l)]
		else:
			result = 0.0
			for j in range(q):
				r = float(r)
				q = float(q)
				try:
					result += (exp( cos(2*pi*j/q)*l ) * cos( sin(2*pi*j/q)*l - (2*pi*r*j/q) ))
				except:
					print("Error!")
					print("j", j)
					print("q", q)
					print("r", r)
					print("l", l)
					sys.exit(1)
			result *= (1.0/q)
			self.memory['f'][(r,q,l)] = result
			return result
	
	def w(self, c, l, m):
		'''
		Not currently used
		'''
		if c<(m+1):
			return self.poisson(l, c)
		else:
			return exp(-l)*self.f(c-(m+1), m+1, l)-self.poisson(l, c-(m+1))
		
	def matrix_power(self, matrix, power, memory_slot = None):
		'''
		Returns the matrix (argument) to the given power (argument). Matrix powers can be stored in memory for faster future computation
			by passing a hashable key to the argument memory_slot.
		'''
		if not memory_slot == None:
			if power == 0:
				identity = numpy.identity(matrix.shape[0])
				if memory_slot not in self.memory:
					self.memory[memory_slot] = {0: identity}
				return identity
			elif power == 1:
				if memory_slot not in self.memory:
					self.memory[memory_slot] = {1: matrix}
				return matrix
			else:
				if memory_slot in self.memory and power in self.memory[memory_slot]:
					return self.memory[memory_slot][power]
				if memory_slot in self.memory and power-1 in self.memory[memory_slot]:
					new_matrix = numpy.dot(self.memory[memory_slot][power-1], matrix)
					self.memory[memory_slot][power] = new_matrix
					return new_matrix
				else:
					old_matrix = matrix
					for i in range(power-1):
						new_matrix = numpy.dot(old_matrix, matrix)
						old_matrix = new_matrix
					if memory_slot in self.memory:
						self.memory[memory_slot][power] = new_matrix
					else:
						self.memory[memory_slot] = {power: new_matrix}
					return new_matrix
		
		else:
	
			if matrix.shape[0]!= matrix.shape[1]:
				print('Error!')
				return 0
			old_matrix = matrix
			if power == 0:
				identity = numpy.identity(matrix.shape[0])
				return identity
		
			elif power == 1:
				return matrix
			else:
				for i in range(power-1):
					new_matrix = numpy.dot(old_matrix, matrix)
					old_matrix = new_matrix
				return new_matrix

