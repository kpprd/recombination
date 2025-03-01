from ast import Param
from cProfile import label
from lib2to3.pgen2.token import STAR
from re import X
import stat
from turtle import title
import numba
import numpy
import random
from numba import jit, njit
import scipy
from scipy.optimize import minimize
from math import exp
import matplotlib.pyplot
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec


"""
A script for performing maximum likelihood analysis on sterility data
"""
class Inversion:
	def __init__(self, name, I, epsilon, observed):
		self.name = name
		self.I = I
		self.epsilon = epsilon
		self.observed = observed

inversions = []



# Data from Coyne et al. 1993
inversions.append(Inversion("273", 0.19, 0.63, -0.0702))
inversions.append(Inversion("238", 0.11, 0, -0.0518))
inversions.append(Inversion("281", 0.24, 0.8, -0.0372))
inversions.append(Inversion("LD31", 0.182, 0.9945, -0.0338))
inversions.append(Inversion("265", 0.144, 0.877, -0.0319))
inversions.append(Inversion("224", 0.085, 0.94, -0.0300))
inversions.append(Inversion("275", 0.201, 0.4975, -0.0263))
inversions.append(Inversion("277", 0.242, 0.991, -0.0258))
inversions.append(Inversion("280", 0.101, 0.99, -0.0258))
inversions.append(Inversion("C190", 0.2, 0.4, -0.0057))
inversions.append(Inversion("234", 0.21, 0.6666, -0.0051 ))
inversions.append(Inversion("260", 0.281, 0.996, 0.0248))
inversions.append(Inversion("LD12", 0.271, 0.996, 0.0251))
inversions.append(Inversion("252", 0.354, 0.79, 0.0469))
inversions.append(Inversion("270", 0.3, 0.933, 0.0512))
inversions.append(Inversion("Sep", 0.27, 0.9259, 0.0672))
inversions.append(Inversion("259", 0.44, 0.9318, 0.0758))
inversions.append(Inversion("267", 0.221, 0.9945, 0.0895))
inversions.append(Inversion("111", 0.835, 0.467, 0.1009))
inversions.append(Inversion("278", 0.48, 0.979, 0.1072))
inversions.append(Inversion("268", 0.395, 0.987, 0.1087))
inversions.append(Inversion("250", 0.57, 0.789, 0.1108))
inversions.append(Inversion("279", 0.58, 0.81, 0.1390))
inversions.append(Inversion("272", 0.485, 0.9381, 0.1622))
inversions.append(Inversion("282", 0.221, 0.9954, 0.1674))
inversions.append(Inversion("257", 0.359, 0, 0.1868))
inversions.append(Inversion("C269", 0.52, 0, 0.1996))
inversions.append(Inversion("LD3", 0.471, 0.9978, 0.1994))
inversions.append(Inversion("208", 0.5, 0.94, 0.2086))
inversions.append(Inversion("271", 0.471, 0.9978, 0.2762))

global_stationary = False # Indicate if you want to use the stationary distribution for the breakpoint (set to False to includue breakpoint interference)


@jit("float64(float64, int64)", nopython = True)
def poisson(l, x):
		p = exp(-l)
		for i in range(1, x+1):
			p = (p*l)/(i)
		return p

def generate_pi_vector(gamma_values):
	'''
	Generates the stationary distribution
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

def sterility_and_rec(I, epsilon, gamma_values_chiasma, gamma_values_breakpoint, stationary = global_stationary, sterility = True):
	s = 0
	m1 = len(gamma_values_chiasma) -1
	m2 = len(gamma_values_breakpoint) -1
	for q in range(m1+1):
		s += (q+1)*gamma_values_chiasma[q]
	M0 = I*epsilon
	M1 = I*(1-epsilon)
	z0 = 0
	z1 = 0
	l0 = 2*M0*s
	l1 = 2*M1*s
	if stationary:
		start_distribution = generate_pi_vector(gamma_values_chiasma)
	else:
		start_distribution = gamma_values_breakpoint
	for q in range(m2+1):
		for c in range(q+1):
			z0 += start_distribution[q]*poisson(l0, c)
			z1 += start_distribution[q]*poisson(l1, c)
	R0 = 0.5*(1-z0)
	R1 = 0.5*(1-z1)
	if sterility:
		return R0*(1-R1) + (1-R0)*R1
	else:
		return R0*R1

def squared_deviation(parameters):
	if len(parameters) == 2:
		q = parameters[0]
		d = parameters[1]
		k = int(q // 1)
		gamma_values =  [0 for i in range(k)]
		gamma_values.append(1-(q%1))
		gamma_values.append(q%1)
		gamma_values_chiasma = gamma_values
		gamma_values_breakpoint = gamma_values
	elif len(parameters) == 3:
		q1 = parameters[0]
		q2 = parameters[1]
		d = parameters[2]
		k1 = int(q1 // 1)
		k2 = int(q2 // 1)
		gamma_values_chiasma =  [0 for i in range(k1)]
		gamma_values_chiasma.append(1-(q1%1))
		gamma_values_chiasma.append(q1%1)
		gamma_values_breakpoint = [0 for i in range(k2)]
		gamma_values_breakpoint.append(1-(q2%1))
		gamma_values_breakpoint.append(q2%1)
	data = []
	predicted = []
	for inversion in inversions:
		data.append(inversion.observed)
		predicted.append(sterility_and_rec(d*inversion.I, inversion.epsilon, gamma_values_chiasma, gamma_values_breakpoint, stationary=global_stationary))
	data = numpy.array(data)
	squared_deviation = sum((data-predicted)**2)
	return squared_deviation


def analyse(gamma_values_chiasma, gamma_values_breakpoint, d = 1):
	for inversion in inversions:
		print("Inversion " + inversion.name + ": expected: "  + str(sterility_and_rec(d*inversion.I, inversion.epsilon, gamma_values_chiasma, gamma_values_breakpoint, global_stationary)) + " observed: " + str(inversion.observed))



def generate_map(max_I, gamma_values_chiasma, gamma_values_breakpoint, dimensions = 100, stationary = global_stationary):
	s = numpy.zeros((dimensions,dimensions))
	Is = numpy.linspace(0,max_I, dimensions)
	epsilons = numpy.linspace(0,1,dimensions)
	for i in range(dimensions):
		for j in range(dimensions):
			s[i][j] = sterility_and_rec(Is[i], epsilons[j], gamma_values_chiasma, gamma_values_breakpoint, stationary = stationary)
	matplotlib.pyplot.imshow(s, interpolation="nearest", origin="lower", norm = "linear", vmin=0, vmax=0.5)
	matplotlib.pyplot.colorbar(label = "Sterility")
	matplotlib.pyplot.ylabel("Inversion length")
	matplotlib.pyplot.xlabel("Centromere position")
	x = numpy.arange(0,1,0.01)
	nx = x.shape[0]
	no_labels = 6 
	step_x = int(nx / (no_labels - 1)) 
	x_positions = numpy.arange(0,nx,step_x) 
	x_labels = x[::step_x] 
	matplotlib.pyplot.xticks(x_positions, x_labels)
	matplotlib.pyplot.show()
	
def get_sterility_and_recombination_matrix(max_I, max_x,max_y, gamma_values, stationary = True, sterility = True):
	s = numpy.zeros((max_y,max_x))
	Is = numpy.linspace(0,max_I, max_y)
	epsilons = numpy.linspace(0,1,max_x)
	for i in range(max_y):
		for j in range(max_x):
			s[i][j] = sterility_and_rec(Is[i], epsilons[j], gamma_values, gamma_values, stationary = stationary, sterility = sterility)
	return s

def plot_1d(epsilon = 0.5, max_I = 0.8, res = 101, sterility = True):
	s0 = numpy.zeros(res)
	s1 = numpy.zeros(res)
	s2 = numpy.zeros(res)
	Is = numpy.linspace(0.0000001,max_I, res)
	for i in range(res):
		s0[i] = sterility_and_rec(Is[i], epsilon, [1], [1], stationary = True, sterility = sterility)
		s1[i] = sterility_and_rec(Is[i], epsilon, [0,0,0,1], [0,0,0,1], stationary= True, sterility= sterility)
		s2[i] = sterility_and_rec(Is[i], epsilon, [0,0,0,1], [0,0,0,1], stationary= False, sterility= sterility)
	
	plt.plot(Is, s0, linestyle='-', label='No interference')
	plt.plot(Is, s1, linestyle='--', label='C-C interference')
	plt.plot(Is, s2, linestyle='-.', label='C-C and B-C interference')

	plt.xlabel('I')
	if sterility:
		plt.ylabel('Sterility')
		plt.title('Sterility in pericentric inversion heterokaryotypes')
		file_name = "sterility_1d.png"
	else:
		plt.ylabel('Recombination rate')
		plt.title('Recombination in pericentric inversion heterokaryotypes')
		file_name = "recombination_1d.png"
	# plt.title('Plots of s0, s1, and s2')
	#plt.yscale('log')
	plt.legend()
	plt.savefig(file_name, dpi=300)
	plt.show()
		
		



def plot_all(sterility = True, nrm = "linear", cmp = "plasma"):
	import matplotlib.pyplot as plt
	import numpy as np
	from matplotlib.gridspec import GridSpec
	max_x = 101
	max_y = 101
	max_I = 0.81
	data1 = get_sterility_and_recombination_matrix(max_I, max_x, max_y, [1], True, sterility)
	data2 = get_sterility_and_recombination_matrix(max_I, max_x, max_y, [0,0,0,0,1], True, sterility)
	data3 = get_sterility_and_recombination_matrix(max_I, max_x, max_y, [0,0,0,0,1], False, sterility)

	# Create figure and a GridSpec layout
	fig = plt.figure(figsize=(16, 5))
	gs = GridSpec(1, 4, width_ratios=[1, 1, 1, 0.05])
	ax1 = fig.add_subplot(gs[0])
	ax2 = fig.add_subplot(gs[1])
	ax3 = fig.add_subplot(gs[2])
	cax = fig.add_subplot(gs[3])
	
	x = numpy.arange(0,1.01,0.01) 
	nx = x.shape[0]
	no_labels = 6 
	step_x = int(nx / (no_labels - 1)) 
	x_positions = numpy.arange(0,nx,step_x)
	x_labels = x[::step_x]
	y_positions = numpy.linspace(0, max_y, 5)
	y_labels = ["0", "0.2", "0.4", "0.6", "0.8"]
	

	# Plotting the data with imshow
	if sterility:
		mx = 0.5
	else:
		mx = np.max([data1, data2, data3])
	if nrm == "linear":
		mn = 0
	else:
		mn = 1e-40
	im1 = ax1.imshow(data1, cmap=cmp, interpolation="nearest", origin="lower", norm = nrm, vmin=mn, vmax=mx)
	im2 = ax2.imshow(data2, cmap=cmp, interpolation="nearest", origin="lower", norm = nrm, vmin=mn, vmax=mx)
	im3 = ax3.imshow(data3, cmap=cmp, interpolation="nearest", origin="lower", norm = nrm, vmin=mn, vmax=mx)
	ax1.set_title("No interference")
	ax2.set_title("C-C interference ($H_0$)")
	ax3.set_title("C-C and B-C interference ($H_1$)")
	ax1.set_ylabel("Inversion length ($I^{'}$) in Morgans")
	ax2.set_xlabel("Centromere position ($\\rho$)")
	for ax in [ax1, ax2, ax3]:
		ax.set_xticks(x_positions, x_labels)
		ax.set_yticks(y_positions, y_labels)

	if sterility:
		legend = "Sterility"
	else:
		legend = "Recombination rate"
	fig.colorbar(im3, cax=cax, label = legend, norm = nrm)
	plt.savefig(legend + '.png', dpi=300)
	plt.show()


def run_nelder_mead(different_breakpoint_interference):
	
	if different_breakpoint_interference:
		parameters = [random.uniform(0,10), random.uniform(0,10), random.uniform(0,1)]
		bounds=((0,30), (0,30), (0,1))
	else:
		parameters = [random.uniform(0,10),random.uniform(0,1)]
		bounds=((0,30),(0,1))
	result = minimize(squared_deviation, parameters, bounds=bounds, method = "Nelder-Mead")
	solution = result['x']
	if different_breakpoint_interference:
		q1 = solution[0]
		q2 = solution[1]
		d = solution[2]
		k1 = int(q1 // 1)
		k2 = int(q2 // 1)
		gamma_values_chiasma =  [0 for i in range(k1)]
		gamma_values_chiasma.append(1-(q1%1))
		gamma_values_chiasma.append(q1%1)
		gamma_values_breakpoint = [0 for i in range(k2)]
		gamma_values_breakpoint.append(1-(q2%1))
		gamma_values_breakpoint.append(q2%1)
		analyse(gamma_values_chiasma, gamma_values_breakpoint, d)
		print(solution)
		print(squared_deviation(solution))
		c = 1		
	else:
		q = solution[0]
		d = solution[1]
		k = int(q // 1)
		gamma_values =  [0 for i in range(k)]
		gamma_values.append(1-(q%1))
		gamma_values.append(q%1)
		analyse(gamma_values, gamma_values, d)
		print(solution)
		print(squared_deviation(solution))
		c = 1
		

if __name__ == "__main__":
	global_stationary = False # Breakpoint interference
	run_nelder_mead(False) # Run Nelder-Mead with the same interference for breakpoints and chiasmata






