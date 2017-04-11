import sys
import numpy as np

class InitializationFunction:
	def __call__(self, *args, **kwargs):
		raise NotImplementedError


class Constant(InitializationFunction):
	"""Constant initilization

	All elements are initialized with the same constant value
	
	Parameters
	----------
	a: float
		Initialization constant
	"""

	def __init__(self, a):
		self.a = a

	def __call__(self, dims, dtype = np.float32):
		return np.full(dims, self.a, dtype = dtype)


class Gaussian(InitializationFunction):
	"""Isotropic Gaussian initialization

	Elements are initialized independently from a Gaussian distribution
	with a given mean and standard deviation
	
	Parameters
	----------
	dev: float, optional
		standard deviation of distribution
	mean: float, optional
		mean of distribution
	fan_in: int, optional
		if this argument is given, 'dev' and 'mean' are ignored. Instead,
		mean 0 and standard deviation: math:'\sqrt{2/fan_in}' is used
	"""

	def __init__(self, dev = 1.0, mean = 0.0, fan_in = None):
		if fan_in is None:
			self.dev = dev
			self.mean = mean
		else:
			self.mean = 0.0
			self.dev = np.sqrt(2.0/fan_in)

	def __call__(self, dims, dtype = np.float32):
		m = np.random.standard_normal(dims)*self.dev + self.mean
		return m.astype(dtype)

class Uniform(InitializationFunction):
	"""Uniform initialization

	Elements are initialized independently from a uniform distribution
	with a given mean and scale

	Parameters
	----------
	scale: float, optional
		scale of distribution
	mean: float, optional
		mean of distribution
	"""

	def __init__(self, scale = 0.01, mean = 0.0):
		self.scale = scale
		self.mean = mean

	def __call__(self, dims, dtype = np.float32):
		return np.random.uniform(
				size = dims,
				low  = mean - 0.5*self.scale,
				high = mean + 0.5*self.scale).astype(dtype)


