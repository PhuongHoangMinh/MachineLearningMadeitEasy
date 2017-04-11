import numpy as np
from model import *
import optim
import pdb


class SolverNN(object):
	"""
	"""

	def __init__(self, data, model, **kwargs):

		self.data = data
		self.model = model

		self.num_epochs = kwargs.pop('num_epochs', 100)
		self.batch_size = kwargs.pop('batch_size', 100)
		self.update_rule = kwargs.pop('update_rule', 'sgd')
		self.optim_config = kwargs.pop('optim_config', {})
		self.lr_decay    = kwargs.pop('lr_decay', 1.0)

		self.number_hidden_layer = kwargs.pop('number_hidden_layer', 2)
		self.number_hidden_node_list = kwargs.pop('number_hidden_node_list', [50, 2])

		self.print_every = kwargs.pop('print_every', 5)
		self.verbose     = kwargs.pop('verbose', True)

		    # Throw an error if there are extra keyword arguments
		if len(kwargs) > 0:
			extra = ', '.join('"%s"' % k for k in kwargs.keys())
			raise ValueError('Unrecognized arguments %s' % extra)

		if not hasattr(optim, self.update_rule):
			raise ValueError("Invalid update rule %s "% self.update_rule)
		self.update_rule_func = getattr(optim, self.update_rule)
		print(type(self.update_rule_func))

		self._reset()

	def _reset(self):
		"""
		set up book-keeping variables for optimization
		"""
		self.epoch = 0
		self.best_val_acc = 0
		self.best_parameters = {}
		self.loss_history = []
		self.train_acc_history = []
		self.val_acc_history = []

		#make a deep copy for optim_config ????
		# self.optim_configs = {}
		# for name, p in self.model.parameters(True):
		# 	d = {k: v for k, v in self.optim_config.items()}
		# 	self.optim_configs[name] = d

	def _step(self):

		#get data by index
		X, Y = self.data

		# X_train, Y_train = X[batch_index,:,:], Y[batch_index, :]

		loss, grads = self.model.loss(X, Y, number_hidden_node_list = self.number_hidden_node_list, number_hidden_layer = self.number_hidden_layer, config = self.optim_config)

		self.loss_history.append(loss)
		
		self.model.update(self.update_rule_func)		

		# #backpropagation update
		# for name, w in self.model.parameters(True):
		# 	dw = grads[name]
		# 	config = self.optim_configs[name]
		# 	next_w, next_config = self.update_rule(w, dw, config)
		# 	self.optim_configs[name] = next_config
		# 	pbd.set_trace()
		# 	print(name)

		# 	if not isinstance(name, tuple):
		# 		raise TypeError('Expected tuple, got %s' % type(name))
		# 	if len(name) == 1:
		# 		# pdb.set_trace()
		# 		self.model.params[name[0]] = next_w
		# 	elif len(name) == 2:
		# 		pdb.set_trace()
		# 		self.model.submodels[name[0]].params[name[1]] = next_w
		# 	elif len(name) == 3:
		# 		pbd.set_trace()
		# 		self.model.submodels[name[0]].submodels[name[1]].params[name[2]] = next_w
	def train(self):

		num_train = self.data[0].shape[0]
		num_iteration_per_epoch = int(max(num_train/self.batch_size, 1))
		# print(num_iteration_per_epoch)
		num_iterations = int(self.num_epochs*num_iteration_per_epoch)

		for t in range(self.num_epochs):
			for i in range(1, num_iteration_per_epoch+1):

				self._step()

				if self.verbose and (t*i)%self.print_every == 0:
					print("iteration %d/%d with loss: %f"%( t*i, num_iterations, self.loss_history[-1]))

			self.epoch += 1
			# for k in self.optim_configs:
			# 		self.optim_configs[k]['learning_rate'] *= self.lr_decay
			# self.model._update_lr(self.lr_decay)

	def evaluate_accuracy(self):
		X, Y = self.data
		self.model.evaluate_accuracy(X, Y)

	def forward(self, inputs):
		return self.model.forward(inputs)