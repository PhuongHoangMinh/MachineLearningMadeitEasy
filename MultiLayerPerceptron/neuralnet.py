from model import *
import numpy as np

class NeuralNet(Model):
	"""
	"""

	def __init__(self, name):
			super().__init__(name)
			self.name = name
			self.grad = {}
			self.net  = {}

	def loss(self, inputs, outputs, number_hidden_node_list = None, number_hidden_layer = 2, config = None):
		
		num_examples = inputs.shape[0]
		self.number_hidden_layer = number_hidden_layer
		self.number_hidden_node_list = number_hidden_node_list

		if number_hidden_node_list is None:
			raise ValueError("Number of hidden nodes in hidden layers must not be empty")
		else:
			#initialized neural net
			self.net['0'] = inputs
			for i, number_node in zip(range(number_hidden_layer), number_hidden_node_list):
				next_layer_name = '%d'%(i+1)
				pre_layer_name  = '%d'%i

				new_config = config.copy()
				
				#initilization
				if next_layer_name not in self.net:
					if i == 0:
						
						self.net[next_layer_name]= LinearRelu(next_layer_name, self.net['0'].shape[-1], number_node, new_config)

						#forward
						self.net[next_layer_name](self.net['0'])
						

					elif i == number_hidden_layer:
						self.net[next_layer_name] = Linear(next_layer_name, self.net[pre_layer_name].outputs.shape[-1], number_node, new_config)
						
						#forward
						self.net[next_layer_name](self.net[pre_layer_name].outputs)

					else:
						self.net[next_layer_name] = LinearRelu(next_layer_name, self.net[pre_layer_name].outputs.shape[-1], number_node, new_config)

						#forward
						self.net[next_layer_name](self.net[pre_layer_name].outputs)

					print('add layer name %s'%next_layer_name)
					# self.add(self.net[next_layer_name])
				else:
					if i == 0:
						self.net[next_layer_name](self.net['0'])
					else:
						self.net[next_layer_name](self.net[pre_layer_name].outputs)

			#estimate loss
			scores = self.net['%d'%number_hidden_layer].outputs
			exp_scores = np.exp(scores)
			probs = exp_scores/np.sum(exp_scores, axis = 1, keepdims = True)

			correct_log_loss = - np.log(probs[range(num_examples), outputs])
			data_loss = np.sum(correct_log_loss)/num_examples
			self.loss_value = data_loss

			#backprop
			dscores = probs
			dscores[range(num_examples), outputs] -= 1
			dscores /= num_examples

			for i in range(number_hidden_layer, 0, -1):
				name = '%d'%i
				next_name = '%d'%(i+1)

				# print('backprop layer %d'%i)	
				if i == number_hidden_layer:
					
					self.grad[name] = self.net[name].backward(dscores)

				else:

					self.grad[name] = self.net[name].backward(self.grad[next_name][0])

			return self.loss_value, self.grad
	
	def update(self, update_rule):
		for i in range(self.number_hidden_layer, 0, -1):
			name = '%d'%i
			# print('update layer %s'%name)
			self.net[name]._update(update_rule)
	
	def evaluate_accuracy(self, inputs, outputs):
		self.net['0'] = inputs

		for i in range(self.number_hidden_layer):
			next_layer_name = '%d'%(i+1)
			pre_layer_name  = '%d'%i

			if i == 0:
				self.net[next_layer_name](self.net['0'])
			else:
				self.net[next_layer_name](self.net[pre_layer_name].outputs)

		scores = self.net['%d'%self.number_hidden_layer].outputs
		exp_scores = np.exp(scores)
		probs = exp_scores/ np.sum(exp_scores, axis = 1, keepdims = True)

		predicted_class = np.argmax(probs, axis = 1)
		print('training accuracy: %.2f' % (np.mean(predicted_class == outputs)))

	def _update_lr(self, lr_decay):
		for i in range(self.number_hidden_layer, 0, -1):
			layer_name = '%d'%i

			self.net[layer_name]._update_lr(lr_decay)

	def forward(self, inputs):
		self.net['0'] = inputs

		for i in range(self.number_hidden_layer):
			next_layer_name = '%d'%(i+1)
			pre_layer_name  = '%d'%i

			if i == 0:
				self.net[next_layer_name](self.net['0'])
			else:
				self.net[next_layer_name](self.net[pre_layer_name].outputs)

		scores = self.net['%d'%self.number_hidden_layer].outputs
		return scores