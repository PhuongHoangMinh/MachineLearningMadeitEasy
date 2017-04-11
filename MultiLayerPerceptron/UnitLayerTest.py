from model import *
from solver import *
from neuralnet import *
import optim
import matplotlib.pyplot as plt

def TestLinearLayer():
	inputs = np.array([[2, 2],[2, 2]]).astype("float32")

	first_linear = Linear('linear', 2, 3)
	print('original filter')
	print(first_linear._w)

	outputs = first_linear(inputs)
	print(outputs.shape)
	print(outputs)

	temp_w = first_linear.parameter(('w',))
	temp_w = np.ones((2,2))
	print(first_linear._w.shape)
	print(first_linear._w)
	print(temp_w)
	print(first_linear.params['w'])


	print('another test with dictionary of parameters:')
	next_w = first_linear.params['w']
	next_w *= 2.0
	print(first_linear._w)
	print(first_linear.params['w'])

	print('test with existing function param():')
	my_w = first_linear.param('w', (2,3))
	my_w += 100.0
	print(my_w)
	print(first_linear._w)


	outputs = first_linear(inputs)
	print(outputs)

	#NOTE su khac biet giua
	# nparray1 = nparray_origin
	# nparray1 = nparray1*2.0 #nparray_origin doesn't change
	# nparray1 *= 2.0 		#nparray_origin CHANGE

	print('architect a nn')
	hidden_node_list = [4, 5, 6]
	net = {}
	net['0'] = inputs
	# for i in range(3):
	# 	current_name = '%d'%i
	# 	current_relu_name = 'relu%d'%i

	# 	name = '%d'%(i+1)
	# 	relu_name = 'relu%d'%(i+1)
	# 	if i == 0: 
	# 		#initialize FC and Relu Layer
	# 		net[name] = Linear(name, net['0'].shape[-1], hidden_node_list[i])
	# 		net[relu_name] = Relu(relu_name)

	# 		#FC Layer1
	# 		outputs1 = net[name](net['0'])

	# 	else:
	# 		net[name] = Linear(name, net[current_relu_name].outputs.shape[-1], hidden_node_list[i])
	# 		net[relu_name] = Relu(relu_name)

	# 		net[name](net[current_name].ouputs)
	# 		net[relu_name]()

	for i in range(3):
		previous_name = '%d'%i
		next_name = '%d'%(i+1)
		if i == 0: 
			net[next_name] = LinearRelu(next_name, net['0'].shape[-1], hidden_node_list[i])

			#forward path
			net[next_name](net['0']) 
			print(net[next_name].outputs.shape)
		else:
			net[next_name] = LinearRelu(next_name, net[previous_name].outputs.shape[-1], hidden_node_list[i])

			#forward path
			net[next_name](net[previous_name].outputs)
			print(net[next_name].outputs.shape)



	# dout = np.ones((2,6)).astype("float32")
	# grad = {}
	# grad['3'] = net['3'].backward(dout)
	# print(grad['3'][0])
	# print(grad['3'][1])
	# print(grad['3'][2])

	#backward
	dout = np.ones((2,6)).astype("float32")
	grad = {}
	for i in range(3, 0, -1):
		name = '%d'%i
		next_name = '%d'%(i+1)

		if i == 3:
			grad[name] = net[name].backward(dout)
		
		else:
			grad[name] = net[name].backward(grad[next_name][0])

	update_rule = 'sgd'
	update_rule_func = getattr(optim, update_rule)
	print(type(update_rule_func))

	# update_rule_func(net['3'].linear._w, grad['3'][1])
	# update(update_rule_func, net['3'].linear._w, grad['3'][1])

	for i in range(3, 0, -1):
		name = '%d'%i
		next_name = '%d'%(i+1)

		net[name]._update(update_rule_func)

def update(update_rule_func, w, dw, config = None):
	update_rule_func(w, dw, config)

	

def GenerateSpiralData():
	np.random.seed(0)
	N = 100 # No of points per class
	D = 2   # dimesionality
	K = 4   # No of classes
	X = np.zeros((N*K, D)).astype("float32")
	num_train_examples = X.shape[0]
	print('X shape: ', X.shape)
	Y = np.zeros(N*K, dtype = 'uint8')
	print('Y shape: ', Y.shape)
	for j in range(K):
		ix = range(N*j, N*(j+1))
		r  = np.linspace(0.0, 1, N)
		t  = np.linspace(j*5, (j+1)*5, N) + np.random.randn(N)*0.2
		X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
		Y[ix] = j
	return X, Y

def TestNeuralNetSpiralData():
	X_data, Y_data = GenerateSpiralData()
	data = (X_data, Y_data)

	#net
	myNet = NeuralNet('Three layer net')
	number_hiden_node_list = [50, 50, 4]
	number_hidden_layer = 3


	#solver
	first_solver = SolverNN(data, myNet, update_rule='sgd',
       num_epochs=10000,
       batch_size=400,
       optim_config={
         'learning_rate': 5e-1,
       },
       lr_decay=0.995,
       verbose=True, print_every=2000,
       number_hidden_layer = number_hidden_layer,
       number_hidden_node_list = number_hiden_node_list)

	first_solver.train()

	first_solver.evaluate_accuracy()

	drawing(X_data, Y_data, first_solver)

def drawing(X, Y,  solver):
	# plot the classifiers- SIGMOID
	h = 0.02
	x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
	y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
	xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
	                 np.arange(y_min, y_max, h))

	# Z = np.dot(sigmoid(np.dot(sigmoid(np.dot(np.c_[xx.ravel(), yy.ravel()], s_W1) + s_b1), s_W2) + s_b2), s_W3) + s_b3
	Z = solver.forward(np.c_[xx.ravel(), yy.ravel()])
	Z = np.argmax(Z, axis=1)
	Z = Z.reshape(xx.shape)
	fig = plt.figure()
	plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
	plt.scatter(X[:, 0], X[:, 1], c=Y, s=40, cmap=plt.cm.Spectral)
	plt.xlim(xx.min(), xx.max())
	plt.ylim(yy.min(), yy.max())
	plt.show()

if __name__ == '__main__':
	# TestLinearLayer()
	TestNeuralNetSpiralData()