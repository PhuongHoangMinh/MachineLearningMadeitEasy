from collections import OrderedDict
import pickle
import sys

import numpy as np
import init
import pdb


class Model(object):
    """Base class for neural network models.
    Attributes
    ----------
    name : str
        Name of the model.
    params : OrderedDict of str -> :class:`theano.compile.sharedvalue.SharedVariable`
        Mapping from parameter names to Theano shared variables. Note that
        submodel parameters are not included, so this should normally not be
        accessed directly, rather use `self.parameters()`.
    regularization : list of Theano symbolic expressions
        These expressions should all be added to the loss function when
        optimizing. Use `self.regularize()` to modify.
    """

    def __init__(self, name):
        """Initialize an empty model.
        Parameters
        ----------
        name : str
            Name of the model.
        """
        self.name = name
        self.params = OrderedDict()
        self.regularization = []
        self.submodels = OrderedDict()

    def loss(self):
        """Part of the loss function that is independent of inputs."""
        terms = [submodel.loss() for submodel in self.submodels.values()] \
              + self.regularization
        return sum(terms, T.as_tensor_variable(0.0))

    def parameters(self, include_submodels=True):
        """Iterate over the parameters of this model and its submodels.

        Each value produced by the iterator is a tuple (name, value), where
        the name is a tuple of strings describing the hierarchy of submodels,
        e.g. ('hidden', 'b'), and the value is a Theano shared variable.
        Parameters
        ----------
        include_submodels : bool
            If ``True`` (default), also iterate over submodel parameters.
        """
        for name, p in self.params.items():
            yield ((name,), p)
        if include_submodels:
            for submodel in self.submodels.values():
                for name, p in submodel.parameters():
                    yield ((submodel.name,) + name, p)

    def summarize(self, grads, f=sys.stdout):
        def tensor_stats(m):
            return ', '.join([
                'norm = %g' % np.sqrt((m*m).sum()),
                'maxabs = %g' % np.abs(m).max(),
                'minabs = %g' % np.abs(m).min()])
        def summarize_parameter(name, p, g):
            p_stats = tensor_stats(p)
            g_stats = tensor_stats(g)
            print('%s\n    parameter %s\n    gradient %s' % (name, p_stats, g_stats)) #,file=f
        params = list(self.parameters())
        assert len(grads) == len(params)
        for (name, p), grad in zip(params, grads):
            summarize_parameter('.'.join(name), p.get_value(), grad)
        f.flush()

    def parameters_list(self, include_submodels=True):
        """Return a list with parameters, without their names."""
        return list(p for name, p in
                self.parameters(include_submodels=include_submodels))

    def parameter(self, name):
        """Return the parameter with the given name.

        Parameters
        ----------
        name : tuple of str
            Path to variable, e.g. ('hidden', 'b') to find the parameter 'b'
            in the submodel 'hidden'.

        Returns
        -------
        value : :class:`theano.compile.sharedvalue.SharedVariable`
        """

        if not isinstance(name, tuple):
            raise TypeError('Expected tuple, got %s' % type(name))
        if len(name) == 1:
            # print('tuple size of 1')
            # print(type(name[0]))
            return self.params[name[0]]
        elif len(name) >= 2:
            # print('tuple size of 2')
            return self.submodels[name[0]].parameter(name[1:])
        else:
            raise ValueError('Name tuple must not be empty!')

    def parameter_count(self):
        """Return the total number of parameters of the model."""
        return sum(p.get_value(borrow=True).size for _,p in self.parameters())

    def param(self, name, dims, init_f=None,
              value=None, dtype=np.float32):
        """Create a new parameter, or share an existing one.
        Parameters
        ----------
        name : str
            Name of parameter, this will be used directly in `self.params`
            and used to create `self._name`.
        dims : tuple
            Shape of the parameter vector.
        value : :class:`theano.compile.sharedvalue.SharedVariable`, optional
            If this parameter should be shared, a SharedVariable instance can
            be passed here.
        init_f : (tuple => numpy.ndarray)
            Function used to initialize the parameter vector.
        dtype : str or numpy.dtype
            Data type (default is `theano.config.floatX`)
        Returns
        -------
        p : :class:`theano.compile.sharedvalue.SharedVariable`
        """
        if name in self.params:
            if not value is None:
                raise ValueError('Trying to add a shared parameter (%s), '
                                 'but a parameter with the same name already '
                                 'exists in %s!' % (name, self.name))
            # print('name %s in params' % name)
            return self.params[name]
        if value is None:
            if init_f is None:
                raise ValueError('Creating new parameter, but no '
                                 'initialization specified!')
            p = init_f(dims, dtype=dtype)
            self.params[name] = p
        else:
            p = value
        setattr(self, '_'+name, p)
        return p

    def regularize(self, p, regularizer):
        """Add regularization to a parameter.
        Parameters
        ----------
        p : :class:`theano.compile.sharedvalue.SharedVariable`
            Parameter to apply regularization
        regularizer : function
            Regularization function, which should return a symbolic
            expression.
        """
        if not regularizer is None:
            self.regularization.append(regularizer(p))

    def add(self, submodel):
        """Import parameters from a submodel.

        If a submodel named "hidden" has a parameter "b", it will be imported
        as "hidden_b", also accessible as `self._hidden_b`.
        Parameters
        ----------
        submodel : :class:`.Model`
        Returns
        -------
        submodel : :class:`.Model`
            Equal to the parameter, for convenience.
        """
        if submodel.name in self.submodels:
            raise ValueError('Submodel with name %s already exists in %s!' % (
                submodel.name, self.name))
        self.submodels[submodel.name] = submodel
        setattr(self, submodel.name, submodel)
        return submodel


class Linear(Model):
    """Fully connected linear layer
    
    Parameters
    ----------
    name: str
        Name of layer
    input_dims : int
        number dimension of input vectors
    output_dims: int
        number of outputs
    """

    def __init__(self, name, input_dims, output_dims, config = None,
                 w = None, w_init = None, w_regularizer = None,
                 b = None, b_init = None, b_regularizer = None,
                 use_bias = True):

        super().__init__(name)

        # super(name)
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.use_bias = use_bias

        if w_init is None: w_init = init.Gaussian(fan_in = input_dims)
        if b_init is None: b_init = init.Constant(0.0)

        self.param('w', (input_dims, output_dims), init_f = w_init, value = w)
        self.regularize(self._w, w_regularizer)

        if use_bias:
            self.param('b', (output_dims, ), init_f = b_init, value = b)
            self.regularize(self._b, b_regularizer)

        self.config_w = config.copy()
        self.config_b = config.copy()

    def __call__(self, inputs):
        self.inputs = inputs.copy()
        if self.use_bias:
            self.cache = (self.inputs, self._w, self._b)
        else:
            self.cache = (self.inputs, self._w)

        outputs = np.dot(inputs, self._w)
        

        if self.use_bias:
            outputs = outputs + self._b

        self.outputs = outputs
        return outputs



    def backward(self, dout):
        """
        Computes the backward pass for an affine layer.

        Inputs:
        - dout: Upstream derivative, of shape (N, M)
        - cache: Tuple of:
        - x: Input data, of shape (N, d_1, ... d_k)
        - w: Weights, of shape (D, M)

        Returns a tuple of:
        - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
        - dw: Gradient with respect to w, of shape (D, M)
        - db: Gradient with respect to b, of shape (M,)
        """
        if self.use_bias:
            x, w, b = self.cache
        else:
            x, w = self.cache

        x_temp = x.reshape(x.shape[0], -1)
        dw = x_temp.T.dot(dout)
        dx = dout.dot(w.T).reshape(x.shape)
        db = np.sum(dout, axis=0)

        self.gradients = (dw, db)
        
        return dx, dw, db

    def _update(self, update_rule, gradients = None):
        
        if gradients is None:
            next_w, next_config_w = update_rule(self._w, self.gradients[0], self.config_w)
            self._w = next_w
            self.config_w = next_config_w.copy()

            if self.use_bias:
                next_b, next_config_b = update_rule(self._b, self.gradients[1], self.config_b)
                self._b = next_b
                self.config_b = next_config_b.copy()

        else:
            next_w, next_config_w = update_rule(self._w, gradients[0], self.config_w)
            self._w = next_w
            self.config_w = next_config_w.copy()

            if self.use_bias:
                # pdb.set_trace()
                next_b, next_config_b = update_rule(self._b, gradients[1], self.config_b)
                self._b = next_b
                self.config_b = next_config_b.copy()

    def _update_lr(self, lr_decay):
        if 'learning_rate' in self.config_w:
            self.config_w['learning_rate'] *= lr_decay
        if 'learning_rate' in self.config_b:
            self.config_b['learning_rate'] *= lr_decay
            
    # def setconfig(self, config = None):
    #     self.config = config

class ReLu(Model):
    """
    """
    def __init__(self, name):
        super().__init__(name)
    
    def __call__(self, inputs):
        outputs = np.maximum(0, inputs)
        self.inputs = inputs.copy()

        self.cache = self.inputs

        self.outputs = outputs

        return outputs

    def backward(self, dout):
        dx, x = None, self.cache

        dx = dout.copy()
        dx[x <= 0] = 0.0

        return dx

class LinearRelu(Model):
    """
    combination of FC layer and relu layer
    """
    def __init__(self, name, input_dims, output_dims, config = None,
                 w = None, w_init = None, w_regularizer = None,
                 b = None, b_init = None, b_regularizer = None,
                 use_bias = True):

        super().__init__(name)
        self.name = name


        self.add(Linear('linear', input_dims, output_dims, config, w, w_init, w_regularizer, b, b_init, b_regularizer, use_bias))
        self.add(ReLu('relu'))

    def __call__(self, inputs):
        self.linear(inputs)
    
        self.outputs = self.relu(self.linear.outputs)

    def backward(self, dout):

        da = self.relu.backward(dout)
        dx, dw, db = self.linear.backward(da)

        self.gradients = (dw, db)
        return dx, dw, db

    def _update(self, update_rule):
        self.linear._update(update_rule = update_rule, gradients = self.gradients)

    def _update_lr(self, lr_decay):
        self.linear._update_lr(lr_decay)

    # def setconfig(self, config = None):
    #     self.linear.setconfig(config)

def affine_backward(dout, cache):
  """
  Computes the backward pass for an affine layer.

  Inputs:
  - dout: Upstream derivative, of shape (N, M)
  - cache: Tuple of:
    - x: Input data, of shape (N, d_1, ... d_k)
    - w: Weights, of shape (D, M)

  Returns a tuple of:
  - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
  - dw: Gradient with respect to w, of shape (D, M)
  - db: Gradient with respect to b, of shape (M,)
  """
  x, w, b = cache
  dx, dw, db = None, None, None
  #############################################################################
  # TODO: Implement the affine backward pass.                                 #
  #############################################################################
  db = np.sum(dout, axis = 0)
  x_temp = x.reshape(x.shape[0],-1)
  dw = x_temp.T.dot(dout)
  dx = dout.dot(w.T).reshape(x.shape)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx, dw, db
