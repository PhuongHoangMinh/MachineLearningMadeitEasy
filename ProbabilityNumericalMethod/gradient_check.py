import numpy as np
from random import randrange

def eval_numerical_gradient(f, x, verbose = True, h = 1.0e-5):
    """
    a naive implementation of numerical gradient of f at x
    :param f: should be a function that takes a single argument
    :param x: is the poinrt to evaluate the gradient at
    :param verbose:
    :param h: epsilon (step to evaluate gradient numerically)
    :return:
    """

    fx = f(x)
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'], op_flags = ['readwrite'])

    while not it.finished:
        ix = it.multi_index
        oldval = x[ix]
        x[ix] = oldval + h
        fxph  = f(x) #evaluate f(x+h)
        x[ix] = oldval - h
        fxmh  = f(x)
        x[ix] = oldval

        grad[ix] = (fxph - fxmh)/(2*h)
        if verbose:
            print(ix, grad[ix])
        it.iternext()
    return grad

def grad_check_sparse(f, x, analytic_grad, num_checks = 10, h=1.0e-5):
    """
    sample a few random elements and only return numerical in these dimensions

    :param f:
    :param x:
    :param analytic_grad:
    :param num_checks:
    :param h:
    :return:
    """
    for i in range(num_checks):
        ix = tuple([randrange(m) for m in x.shape])

        oldval =  x[ix]
        x[ix]  = oldval + h
        fxph   = f(x)
        x[ix]  = oldval - h
        fxmh   = f(x)
        x[ix]  = oldval

        grad_numerical = (fxph - fxmh)/(2*h)
        grad_analytic  = analytic_grad[ix]
        rel_error = abs(grad_numerical - grad_analytic)/(abs(grad_analytic) + abs(grad_numerical))
        print('numerical: %f analytic: %f relative error %e' %(grad_numerical, grad_analytic, rel_error))


def eval_numerical_gradient_array(f, x, df, h=1.0e-5):
    """
    evaluate a numeric gradient for a function that accepts a numpy array and returns a numpy array (vector value function)
    :param f:
    :param x:
    :param df:
    :param h:
    :return:
    """
    grad = np.zeros_like(x)
    it   = np.nditer(x, flags = ['multi_index'], op_flags = ['readwrite'])
    while not it.finished:
        ix = it.multi_index

        oldval = x[ix]
        x[ix]  = oldval + h
        pos    = f(x).copy()
        x[ix]  = oldval - h
        neg    = f(x).copy()
        x[ix]  = oldval

        grad[ix] = np.sum((pos - neg)*df)/(2*h)
        it.iternext()
    return grad


def eval_numerical_gradient_blobs(f, inputs, output, h=1e-5):
    """
    Compute numeric gradients for a function that operates on input and output blobs

    we assume that f accepts several input blobs as arguments, followed by a blob into which
    outputs will be written. For example, f might be called like this:

    f(x, w, out)

    :param f:
    :param inputs:
    :param output:
    :param h:
    :return:
    """

    numeric_diffs = []
    for input_blob in inputs:
        diff = np.zeros_like(input_blob)
        it = np.nditer(input_blob.vals, flags = ['multi_index'], op_flags = ['readwrite'])

        while not it.finished:
            idx = it.multi_index
            orig = input_blob.vals[idx]

            input_blob.vals[idx] = orig + h
            f(*(inputs + (output,)))
            pos = np.copy(output.vals)

            input_blob.vals[idx] = orig - h
            f(*(inputs + (output)))
            neg = np.copy(output.vals)
            input_blob.vals[idx] = orig

            diff[idx] = np.sum(pos - neg)*output.diffs/(2.0*h)

            it.iternext()
        numeric_diffs.append(diff)
        return numeric_diffs


