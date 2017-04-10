import numpy as np
#padding
a = np.array([1,2, 3, 4, 5])
b = np.pad(a, (1,1), 'constant', constant_values= (0.0))
print(b)

#Convolution 1D
def Convol1D(input, filter, config = None):
    """
    naive implementation of convolution 1D
    :param input: 1D numpy array
    :param filter: 1D numpy array
    :return: 1D output
    """
    if config is None: config = {}
    config.setdefault('pad', 1)
    config.setdefault('stride', 1)

    pad = config['pad']
    stride = config['stride']

    len_input = input.shape[0]
    len_filter = filter.shape[0]

    len_out = int(1 + (len_input + 2*pad - len_filter)/stride)
    out = np.zeros((len_out,1)).astype("float32")


    x_pad = np.pad(input, (1,1), 'constant', constant_values=(0.0))
    for i in range(len_out):
        out[i,0] = np.sum(x_pad[i*stride: i*stride + len_filter]*filter)

    return out


def test_convol_1D():
    input = np.array([1., 2., 1., 2., 1.]).astype("float32")
    filter= np.array([-1., 2., 1.]).astype("float32")

    out = Convol1D(input, filter)
    print(out)

test_convol_1D()


def Convol2D(input, filter, config = None):
    """
    naive implementation of convolution 2D
    :param input: 2D numpy array
    :param filter:2D numpy array
    :param config:
    :return:
    """

    if config is None: config = {}
    config.setdefault('pad', (1,1))
    config.setdefault('stride', (1,1))

    pad = config['pad']
    stride = config['stride']

    H, W = input.shape
    HH, WW= filter.shape

    H_out = (int)(1 + (H + 2*pad[0] - HH)/stride[0])
    W_out = (int)(1 + (W + 2*pad[1] - WW)/stride[1])

    out = np.zeros((H_out, W_out)).astype("float32")

    input_pad = np.pad(input, pad, 'constant', constant_values=(0.0))
    for i in range(H_out):
        for j in range(W_out):
            out[i,j] = np.sum(input_pad[i*stride[0]:i*stride[0] + HH, j*stride[1]:j*stride[1]+WW]*filter)

    return out

def test_convol2D():
    input = np.array([[1.,2.,3.], [4.,5.,6.], [7.,8.,9.]]).astype("float32")
    filter = np.array([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]).astype("float32")

    out = Convol2D(input, filter)
    print(out)

test_convol2D()