
from numpy.lib.stride_tricks import as_strided as numpy_as_strided
from torch import as_strided as torch_as_strided


def _roll_shape_and_strides(
        input_shape, input_strides, window_shape, dx=1, dy=1, dz=None):
    '''
    Rolling 3D window for array.
        input_shape (np.array): input array shape
        input_strides (np.array): input array strides
        window_shape (np.array): rolling 2D window array shape
        dx (int): horizontal step, abscissa, number of columns
        dy (int): vertical step, ordinate, number of rows
        dz (int): transverse step, applicate, number of layers. Only used with
        3D window
    '''
    if dz is not None:
        shape = input_shape[:-3] + (
            (input_shape[-3] - window_shape[-3]) // dz + 1,
        )
        strides = input_strides[:-3] + (input_strides[-3] * dz,)
    else:
        shape = input_shape[:-2]
        strides = input_strides[:-2]

    shape += \
        ((input_shape[-2] - window_shape[-2]) // dy + 1,) + \
        ((input_shape[-1] - window_shape[-1]) // dx + 1,) + \
        window_shape  # multidimensional "sausage" with 3D cross-section
    strides += \
        (input_strides[-2] * dy,) + (input_strides[-1] * dx,)

    if dz is not None:
        strides += input_strides[-3:]
    else:
        strides += input_strides[-2:]

    return shape, strides


def roll_numpy(input_array, window_array, dx=1, dy=1, dz=None):
    '''
    Rolling 3D window for numpy array. This function is only used with
    Qiskit backends.
        input_array (np.array): input array
        window_array (np.array): rolling 2D window array
        dx (int): horizontal step, abscissa, number of columns
        dy (int): vertical step, ordinate, number of rows
        dz (int): transverse step, applicate, number of layers. Only used with
        3D window
    '''
    shape, strides = _roll_shape_and_strides(
        input_array.shape, input_array.strides, window_array.shape,
        dx, dy, dz)
    return numpy_as_strided(input_array, shape=shape, strides=strides), shape


def roll_torch(input_array, window_array, dx=1, dy=1, dz=None):
    '''
    Rolling 3D window for pytorch tensor. This function is only used with
    Pytorch backends.
        input_array (tensor): input array, shape (n_samples, N,N,N)
        window_array (tensor): rolling 3D window array, shape (n,n,n)
        dx (int): horizontal step, abscissa, number of columns
        dy (int): vertical step, ordinate, number of rows
        dz (int): transverse step, applicate, number of layers
    '''
    shape, strides = _roll_shape_and_strides(
        input_array.shape, input_array.stride(), window_array.shape,
        dx, dy, dz)
    return torch_as_strided(input_array, shape, strides), shape
