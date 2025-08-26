import numpy as np

def he_init(shape, fan_in):
    std = np.sqrt(2.0 / fan_in)
    return np.random.randn(*shape) * std


def xavier_init(shape, fan_in, fan_out=None):
    if fan_out is None:
        fan_out = fan_in
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    return np.random.uniform(-limit, limit, size=shape)