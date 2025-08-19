import matplotlib.pyplot as plt
import numpy as np
from relu_nn import relu_nn

class plot_bound:

    def __init__(self, n_abs, n_ord):
        self.n_abs = n_abs
        self.n_ord = n_ord
    
    def plot(self, nn : relu_nn, data, labels, factor, err=0.01):
        x = np.linspace(-1, 1, self.n_abs)
        y = np.linspace(-1, 1, self.n_ord)
        clase0 = labels == 0
        clase1 = labels == 1

        p = []

        for i in x:
            for j in y:
                x_t = np.array([i, j]).reshape(2, 1)
                nn.forward_pass(x_t)
                z_k_1 = nn.values[f'z{nn.K+1}']*factor

                r = np.abs(np.subtract.reduce(z_k_1, axis=0))

                if r <= err:
                    p.append([i, j])

        arr = np.array(p)

        

        plt.scatter(arr[:,0], arr[:,1], s=1, color='black')
        plt.scatter(data[0, clase0], data[1, clase0], color='blue')
        plt.scatter(data[0, clase1], data[1, clase1], color='red')