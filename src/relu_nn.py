import numpy as np


class relu_nn:
      """Red neuronal ReLU, con d entradas, K capas de D neuronas, S salidas"""
      params = {}                   ## usar a partir de python 3.7
      values = {}                   ## usar a partir de python 3.7
      grads = {}                    ## usar a partir de python 3.7

      def __init__(self, K, D, d, S):
            self.K = K
            self.D = D
            self.d = d
            self.S = S
            self.params['W1'] = np.random.rand(D, d) - 0.5
            self.params['b1'] = np.random.rand(D, 1) - 0.5

            for i in range(K-1):                                        ## los params de la primera capa ya están inicializados
                  self.params[f'W{i+2}'] = np.random.rand(D, D) - 0.5
                  self.params[f'b{i+2}'] = np.random.rand(D, 1) - 0.5
            
            self.params[f'W{K+1}'] = np.random.rand(S, D) - 0.5                             ## la capa de salida no se contempla en K
            self.params[f'b{K+1}'] = np.random.rand(S, 1) - 0.5


      def relu(self, x):
           return np.maximum(0,x)
      
      
      def d_relu(self, x):
            return (x > 0).astype(int)


      def softmax(self, x):
            exps = np.exp(x - np.max(x, axis=0, keepdims=True))
            return exps / np.sum(exps, axis=0, keepdims=True)


      def forward_pass(self, x):
            self.values['z1'] = self.params['W1'].dot(x) + self.params['b1']
            self.values['a1'] = self.relu(self.values['z1'])

            for i in range(self.K-1):
                  self.values[f'z{i+2}'] = self.params[f'W{i+2}'].dot(self.values[f'a{i+1}']) + self.params[f'b{i+2}']
                  self.values[f'a{i+2}'] = self.relu(self.values[f'z{i+2}'])
            
            self.values[f'z{self.K+1}'] = self.params[f'W{self.K+1}'].dot(self.values[f'a{self.K}']) + self.params[f'b{self.K+1}']
            self.values[f'a{self.K+1}'] = self.softmax(self.values[f'z{self.K+1}'])


      def one_hot(self, Y):
            one_hot_Y = np.zeros((Y.size, Y.max() + 1))
            one_hot_Y[np.arange(Y.size), Y] = 1                     # one_hot_Y[i, Y[i]] = 1
            one_hot_Y = one_hot_Y.T
            return one_hot_Y


      def back_propagation(self, x, Y):
            p = Y.size
            one_hot_Y = self.one_hot(Y)
            
            self.grads[f'dz{self.K+1}'] = self.values[f'a{self.K+1}'] - one_hot_Y

            for i in range(self.K):
                  self.grads[f'dW{self.K-i+1}'] = 1/p * self.grads[f'dz{self.K-i+1}'].dot(self.values[f'a{self.K-i}'].T)
                  self.grads[f'db{self.K-i+1}'] = 1/p * np.sum(self.grads[f'dz{self.K-i+1}'], 1)
                  self.grads[f'dz{self.K-i}'] = self.params[f'W{self.K-i+1}'].T.dot(self.grads[f'dz{self.K-i+1}']) * self.d_relu(self.values[f'z{self.K-i}'])

            self.grads[f'dW1'] = 1/p * self.grads[f'dz1'].dot(x.T)
            self.grads[f'db1'] = 1/p * np.sum(self.grads[f'dz1'], 1)


      def grad_desc(self, mu):
            for i in range(self.K+1):
                  self.params[f'W{i+1}'] = self.params[f'W{i+1}'] - mu * self.grads[f'dW{i+1}']
                  self.params[f'b{i+1}'] = self.params[f'b{i+1}'] - mu * self.grads[f'db{i+1}'].reshape(self.params[f'b{i+1}'].shape)


      def cross_entropy(self, x, y):
            return sum(-np.log(x[np.arange(y.size), y]))


      def get_prediction(self):
            return np.argmax(self.values[f'a{self.K+1}'], axis=0)


      def get_accuracy(self, pred, y):
            return np.sum(pred == y)/y.size


      def learning(self, n_epoch, n_batches, x, Y, learning_rate, n_doc=10):
            batch_size = Y.size//n_batches
            
            f_mod = (n_batches*n_epoch)/n_doc

            for i in range(n_batches):
                  for k in range(n_epoch):
                        self.forward_pass(x[:,i*batch_size : (i+1)*batch_size])
                        self.back_propagation(x[:, i*batch_size : (i+1)*batch_size], Y[:, i*batch_size : (i+1)*batch_size])
                        self.grad_desc(learning_rate)

                        if ((k*i)%f_mod==0):
                              print(f'iteración:{k+1}; {i+1}/{n_batches} batch')
                              print("presición", self.get_accuracy(self.get_prediction(), Y[:, i*batch_size : (i+1)*batch_size]))

      
      def test_predic(self, x):
            self.forward_pass(x)
            return (self.get_prediction())
      

      def save_params(self, src):
            try:
                  np.savez(src, n_params=self.K+1, params=self.params)
                  print(f'se guardó los parámetros en {src}')
            except:
                  print(f'no se pudo guardar los parámetros')
