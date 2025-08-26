import numpy as np
import pickle
from numpy.lib.stride_tricks import as_strided
from scipy import signal
from scipy.special import expit
from IPython.display import clear_output
import matplotlib.pyplot as plt
import initialization as init

class model:

    """
    Linear layers
    Convolutional layers
    Max-pooling layers
    Transpose convolutional layers
    Batch normalization layers
    """

    ## los diccionarios de capas y demás conservan el orden de creación a partir de python 3.7

    def __init__(self):
        self.layers = {}
        self.n_conv_layers = 0
        self.n_deconv_layers = 0
        self.n_lin_layers = 0
        self.n_mpool_layers = 0
        self.n_layers = 0
        self.cache = {}
        self.grads = {}
        self.optimSGD = True                                              ## por defecto el optimizador es Stochastic Gradient Descent
        self.optimSGDM = False
        self.pond = {}
        self.momentum = 0


    def optimizer_SGD(self, momentum=0.9):
        self.optimSGDM = True                                             ## es posible cambiar el tipo de optimización a Stochastic Gradient Descent con Momentum
        self.optimSGD = False
        self.momentum = momentum


    
    def first_conv_layer(self, input_shape, n_chan, kernel_size, padding=0, stride=1, bias=True):
        self.n_conv_layers += 1
        self.n_layers += 1

        c, h, w = input_shape

        shape = (n_chan, c, kernel_size, kernel_size)
        fan_in = c * kernel_size * kernel_size

        kernel_list = init.he_init(shape, fan_in)

        b = np.zeros((n_chan,))

        params = {
            'layer': self.n_conv_layers,
            'input': input_shape,
            'kernel_list':kernel_list,
            'output':(n_chan, (h-kernel_size + 2*padding)//stride + 1, (w-kernel_size + 2*padding)//stride + 1),
            'stride': stride,
            'padding': padding
            }

        if bias:
            params['bias'] = b

        self.layers[f'conv{self.n_conv_layers}'] = params

        if self.optimSGDM:
            self.pond[f'conv{self.n_conv_layers}'] = {'VdW':np.zeros((n_chan, c, kernel_size, kernel_size)), 'Vdb': np.zeros(n_chan,)}          ## Si el optimizador es SGDM, inicio sus vectores de ponderación
    


    def conv_layer(self, n_chan, kernel_size, padding=0, stride=1, bias=True):
        self.n_conv_layers += 1
        self.n_layers += 1

        last_key = list(self.layers.keys())[-1]
        c, h, w = self.layers[last_key]['output']
        
        shape = (n_chan, c, kernel_size, kernel_size)
        fan_in = c * kernel_size * kernel_size

        kernel_list = init.he_init(shape, fan_in)

        b = np.zeros((n_chan,))

        params = {
                  'layer': self.n_conv_layers,
                  'input': (c,h,w),
                  'kernel_list':kernel_list,
                  'output':(n_chan, (h-kernel_size + 2*padding)//stride + 1, (w-kernel_size + 2*padding)//stride + 1),
                  'stride':stride,
                  'padding':padding
                  }

        if bias:
            params['bias'] = b

        self.layers[f'conv{self.n_conv_layers}'] = params

        if self.optimSGDM:
            self.pond[f'conv{self.n_conv_layers}'] = {'VdW':np.zeros((n_chan, c, kernel_size, kernel_size)), 'Vdb': np.zeros(n_chan,)}



    def lin_layer_only(self, n_input, n_neurons, activ='relu'):
        self.n_lin_layers += 1
        self.n_layers += 1

        relu = False
        softmax = False
        n_in = n_input

        shape = (n_neurons, n_in)
        fan_in = n_in

        if activ == 'relu':
            relu = True
            W = init.he_init(shape, fan_in)

        else:
            fan_out = n_neurons
            W = init.xavier_init(shape, fan_in, fan_out)

        if activ == 'softmax':
            softmax = True


        b = np.random.rand(n_neurons, 1) - 0.5
        self.layers[f'lino{self.n_lin_layers}'] = {'layer': self.n_lin_layers,'W': W, 'b': b, 'relu': relu, 'softmax': softmax, 'output': (n_neurons,)}

        if self.optimSGDM:
            self.pond[f'lino{self.n_lin_layers}'] = {'VdW':np.zeros((n_neurons, n_in)), 'Vdb': np.zeros(n_neurons,1)}



    def lin_layer(self, n_neurons, activ='relu'):

        """
        activ : (relu, softmax, None)
        """

        self.n_lin_layers += 1
        self.n_layers += 1

        last_layer = list(self.layers.keys())[-1]

        relu = False
        softmax = False

        if last_layer[0] == 'c':
            c, h, w = self.layers[f'conv{self.n_conv_layers}']['output']
            n_in = c * h * w

        elif last_layer[0] == 'm':
            c, h, w = self.layers[f'mpool{self.n_mpool_layers}']['output']
            n_in = c*h*w

        elif self.n_lin_layers > 1:
            try:
                n_in = self.layers[f'lin{self.n_lin_layers-1}']['output'][0]
            except:
                n_in = self.layers[f'lino{self.n_lin_layers-1}']['output'][0]

        else:
            raise ValueError("Debe existir al menos una capa previa antes de agregar una capa lineal.")
        
        shape = (n_neurons, n_in)
        fan_in = n_in
        
        if activ == 'relu':
            relu = True
            W = init.he_init(shape, fan_in)

        else:
            fan_out = n_neurons
            W = init.xavier_init(shape, fan_in, fan_out)

        if activ == 'softmax':
            softmax = True

        b = np.zeros((n_neurons, 1))

        self.layers[f'lin{self.n_lin_layers}'] = {'layer': self.n_lin_layers,'W': W, 'b': b, 'relu': relu, 'softmax': softmax, 'output': (n_neurons,)}

        if self.optimSGDM:
            self.pond[f'lin{self.n_lin_layers}'] = {'VdW':np.zeros((n_neurons, n_in)), 'Vdb': np.zeros(n_neurons,1)}

    

    def first_deconv_layer(self, input_shape, n_chan, kernel_size, padding=0, stride=1, bias=True, activation='relu'):
        """
        activation puede ser relu o sigmoid
        """
        
        self.n_deconv_layers += 1
        self.n_layers += 1
        
        relu = False
        sigmoid = False

        c, h, w = input_shape
        
        shape = (n_chan, c, kernel_size, kernel_size)
        fan_in = c * kernel_size * kernel_size

        if activation == 'relu':
            relu = True
            kernel_list = init.he_init(shape, fan_in)

        if activation == 'sigmoid':
            sigmoid = True
            fan_out = n_chan * kernel_size * kernel_size
            kernel_list = init.xavier_init(shape, fan_in, fan_out)


        b = np.zeros((n_chan,))

        h_out = h + (stride-1)*(h-1) + 2*(kernel_size-1-padding) - kernel_size + 1
        w_out = w + (stride-1)*(w-1) + 2*(kernel_size-1-padding) - kernel_size + 1


        last_layer = list(self.layers.keys())[-1]

        if last_layer[0] == 'l':
            self.lin_layer(c*h*w)


        params = {
                  'layer': self.n_deconv_layers,
                  'input': (c,h,w),
                  'kernel_list':kernel_list,
                  'output':(n_chan, h_out, w_out),
                  'stride':stride,
                  'padding':padding,
                  'relu': relu,
                  'sigmoid': sigmoid
                  }

        if bias:
            params['bias'] = b

        self.layers[f'deconv{self.n_deconv_layers}'] = params

        if self.optimSGDM:
            self.pond[f'deconv{self.n_deconv_layers}'] = {'VdW':np.zeros((n_chan, c, kernel_size, kernel_size)), 'Vdb': np.zeros(n_chan,)}



    def deconv_layer(self, n_chan, kernel_size, padding=0, stride=1, bias=True, activation='relu'):                ## si hay n capas convolucionales, i-ésima deconv_layer --> (input) = (n+1)-i-ésima conv_layer --> (output)
        """
        activation puede ser relu o sigmoid
        """
        
        self.n_deconv_layers += 1
        self.n_layers += 1

        try:
            last_key = list(self.layers.keys())[-1]

            if (np.strings.startswith(last_key, 'lin') and not np.strings.startswith(last_key, 'lino')):
                try:
                    c, h, w = self.layers[f'conv{self.n_conv_layers+1-self.n_deconv_layers}']['output']
                
                except KeyError:
                    print('no se puede agregar una capa deconvolucional después de una capa lineal sin capas convolucionales anteriores')

            else:
                c, h, w = self.layers[last_key]['output']

        except IndexError:
            print('para agregar una capa deconvolucional como primera capa utiliza "first_deconv_layer"')

        
        shape = (n_chan, c, kernel_size, kernel_size)
        fan_in = c * kernel_size * kernel_size

        relu = False
        sigmoid = False

        if activation == 'relu':
            relu = True
            kernel_list = init.he_init(shape, fan_in)

        if activation == 'sigmoid':
            sigmoid = True
            fan_out = n_chan * kernel_size * kernel_size
            kernel_list = init.xavier_init(shape, fan_in, fan_out)

        b = np.zeros((n_chan,))

        h_out = h + (stride-1)*(h-1) + 2*(kernel_size-1-padding) - kernel_size + 1
        w_out = w + (stride-1)*(w-1) + 2*(kernel_size-1-padding) - kernel_size + 1

        last_layer = list(self.layers.keys())[-1]

        if last_layer[0] == 'l':
            self.lin_layer(c*h*w, activ=None)

        params = {
                  'layer': self.n_deconv_layers,
                  'input': (c,h,w),
                  'kernel_list':kernel_list,
                  'output':(n_chan, h_out, w_out),
                  'stride':stride,
                  'padding':padding,
                  'relu': relu,
                  'sigmoid': sigmoid
                  }

        if bias:
            params['bias'] = b

        self.layers[f'deconv{self.n_deconv_layers}'] = params

        if self.optimSGDM:
            self.pond[f'deconv{self.n_deconv_layers}'] = {'VdW':np.zeros((n_chan, c, kernel_size, kernel_size)), 'Vdb': np.zeros(n_chan,)}



    def maxpooling_layer(self, kernel_size=2, stride=2):
        self.n_mpool_layers += 1
        self.n_layers += 1

        c, h, w = self.layers[f'conv{self.n_conv_layers}']['output']

        self.layers[f'mpool{self.n_mpool_layers}'] = {
                                                      'layer': self.n_layers - 1,
                                                      'input': (c, h, w),
                                                      'output': (c, (h - kernel_size)//stride + 1, (w - kernel_size)//stride + 1),
                                                      'stride': stride,
                                                      'kernel_size':kernel_size
                                                     }


    
    def relu(self, x):
        return np.maximum(0,x)
    
    
    def d_relu(self, x):
            return (x > 0).astype(int)
    

    def one_hot(self, Y, num_classes):
        y = np.asarray(Y).ravel()                           ## aplana Y para que sea un arreglo 1D
        m = y.size
        oh = np.zeros((num_classes, m), dtype=float)
        oh[y, np.arange(m)] = 1.0
        return oh

    
    def softmax(self, x):
            exps = np.exp(x - np.max(x, axis=0, keepdims=True))
            return exps / np.sum(exps, axis=0, keepdims=True)


    def cross_entropy(self, y_hat, y):
        eps = 1e-12
        y_hat = np.clip(y_hat, eps, 1.0 - eps)          # (n_classes, m)
        y = np.asarray(y).ravel()
        m = y.size
        return -np.mean(np.log(y_hat[y, np.arange(m)]))



    def sigmoid(self, x):
        return expit(x)



    def binary_cross_entropy(self, y_true, last_layer, eps=1e-15):
        """
        Calcula Binary Cross Entropy entre valores reales y predicciones.
        
        Parámetros:
        y_true: np.array con etiquetas reales (0 o 1, o valores en [0,1]).
        y_pred: np.array con predicciones en [0,1] (salida de la sigmoide).
        eps: valor pequeño para evitar log(0).
        """
        # Clip para evitar log(0)
        y_pred = self.cache[f'{last_layer}_act']
        y_pred = np.clip(y_pred, eps, 1 - eps)
        
        # Fórmula BCE
        loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return loss



    def model_info(self):
        for i, (tipo, layer) in enumerate(self.layers.items()):
            print(f'capa {i}')
            if (tipo[0] == 'c'):
                print('\ttipo:', tipo)
                print('\tinput shape:', layer['input'])
                print('\tkernel list shape:', layer['kernel_list'].shape)
                print('\toutput shape:', layer['output'])

            elif (tipo[0] == 'l'):
                print('\ttipo:', tipo)
                print('\tW shape:', layer['W'].shape)
                act = 'relu'
                if layer['softmax']:
                    act = 'softmax'
                elif not layer['relu']:
                    act = None

                print('\tactivation:', act)

            elif (np.strings.startswith(tipo, 'deconv')):
                print('\ttipo:', tipo)
                print('\tinput shape:', layer['input'])
                print('\tkernel list shape:', layer['kernel_list'].shape)
                print('\toutput shape:', layer['output'])
                act = 'relu'
                if layer['sigmoid']:
                    act = 'sigmoid'
                print('\tactivation:', act)

            else:
                print('\ttipo:', tipo)
                print('\tinput shape:', layer['input'])
                print('\toutput shape:', layer['output'])
            
            print('\n')



    def conv_batch(self, x, kernel_list, padding=0, stride=1):
        """
        x: (batch, c_in, h, w)
        kernel_list: (c_out, c_in, h_k, w_k)
        padding: int
        stride: int
        devuelve: (batch, c_out, h_out, w_out)
        """
        batch, c_in, h, w = x.shape
        _, _, h_k, w_k = kernel_list.shape

        # Padding
        if padding != 0:
            x = np.pad(x, ((0,0),(0,0),(padding,padding),(padding,padding)), mode='constant')

        # Tamaño de la salida
        h_out = (h + 2*padding - h_k)//stride + 1
        w_out = (w + 2*padding - w_k)//stride + 1

        # Extraer todas las ventanas del batch
        shape = (batch, c_in, h_out, w_out, h_k, w_k)
        strides = (x.strides[0], x.strides[1], stride*x.strides[2], stride*x.strides[3], x.strides[2], x.strides[3])
        windows = as_strided(x, shape=shape, strides=strides)

        # Tensordot: multiplica cada ventana por cada kernel
        out = np.tensordot(windows, kernel_list, axes=([1,4,5],[1,2,3]))

        # tensordot devuelve (batch, h_out, w_out, c_out) → transponemos a (batch, c_out, h_out, w_out)
        out = out.transpose(0,3,1,2)

        return out



    def deconv_batch(self, x, kernel_list, output_shape, padding=0, stride=1):
        batch, c_in, _, _ = x.shape
        c_out, h_out, w_out = output_shape
        _, _, h_k, w_k = kernel_list.shape

        out = np.zeros((batch, c_out, h_out, w_out))

        k_rot = kernel_list[:, :, ::-1, ::-1]

        _, _, h_in, w_in = x.shape

        h_dil = (h_in-1) * stride + 1
        w_dil = (w_in-1) * stride + 1

        for b in range(batch):
            for cin in range(c_in):
                # "upsample" x insertando ceros según stride
                upsampled = np.zeros((h_dil, w_dil))
                upsampled[::stride, ::stride] = x[b, cin]
                for cout in range(c_out):
                    # correlación con kernel
                    out[b, cout] += signal.correlate2d(
                        upsampled,
                        k_rot[cout, cin],  # ojo: eje correcto
                        mode="full"
                    )[padding:padding+h_out, padding:padding+w_out]

        return out



    def maxpooling_batch(self, x, kernel_size=2, stride=2):
        """
        x: (batch, c, h, w)
        kernel_size: tamaño del kernel
        stride: stride
        devuelve: (batch, c, h_out, w_out)
        """
        batch, c, h, w = x.shape
        h_out = (h - kernel_size)//stride + 1
        w_out = (w - kernel_size)//stride + 1

        shape = (batch, c, h_out, w_out, kernel_size, kernel_size)
        strides = (x.strides[0], x.strides[1], stride*x.strides[2], stride*x.strides[3], x.strides[2], x.strides[3])
        windows = as_strided(x, shape=shape, strides=strides)

        out = windows.max(axis=(4,5))
        flat_windows = windows.reshape(batch, c, h_out, w_out, kernel_size*kernel_size)
        idx = flat_windows.argmax(axis=-1)
        return out, idx


    def maxpooling_backward(self, dout, idx, input_shape, kernel_size=2, stride=2):
        batch, ch, h, w = input_shape

        dx = np.zeros(input_shape, dtype=dout.dtype)
        h_out = (h - kernel_size)//stride + 1
        w_out = (w - kernel_size)//stride + 1

        for b in range(batch):
            for c in range(ch):
                for i in range(h_out):
                    for j in range(w_out):
                        idx_flat = idx[b, c, i, j]

                        di, dj = divmod(idx_flat, kernel_size)

                        row = i * stride + di
                        col = j * stride + dj

                        dx[b, c, row, col] += dout[b, c, i, j]
        
        return dx



    def conv2d_backward_weights(self, x, dout, kernel_shape, padding=0, stride=1):
        batch, _, _, _ = x.shape
        _, _, h_o, w_o = dout.shape

        _, c_in, h_k, w_k = kernel_shape

        if padding != 0:
            x = np.pad(x, ((0,0),(0,0),(padding,padding),(padding,padding)), mode='constant')

        shape = (batch, c_in, h_o, w_o, h_k, w_k)
        strides = (x.strides[0], x.strides[1], stride*x.strides[2], stride*x.strides[3], x.strides[2], x.strides[3])
        windows = as_strided(x, shape=shape, strides=strides)

        out = np.tensordot(windows, dout, axes=([0,2,3],[0,2,3]))
        out = out.transpose(3,0,1,2)   # (c_out, c_in, h_k, w_k)

        return out



    def conv2d_backward_input(self, dout, kernel, padding=0, stride=1):
        # dout: (B, C_out, H_out, W_out)
        # kernel: (C_out, C_in, K_h, K_w)
        B, C_out, H_out, W_out = dout.shape
        C_out_k, C_in, K_h, K_w = kernel.shape
        assert C_out_k == C_out

        # dilatar si stride > 1
        if stride > 1:
            H_dil = (H_out-1) * stride + 1
            W_dil = (W_out-1) * stride + 1
            dout_dil = np.zeros((B, C_out, H_dil, W_dil), dtype=dout.dtype)
            dout_dil[:, :, ::stride, ::stride] = dout
        else:
            dout_dil = dout

        # padding "traspuesto"
        pad_h = K_h - 1 - padding
        pad_w = K_w - 1 - padding
        dout_pad = np.pad(dout_dil, ((0,0),(0,0),(pad_h, pad_h),(pad_w, pad_w)), mode='constant')

        # kernel rotado 180°
        k_rot = kernel[:, :, ::-1, ::-1]  # (C_out, C_in, K_h, K_w)

        # correlación válida por canal
        B, _, Hp, Wp = dout_pad.shape
        H_in = Hp - K_h + 1
        W_in = Wp - K_w + 1
        dX = np.zeros((B, C_in, H_in, W_in), dtype=dout.dtype)

        for b in range(B):
            for cin in range(C_in):
                acc = np.zeros((H_in, W_in), dtype=dout.dtype)
                for cout in range(C_out):
                    acc += signal.correlate2d(dout_pad[b, cout], k_rot[cout, cin], mode="valid")
                dX[b, cin] = acc
        return dX



    def deconv2d_backward_weights(self, dout, x, kernel_shape, padding=0, stride=1):                ## padding = padding utilizado en la respectiva capa
        ## dout : (batch, c_out, h_out, w_out)

        if padding != 0:
            dout = np.pad(dout, ((0,0),(0,0),(padding,padding),(padding,padding)), mode='constant')

        _, c_out, h_out, w_out = dout.shape
        batch, c_in, h_in, w_in = x.shape

        _, _, h_k, w_k = kernel_shape

        h_mod = (stride-1)*(h_in-1) + h_in
        w_mod = (stride-1)*(w_in-1) + w_in

        ups = np.zeros((batch, c_in, h_mod, w_mod), dtype=x.dtype)
        ups[:, :, ::stride, ::stride] = x

        shape = (batch, c_out, h_k, w_k, h_mod, w_mod)
        strides = (dout.strides[0], dout.strides[1], stride*dout.strides[2], stride*dout.strides[3], dout.strides[2], dout.strides[3])
        windows = as_strided(dout, shape=shape, strides=strides)

        out = np.tensordot(windows, ups, axes=([0,4,5],[0,2,3]))
        out = out.transpose(0,3,1,2)   # (c_out, c_in, h_k, w_k)

        return out



    def deconv2d_backward_input(self, dout, kernel, input_shape, padding=0, stride=1):
        c_in, h_in, w_in = input_shape
        batch, c_out, h_out, w_out = dout.shape

        if padding != 0:
            dout = np.pad(dout, ((0,0),(0,0),(padding,padding),(padding,padding)), mode='constant')

        out = np.zeros((batch, c_in, h_out, w_out))

        k_rot = kernel[:, :, ::-1, ::-1]

        for b in range(batch):
            for cout in range(c_out):
                for cin in range(c_in):

                    out[b, cin] += signal.correlate2d(
                        dout[b, cout],
                        k_rot[cout, cin],
                        mode="valid")

        out = out[:, :, ::stride, ::stride]

        return out



    def forward_pass(self, x):
        for tipo, layer in self.layers.items():
            if tipo[0] == 'c':
                self.cache[f'{tipo}_in'] = x
                t_out = self.conv_batch(x, layer['kernel_list'], layer['padding'], layer['stride'])
                if 'bias' in layer:
                    t_out += layer['bias'][None,:, None, None]
                
                x = self.relu(t_out)

                self.cache[f'{tipo}_out'] = t_out

            elif tipo[0] == 'm':
                self.cache[f'{tipo}_in'] = x
                t_out, idx = self.maxpooling_batch(x, layer['kernel_size'], layer['stride'])
                x = t_out

                self.cache[f'{tipo}_out'] = x
                self.cache[f'{tipo}_idx'] = idx

            elif tipo[0] == 'l':
                if tipo == 'lin1' or tipo == 'lino1':
                    cant, _, _, _ = x.shape
                    x = x.reshape(cant, -1).T
                    

                self.cache[f'{tipo}_in'] = x
                t_out = layer['W'].dot(x) + layer['b']

                if layer['relu']:
                    x = self.relu(t_out)
                    self.cache[f'{tipo}_out'] = t_out

                elif layer['softmax']:
                    x = self.softmax(t_out)
                    self.cache[f'{tipo}_out'] = t_out
                    self.cache[f'{tipo}_act'] = x

                else:
                    x = t_out
                    self.cache[f'{tipo}_out'] = t_out
                

            elif tipo[0] == 'd':
                if x.ndim != 4:
                    c, h, w = layer['input']
                    x = x.reshape(cant, c, h, w)

                self.cache[f'{tipo}_in'] = x
                t_out = self.deconv_batch(x, layer['kernel_list'], layer['output'], layer['padding'], layer['stride'])

                if 'bias' in layer:
                    t_out += layer['bias'][None,:, None, None]

                self.cache[f'{tipo}_out'] = t_out
                if layer['relu']:
                    x = self.relu(t_out)
                
                else:
                    x = self.sigmoid(t_out)
                    self.cache[f'{tipo}_act'] = x
                
            else:
                raise ValueError("error inesperado")

        return x


    def back_propagation(self, x, Y):
        capas= list(self.layers.keys())
        y = np.asarray(Y).ravel()          # asegurar 1D
        p = y.size

        dout = None 
        
        for tipo in reversed(range(self.n_layers)):
            if capas[tipo][0] == 'c':
                dout = self.d_relu(self.cache[f'{capas[tipo]}_out']) * dout     ## dout lado derecho = X de la prox capa conv = salida activada de la capa conv actual

                self.grads[capas[tipo]] = {'dW': 1/p * self.conv2d_backward_weights(self.cache[f'{capas[tipo]}_in'], dout, self.layers[f'{capas[tipo]}']['kernel_list'].shape),
                                           'db': 1/p * np.sum(dout, axis=(0, 2, 3), keepdims=True)}

                dout = self.conv2d_backward_input(dout, self.layers[capas[tipo]]['kernel_list'], self.layers[capas[tipo]]['padding'], self.layers[capas[tipo]]['stride'])        ## dout = dL/dX, X = entrada de la conv layer

                if self.optimSGDM:
                    self.pond[capas[tipo]]['VdW'] = self.momentum*self.pond[capas[tipo]]['VdW'] + (1-self.momentum)*self.grads[capas[tipo]]['dW']
                    self.pond[capas[tipo]]['Vdb'] = self.momentum*self.pond[capas[tipo]]['Vdb'] + (1-self.momentum)*self.grads[capas[tipo]]['db']

            elif capas[tipo][0] == 'm':
                dout = self.maxpooling_backward(dout, self.cache[f'{capas[tipo]}_idx'], self.cache[f'{capas[tipo]}_in'].shape, self.layers[f'{capas[tipo]}']['kernel_size'], self.layers[f'{capas[tipo]}']['stride'])

            elif capas[tipo][0] == 'l':
                name = capas[tipo]
                if self.layers[capas[tipo]]['softmax']:
                    n_classes = self.layers[name]['output'][0]
                    one_hot_y = self.one_hot(Y, n_classes)
                    dout = self.cache[f'{capas[tipo]}_act'] - one_hot_y             ## dout.shape = (features, batch_size)
                
                elif self.layers[capas[tipo]]['relu']:
                    if capas[tipo+1][0] == 'l':
                        dout = self.layers[f'{capas[tipo+1]}']['W'].T.dot(dout) * self.d_relu(self.cache[f'{capas[tipo]}_out'])

                else:
                    if capas[tipo+1][0] == 'l':
                        dout = self.layers[f'{capas[tipo+1]}']['W'].T.dot(dout)

                self.grads[capas[tipo]] = {'dW': 1/p * dout.dot(self.cache[f'{capas[tipo]}_in'].T), 'db': 1/p * np.sum(dout, axis=1, keepdims=True)}
                

                if name == 'lin1':
                    prev_name = capas[tipo-1]                                       ## debe ser conv o pool
                    dout = self.layers[capas[tipo]]['W'].T.dot(dout)
                    dout = dout.reshape(self.cache[f'{prev_name}_out'].shape)
                
                if self.optimSGDM:
                    self.pond[capas[tipo]]['VdW'] = self.momentum*self.pond[capas[tipo]]['VdW'] + (1-self.momentum)*self.grads[capas[tipo]]['dW']
                    self.pond[capas[tipo]]['Vdb'] = self.momentum*self.pond[capas[tipo]]['Vdb'] + (1-self.momentum)*self.grads[capas[tipo]]['db']



            elif capas[tipo][0] == 'd':
                layer = self.layers[capas[tipo]]

                if layer['sigmoid']:
                    dout = self.cache[f'{capas[tipo]}_out'] - x
                    
                else:
                    dout = self.d_relu(self.cache[f'{capas[tipo]}_out']) * dout

                self.grads[capas[tipo]] = {'dW': 1/p * self.deconv2d_backward_weights(dout, self.cache[f'{capas[tipo]}_in'], layer['kernel_list'].shape, layer['padding'], layer['stride']),
                                           'db': 1/p * np.sum(dout, axis=(0, 2, 3), keepdims=True)}
                
                dout = self.deconv2d_backward_input(dout, layer['kernel_list'], layer['input'], layer['padding'], layer['stride'])
    

                if capas[tipo-1][0] == 'l':
                    dout = dout.reshape(self.cache[f'{capas[tipo-1]}_out'].shape)

                if self.optimSGDM:
                    self.pond[capas[tipo]]['VdW'] = self.momentum*self.pond[capas[tipo]]['VdW'] + (1-self.momentum)*self.grads[capas[tipo]]['dW']
                    self.pond[capas[tipo]]['Vdb'] = self.momentum*self.pond[capas[tipo]]['Vdb'] + (1-self.momentum)*self.grads[capas[tipo]]['db']
            
            else:
                if self.layers[capas[tipo]]['softmax']:
                    n_classes = self.layers[name]['output'][0]
                    one_hot_y = self.one_hot(Y, n_classes)
                    dout = self.cache[f'{capas[tipo]}_act'] - one_hot_y             ## dout.shape = (features, batch_size)
                
                elif self.layers[capas[tipo]]['relu']:
                    dout = self.layers[f'{capas[tipo+1]}']['W'].T.dot(dout) * self.d_relu(self.cache[f'{capas[tipo]}_out'])

                self.grads[f'{capas[tipo]}'] = {'dW': 1/p * dout.dot(self.cache[f'{capas[tipo]}_in'].T), 'db': 1/p * np.sum(dout, axis=1, keepdims=True)}

                if self.optimSGDM:
                    self.pond[capas[tipo]]['VdW'] = self.momentum*self.pond[capas[tipo]]['VdW'] + (1-self.momentum)*self.grads[capas[tipo]]['dW']
                    self.pond[capas[tipo]]['Vdb'] = self.momentum*self.pond[capas[tipo]]['Vdb'] + (1-self.momentum)*self.grads[capas[tipo]]['db']


    
    def grad_descent(self, mu):
        if self.optimSGD:
            for tipo in self.layers.keys():
                if tipo[0] == 'c':
                    self.layers[tipo]['kernel_list'] = self.layers[tipo]['kernel_list'] - self.grads[tipo]['dW']*mu
                    self.layers[tipo]['bias'] = self.layers[tipo]['bias'] - np.squeeze(self.grads[tipo]['db'])*mu

                if tipo[0] == 'l':
                    self.layers[tipo]['W'] = self.layers[tipo]['W'] - self.grads[tipo]['dW']*mu
                    self.layers[tipo]['b'] = self.layers[tipo]['b'] - self.grads[tipo]['db']*mu

                if tipo[0] == 'd':
                    self.layers[tipo]['kernel_list'] = self.layers[tipo]['kernel_list'] - self.grads[tipo]['dW']*mu
                    self.layers[tipo]['bias'] = self.layers[tipo]['bias'] - np.squeeze(self.grads[tipo]['db'])*mu

        elif self.optimSGDM:
            for tipo in self.layers.keys():
                if tipo[0] == 'c':
                    self.layers[tipo]['kernel_list'] = self.layers[tipo]['kernel_list'] - mu*self.pond[tipo]['VdW']
                    self.layers[tipo]['bias'] = self.layers[tipo]['bias'] - mu*np.squeeze(self.pond[tipo]['Vdb'])

                if tipo[0] == 'l':
                    self.layers[tipo]['W'] = self.layers[tipo]['W'] - mu*self.pond[tipo]['VdW']
                    self.layers[tipo]['b'] = self.layers[tipo]['b'] - mu*self.pond[tipo]['Vdb']

                if tipo[0] == 'd':
                    self.layers[tipo]['kernel_list'] = self.layers[tipo]['kernel_list'] - mu*self.pond[tipo]['VdW']
                    self.layers[tipo]['bias'] = self.layers[tipo]['bias'] - mu*np.squeeze(self.pond[tipo]['Vdb'])


    def learning(self, epoch, batch_size, learning_rate, x, Y, n_doc=10):
        n_batches = Y.size//batch_size
        last_layer = list(self.layers.keys())[-1]
        
        f_mod = (n_batches*epoch)/n_doc

        for e in range(epoch):
            for batch in range(n_batches):
                
                self.forward_pass(x[batch*batch_size:(batch+1)*batch_size])
                self.back_propagation(x[batch*batch_size:(batch+1)*batch_size],Y[:,batch*batch_size:(batch+1)*batch_size])
                self.grad_descent(learning_rate)

                x_sample = self.cache[f'{last_layer}_act'][:4, 0]

                clear_output(wait=True)

                for i in range(len(x_sample)):
                    plt.subplot(2, len(x_sample), i+1)
                    plt.imshow(x_sample[i], cmap='gray')  # reshape según tus datos
                    plt.axis('off')

                plt.suptitle(f"Epoch {epoch+1}")
                plt.suptitle(f'epoch:{e+1}; {batch+1}/{n_batches} batch')
                plt.show()
                print("pérdida", self.binary_cross_entropy(x[batch*batch_size:(batch+1)*batch_size],last_layer))




    def get_prediction(self):
        return np.argmax(self.cache[f'lin{self.n_lin_layers}_act'], axis=0)


    def get_accuracy(self, pred, y):
        return np.sum(pred == y)/y.size
    

    def test_predic(self, x):
        self.forward_pass(x)
        return (self.get_prediction())
    
    def output(self):
        last_layer = list(self.layers.keys())[-1]
        return self.cache[f'{last_layer}_act']
      

    def save_params(self, src):
        with open(f'{src}.pkl', 'wb') as f:
            pickle.dump(self.layers, f)

    
    def load_params(self, src):
        with open(f'{src}.pkl', 'rb') as f:
            self.layers_loaded = pickle.load(f)
