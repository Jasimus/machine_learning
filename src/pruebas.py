import numpy as np
from scipy import signal


def deconv_batch(x, kernel_list, output_shape, padding=0, stride=1):
        c_in, h_in, w_in = x.shape
        c_out, h_out, w_out = output_shape

        out = np.zeros((c_out, h_out, w_out))

        for cin in range(c_in):
            for cout in range(c_out):
                # "upsample" x insertando ceros según stride
                k_rot = np.rot90(kernel_list[cout], 2)
                upsampled = np.zeros((h_in*stride, w_in*stride))
                upsampled[::stride, ::stride] = x[cin]

                # correlación con kernel
                out[cout] += signal.correlate2d(
                    upsampled,
                    k_rot,  # ojo: eje correcto
                    mode="full"
                )[padding:padding+h_out, padding:padding+w_out]

        return out

## upsampling

# matriz original
M = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

stride = 2  # cuántos ceros insertar entre elementos

# 1) crear matriz expandida con ceros
h, w = M.shape
H, W = h*stride - (stride-1), w*stride - (stride-1)  # tamaño final
upsampled = np.zeros((H, W), dtype=M.dtype)

# 2) rellenar posiciones correspondientes
upsampled[::stride, ::stride] = M

print(upsampled)


# kernel = np.array([[[-1, 1, 0],
#         [-2, 3, 1],
#         [1, 2, -1]]])

# kernel_rot = kernel[:, ::-1, ::-1]

# x = np.array([[[2, 3],[-2, 1]]])


# c_in, h_in, w_in = x.shape
# _, h_k, w_k = kernel.shape

# s = 2
# p = 1

# h_out = h_in + (s-1)*(h_in-1) + 2*(h_k-1-p) - h_k + 1
# w_out = w_in + (s-1)*(w_in-1) + 2*(w_k-1-p) - w_k + 1

# print(deconv_batch(x, kernel, (1, h_out, w_out), p, s))


## conclusión: en deconv_batch es necesario utilizar el kernel rotado para obtener el resultado correcto