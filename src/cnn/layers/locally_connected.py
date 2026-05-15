import numpy as np
from src.shared.activations import Activation

class LocallyConnected2DLayer:
    def __init__(self, keras_layer=None, weights=None, config=None):
        if keras_layer is not None:
            weights = keras_layer.get_weights()
            cfg = keras_layer.get_config()
        else:
            cfg = config

        self.kernel = weights[0]
        self.bias = weights[1]

        self.kH, self.kW = cfg["kernel_size"]
        self.stride = cfg['strides'][0]
        self.activation = cfg['activation']

    def forward(self, x: np.ndarray) -> np.ndarray:
        H, W, C_in = x.shape
        out_H = (H - self.kH) // self.stride + 1
        out_W = (W - self.kW) // self.stride + 1
        out = np.zeros((out_H, out_W, self.kernel.shape[-1]))

        for i in range(out_H):
            for j in range(out_W):
                pos_idx = i * out_W + j
                patch = x[i*self.stride : i * self.stride+self.kH,
                          j*self.stride : j * self.stride+self.kW, :]
                patch_flat = patch.ravel()
                out[i, j, :] = np.dot(patch_flat, self.kernel[pos_idx]) + self.bias[pos_idx]

        return self._apply_activation(out)

    def _apply_activation(self, x):
        if self.activation == 'relu':
            return Activation().relu(x)
        elif self.activation == 'sigmoid':
            return Activation().sigmoid(x)
        elif self.activation == 'tanh':
            return Activation().tanh(x)
        elif self.activation == 'softmax':
            return Activation().softmax(x)
        else:
            return x

    # TODO: Implement backward pass (BONUS)
    def backward(self):
        pass