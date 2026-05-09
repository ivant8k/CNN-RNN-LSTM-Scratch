import numpy as np
from src.shared.activations import Activation

class LocallyConnected2DLayer:
    def __init__(self, keras_layer):
        # TODO: load bobot (harusnya masih sama kayak conv2d)
        weights = keras_layer.get_weights()
        self.kernel = weights[0]
        self.bias = weights[1]

        cfg = keras_layer.get_config()
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

                out[i, j, :] = np.dot(patch_flat, self.kernel[:, pos_idx]) + self.bias

        return self._apply_activation(out)

    # TODO: Implement backward pass (BONUS)
    def backward(self):
        pass