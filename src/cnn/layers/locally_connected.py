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
        self.padding = cfg.get('padding', 'valid')
        self.activation = cfg['activation']

    def _get_pad(self, H, W):
        if self.padding == 'valid':
            return 0, 0
        pad_h = max((H - 1) * self.stride + self.kH - H, 0)
        pad_w = max((W - 1) * self.stride + self.kW - W, 0)

        return pad_h // 2, pad_w // 2

    def forward(self, x: np.ndarray) -> np.ndarray:
        H, W, C_in = x.shape
        
        pad_h, pad_w = self._get_pad(H, W)
        if pad_h > 0 or pad_w > 0:
            x = np.pad(x, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode="constant")
            H, W = x.shape[:2]

        out_H = (H - self.kH) // self.stride + 1
        out_W = (W - self.kW) // self.stride + 1
        
        expected_output_pixels = out_H * out_W
        assert self.kernel.shape[0] == expected_output_pixels, \
            f"Mismatch: Kernel memiliki {self.kernel.shape[0]} posisi, tetapi output butuh {expected_output_pixels}"

        out = np.zeros((out_H, out_W, self.kernel.shape[-1]))

        for i in range(out_H):
            for j in range(out_W):
                pos_idx = i * out_W + j
                
                patch = x[i*self.stride : i*self.stride + self.kH,
                          j*self.stride : j*self.stride + self.kW, :]
                
                patch_flat = patch.ravel()
                
                out[i, j, :] = np.dot(patch_flat, self.kernel[pos_idx]) + self.bias[pos_idx]

        return self._apply_activation(out)

    def _apply_activation(self, x):
        """Abstraksi fungsi aktivasi."""
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

    def backward(self):
        # BONUS: Belum diimplementasikan
        pass