import numpy as np
from src.shared.activations import Activation

class Conv2DLayer:
    def __init__(self, keras_layer):
        weights = keras_layer.get_weights()
        # weights[0] -> kernel; shape (kH, kW, C_in, C_out)
        # weights[1] -> bias; shape (C_out,)
        self.kernel = weights[0]
        self.bias = weights[1]

        cfg = keras_layer.get_config()
        self.strides = cfg['strides'][0]
        self.padding = cfg['padding']
        self.activation = cfg['activation']

        self.kH, self.kW, self.C_in, self.C_out = self.kernel.shape
        
    # Padding helper (valid vs same)
    def _get_pad(self, H, W):
        if self.padding == 'valid':
            return 0,0
        
        pad_h = max((H - 1) * self.stride + self.kH - H, 0)
        pad_w = max((W - 1) * self.stride + self.kW - W, 0)

        return pad_h // 2, pad_w // 2
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        # TODO: Implement forward pass
        # pass
        H, W, C_in = x.shape
        assert C_in == self.C_in

        # pad_h, pad_w = self.kH // 2, self.kW // 2
        pad_h, pad_w = self._get_pad(H, W)
        if pad_h > 0 or pad_w > 0:
            x = np.pad(x, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode="constant")
            H, W = x.shape[:2]

        out_H = (H - self.kH) // self.strides + 1
        out_W = (W - self.kW) // self.strides + 1
        out = np.zeros((out_H, out_W, self.C_out))

        # Naive approaches -> TODO: Klo ada waktu, cari optimasi lain
        for i in range(out_H):
            for j in range(out_W):
                patch = x[i*self.stride : i * self.stride+self.kH,
                          j*self.stride : j * self.stride+self.kW, :]
                out[i, j, :] = np.einsum("hwc, hwck -> k", patch, self.kernel) + self.bias

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
