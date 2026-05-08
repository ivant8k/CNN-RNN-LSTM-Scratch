import numpy as np

class MaxPooling2DLayer:
    def __init__ (self, keras_layer):
        cfg = keras_layer.get_config()
        self.pool_h, self.pool_w = cfg["pool_size"]
        self.stride_h, self.stride_w = cfg['strides']

    def forward(self, x: np.ndarray) -> np.ndarray:
        H, W, C = x.shape
        out_H = (H - self.pool_h) // self.stride_h + 1
        out_W = (W - self.pool_w) // self.stride_w + 1
        out = np.zeros((out_H, out_W, C))

        for i in range(out_H):
            for j in range(out_W):
                patch = x[i*self.stride_h : i * self.stride_h+self.pool_h,
                          j*self.stride_w : j * self.stride_w+self.pool_w, :]
                out[i, j, :] = np.max(patch, axis=(0, 1))

        return out
        

class AveragePooling2DLayer:
    def __init__ (self, keras_layer):
        cfg = keras_layer.get_config()
        self.pool_h, self.pool_w = cfg["pool_size"]
        self.stride_h, self.stride_w = cfg['strides']

    def forward(self, x: np.ndarray) -> np.ndarray:
        H, W, C = x.shape
        out_H = (H - self.pool_h) // self.stride_h + 1
        out_W = (W - self.pool_w) // self.stride_w + 1
        out   = np.zeros((out_H, out_W, C), dtype=np.float32)

        for i in range(out_H):
            for j in range(out_W):
                window = x[i*self.stride_h : i*self.stride_h + self.pool_h,
                           j*self.stride_w : j*self.stride_w + self.pool_w, :]
                out[i, j, :] = np.mean(window, axis=(0, 1))

        return out
    
class GlobalAveragePooling2DLayer:
    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.mean(x, axis=(0, 1))

class GlobalMaxPooling2DLayer:
    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.max(x, axis=(0, 1))
    