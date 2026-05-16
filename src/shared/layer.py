import numpy as np

class Dense:
    def __init__(self, in_features: int, out_features:int):
        self.in_features = in_features
        self.out_features = out_features

        # Buat sementara zero -> nanti diganti lewat keras
        self.w = np.zeros((in_features, out_features), dtype=np.float64)
        self.b = np.zeros(out_features, dtype=np.float64)

    def set_weights(self, weights: np.ndarray, bias: np.ndarray):
        assert weights.shape == (self.in_features, self.out_features)
        assert bias.shape == (self.out_features,)
        self.w = np.asarray(weights, dtype=np.float64)
        self.b = np.asarray(bias, dtype=np.float64)

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        # self.cache = x 
        return x @ self.w + self.b