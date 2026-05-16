import numpy as np
import tensorflow as tf

from src.cnn.layers.conv2d import Conv2DLayer
from src.cnn.layers.locally_connected import LocallyConnected2DLayer
from src.cnn.layers.pooling import (MaxPooling2DLayer, AveragePooling2DLayer, GlobalMaxPooling2DLayer, GlobalAveragePooling2DLayer)
from src.cnn.layers.flatten import FlattenLayer
from src.shared.layer import Dense
from src.shared.activations import Activation


class ActivationLayer:
    def __init__(self, activation_name: str):
        self.activation_name = activation_name
        self.activations = Activation()

    def forward(self, x: np.ndarray) -> np.ndarray:
        if self.activation_name == 'relu':
            return self.activations.relu(x)
        elif self.activation_name == 'sigmoid':
            return self.activations.sigmoid(x)
        elif self.activation_name == 'tanh':
            return self.activations.tanh(x)
        elif self.activation_name == 'softmax':
            return self.activations.softmax(x)
        return x


class CNNModel:
    LAYER_MAP = {
        'Conv2D': Conv2DLayer,
        'LocallyConnected2D': LocallyConnected2DLayer,
        'MaxPooling2D': MaxPooling2DLayer,
        'AveragePooling2D': AveragePooling2DLayer,
        'GlobalMaxPooling2D': GlobalMaxPooling2DLayer,
        'GlobalAveragePooling2D': GlobalAveragePooling2DLayer,
        'Flatten': FlattenLayer,
        'Dense': Dense,
    }

    def __init__(self, keras_model: tf.keras.Model):
        self.layers = self._build_from_keras(keras_model)

    def _build_from_keras(self, keras_model):
        scratch_layers = []

        for keras_layer in keras_model.layers:
            layer_type = type(keras_layer).__name__

            if layer_type == 'InputLayer':
                continue

            if layer_type not in self.LAYER_MAP:
                raise ValueError(
                    f"Layer tipe '{layer_type}' belum didukung. "
                    f"Tambahkan ke LAYER_MAP atau implementasikan layernya."
                )

            ScratchClass = self.LAYER_MAP[layer_type]

            if layer_type in ('Flatten', 'GlobalMaxPooling2D', 'GlobalAveragePooling2D'):
                scratch_layers.append(ScratchClass())
            elif layer_type == 'Dense':
                layer_weights = keras_layer.get_weights()
                weights = layer_weights[0]
                if len(layer_weights) > 1:
                    biases = layer_weights[1]
                else:
                    biases = np.zeros(weights.shape[1], dtype=np.float64)
                
                in_features, out_features = weights.shape
                dense_layer = ScratchClass(in_features, out_features)
                dense_layer.set_weights(weights, biases)
                scratch_layers.append(dense_layer)

                activation_name = keras_layer.get_config().get('activation', 'linear')
                if activation_name != 'linear':
                    scratch_layers.append(ActivationLayer(activation_name))
            else:
                scratch_layers.append(ScratchClass(keras_layer))

        return scratch_layers

    def forward(self, x: np.ndarray) -> np.ndarray:
        out = x.copy()
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def predict(self, x: np.ndarray) -> int:
        probs = self.forward(x)
        return int(np.argmax(probs))

    @classmethod
    def from_layers(cls, layers: list):
        instance = cls.__new__(cls)
        instance.layers = layers
        return instance
