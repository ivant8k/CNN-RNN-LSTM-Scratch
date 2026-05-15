import numpy as np

class FlattenLayer:
    def forward(self, input):
        self.input_shape = input.shape
        return input.flatten()
    
    def backward():
        pass