import numpy as np

class Flatten:
    def forward(self, input):
        self.input_shape = input.shape
        return input.reshape(input.shape[0], -1)
    
    def backward():
        pass