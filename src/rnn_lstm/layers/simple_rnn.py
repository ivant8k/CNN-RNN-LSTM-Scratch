import numpy as np
from typing import List

class SimpleRNN:
    
    input_dim: int
    hidden_size: int
    W_xh: np.ndarray
    W_hh: np.ndarray
    b_h: np.ndarray

    def __init__(self, input_dim: int, hidden_size: int) -> None:
        """
        Parameters:
        - input_dim     : Dimensi input fitur (embed_dim)
        - hidden_size   : Ukuran hidden state
        """
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        
        # Inisialisasi bobot sementara
        self.W_xh = np.zeros((input_dim, hidden_size), dtype=np.float32)
        self.W_hh = np.zeros((hidden_size, hidden_size), dtype=np.float32)
        self.b_h = np.zeros((hidden_size,), dtype=np.float32)

    def set_weights(self, keras_weights: List[np.ndarray]) -> None:
        """
        Parameters:
        - keras_weights: [kernel, recurrent_kernel, bias]
        """
        if len(keras_weights) != 3:
            raise ValueError(f"SimpleRNN membutuhkan 3 matriks bobot, menerima {len(keras_weights)}")
            
        W_xh, W_hh, b_h = keras_weights
        
        # Validasi dimensi bobot
        assert W_xh.shape == (self.input_dim, self.hidden_size), \
            f"Dimensi W_xh salah. Ekspektasi {(self.input_dim, self.hidden_size)}, dapat {W_xh.shape}"
        assert W_hh.shape == (self.hidden_size, self.hidden_size), \
            f"Dimensi W_hh salah. Ekspektasi {(self.hidden_size, self.hidden_size)}, dapat {W_hh.shape}"
        assert b_h.shape == (self.hidden_size,), \
            f"Dimensi b_h salah. Ekspektasi {(self.hidden_size,)}, dapat {b_h.shape}"
            
        self.W_xh = W_xh
        self.W_hh = W_hh
        self.b_h = b_h

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Parameters:
        - inputs: Tensor 3D dari embedding layer
        - Shape : (batch_size, sequence_length, input_dim)
        
        Returns:
        - np.ndarray: Tensor 3D hidden state untuk seluruh timestep
        - Shape     : (batch_size, sequence_length, hidden_size)
        """
        batch_size, seq_len, current_input_dim = inputs.shape
        
        if current_input_dim != self.input_dim:
            raise ValueError(f"Input dim tidak cocok. Ekspektasi {self.input_dim}, dapat {current_input_dim}")

        # Inisialisasi h_0 dan array Ouput
        h_t = np.zeros((batch_size, self.hidden_size), dtype=np.float32)
        outputs = np.zeros((batch_size, seq_len, self.hidden_size), dtype=np.float32)

        for t in range(seq_len):
            x_t = inputs[:, t, :] # Shape: (batch_size, input_dim)
            
            # Calculate: (x_t * W_xh) + (h_{t-1} * W_hh) + b_h
            linear_combination = np.dot(x_t, self.W_xh) + np.dot(h_t, self.W_hh) + self.b_h
            
            # Activation
            h_t = np.tanh(linear_combination)
            
            # Output h_t
            outputs[:, t, :] = h_t
            
        return outputs