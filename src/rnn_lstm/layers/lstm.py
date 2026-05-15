import numpy as np
from typing import List

class LSTM:
    input_dim: int
    hidden_size: int
    W_x: np.ndarray
    W_h: np.ndarray
    b: np.ndarray

    def __init__(self, input_dim: int, hidden_size: int) -> None:
        """
        Parameters:
        - input_dim     : Dimensi fitur input (embed_dim)
        - hidden_size   : Ukuran hidden state / memori sel
        """
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        
        # 4 gate (i, f, c, o) digabung jadi 1 matriks
        self.W_x = np.zeros((input_dim, hidden_size * 4), dtype=np.float32)
        self.W_h = np.zeros((hidden_size, hidden_size * 4), dtype=np.float32)
        self.b = np.zeros((hidden_size * 4,), dtype=np.float32)

    def set_weights(self, keras_weights: List[np.ndarray]) -> None:
        """
        Parameters:
        - keras_weights: [kernel (W_x), recurrent_kernel (W_h), bias (b)]
        """
        if len(keras_weights) != 3:
            raise ValueError(f"LSTM membutuhkan 3 matriks bobot, menerima {len(keras_weights)}")
            
        W_x, W_h, b = keras_weights
        
        h_4 = self.hidden_size * 4
        assert W_x.shape == (self.input_dim, h_4), f"Dimensi W_x salah. Ekspektasi {(self.input_dim, h_4)}, dapat {W_x.shape}"
        assert W_h.shape == (self.hidden_size, h_4), f"Dimensi W_h salah. Ekspektasi {(self.hidden_size, h_4)}, dapat {W_h.shape}"
        assert b.shape == (h_4,), f"Dimensi bias salah. Ekspektasi {(h_4,)}, dapat {b.shape}"
            
        self.W_x = W_x
        self.W_h = W_h
        self.b = b

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Parameters:
        - inputs: Tensor 3D
        - Shape : (batch_size, sequence_length, input_dim)
            
        Returns:
        - np.ndarray: Hidden state untuk seluruh timestep
        - Shape     : (batch_size, sequence_length, hidden_size)
        """
        batch_size, seq_len, current_input_dim = inputs.shape
        
        if current_input_dim != self.input_dim:
            raise ValueError(f"Input dim tidak cocok. Ekspektasi {self.input_dim}, dapat {current_input_dim}")

        # Inisialisasi h, C, dan output
        h_t = np.zeros((batch_size, self.hidden_size), dtype=np.float32)
        C_t = np.zeros((batch_size, self.hidden_size), dtype=np.float32)
        outputs = np.zeros((batch_size, seq_len, self.hidden_size), dtype=np.float32)

        h = self.hidden_size

        for t in range(seq_len):
            x_t = inputs[:, t, :]
            
            # Calculate 4 Gates
            z = np.dot(x_t, self.W_x) + np.dot(h_t, self.W_h) + self.b  # Shape: (batch_size, 4 * hidden_size)
            
            # Split: i, f, c, o
            z_i = z[:, :h]      # Input gate
            z_f = z[:, h:2*h]   # Forget gate
            z_c = z[:, 2*h:3*h] # Cell candidate
            z_o = z[:, 3*h:]    # Output gate
            
            # Activation
            i = self._sigmoid(z_i)
            f = self._sigmoid(z_f)
            C_tilde = np.tanh(z_c)
            o = self._sigmoid(z_o)
            
            # Update Cell State
            C_t = f * C_t + i * C_tilde
            
            # Update Hidden State
            h_t = o * np.tanh(C_t)
            
            # Output h_t
            outputs[:, t, :] = h_t
            
        return outputs