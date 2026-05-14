import numpy as np

class Embedding:
    
    vocab_size: int
    embed_dim: int
    weights: np.ndarray

    def __init__(self, vocab_size: int, embed_dim: int) -> None:
        """
        Parameters:
        - vocab_size: Ukuran total vocabulary
        - embed_dim : Dimensi vektor embedding yang dihasilkan
        """
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.weights = np.zeros((vocab_size, embed_dim), dtype=np.float32)

    def set_weights(self, keras_weights: np.ndarray) -> None:
        """
        Parameters:
        - keras_weights: Bobot embedding dari Keras
        """
        if isinstance(keras_weights, list):
            weight_matrix: np.ndarray = keras_weights[0]
        else:
            weight_matrix: np.ndarray = keras_weights
            
        if weight_matrix.shape != (self.vocab_size, self.embed_dim):
            raise ValueError(
                f"Dimensi bobot Keras {weight_matrix.shape} tidak cocok "
                f"dengan spesifikasi Embedding {(self.vocab_size, self.embed_dim)}"
            )
            
        self.weights = weight_matrix

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Parameters:
        - inputs: Matriks 2D berisi indeks token dengan tipe integer
        - Shape : (batch_size, sequence_length)
            
        Returns:
        - np.ndarray: Tensor 3D hasil pemetaan indeks ke vektor. 
        - Shape     : (batch_size, sequence_length, embed_dim)
        """
        if not np.issubdtype(inputs.dtype, np.integer):
            raise TypeError("Input untuk Embedding layer harus berupa array integer (indeks token).")

        out: np.ndarray = self.weights[inputs]
        return out