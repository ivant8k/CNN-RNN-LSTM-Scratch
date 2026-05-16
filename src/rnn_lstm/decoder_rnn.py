import numpy as np
from typing import List, Dict, Any
from rnn_lstm.layers.embedding import Embedding
from rnn_lstm.layers.simple_rnn import SimpleRNN
from shared.layer import Dense
from shared.activations import Activation

class DecoderRNN:
    def __init__(
        self,
        vocab_size: int, 
        embed_dim: int, 
        hidden_size: int, 
        num_layers: int = 1,
    ) -> None:
        """
        Parameters:
        - vocab_size    : Ukuran total kosakata
        - embed_dim     : Dimensi vektor embedding
        - hidden_size   : Dimensi hidden state RNN
        - num_layers    : Jumlah layer RNN
        """
        self.num_layers = num_layers
        self.activation = Activation()

        # Embedding Layer
        self.embedding = Embedding(vocab_size, embed_dim)

        # Dense Projection Layer
        self.projection_layer = Dense(in_features=2048, out_features=embed_dim)

        # Recurrent Layers
        self.rnn_layers: List[SimpleRNN] = []
        for i in range(num_layers):
            if i == 0:
                self.rnn_layers.append(SimpleRNN(input_dim=embed_dim, hidden_size=hidden_size))
            else:
                self.rnn_layers.append(SimpleRNN(input_dim=hidden_size, hidden_size=hidden_size))

        # Output Layer
        self.output_layer = Dense(in_features=hidden_size, out_features=vocab_size)


    def set_model_params(self, params: Dict[str, Any]) -> None:
        """
        Set bobot dari .weights.h5.
        """
        # Load Embedding
        self.embedding.set_weights(params['embedding'])
        
        # Load Projection
        self.projection_layer.set_weights(params['proj_w'], params['proj_b'])
        
        # Load RNN
        for i in range(self.num_layers):
            self.rnn_layers[i].set_weights(params[f'rnn_{i}'])
            
        # Load Output Dense
        self.output_layer.set_weights(params['out_w'], params['out_b'])


    def forward(self, image_features: np.ndarray, caption_tokens: np.ndarray) -> np.ndarray:
        """
        Forward propagation untuk satu batch data
        - image_features: Fitur hasil ekstraksi CNN (batch_size, 2048)
        - caption_tokens: Indeks token caption      (batch_size, seq_len)
        """
        # Proyeksi Fitur Gambar
        img_x = self.projection_layer.forward(image_features)
        img_x = self.activation.relu(img_x)             # (batch_size, embed_dim)
        img_x = np.expand_dims(img_x, axis=1)           # (batch_size, 1, embed_dim)

        # Word Embedding
        word_x = self.embedding.forward(caption_tokens) # (batch_size, seq_len, embed_dim)

        # Pre-Injection
        x = np.concatenate([img_x, word_x], axis=1)     # (batch_size, seq_len + 1, embed_dim)

        # Pass through RNN
        for rnn_layer in self.rnn_layers:
            x = rnn_layer.forward(x)

        # Slicing Output -> ambil hasil timestep teks
        x_words = x[:, 1:, :]                           # (batch_size, seq_len, hidden_size)

        # Dense & Softmax
        logits = self.output_layer.forward(x_words)
        probs = self.activation.softmax(logits)         # (batch_size, seq_len, vocab_size)

        return probs