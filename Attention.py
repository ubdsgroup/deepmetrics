import keras
import numpy as np
from keras.backend import expand_dims, sum
from keras.activations import tanh, softmax

""" Custom layer that implements an attention mechanism, for use with LSTM layers 

@:param units:  The number of units in the layer. Should be set equal to the
                window size used by the preceding LSTM layer
                
@:note:         Input should be a list containing (1) the output of the preceding
                LSTM layer, and (2) the hidden state of the preceding LSTM layer.
"""

class Attention(keras.layers.Layer):

    def __init__(self, units, **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.output_dim = units

    def build(self, input_shape):
        self.W1 = keras.layers.Dense(self.output_dim, name="Attention_W1")
        self.W2 = keras.layers.Dense(self.output_dim, name="Attention_W2")
        self.V = keras.layers.Dense(1, name="Attention_final")
        super().build(input_shape)

    def call(self, inputs, training=None, mask=None):
        # Uses standard LSTM notation: c is output states, h is hidden states
        c = inputs[0]
        h = inputs[1]
        hidden_with_time_axis = expand_dims(h, 1)
        score = tanh(self.W1(c) + self.W2(hidden_with_time_axis))
        attention_weights = softmax(self.V(score), axis=1)
        context_vector = attention_weights * c
        context_vector = sum(context_vector, axis=1, keepdims=False)

        return context_vector

    def compute_output_shape(self, input_shape):
        return input_shape[1]

    def get_config(self):
        base_config = super().get_config()
        config = {'units': self.output_dim}
        return dict(list(base_config.items()) + list(config.items()))
