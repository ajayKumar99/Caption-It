import tensorflow as tf
from Model.Attention import Attention



class RNN_Decoder(tf.keras.Model):
  def __init__(self , embedding_dim , units , vocab_size):
    super(RNN_Decoder , self).__init__()
    self.units = units
    self.embedding = tf.keras.layers.Embedding(vocab_size , embedding_dim)
    self.gru = tf.keras.layers.GRU(self.units , return_sequences = True , return_state = True , recurrent_initializer = 'glorot_uniform')
    self.fc1 = tf.keras.layers.Dense(self.units)
    self.fc2 = tf.keras.layers.Dense(vocab_size)
    self.attention = Attention(self.units)
    
  def call(self , x , features , hidden):
    context_vector , attention_weights = self.attention(features , hidden)
    x = self.embedding(x)
    x = tf.concat([tf.expand_dims(context_vector , 1) , x] , axis = -1)
    output , state = self.gru(x)
    x = self.fc1(output)
    x = tf.reshape(x , (-1 , x.shape[2]))
    x = self.fc2(x)

    return x , state , attention_weights

  def reset_state(self , batch_size):
    return tf.zeros((batch_size , self.units))