import tensorflow as tf



class CNN_Encoder(tf.keras.Model):
  def __init__(self , embedding_dim):
    super(CNN_Encoder , self).__init__()
    self.fc = tf.keras.layers.Dense(embedding_dim)
    
  def call(self , x):
    x = self.fc(x)
    x = tf.nn.relu(x)
    return x