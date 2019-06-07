import tensorflow as tf



def restore_model(checkpoint_path , encoder , decoder , optimizer):
    ckpt = tf.train.Checkpoint(encoder = encoder,
                          decoder = decoder,
                          optimizer = optimizer)

    ckpt.restore(tf.train.latest_checkpoint(checkpoint_path))
