import tensorflow as tf


def Load_Inception():
    image_model = tf.keras.applications.InceptionV3(include_top = False ,
                                               weights = 'imagenet')

    new_input = image_model.input
    hidden_layer = image_model.layers[-1].output

    image_features_extract_model = tf.keras.Model(new_input , hidden_layer)
    return image_features_extract_model