import tensorflow as tf
import numpy as np
import Model.load_image as li



def evaluate(image_path , max_length , attention_features_shape , encoder , decoder , image_features_extract_model , tokenizer):
    
    attention_plot = np.zeros((max_length , attention_features_shape))

    
    
    hidden = decoder.reset_state(batch_size = 1)
    
    temp_input = tf.expand_dims(li.load_image(image_path)[0] , 0)
    img_tensor_val = image_features_extract_model(temp_input)
    img_tensor_val = tf.reshape(img_tensor_val , (img_tensor_val.shape[0] , -1 , img_tensor_val.shape[3]))
    
    features = encoder(img_tensor_val)
    
    dec_input = tf.expand_dims([tokenizer.word_index['<start>']] , 0)
    result = []
    
    for i in range(max_length):
        predictions , hidden , attention_weights = decoder(dec_input , features , hidden)
        
        attention_plot[i] = tf.reshape(attention_weights , (-1, )).numpy()
        
        predicted_id = tf.argmax(predictions[0]).numpy()
        result.append(tokenizer.index_word[predicted_id])
        
        if tokenizer.index_word[predicted_id] == '<end>' :
            return result , attention_plot
        
        dec_input = tf.expand_dims([predicted_id] , 0)
        
    attention_plot = attention_plot[:len(result) , :]
    
    return result , attention_plot