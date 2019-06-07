import tensorflow as tf


def Tokenizer(train_captions):
    top_k = 6000
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words = top_k,
                                                    oov_token = "<unk>",
                                                    filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')

    tokenizer.fit_on_texts(train_captions)
    tokenizer.word_index['<pad>'] = 0
    tokenizer.index_word[0] = '<pad>'
    train_seqs = tokenizer.texts_to_sequences(train_captions)
    cap_vector = tf.keras.preprocessing.sequence.pad_sequences(train_seqs , padding = 'post')

    return tokenizer
