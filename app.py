from flask import Flask , render_template , url_for , redirect , request , flash
import os
import tensorflow as tf
import Model.Evaluate as evl
import Model.Tokenizer as tk
import Model.Inception as inp
import Model.load_image as li
import Model.caption_extract as ce
import Model.Weight_Restore as wr
from Model.CNN_Encoder import CNN_Encoder
from Model.RNN_Decoder import RNN_Decoder

app = Flask(__name__)
APP_ROOT = os.path.dirname(os.path.abspath(__file__))

train_captions = ce.Caption_Extract()
image_features_extract_model = inp.Load_Inception()
tokenizer = tk.Tokenizer(train_captions)
max_length = 51
attention_features_shape = 64
embedding_dim = 256
units = 512
vocab_size = len(tokenizer.word_index) + 1

encoder = CNN_Encoder(embedding_dim)
decoder = RNN_Decoder(embedding_dim , units , vocab_size)
optimizer = tf.keras.optimizers.Adam()

checkpoint_path = 'Model/checkpoints/train'
wr.restore_model(checkpoint_path , encoder , decoder , optimizer)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/caption' , methods = ['GET' , 'POST'])
def caption():
    target = os.path.join(APP_ROOT , 'static/')
    if not os.path.isdir(target):
        os.mkdir(target)
    filename =""
    for file in request.files.getlist('file'):
        filename = file.filename
        if filename == "":
            flash('Please input an image...')
            return redirect(url_for('index'))
        dest = '/'.join([target , filename])
        file.save(dest)

    image_path = 'static/'+filename
    result , attention_plot = evl.evaluate(image_path , max_length , attention_features_shape , encoder , decoder , image_features_extract_model , tokenizer)
    output = ' '.join(result[:-1])
    return render_template('caption.html' , output=output , image_path=image_path)


if __name__ == '__main__':
    app.run(debug=False)