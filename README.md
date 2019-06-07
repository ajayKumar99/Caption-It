# Caption It
This is an automatic image captioner which gives captions to images taking into account the pic's subject and attention also into account.
It achieves this using a GRU Recurrent Network to generate setences and a pre-trained Inception model to classify objects in the image.
## Architecture
![Network Structure](https://github.com/Hvass-Labs/TensorFlow-Tutorials/raw/aa0d6796c6bb61a4c81ab1f8d0dc425cc034095e/images/22_image_captioning_flowchart.png)
* Firstly, Transfer Learning is applied to a pre-trained inception model to classify several objects inside the image. These classified objects accounts to the subject recognition of the image.
* The transfer learning is achieved by omitting the top fully-connected soft-max layer and taking the bottom features, which is then fed into the RNN Decoder.
* Next, a Recurrent Neural Network with GRUs(Gated Recurrent Units) is used which takes in the features extracted from Inception and the initial sentence tokens and generates output sentences based on the attention in the image.
* Attention is implemented using a Bahdanau style (additive) attention.

## Dataset
* The [MS COCO](http://images.cocodataset.org/annotations/annotations_trainval2014.zip) dataset has been used.
* The dataset contains over 82,000 images, each of which has at least 5 different caption annotations.

## Dependencies
Tensorflow 2.0.0 alpha has been used to train the model. Further , a Flask app has been made to deploy the model into a web app.
* Tensorflow
* Flask

All the dependencies are in the requirements.txt file with the versions used.
A virtual environment is recommended, the following command can be used to build a virtual environment in the project root folder.
```
python -m venv env
```

Then , all the dependencies can be installed from the requirements.txt file using the following command.
```
pip install -r requirements.txt
```

## Training the model
The model was built using tensorflow on Google Colab which provides free K80 GPU support for training the model. 
* All the network models have been split and made into different class objects.
* Checkpoints are saved at every 5 epochs and the last epoch weights were saved and extracted along with train captions data to be loaded while inferring.
* An Adam Optimizer has been used.
* Sparse categorical crossentropy loss object has been used with custom loss-function.
All the training details and code can be found in this jupyter notebook :- [Caption It Training](https://github.com/ajayKumar99/Caption-It/blob/master/training/I_Caption.ipynb) 

### Post Training
* After training the model, the saved checkpoints folder should be placed inside the Model folder.
* Also, the training captions saved, training_captions.npy, should be kept inside the checkpoint folder inside Model.

## APIs Used
* [Material Design Lite](https://getmdl.io/started/index.html#download) - A light weight material design ui for static html pages. 

## Setting up the app
* Install all the dependencies as mentioned above.
### Running the app in debug mode
* The flask server has to be restarted everytime a change is made to see the changes. So, the server can be used in debug mode which enables auto-restarting of server whenever a change is reflected on any file.
Change the app.run() as follows:
```
app.run(debug=True)
```
### Starting the server
* The flask server can be started by the following command:
```
python app.py
```
All the details of the host address(http://127.0.0.1:5000) will be given just go to the address and the app can be used.
