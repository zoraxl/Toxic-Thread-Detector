from flask import Flask,render_template,url_for,request
import pandas as pd
# Package Imports
import pandas as pd
import nltk
import numpy as np
import matplotlib.pyplot as plt
import json
import myutils
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from tensorflow.keras import regularizers, initializers, optimizers, callbacks
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.utils.np_utils import to_categorical
from tensorflow.keras.layers import *

from keras.utils import CustomObjectScope
from keras.initializers import glorot_uniform
from tensorflow.keras.models import load_model
from keras_preprocessing.text import tokenizer_from_json

# Download Stopwords
nltk.download('stopwords')

# Data Path
dataPath = "../../Data/Toxic-Thread-Detector/jigsaw-toxic"
MAX_NB_WORDS = 100000    # max no. of words for tokenizer
MAX_SEQUENCE_LENGTH = 200 # max length of each entry (sentence), including padding
VALIDATION_SPLIT = 0.2   # data for validation (not used in training)
EMBEDDING_DIM = 100      # embedding dimensions for word vectors (word2vec/GloVe)
GLOVE_DIR = "../../Data/Toxic-Thread-Detector/glove/glove.6B."+str(EMBEDDING_DIM)+"d.txt"
labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
    model = load_model('doc/ltsm_model.h5')

with open('doc/tokenizer.json') as f:
    data = json.load(f)
    tokenizer = tokenizer_from_json(data)

app = Flask(__name__)


@app.route('/')
def home():
    result = [0,0,0,0,0,0]
    my_prediction = dict(zip(labels, result))

    return render_template('home.html',result=my_prediction)


@app.route('/predict', methods=['POST'])
def predict():

    comments = []
    if request.method == 'POST':
        message = request.form['message']
        data = pd.Series(message)
        vect = myutils.preprocessing_test(data, tokenizer)
        pred = model.predict(vect)
        result = []
        for p in pred:
            result.append(round(float(p[0]),4))
        my_prediction = dict(zip(labels, result))

    for l, p in my_prediction.items():
        if p > 0.5:
            comments.append(l)

    return render_template('home.html', result=my_prediction, comments = comments)


if __name__ == '__main__':
    app.run(debug=True)