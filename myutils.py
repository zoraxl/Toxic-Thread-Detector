import re
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import *



def clean_text(text, remove_stopwords = True):
    output = ""
    text = str(text).replace("\n", "")
    text = re.sub(r'[^\w\s]','',text).lower()
    if remove_stopwords:
        text = text.split(" ")
        for word in text:
            if word not in stopwords.words("english"):
                output = output + " " + word
    else:
        output = text
    return str(output.strip())[1:-3].replace("  ", " ")

def preprocessing_train(test):
    MAX_NB_WORDS = 100000    # max no. of words for tokenizer
    MAX_SEQUENCE_LENGTH = 200 # max length of each entry (sentence), including padding
    texts = []
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    for line in test:
        texts.append(clean_text(line))
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    data = pad_sequences(sequences, padding = 'post', maxlen = MAX_SEQUENCE_LENGTH)
    return data, tokenizer

def preprocessing_test(test, tokenizer):
    MAX_SEQUENCE_LENGTH = 200 # max length of each entry (sentence), including padding
    texts = []

    for line in test:
        texts.append(clean_text(line))
    sequences = tokenizer.texts_to_sequences(texts)
    data = pad_sequences(sequences, padding='post', maxlen = MAX_SEQUENCE_LENGTH)
    return data
