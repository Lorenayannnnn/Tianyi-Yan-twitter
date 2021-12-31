import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

EMBEDDING_DIM = 50

def load_tweets_data():
    # Load tweets data
    tweets_data = pd.read_csv("data.csv")

    # 1. Load tweets and their corresponding valence
    print("---- Loading tweets and labels ----")
    tweets = tweets_data['tweet'].tolist()
    labels = np.array(tweets_data['valence'].tolist())

    # 2. Tokenize the tweets
    print("---- Tokenize the tweets ----")
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(tweets)
    sequences = tokenizer.texts_to_sequences(tweets)
    word_index = tokenizer.word_index

    # 3. Pad the sequences to make all tweets having the same size
    print("---- Pad the sequences to make all tweets having the same size ----")
    training_data = pad_sequences(sequences)

    # 4. Load pretrained word embeddings
    print("---- Load pre-trained word embeddings: GloVe embedding is used ----")
    embeddings_index = {}
    f = open('glove.6B.txt', 'rb')
    for line in f:
        values = line.split()
        word = values[0].decode('UTF-8')
        coef = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coef
    f.close()

    # Get word embeddings for all words in the tweets
    print("---- Get word embeddings for all words in the tweets ----")
    # prepare word embedding matrix
    num_words = len(word_index) + 1
    embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
    for word, i in word_index.items():
        if i >= num_words:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    print("---- One-hot encoding ----")
    # One-hot encoding of valence labels

    # All labels of valence
    all_labels = [0, 2, 4]

    mapping = {}
    for x in range(len(all_labels)):
        mapping[all_labels[x]] = x

    for x in range(len(labels)):
        labels[x] = mapping[labels[x]]

    labels_to_categorical = to_categorical(labels)

    print("labels to categorical example: ")
    print(labels[1599999], labels_to_categorical[1599999])

    return tweets, training_data, labels_to_categorical, word_index, embedding_matrix
