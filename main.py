from sklearn.model_selection import train_test_split
from load_tweets_data import load_tweets_data, EMBEDDING_DIM
from keras.models import Sequential
from keras.layers import Embedding
from keras.layers.core import Dense
from keras.layers.recurrent import LSTM

tweets, tweets_preprocessed, labels, word_index, embedding_matrix = load_tweets_data()

# TODO try different portion
TRAIN_PERCENTAGE = 0.8
TEST_PERCENTAGE = 0.2

# Training data size, number of tweets, and label size
print("Training Data Size: ", tweets_preprocessed.shape)
print("Number of Tweets: ", tweets_preprocessed.shape[0])
print("Max Tweet Length: ", tweets_preprocessed.shape[1])
print("Labels Size: ", labels.shape)

print("---- Split data into training and testing data ----")
# Ratio (Train : Validate : Test) =  3 : 1 : 1
# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(tweets_preprocessed, labels, train_size=TRAIN_PERCENTAGE,
                                                    test_size=TEST_PERCENTAGE, random_state=42, shuffle=True)
# Split train into train and data for validation
X_train, X_validate, y_train, y_validate = train_test_split(X_train, y_train, train_size=0.75,
                                                            test_size=0.25, random_state=42, shuffle=True)

# Construct model
print("---- Construct Model ----")
model = Sequential()
model.add(Embedding(len(word_index) + 1,
                    EMBEDDING_DIM,
                    weights=[embedding_matrix],
                    input_length=tweets_preprocessed.shape[1]))

model.add(LSTM(EMBEDDING_DIM, dropout=0.2, recurrent_dropout=0.2, return_sequences=True))

model.add(LSTM(EMBEDDING_DIM, dropout=0.5, recurrent_dropout=0.5))

model.add(Dense(3, activation="softmax"))

print(model.summary())

LOSS = 'categorical_crossentropy'
OPTIMIZER = 'RMSprop'  # RMSprop tends to work well for recurrent models

print("---- Compile model ----")
model.compile(loss=LOSS, optimizer=OPTIMIZER, metrics=['accuracy'])

# Train the model
EPOCHS = 8
BATCH_SIZE = 2048

print("---- Train the model ----")
model.fit(X_train, y_train,
          epochs=EPOCHS,
          batch_size=BATCH_SIZE,
          validation_data=(X_validate, y_validate))

print("---- Model Evaluation ----")
result = model.evaluate(X_test, y_test, batch_size=BATCH_SIZE)
print("Final test cost/loss: ", result[0])
print("Final test accuracy: ", result[1])
