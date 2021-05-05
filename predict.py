import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense , Input , LSTM , Embedding, Dropout , Activation, Flatten
from tensorflow.keras.layers import Bidirectional, GlobalMaxPool1D, SpatialDropout1D
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import initializers, regularizers, constraints, optimizers, layers

def tokenize(sentence):
  max_features = 6000
  tokenizer = Tokenizer(num_words=max_features)
  tokenizer.fit_on_texts(list(train_reviews))
  list_tokenized_test = tokenizer.texts_to_sequences(sentence)
  max_length = 360
  idv_test = pad_sequences(list_tokenized_test, maxlen=max_length)

  return idv_test

class Model_C():
  def __new__(self):
    embed_size = 128
    model = Sequential()
    model.add(Embedding(6000, embed_size))
    model.add(Bidirectional(LSTM(32, return_sequences = True)))
    model.add(GlobalMaxPool1D())
    model.add(Dense(20, activation="relu"))
    model.add(Dropout(0.05))
    model.add(Dense(1, activation="sigmoid"))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model
    
model_c = Model_C()

model_c.load_weights('./cp-0002.ckpt')

tokenized_word = tokenize('I hate this movie')
prediction = model_c.predict(tokenized_word)
y_pred = (prediction > 0.5)