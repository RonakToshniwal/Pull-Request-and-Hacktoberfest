import pandas as pd
import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split



hello=pd.read_json(path_or_buf="C:\\Users\\asus\\Desktop\\Sarcasm_Headlines_Dataset.json",lines=True)
hello=pd.DataFrame(hello)
print(hello.shape)
train,test=train_test_split(hello,test_size=0.2)

token=Tokenizer(oov_token="<oov>")
token.fit_on_texts(train['headline'])
train_toker=token.texts_to_sequences(train['headline'])
test_talker=token.texts_to_sequences(test["headline"])
train_padded=pad_sequences(train_toker,maxlen=40)
test_padded=pad_sequences(test_talker,maxlen=40)
print(train_padded.shape)
print(test_padded.shape)
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(26709,64,input_length=40),
    tf.keras.layers.Conv1D(32, 5, activation='relu'),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
training_labels_final = np.array(train.is_sarcastic)
testing_labels_final = np.array(test.is_sarcastic)
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()
print(train.columns)
model.fit(train_padded, training_labels_final, epochs=2, validation_data=(test_padded, testing_labels_final))