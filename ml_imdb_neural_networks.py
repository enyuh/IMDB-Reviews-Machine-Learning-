#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import re
import nltk
import os
import seaborn as sns
from nltk.corpus import stopwords
from numpy import array
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers.core import Activation, Dropout, Dense
from keras.layers import Flatten
from keras.layers import Conv1D
from keras.layers import GlobalMaxPooling1D
from keras.layers.embeddings import Embedding
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import LSTM


# In[2]:


import tensorflow as tf
tf.version.VERSION


# In[3]:


import keras
keras.__version__


# In[4]:


def preprocess_text(sen):
    # Removing html tags
    sentence = remove_tags(sen)

    # Remove punctuations and numbers
    sentence = re.sub('[^a-zA-Z]', ' ', sentence)

    # Single character removal
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)

    # Removing multiple spaces
    sentence = re.sub(r'\s+', ' ', sentence)

    return sentence
TAG_RE = re.compile(r'<[^>]+>')

def remove_tags(text):
    return TAG_RE.sub('', text)


# In[5]:


path1 = "C:/Users/enyu8/Desktop/Angus/Course Material/Comp273A Machine Learning/project/aclImdb/train/neg"
path2 = "C:/Users/enyu8/Desktop/Angus/Course Material/Comp273A Machine Learning/project/aclImdb/train/pos"
path3 = "C:/Users/enyu8/Desktop/Angus/Course Material/Comp273A Machine Learning/project/aclImdb/test/neg"
path4 = "C:/Users/enyu8/Desktop/Angus/Course Material/Comp273A Machine Learning/project/aclImdb/test/pos"


# In[6]:


#train data
all_files = os.listdir(path1)
all_files.sort(key=lambda x:int(x[:-4])) # let the text file been read in sequence
x = []
y = []
for file in all_files: # store negative review into x and label as 0 in ya
    with open(os.path.join(path1, file),'r',encoding="utf8") as f:
        x.append(preprocess_text(f.read()))
        y.append(0)
        
all_files = os.listdir(path2) # store negative review into x and label as 1 in y
all_files.sort(key=lambda x:int(x[:-4])) # let the text file been read in sequence
for file in all_files:
    with open(os.path.join(path2, file),'r',encoding="utf8") as f:
        x.append(preprocess_text(f.read()))
        y.append(1)


# In[7]:


#test data
all_files = os.listdir(path3)
all_files.sort(key=lambda x:int(x[:-4])) # let the text file been read in sequence
x_predict = []
y_predict = []
for file in all_files: # store negative review into x and label as 0 in ya
    with open(os.path.join(path3, file),'r',encoding="utf8") as f:
        x_predict.append(preprocess_text(f.read()))
        y_predict.append(0)
        
all_files = os.listdir(path4) # store negative review into x and label as 1 in y
all_files.sort(key=lambda x:int(x[:-4])) # let the text file been read in sequence
for file in all_files:
    with open(os.path.join(path4, file),'r',encoding="utf8") as f:
        x_predict.append(preprocess_text(f.read()))
        y_predict.append(1)


# In[8]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(x_train)

x_train = tokenizer.texts_to_sequences(x_train)
x_test = tokenizer.texts_to_sequences(x_test)
x_predict = tokenizer.texts_to_sequences(x_predict) # create x_predict to store test data for prediction 


# In[30]:


# Adding 1 because of reserved 0 index
vocab_size = len(tokenizer.word_index) + 1

maxlen = 100

x_train = pad_sequences(x_train, padding='post', maxlen=maxlen)
x_test = pad_sequences(x_test, padding='post', maxlen=maxlen)
x_predict = pad_sequences(x_predict, padding='post', maxlen=maxlen)


# In[11]:


from numpy import array
from numpy import asarray
from numpy import zeros

embeddings_dictionary = dict()
glove_file = open('C:/Users/enyu8/Desktop/Angus/Course Material/Comp273A Machine Learning/project/glove.6B.200d.txt', encoding="utf8")

for line in glove_file:
    records = line.split()
    word = records[0]
    vector_dimensions = asarray(records[1:], dtype='float32')
    embeddings_dictionary [word] = vector_dimensions 
glove_file.close()


# In[36]:


embedding_matrix = zeros((vocab_size, 200))
for word, index in tokenizer.word_index.items():
    embedding_vector = embeddings_dictionary.get(word)
    if embedding_vector is not None:
        embedding_matrix[index] = embedding_vector


# In[13]:


#CNN
model = Sequential()
embedding_layer = Embedding(vocab_size, 200, weights=[embedding_matrix], input_length=maxlen , trainable=False)
model.add(embedding_layer)
model.add(Conv1D(200, 5, activation='relu'))
model.add(GlobalMaxPooling1D())
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])


# In[14]:


print(model.summary())


# In[15]:


x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)

history = model.fit(x_train, y_train, batch_size=128, epochs=6, verbose=1, validation_split=0.2)
score = model.evaluate(x_test, y_test, verbose=1)
print("CNN Validation Loss:", score[0])
print("CNN Validation Accuracy:", score[1])
    


# In[16]:


# Use the model above to predict test data
prediction = model.predict(x_predict)
x_predict = np.array(x_predict)
y_predict = np.array(y_predict)
# If value greater than 0.5, classify it as positive, vice versa
for i in range(len(prediction)):
    if prediction[i] > 0.5:
        prediction[i] = 1
    elif prediction[i] <= 0.5:   
        prediction[i] = 0
prediction = np.reshape(prediction,25000)
prediction = prediction.astype(int)


# In[17]:


# Create a csv file to store y_predict and y_truth for confusion matrix
import pandas as pd
predictions_df = pd.DataFrame()
predictions_df['y_prediction'] = prediction
predictions_df['y_truth'] = y_predict
predictions_df.to_csv('predictions_CNN.csv')


# In[18]:


import matplotlib.pyplot as plt

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','test'], loc = 'upper left')
plt.show()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','test'], loc = 'upper left')
plt.show()


# In[39]:


# RNN
model = Sequential()
embedding_layer = Embedding(vocab_size, 200, weights=[embedding_matrix], input_length=maxlen , trainable=False)
model.add(embedding_layer)
model.add(LSTM(128))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])


# In[34]:


print(model.summary())


# In[37]:


x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)
history = model.fit(x_train, y_train, batch_size=128, epochs=6, verbose=1, validation_split=0.2)
score = model.evaluate(x_test, y_test, verbose=1)
print("RNN Validation Loss:", score[0])
print("RNN Validation Accuracy:", score[1])


# In[22]:


# Use the model above to predict test data
prediction = model.predict(x_predict)
x_predict = np.array(x_predict)
y_predict = np.array(y_predict)
# If value greater than 0.5, classify it as positive, vice versa
for i in range(len(prediction)):
    if prediction[i] > 0.5:
        prediction[i] = 1
    elif prediction[i] <= 0.5:   
        prediction[i] = 0
prediction = np.reshape(prediction,25000)
prediction = prediction.astype(int)


# In[23]:


# Create a csv file to store y_predict and y_truth for confusion matrix
import pandas as pd
predictions_df = pd.DataFrame()
predictions_df['y_prediction'] = prediction
predictions_df['y_truth'] = y_predict
predictions_df.to_csv('predictions_RNN.csv')


# In[24]:


import matplotlib.pyplot as plt
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','test'], loc='upper left')
plt.show()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss') 
plt.xlabel('epoch')
plt.legend(['train','test'], loc='upper left')
plt.show()

