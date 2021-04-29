import nltk.classify.util
from nltk.corpus import movie_reviews
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfTransformer,CountVectorizer
from sklearn.linear_model import LogisticRegression
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import random
import string
import re
import gensim
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from gensim.models import Word2Vec
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.layers.embeddings import Embedding





lemm = WordNetLemmatizer()
st = PorterStemmer()
stops = set(stopwords.words("english"))

def cleanData(text):

    txt = str(text)
    txt = re.sub(r'[^a-zA-Z. ]+|(?<=\\d)\\s*(?=\\d)|(?<=\\D)\\s*(?=\\d)|(?<=\\d)\\s*(?=\\D)',r'',txt)
    txt = re.sub(r'\n',r' ',txt)
    txt = " ".join([w.lower() for w in txt.split()])
    txt = " ".join([w for w in txt.split() if w not in stops])
    #txt = " ".join([st.stem(w) for w in txt.split()])
    txt = " ".join([lemm.lemmatize(w) for w in txt.split()])
    return txt

documents = [(cleanData(movie_reviews.sents(fileid)), 1 if category=='pos' else 0) for category in movie_reviews.categories() for fileid in movie_reviews.fileids(category)]

X,y=[],[]

for doc in documents:
    X.append(doc[0])
    y.append(int(doc[1]))

# train data and test data split into 90% and 10% respectively
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.1,random_state=42)

X_train_list = [word_tokenize(sent) for sent in X]
#word2vec model training using gensim lib
word2vec_model = Word2Vec(X_train_list,min_count=1,vector_size=300)
word2vec_model.train(X_train_list,total_examples=word2vec_model.corpus_count,epochs=30)
print(word2vec_model)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_test)
sequences = tokenizer.texts_to_sequences(X_test)
word_index = tokenizer.word_index

max_length = 0
for review_number in range(len(sequences)):
    numberofwords = len(sequences[review_number])
    if(numberofwords > max_length):
        max_length = numberofwords
print(max_length)

data = pad_sequences(sequences, maxlen=max_length)
y_test = np.asarray(y_test)

print(data.shape, y_test.shape)

unique_words = len(word_index)
total_words = unique_words+1
skipped_words = 0
embedding_dim = 300
embedding_matrix = np.zeros((total_words,embedding_dim))

#converting testdata to vector using trained word2vec model
for word, index in tokenizer.word_index.items():
    try:
        embedding_vector = word2vec_model.wv[word]
    except:
        skipped_words= skipped_words+1
        pass
    if embedding_vector is not None:
        embedding_matrix[index] = embedding_vector


print(embedding_matrix.shape,embedding_vector.shape)


embedding_layer = Embedding(total_words, embedding_dim,weights=[embedding_matrix], input_length=max_length, trainable=False)
model = tf.keras.models.Sequential()
model.add(embedding_layer)
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(5, activation='relu'))
model.add(tf.keras.layers.Dense(5, activation='relu'))
model.add(tf.keras.layers.Dense(5, activation='relu'))
model.add(tf.keras.layers.Dense(5, activation='relu'))
model.add(tf.keras.layers.Dense(5, activation='relu'))
model.add(tf.keras.layers.Dense(5, activation='relu'))
model.add(tf.keras.layers.Dense(5, activation='relu'))
model.add(tf.keras.layers.Dense(5, activation='relu'))
model.add(tf.keras.layers.Dense(5, activation='relu'))
model.add(tf.keras.layers.Dense(5, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
model.summary()

model.compile(optimizer='adam', loss='binary_crossentropy', metrics = ['accuracy'])
model.fit(data,y_test,epochs=10, batch_size=128,verbose=1)
