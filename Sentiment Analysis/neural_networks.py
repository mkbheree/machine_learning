import nltk.classify.util
from nltk.corpus import movie_reviews
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.neural_network import MLPClassifier
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



X_list = [word_tokenize(sent) for sent in X]
#word2vec model training using gensim lib
word2vec_model = Word2Vec(X_list,min_count=1,vector_size=300)
word2vec_model.train(X_list,total_examples=word2vec_model.corpus_count,epochs=30)
print(word2vec_model)


word2vecList = []
for sent in X:
    vec_list = []
    for word in sent:
        if word in word2vec_model.wv.index_to_key:
            vec_list.append(word2vec_model.wv[word])
        else:
            vec_list.append(np.zeros(300))
    vec_list = np.array(vec_list).mean(axis=0)
    word2vecList.append(vec_list)


print(len(word2vecList),len(y))

# train data and test data split into 90% and 10% respectively
X_train, X_test, y_train, y_test = train_test_split(word2vecList, y,test_size=0.1,random_state=42)

clf = MLPClassifier(hidden_layer_sizes=(5,5,5,5,5,5,5,5,5,5), max_iter=100000, warm_start=True)
clf.fit(X_train, y_train)

print("Accuracy :"+str(clf.score(X_test, y_test)*100)+"%")
