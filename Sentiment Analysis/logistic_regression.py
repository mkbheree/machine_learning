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
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords


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
#random.shuffle(documents)
X,y=[],[]

for doc in documents:
    X.append(doc[0])
    y.append(int(doc[1]))

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3,random_state=10)


logistic_reg = Pipeline([('vect', CountVectorizer(binary=True, max_df=1.0,ngram_range=(1,2))),
                         ('tfidf', TfidfTransformer()),
                         ('clf', LogisticRegression( solver='saga',max_iter=500)),
                        ])
logistic_reg.fit(X_train, y_train)


y_pred = logistic_reg.predict(X_test)

print('Accuracy', accuracy_score(y_pred, y_test)*100,"%")
