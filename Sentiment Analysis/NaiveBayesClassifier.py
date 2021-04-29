import nltk.classify.util
import random
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import movie_reviews
from nltk.tokenize import word_tokenize

'''
 all_words = nltk.FreqDist(w.lower() for w in movie_reviews.words())
 f_words = list(all_words)

 def typeCast_input(words):
     word_set = set(words)
     features = {}
     for w in f_words:
             features[w] = (w in word_set)
     return features
'''


#Setting the input as per the classifier arguments
def typeCast_input(words):
    words_dict = dict([(word, True) for word in words])
    return words_dict

#getting movie reviews and setting according to their category('pos','neg')
documents = [(typeCast_input(movie_reviews.words(fileid)), category) for category in movie_reviews.categories() for fileid in movie_reviews.fileids(category)]

#random.shuffle(documents)
print("Length of Movie Review Docs:",len(documents))

#setting training set with 70% i.e 1400 documents out of 2000 docs available and rest to test set
train_set = documents[:1400]
test_set =  documents[1400:]

classifier = NaiveBayesClassifier.train(train_set)
accuracy = nltk.classify.util.accuracy(classifier, test_set)


print("Accuary for Model(with 70% Training & 30% testing set) is",accuracy*100)

#testing the model with pre defined postive & negative movie Reviews

positive_movie_review = """Avengers Assemble' ('The Avengers') is a truly enjoyable superhero film that lives up to its hype and creates a story that allows for four of the greatest superheroes to connect in this mega-blockbuster extravaganza. Joss Whedon has created one of the most action-packed Marvel films to have graced the screen, full of humour, thrills and a great cast of characters, all of which impel this visual effects-driven spectacle. Whilst I had the great opportunity to watch this epic in the cinema in 3D, the film is equally as stunning on an average television set, with the final battle between the Avengers and Loki's army being one of the most spectacular scenes in a superhero movie. An impressive and remarkable fantastical superhero flick from Whedon."""

positive_mr_words = typeCast_input(word_tokenize(positive_movie_review))
print("Testing model with Positive Movie Review: ",classifier.classify(positive_mr_words))

negative_movie_review = """I would give this movie 1 star if it weren't for some good acting by Bryan Cranston in the beginning. Everything about this movie is horrible. I have watched loads and loads of movies from all over the world and this has to be one of the worst. The acting is just terrible and the story is even worse. The plot twists are horrendous. It just didn't make sense that Godzilla swam all the way from japan to San Francisco to fight the MUTOs and then he just swam back after smiling for the cameras. And this movie is rated higher than the movie which came out in 1998. That movie was better off by miles. Some people might have liked it after watching all the CGI in theaters. If you watch it on a small screen you will realize how bad the movie actually is."""

negative_mr_words = typeCast_input(word_tokenize(negative_movie_review))
print("Testing model with Negative Movie Review: ",classifier.classify(negative_mr_words))

#testing the model through random inputs by user
'''
testReview_words = typeCast_input(word_tokenize(input('Please enter the movie review : ')))
print("Sentiment polarity for user test review: ",classifier.classify(testReview_words))
'''
