from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk.corpus import movie_reviews
import math

def term_frequency(f_word,num_words):
    return math.log((f_word/num_words) + 1)

def inv_doc_frequency(word):
    return math.log(num_docs/fdist[word])


def cosineSimlarity(tfidf_1,tfidf_2):
    dotProduct=squares_1=squares_2=0
    for i in range(len(tfidf_1)):
         dotProduct+= tfidf_1[i]*tfidf_2[i]
         squares_1+= tfidf_1[i]*tfidf_1[i]
         squares_2+= tfidf_2[i]*tfidf_2[i]
    denominator = math.sqrt(squares_1) * math.sqrt(squares_2)
    return dotProduct/denominator

fdist = FreqDist()
num_docs = 2

def avg_cosine_similarity(category):
    sum=count=0
    for i in range(0,len(movie_reviews.fileids(category)),2):
        doc_1_tokens = movie_reviews.words(movie_reviews.fileids(category)[i])
        doc_2_tokens = movie_reviews.words(movie_reviews.fileids(category)[i+1])
        vocab = set()
        for doc in doc_1_tokens,doc_2_tokens:
            for word in doc:
                fdist[word]+=1
                vocab.add(word)

        num_words_doc1 = len(doc_1_tokens)
        num_words_doc2 = len(doc_2_tokens)

        fdist_1 = FreqDist(doc_1_tokens)
        fdist_2 = FreqDist(doc_2_tokens)

        tfidf_1 = []
        tfidf_2 =[]

        for word in vocab:
            tfidf_1.append(term_frequency(fdist_1[word],num_words_doc1) * inv_doc_frequency(word))
            tfidf_2.append(term_frequency(fdist_2[word],num_words_doc2) * inv_doc_frequency(word))
        sum+=cosineSimlarity(tfidf_1,tfidf_2)
        count+=1
        return sum/count

print("Avg Cosine Simlarity for postive movie reviews",avg_cosine_similarity("pos") )
print("Avg Cosine Simlarity for negative movie reviews",avg_cosine_similarity("neg") )
