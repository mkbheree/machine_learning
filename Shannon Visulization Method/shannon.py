from nltk.corpus import brown
import nltk
from nltk.lm.preprocessing import pad_both_ends
from nltk.tokenize import word_tokenize, sent_tokenize

#Collecting sentences from brown corpus lib
sentences =  brown.sents(categories='news')
print("sentences length", len(sentences))

bigrams = []
trigrams = []

# padding the sentence with <s> at start  & </s> at the and and breaking them into bigrams & trigrams
for sent in sentences:
    bigrams.extend(list(nltk.bigrams(pad_both_ends(sent, n=2))))
    trigrams.extend(list(nltk.trigrams(pad_both_ends(sent, n=3))))

# to record no. of times each sample occured
cfd = nltk.ConditionalFreqDist(bigrams)

def generate_model(cfdist, word, num=30):
    content=" "
    for i in range(num):
        word = cfdist[word].max() # finding next word with high probablity
        if word == '</s>': # stop at end of the Senetence
            break
        content= content + " " + word
    print("Random Senetence using bigrams:", content)
generate_model(cfd,"The")

condition_pairs = (((w0, w1), w2) for w0, w1, w2 in trigrams)
cfd_tri = nltk.ConditionalFreqDist(condition_pairs)

def generate_model_tri(cfdist, word, num=30):
    content=" "
    w1 = str(word[0])
    w2 = str(word[1])
    for i in range(num):
        word = cfdist[w1,w2].max()
        if word == '</s>':
            break
        content= content + " " + word
        temp = w2
        w2 = word
        w1 = temp
    print("Random Senetence using trigrams:", content)
generate_model_tri(cfd_tri,("of","the"))
