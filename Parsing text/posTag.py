import nltk
from nltk.tokenize import word_tokenize,sent_tokenize



text_8 = word_tokenize("I have finished that infernal assignment by midnight")
print(nltk.pos_tag(text_8))
