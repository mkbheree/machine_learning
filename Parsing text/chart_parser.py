import nltk
import sys
grammar = nltk.CFG.fromstring("""
  S  -> NP VP | AUX NP VP
  NP ->  NP PP | NP NP | DT NP | NP CC NP | CD N | PRP | N | NNP | EX
  VP -> V NP | V VP | V | TO V PP | ε NP
  PP -> P NP
  PRP -> 'I'
  PropN -> 'Buster' | 'Chatterer' | 'Joe'
  DT -> 'an' | 'a'
  NNP -> 'Calgary' | 'Dubai' | 'Mumbai' | 'Emirates'
  V ->  'need'  | 'would' | 'like' | 'fly' | 'have'
  P -> 'between' | 'from' | 'to' | 'on'
  N -> 'return' | 'flight' | 'am'
  CC -> 'and'
  TO -> 'to'
  AUX -> 'Is' | 'Does'
  EX -> 'there'
  CD -> '5:00' | '6:00'
  ε -> 'empty'
  """)

sys.setrecursionlimit(15000)
sentence =  "Does Emirates have a flight between 5:00 am and 6:00 am from Dubai to Mumbai ".split()
def parse(sent):
    #Returns nltk.Tree.Tree format output
    a = []
    parser = nltk.ChartParser(grammar)
    for tree in parser.parse(sent):
        a.append(tree)
    return(a[0])

#Gives output as structured tree
print(parse(sentence))
