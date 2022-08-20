import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import json
import tflearn
import numpy
import tensorflow


with open("intents.json") as file:
    data = json.load(file)

print(data["intents"])

words = []
labels = []
docs = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        wrds = nltk.word_tokenize(pattern)
        words.extend(wrds)
        docs.append(pattern)

    if(intent["tag"] not in labels):
        labels.append(intent["tag"])