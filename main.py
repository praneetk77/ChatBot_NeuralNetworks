import nltk
nltk.download('punkt')
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import json
import tflearn
import numpy as np
import tensorflow


with open("intents.json") as file:
    data = json.load(file)

words = []
labels = []
docs_x = []
docs_y = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        wrds = nltk.word_tokenize(pattern)
        words.extend(wrds)
        docs_x.append(pattern)
        docs_y.append(intent["tag"])

    if(intent["tag"] not in labels):
        labels.append(intent["tag"])

words = [stemmer.stem(w.lower()) for w in words]
words = sorted(list(set(words)))

labels = sorted(labels)

training = []
output = []

out_empty = [0 for _ in range(len(labels))]

#creating bag of words and output array for every sentence
for x, doc in enumerate(docs_x):
    bag = []
    wrds = [stemmer.stem(w) for w in doc]

    for w in words:
        if(w in wrds):
            bag.append(1)
        else:
            bag.append(0)

    #copying empty array of len = #tags
    output_row = out_empty[:]
    #making the tag of this sentence 1
    output_row[labels.index(docs_y[x])] = 1

    training.append(bag)
    output.append(output_row)

training = np.array(training)
output = np.array(output)


