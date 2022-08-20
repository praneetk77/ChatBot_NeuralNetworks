import random

import nltk
nltk.download('punkt')
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
import pickle

import json
import tflearn
import numpy as np
import tensorflow


with open("intents.json") as file:
    data = json.load(file)

try:
    with open("data.pickle","rb") as f:
        words, labels, training, output = pickle.load(f)
except:
    words = []
    labels = []
    docs_x = []
    docs_y = []

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])

        if(intent["tag"] not in labels):
            labels.append(intent["tag"])

    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
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

    with open("data.pickle","wb") as f:
        pickle.dump((words, labels, training, output), f)



net = tflearn.input_data(shape=[None,len(training[0])])
net = tflearn.fully_connected(net,8)
net = tflearn.fully_connected(net,8)
net = tflearn.fully_connected(net,len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

try:
    model.load("model.tflearn")
    print("herwerwerwer")
except:
    model.fit(training, output,n_epoch=1000, batch_size=8, show_metric=True)
    model.save("model.tflearn")


def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(w.lower()) for w in s_words]

    for i,word in enumerate(words):
        if(word in s_words):
            bag[i] = 1

    return np.array(bag)

def chat():
    print("Start talking with the bot:-")
    while(True):
        inp = input("You : ")
        if(inp.lower()=="quit"): break

        result = model.predict([bag_of_words(inp,words)])[0]
        if(result[np.argmax(result)]>0.7):
            tag = labels[np.argmax(result)]
            for tg in data["intents"]:
                if(tg["tag"]==tag):
                    responses = tg["responses"]
                    print(random.choice(responses))
        else:
            print("I didn't get that, try again.")


chat()